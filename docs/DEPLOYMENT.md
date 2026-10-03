# Deploying the public demo

This guide puts PRAMANA on a public HTTPS URL at no cost, using a free Render web
service and a free Google AI Studio key. A free account on each is all you need;
no credit card is required. The deployed site is the same evidence desk as the
local demo, running in **public mode**. That mode adds the controls an anonymous
audience needs:

| Control | What it does |
|---|---|
| Per-visitor documents | An upload is indexed only for the browser tab that sent it. Other visitors keep the sample policy. Uploads live in memory, expire after 30 idle minutes, and at most 24 sessions are held. |
| Visitor limit | Each visitor gets 10 checks per hour (ask, audit and upload each count as one) and receives HTTP 429 with `Retry-After` beyond that. |
| Daily budget | 150 checks per day across all visitors, reset at 00:00 UTC, so the free API quota cannot be exhausted. |
| Short queue | If two visitors overlap, the second waits up to 20 s for the single inference slot instead of failing at once. |
| Same-origin writes | POST/DELETE requests from other websites are refused, as are requests whose `Host` is not the deployment's own hostname. |
| No disk writes | Prompt caching is disabled, so prompts and uploaded text are never written to disk. |
| Hardened container | Non-root user, no secrets in the image, and a strict Content-Security-Policy on the page. |

The remaining limitations are listed under [Free-tier behaviour](#free-tier-behaviour)
and in [VALIDATION.md](VALIDATION.md).

## Choosing a host (checked October 2026)

| Option | Cost | Fit |
|---|---|---|
| **Render free web service** (recommended) | Free, no card | Docker supported, 512 MB RAM, HTTPS on `*.onrender.com`. Sleeps after 15 idle minutes; wakes in about a minute. 750 free hours a month. |
| Azure for Students | $100 credit, no card, renews yearly while a student | Always-on via Azure Container Apps or App Service; more setup. Uses the same image and `--public` command. |
| Any Docker host / VM | Varies | `docker run` command under [Other hosts](#other-hosts). |
| Hugging Face Docker Spaces | Needs a PRO plan to create | Not free any more for Docker SDK Spaces. |
| Koyeb | Free instance, but a $29 card hold for new users | Works with the same image. |

Free tiers change often. Check the provider's pricing page before relying on
these terms.

## Deploy to Render (about 10 minutes)

1. **Pick the repository.** Render builds from GitHub. Use the public
   repository, <https://github.com/pranav7368/pramana>, or your fork of it. It
   contains `render.yaml`, `Dockerfile` and the source. `.env` is git-ignored
   and must stay that way.
2. **Get an API key.** Create a free key at <https://aistudio.google.com/apikey>.
   A Groq key from <https://console.groq.com/keys> is optional and is used as a
   failover. Never put a key in a file that is committed.
3. **Create the service.** Click the *Deploy to Render* button in the README,
   or in the Render dashboard choose **New → Blueprint** and select the
   repository. Render reads `render.yaml` and asks for `GOOGLE_API_KEY` and
   `GROQ_API_KEY`; leave either blank if you don't have it. Click **Apply**.
4. **Wait for the first build** (about 3–6 minutes). The log ends with
   `PRAMANA public demo on port 8000; hosts: <name>.onrender.com`.
5. **Open the URL** shown on the service page, e.g. `https://pramana.onrender.com`.
   The header shows *Public demo* and *Live · google* (or *Offline · fixtures*
   if no key was supplied).
6. **Verify it.** Run these against your URL:

   ```bash
   curl https://<your-service>.onrender.com/v1/health   # {"status":"ok", ...}
   curl https://<your-service>.onrender.com/v1/ready    # retrieval mode, languages, chunk counts
   ```

   Interactive API documentation is at `/docs`.

Render redeploys automatically after every push to the default branch, once the
GitHub *Checks* workflow has passed (`autoDeployTrigger: checksPass`). A failing
test therefore never reaches the public URL.

### Optional: keep it awake

A free instance sleeps after 15 minutes without traffic, and the next visitor
waits about a minute. To keep it awake before a presentation or interview,
enable the *Public demo uptime* workflow
(`.github/workflows/keep-warm.yml`). In GitHub, go to **Settings → Secrets and
variables → Actions → Variables** and add `PUBLIC_DEMO_URL` with your Render URL.
The workflow pings `/v1/health` every 10 minutes from 08:30 to 00:30 IST and
fails if the demo is down, which doubles as uptime monitoring. GitHub pauses
scheduled workflows after 60 days without repository activity.

### Custom domain

Add the domain in Render (**Settings → Custom Domains**), then set
`PRAMANA_PUBLIC_HOST` to it. To keep the `onrender.com` address working too,
use a comma-separated list, e.g. `pramana.example.com,pramana.onrender.com`.
Requests for any hostname not on the list get HTTP 400.

## Configuration

Public mode reads the same variables as the local demo, plus these. Set them in
the Render dashboard (**Environment**) or in `render.yaml`.

| Variable | Default (public) | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` / `GROQ_API_KEY` | — | Provider keys (secrets). Google also enables semantic search. |
| `PRAMANA_VISITOR_LIMIT` | `10` | Checks per visitor per window; `0` disables. |
| `PRAMANA_VISITOR_WINDOW_S` | `3600` | Length of the visitor window in seconds. |
| `PRAMANA_DAILY_LIMIT` | `150` | Checks per UTC day for all visitors together; `0` disables. |
| `PRAMANA_QUEUE_WAIT_S` | `20` | How long a request waits for the inference slot. |
| `PRAMANA_SESSION_TTL_S` | `1800` | Idle seconds before a visitor's upload is dropped. |
| `PRAMANA_MAX_SESSIONS` | `24` | Uploaded-document sessions held in memory at once. |
| `PRAMANA_PUBLIC_HOST` | platform value | Public hostname(s). Render's `RENDER_EXTERNAL_HOSTNAME` is used automatically. |
| `PRAMANA_CLIENT_IP_HEADER` | `true-client-ip` on Render | Header carrying the visitor address from the platform's proxy. |
| `PRAMANA_LOG_FORMAT` | `json` in `render.yaml` | One JSON line per event with a request id; no prompts or documents. |

**Sizing the daily budget.** One *ask* makes about 4–8 model calls (draft, claim
split, one verification per claim, and any corrections). One *audit* makes
about 2–5. With the configured free quotas (Gemini Flash-Lite about 1,500
requests a day, Groq about 1,000), 150 checks a day leaves headroom. If you only
have one key, or your console shows a lower quota, lower `PRAMANA_DAILY_LIMIT`
to roughly *daily requests ÷ 8*.

## Free-tier behaviour

- **Cold start:** after 15 idle minutes the instance sleeps. The first request
  then waits about a minute, and the page says so if a request times out.
- **Memory only:** uploads and usage counters reset whenever the instance
  restarts, sleeps or redeploys. The sample policy is always available.
- **One inference at a time:** the free instance has a fraction of a CPU. Requests
  queue for up to 20 s, then get HTTP 503 with `Retry-After`.
- **Not for confidential data:** questions and retrieved passages are sent to
  the configured model APIs. The page tells visitors not to upload confidential
  or personal documents.
- **Accuracy:** a public demo does not change the evaluation status in
  [VALIDATION.md](VALIDATION.md). Confidence is heuristic unless a fitted
  calibrator is configured.

## Operating it

| Endpoint | Use |
|---|---|
| `/v1/health` | Liveness. Render's health check uses it. |
| `/v1/ready` | Loaded components: retrieval mode, languages, chunk counts. |
| `/v1/runtime` | Provider counters, checks used today, active upload sessions. |
| `/v1/metrics` | Prometheus text: requests by route and status, latency, router counters. |
| `/docs` | OpenAPI documentation for `/v1/ask`, `/v1/verify` and the rest. |

**Troubleshooting**

| Symptom | Cause and fix |
|---|---|
| Every page returns `Untrusted or malformed request host` | The hostname is not trusted. Set `PRAMANA_PUBLIC_HOST` (needed for custom domains and non-Render hosts). |
| `Answer service unavailable; retry later` | A provider call failed or hit its quota. Check the Render logs and the AI Studio quota page; add `GROQ_API_KEY` as a failover. |
| `Demo limit reached` / `today's shared budget` | The visitor or daily limit is working as designed. Raise it only if your provider quota allows. |
| Header shows *Offline · fixtures* | No key reached the container. Add `GOOGLE_API_KEY` under **Environment** and redeploy. |
| Build fails at `pip install` | Rerun the deploy; if it repeats, check that `requirements-pilot.txt` installs on Python 3.12 locally. |

**If a key leaks**, revoke it in the provider console first, then replace it in
Render. Render does not show secret values after saving them, and PRAMANA never
logs them.

## Other hosts

The image and command are the same everywhere:

```bash
docker build -t pramana .
docker run -d --name pramana -p 80:8000 --read-only --tmpfs /tmp --cap-drop ALL \
  -e PRAMANA_PUBLIC_HOST=demo.example.com \
  -e GOOGLE_API_KEY=... \
  pramana python scripts/serve_demo.py --public
```

Put a TLS-terminating proxy or the platform's HTTPS ingress in front of it. If the
proxy passes the visitor address in a header, name it with
`PRAMANA_CLIENT_IP_HEADER`. Otherwise every visitor shares the proxy's address and
the per-visitor limit applies to the whole site. `PORT` selects the listening
port (default 8000).

To try public mode locally without Docker:

```powershell
$env:PRAMANA_PUBLIC_HOST = "localhost"
python scripts/serve_demo.py --public --port 8780
```
