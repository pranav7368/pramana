# PRAMANA — clone once, build once, restart from Docker

**Host par Python/GPU/local models ki zaroorat nahi hai.** Docker Desktop ka
Linux engine installed/running hona chahiye. First build ko internet aur disk
space chahiye. Live APIs ko internet, valid key aur available quota chahiye.

## First time — Windows

Clone your actual repository, or extract the latest Docker-ready ZIP:

```powershell
git clone YOUR_REPOSITORY_URL pramana
cd pramana
.\docker-run.cmd setup
notepad .env
```

`setup` missing `.env` create karta hai, existing config overwrite nahi karta.
Apni `GOOGLE_API_KEY` daalo, save karo. `GROQ_API_KEY` backup optional hai.
Then:

```powershell
.\docker-run.cmd
```

First run image build karke container create/start karega, health ready hone
tak wait karega, phir URL print karega. Open **http://127.0.0.1:8765/**.
Terminal close kar sakte ho; container background mein chalta rahega.

Default project: **pramana-local**. Container: **pramana-local-pramana-1**.
Fictional EN/HI/TA documents image mein bundled hain, host corpus mount required
nahi hai. Google embeddings + sparse retrieval auto-enable hain with a Google
key; live generation/verification/correction run in the same container.

## Next time — no clone/build/setup again

Docker Desktop open/start karo, then either:

```powershell
.\docker-run.cmd start
```

Or Docker Desktop ke Containers page mein **pramana-local / pramana** ka Start
button click karo. Same container, image aur saved runtime environment reuse
hote hain. Open the same URL. Account access/quota still required.

Stop without deleting the container:

```powershell
.\docker-run.cmd stop
```

More commands:

```powershell
.\docker-run.cmd status
.\docker-run.cmd logs
.\docker-run.cmd restart
.\docker-run.cmd build           # apply changed source after git pull
```

Normal `docker-run.cmd` builds only if the image is missing, otherwise uses it;
`build` explicitly rebuilds. If `.env`/port/key settings change, use the normal
`docker-run.cmd` again to recreate/update configuration. `start`/`restart` reuse
the existing container's stored environment and do NOT apply new `.env` values.

Do not delete the container/image/volume in Docker Desktop if you want reuse.
`docker compose down` removes containers, so next time creation is needed again.
Restart policy is `unless-stopped`: engine restarts can resume a running app,
but a deliberately stopped app must be started manually.

## macOS/Linux or direct Compose

Copy `.env.example` to `.env` **only if missing**, add your key, then:

```sh
docker compose -p pramana-local -f compose.local.yaml up -d --wait
# Later:
docker compose -p pramana-local -f compose.local.yaml stop
docker compose -p pramana-local -f compose.local.yaml start
```

The Windows helper gives extra key/engine checks and health waiting after start;
direct `compose start` does not wait for readiness. Watch status/health before use.
Always specify `compose.local.yaml`; the existing default `compose.yaml` remains
the separately configured authenticated pilot, not this simple live demo.

## Configuration / limits

Optional `.env` values:

```dotenv
PRAMANA_PORT=8765
PRAMANA_SEMANTIC=auto
PRAMANA_DOCKER_MEMORY=768m
```

Port change example: `PRAMANA_PORT=8770`, then normal `docker-run.cmd`.
For embedding API outages, `PRAMANA_SEMANTIC=false` selects sparse retrieval;
the live generation/verification provider still needs access/quota. Groq-only
keys use sparse retrieval. Never silently substitute fixtures as live results.

The container is non-root, read-only, localhost-only, one CPU, 768 MiB cap and
PID limit 128, with capabilities dropped and no-new-privileges. Docker Desktop
itself also uses host RAM; this cap is not a total laptop-memory guarantee.
No local model weights load. Keys are supplied at runtime, not copied into the
image; administrators can still inspect container environment, so do not share
container inspect/export output containing secrets.

**Uploaded PDFs are currently in-memory.** Stop/restart/recreation restores the
bundled sample corpus; upload your approved PDF again. A named state volume exists,
but it does not make the in-memory PDF index persistent. API counters also reset.

This is a local demonstration, not public production hosting. Cloud APIs receive
questions/retrieved text, and Google embeddings receive document text. Use
approved or fictional content. Confidence remains heuristic until fitted.
For approved static private corpus/authenticated hosting, use the separate
[pilot runbook](docs/11_ENTERPRISE_PILOT.md).

`compose.demo.yaml` remains an explicitly offline fixture rehearsal. The new
`compose.local.yaml` is the **live API** easy-launch profile.
