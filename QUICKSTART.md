# PRAMANA — simple clone, configure, run

**Docker use karna hai?** [Clone/build-once/restart guide](DOCKER_QUICKSTART.md)
use karo: `docker-run.cmd` first time, `docker-run.cmd start` next time. Host
Python needed nahi hai. Neeche wala flow Docker ke bina native Python setup hai.

Windows ke liye sirf **Python 3.12 + Git** chahiye. Docker, GPU, PyTorch aur local
model download zaroori nahi hain. Launcher Python 3.11–3.13 allow karta hai;
3.12 validated hai. First setup Python packages download karta hai.

## 1. Clone / download

Replace `YOUR_REPOSITORY_URL` with your actual repository URL:

```powershell
git clone YOUR_REPOSITORY_URL pramana
cd pramana
```

ZIP download bhi chalega: extract karo aur us folder mein terminal kholo.
Share karne wale repository/ZIP mein ye changes include hone chahiye. Is work
se repository publish ya source upload nahi hua hai.

## 2. Setup + one key

```powershell
.\run.cmd setup
notepad .env
```

Setup `.venv` banata hai, pinned API-only dependencies install karta hai, aur
missing `.env` create karta hai. **Existing configuration overwrite nahi hoti.**

Apni Google API key is line mein daalo, save karo, editor close karo:

```dotenv
GOOGLE_API_KEY=YOUR_REAL_KEY
```

Optional: `GROQ_API_KEY` generation/verification failover ke liye. Google embedding
service separate dependency hai; Groq uska replacement nahi hai. Real key Git,
README, screenshot ya public chat mein mat daalo. Populated `.env` commit mat karo.

## 3. Run the entire local system

```powershell
.\run.cmd
```

Open **http://127.0.0.1:8765/**. Terminal open rakho; **Ctrl+C** se stop karo.
Configuration save karne ke baad `run.cmd` double-click bhi kar sakte ho.
Old `run_demo.cmd` compatible hai.

UI/API, English/Hindi/Tamil sample indexes, Google API semantic + sparse search,
live generation, claim verification, heuristic confidence, citations aur bounded
correction/abstention ek saath start hote hain. Google first, optional Groq backup.
Separate frontend/backend/indexing setup nahi chahiye. Fictional sample documents
automatically load hote hain; approved text PDF/TXT/MD upload optional hai.

Questions/document text APIs ko bheje jate hain. Quotas/charges apply ho sakte hain.
Missing key par error aata hai—silent fixture fallback nahi hota. Google embedding
access/quota unavailable ho to startup fail ho sakta hai; access fix karo ya
explicit sparse-search option use karo.

## Useful commands

```powershell
.\run.cmd doctor                 # local config/dependencies/corpus/port; no model API call
.\run.cmd --port 8770            # another local port
.\run.cmd --no-semantic          # live APIs + sparse search, no embedding calls
.\run.cmd --provider groq --no-semantic  # Groq-only, needs GROQ_API_KEY
.\run.cmd --offline              # explicit fixtures, not live answers
.\run.cmd doctor --offline       # diagnosis without a provider key
```

Offline mode ko bhi first-time package installation chahiye. Planted-error
fixtures quality results nahi hain; uploads se answers generate karne ke liye
live provider chahiye. `doctor` account quota/credit/answer quality check nahi
karta aur dependencies install nahi karta; missing environment par `setup` bolo.

Optional `.env` settings:

```dotenv
PRAMANA_PORT=8765
PRAMANA_SEMANTIC=auto
```

`auto`: Google key ho to API embeddings; Groq-only par sparse search. `false`:
embeddings off. CLI options in settings ko override karte hain. Process
environment `.env` values se priority rakhta hai.

## macOS / Linux

```sh
sh run.sh setup
# Edit .env and add your own GOOGLE_API_KEY.
sh run.sh
```

Same flags work. Portable alternative:

```sh
python run_demo.py setup
python run_demo.py
```

Use `python3` if needed. Activation step nahi chahiye. Scripts apne project folder
ko locate karte hain, chahe kisi aur directory se launch karo.

## Troubleshooting

- **Python not found:** Python 3.12 install karo, launcher/PATH option enable karo.
- **Install failed:** internet, disk space, proxy/Python check karo; `setup` rerun
  karo. Package installation ke liye API key nahi chahiye.
- **Port busy:** old terminal stop karo ya `--port 8770`.
- **Missing key/quota/unavailable provider:** provider access fix karo. Sirf
  embeddings blocked hon to `--no-semantic` use karo; doctor quota certify nahi karta.
- **Scanned PDF:** pehle OCR/text conversion chahiye. Limit: 5 MB, 25 pages.
- **No answer:** retrieved passages inspect karo; missing fact par abstention
  expected hai. Verification disable karke forced answer mat lo.

Ye launcher **localhost demo** hai, public enterprise deployment nahi. Uploads
memory mein hain aur restart par disappear hote hain. Port ko network par expose
mat karo; Host/Origin controls disable mat karo. Approved static documents,
authenticated pilot, TLS/ingress aur production gates ke liye
[separate pilot runbook](docs/11_ENTERPRISE_PILOT.md) use karo.

Docker optional hai: [easy live Docker flow](DOCKER_QUICKSTART.md) uses
`compose.local.yaml`; `compose.demo.yaml` fictional offline rehearsal hai;
`compose.yaml` separately configured authenticated pilot hai. Native Python
launcher unko automatically start nahi karta.
