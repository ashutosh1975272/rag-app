# RAG Knowledge Assistant

Upload documents, chat over them with cited answers, and ask for live
weather — one Postgres, one Streamlit app, no heavy infra.

- RAG over your docs: PDF/text upload, chunk + embed, hybrid retrieval
  (vector + full-text RRF), answers with `[doc #chunk]` citations and an
  exact refusal when nothing matches.
- Chat: multi-conversation history, pronoun/antecedent rewrite, strict
  topic routing (`docs` vs `weather` vs `general` vs `compare`), no fake
  citations.
- Live weather via Open-Meteo (geocoding + forecast, no key), cached 10 min.
- Caching: DB embedding L2, exact + semantic (0.95) answer cache, weather
  tool cache, Redis L1 when `REDIS_URL` is set.

## Stack

- DB: Postgres 18 + pgvector, SQLAlchemy 2, Alembic migrations
  (`alembic/versions/`). No HNSW index: 2048 dims exceed pgvector's 2000
  cap, so retrieval uses exact cosine `<=>` plus a GIN `tsv` column.
- RAG: LangChain (`langchain-core`, `langchain-text-splitters`,
  `langchain-nvidia-ai-endpoints` for embeddings only, `langchain-openai`
  for chat) over a custom `ChunksVectorStore` on our own `chunks` table —
  one vector store, no duplicated embeddings.
- Models: NVIDIA NIM (`meta/llama-3.2-11b-vision-instruct` chat,
  `nvidia/nemotron-3-embed-1b` embeddings, 2048-dim, measured live).
  Chat goes through `ChatOpenAI` because `ChatNVIDIA` hard-routes to
  `ai.api.nvidia.com` (HTTP 451); the integrate endpoint returns 200.
- UI: Streamlit (`app.py` + `pages/`), localhost-only dev server.
- Status dashboard: workspace `scripts/build_dashboard.py` renders
  `db/reports/index.html` from STATUS files (deterministic).

## Branch rules

- `main`: stable releases only (PR from `beta`, never direct pushes)
- `beta`: integration and testing
- `feature/T-xxx-name`: one branch per ticket, from `beta`

Commits use Conventional Commits and always reference the ticket id, e.g.
`feat(ingest): add PDF text extraction (T-003)`.

## Quickstart

Prereqs: Docker, Python 3.11+.

```bash
cd rag-app
docker compose up -d db redis
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in NVIDIA_API_KEY (never commit .env)
set -a; source .env; set +a
alembic upgrade head   # create tables (tests migrate their own DBs)
pytest -q              # unit suite (localhost DBs only)
streamlit run app.py --server.address=127.0.0.1   # http://localhost:8501
```

Upload a PDF on the main page (or `pages/1_Upload.py`), then ask questions.
`pages/2_History.py` lists conversations, `pages/3_Settings.py` shows
models/thresholds. Every dev-loop check in one shot:

```bash
bash scripts/smoke.sh   # migrations current + unit suite + Streamlit health
```

Live checks (need `NVIDIA_API_KEY`):

```bash
pytest -m live -q            # NIM embed/chat, retrieval, answering, weather
python scripts/nim_probe.py  # model discovery (exit 0 on success)
```

## Layout

- `rag/` — library: `models.py`, `db.py`, `migrate.py`, `nim_client.py`,
  `embeddings.py`, `chunking.py`, `vectorstore.py`, `ingest.py`,
  `retrieve.py`, `prompts.py`, `llm.py`, `chat.py` (history + answering),
  `cache.py`/`toolcache.py`, `router.py`, `tools_weather.py`,
  `ui_helpers.py`, `probe.py`
- `app.py`, `pages/` — Streamlit UI
- `alembic/` — migrations; `tests/` (+`fixtures/`); `scripts/` — smoke,
  NIM probe (dashboard builder lives in the workspace, outside this repo)
- `docs/` — `operations.md` (rotations, schedules), `troubleshooting.md`

## Docs

- [Operations](docs/operations.md) — beta DB, schedules, webhook, dashboard
- [Troubleshooting](docs/troubleshooting.md) — common failures and fixes

Planning files (brief, PRD, tickets, memory) live in the project workspace
outside this repo; see tickets `T-000`–`T-013` for the build order.
