# RAG Chat App

Python + Streamlit RAG application using **Neon Postgres + pgvector** (storage)
and **NVIDIA NIM** (embeddings + chat). Planned features: document upload and
ingestion, cited answers, chat history, multi-layer caching, a free weather
tool inside chat, and automated checks.

## Stack

- App: Python 3.12, Streamlit
- DB: Neon Postgres + pgvector (local Docker pgvector for dev/test)
- Models: NVIDIA NIM (OpenAI-compatible API)
- Live data: Open-Meteo, Wikipedia, Frankfurter (free, no keys)

## Branch rules

- `main`: stable releases only (PR from `beta`, never direct pushes)
- `beta`: integration and testing
- `feature/T-xxx-name`: one branch per ticket, from `beta`

Commits use Conventional Commits and always reference the ticket id, e.g.
`feat(ingest): add PDF text extraction (T-003)`.

## Quickstart (full setup lands with later tickets)

```bash
cd rag-app
docker compose up -d db redis
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your own keys (never commit .env)
pytest -q
streamlit run app.py
```

Planning files (brief, PRD, tickets, memory) live in the project workspace
outside this repo; see tickets `T-000`–`T-013` for the build order.
