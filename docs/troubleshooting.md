# Troubleshooting

## `alembic upgrade head` fails to connect

- Is Postgres up? `docker compose ps db`, then
  `docker compose logs db | tail -n 20`.
- Is `DATABASE_URL` exported? Alembic reads the env, not `.env`:
  `set -a; . ./.env; set +a`. Check with `echo "$DATABASE_URL"`.

## Tests fail with connection refused

Tests only touch localhost DBs (`127.0.0.1`/`localhost` guard in
`tests/conftest.py`). Start the stack first: `docker compose up -d db redis`.

- `test_db_url` unreachable → create it:
  `psql "$DATABASE_URL" -c 'CREATE DATABASE rag_test'`
  (same for `rag_empty_test` when running the full suite).

## Live tests fail (`pytest -m live`)

- `NVIDIA_API_KEY` missing/invalid: `python scripts/nim_probe.py` shows the
  raw status. Keys rotate — update `.env` and re-export.
- HTTP 451 from chat: you are hitting `ai.api.nvidia.com`. The app uses
  `ChatOpenAI` against the integrate endpoint — do not swap in `ChatNVIDIA`.
- Weather failures: Open-Meteo/geocoding network issue. Re-run; the tool
  cache (`tool_cache`, 10 min TTL) may serve stale data meanwhile.

## Streamlit shows "Database unavailable" / errors

- Migrate: `alembic upgrade head`, then `alembic current` (expect the newest
  revision in `alembic/versions/`).
- The app surfaces friendly messages, never tracebacks. For the real error,
  run the failing call from a shell with the same env.

## `smoke.sh` reports `DRIFT`

Local DB is behind `alembic/versions/`: run `alembic upgrade head` on that
database and re-run the script. CI fails the same way — migrations must be
committed with the code that needs them.

## Lint fails (`ruff check .`)

Run `.venv/bin/ruff check --fix .` for auto-fixable rules, then fix the rest
by hand. CI blocks merge on any finding.

## Redis

Optional. Without `REDIS_URL` the app falls back to in-process + Postgres
caches (Redis tests skip). With Docker: `docker compose up -d redis`.
