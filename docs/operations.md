# Operations: smoke test, schedule, webhook

## Nightly smoke (`scripts/smoke.sh`)

Runs three checks and prints `SMOKE PASS/FAIL` lines (never secret values):

1. `SELECT 1` on `DATABASE_URL`.
2. Alembic `current` equals `heads` (fails loudly on migration drift).
3. Ingests a timestamped copy of `tests/fixtures/rivers.txt`, asks the
   known question, asserts a `[doc #chunk]` citation, then deletes the
   smoke document and its cache rows.

Required env: `DATABASE_URL`, `NVIDIA_API_KEY`, `NIM_CHAT_MODEL`,
`NIM_EMBED_MODEL`. Uses `.venv/bin/python` when present.

```bash
cd rag-app
DATABASE_URL=... NVIDIA_API_KEY=... NIM_CHAT_MODEL=... NIM_EMBED_MODEL=... \
  bash scripts/smoke.sh
```

## Nightly schedule

- Name: `Nightly RAG smoke`, cron `30 2 * * *`, timezone `UTC`.
- Destination: main project chat (`crew_chat`).
- Each run executes `scripts/smoke.sh` against the Neon `rag_beta`
  database (derived from the Neon secret by replacing `/neondb` with
  `/rag_beta`) and reports the PASS/FAIL lines.

## Webhook trigger (Slack `/ask`)

- Endpoint: `https://agents.excellencetechnologies.in/api/hooks/product/c1b830cb-778a-457d-98e7-d55945cd78d5`
- Auth: `Authorization: Bearer <one-time secret>` (delivered once at
  creation; it cannot be listed again — rotate if lost).
- Behavior: each authenticated JSON delivery queues the saved instruction
  with the payload path into the trigger's own conversation (`isolated`).
  The agent answers via `ask_with_history` and POSTs plain text to the
  payload's `response_url` when present.

### Wiring Slack `/ask` (owner step, ~5 minutes)

1. Open the Slack workspace's app settings (api.slack.com/apps) and pick
   (or create) the app used for this project.
2. Add a slash command: command `/ask`, request URL = the endpoint above,
   short description `Ask the RAG assistant`.
3. Slack signs deliveries with its own signing secret — that verifies
   Slack-to-app; the trigger bearer secret goes in an app-level proxy
   header only if you front the endpoint. For a direct setup, note that
   Slack cannot send custom bearer headers: put the endpoint behind your
   own tiny forwarder, or accept Slack-signed delivery and validate the
   signature in the trigger instruction instead.
4. In Slack, run `/ask what is the longest river in Africa?` — the answer
   posts back via `response_url`.

### Rotating the trigger secret

Use the `update_project_trigger` platform tool with the trigger id
`c1b830cb-778a-457d-98e7-d55945cd78d5` and rotation enabled; the new
secret is returned once. Update the Slack forwarder, then re-prove with:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST \
  -H "Authorization: Bearer <new-secret>" \
  -H "Content-Type: application/json" \
  -d '{"command":"/ask","text":"ping"}' \
  https://agents.excellencetechnologies.in/api/hooks/product/c1b830cb-778a-457d-98e7-d55945cd78d5
# expect 202; a wrong secret answers 401
```
