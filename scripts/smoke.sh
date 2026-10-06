#!/usr/bin/env bash
# Nightly smoke test: DB reachable, migrations current, one cited answer.
#
# Required env: DATABASE_URL, NVIDIA_API_KEY, NIM_CHAT_MODEL, NIM_EMBED_MODEL.
# Ingests a timestamped copy of the rivers fixture, asks a known question,
# asserts a citation, then deletes the smoke document. Prints check names and
# PASS/FAIL only — never secret values.
set -euo pipefail

fail() { echo "SMOKE FAIL: $1"; exit 1; }
pass() { echo "SMOKE PASS: $1"; }

for var in DATABASE_URL NVIDIA_API_KEY NIM_CHAT_MODEL NIM_EMBED_MODEL; do
  if [ -z "${!var:-}" ]; then
    fail "missing env $var"
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
else
  PY=python3
fi

# 1. Database reachable.
"$PY" -c "
import os, psycopg
with psycopg.connect(os.environ['DATABASE_URL'], connect_timeout=15) as c:
    assert c.execute('SELECT 1').fetchone()[0] == 1
" || fail "database SELECT 1"
pass "database reachable"

# 2. Migrations current (current revision == head).
CURRENT="$("$PY" -m alembic current 2>/dev/null | tail -n 1)"
HEADS="$("$PY" -m alembic heads 2>/dev/null | tail -n 1)"
if [ -z "$CURRENT" ] || [ -z "$HEADS" ]; then
  fail "could not read alembic current/heads"
fi
CURRENT_REV="$(echo "$CURRENT" | awk '{print $1}')"
HEADS_REV="$(echo "$HEADS" | awk '{print $1}')"
if [ "$CURRENT_REV" != "$HEADS_REV" ]; then
  fail "database at $CURRENT_REV, head is $HEADS_REV"
fi
pass "migrations current ($CURRENT_REV)"

# 3. Ingest -> ask -> cited answer -> cleanup.
"$PY" - <<'EOF' || fail "fixture ask did not return a cited answer"
import os, re, time
import psycopg
from pathlib import Path
from rag.chat import ask
from rag.ingest import ingest_file
from rag.nim_client import NimClient

url = os.environ["DATABASE_URL"]
client = NimClient(
    chat_model=os.environ["NIM_CHAT_MODEL"],
    embed_model=os.environ["NIM_EMBED_MODEL"],
)
name = f"smoke-{int(time.time())}-rivers.txt"
data = (Path("tests/fixtures/rivers.txt")).read_bytes()
doc_id = ingest_file(data, name, database_url=url, client=client,
                     embed_model=os.environ["NIM_EMBED_MODEL"])["document_id"]
try:
    res = ask("Which river is the longest in Africa?",
              database_url=url, client=client)
    assert not res["refused"], "unexpected refusal"
    assert "Nile" in res["answer"], "answer missing known fact"
    assert re.search(r"\[.+? #\d+\]", res["answer"]), "answer missing citation"
    print("SMOKE PASS: cited answer (" + name + ")")
finally:
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.execute("DELETE FROM answer_cache WHERE corpus_version = "
                     "(SELECT value::bigint FROM app_meta "
                     "WHERE key = 'corpus_version')")
        conn.execute("DELETE FROM embedding_cache")
EOF

echo "SMOKE OK: all checks passed"
