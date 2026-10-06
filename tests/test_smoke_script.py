"""Smoke-script tests: usage failure offline, full run marked live."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SMOKE = Path(__file__).parent.parent / "scripts" / "smoke.sh"


def test_smoke_fails_cleanly_without_env():
    env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    proc = subprocess.run(
        ["bash", str(SMOKE)],
        capture_output=True,
        text=True,
        check=False,
        env={**env, "DATABASE_URL": ""},
        timeout=120,
    )
    assert proc.returncode != 0
    assert "SMOKE FAIL: missing env DATABASE_URL" in proc.stdout


@pytest.mark.live
def test_smoke_full_run_green():
    for var in (
        "DATABASE_URL",
        "NVIDIA_API_KEY",
        "NIM_CHAT_MODEL",
        "NIM_EMBED_MODEL",
    ):
        if not os.environ.get(var):
            pytest.skip(f"needs {var}")
    proc = subprocess.run(
        ["bash", str(SMOKE)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SMOKE OK: all checks passed" in proc.stdout
