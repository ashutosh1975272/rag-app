"""T-000: config loader tests (written first, TDD)."""
import pytest

from rag.config import Config, ConfigError, load_config, require


def test_defaults_load_with_empty_env():
    cfg = load_config(env={})
    assert isinstance(cfg, Config)
    assert cfg.chunk_size == 800
    assert cfg.chunk_overlap == 100
    assert cfg.top_k == 6
    assert cfg.semantic_threshold == 0.95
    assert cfg.nim_base_url == "https://integrate.api.nvidia.com/v1"
    assert cfg.database_url is None
    assert cfg.nvidia_api_key is None


def test_env_overrides_are_parsed():
    cfg = load_config(env={
        "DATABASE_URL": "postgresql://x",
        "CHUNK_SIZE": "500",
        "CHUNK_OVERLAP": "50",
        "TOP_K": "3",
        "SEMANTIC_THRESHOLD": "0.9",
    })
    assert cfg.database_url == "postgresql://x"
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap == 50
    assert cfg.top_k == 3
    assert cfg.semantic_threshold == 0.9


def test_require_raises_named_error_without_value():
    with pytest.raises(ConfigError) as exc:
        require(None, "NVIDIA_API_KEY")
    assert "NVIDIA_API_KEY" in str(exc.value)


def test_require_passes_value_through():
    assert require("abc", "X") == "abc"


def test_invalid_int_raises_config_error():
    with pytest.raises(ConfigError):
        load_config(env={"CHUNK_SIZE": "not-a-number"})
