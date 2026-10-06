"""Settings page: retrieval options, caches, prompt versions."""

import streamlit as st

from rag import ui_helpers
from rag.cache import clear_caches
from rag.chat import DEFAULT_THRESHOLD
from rag.db import is_local_database_url
from rag.prompts import add_version, get_active

st.set_page_config(page_title="Settings")

cfg, error = ui_helpers.load_or_error()
if error:
    st.error(error)
    st.stop()
assert cfg is not None and cfg.database_url is not None
db_url = cfg.database_url

st.title("Settings")

st.subheader("Retrieval (this session)")
st.session_state.cfg_top_k = st.number_input(
    "Top K", min_value=1, max_value=20,
    value=st.session_state.get("cfg_top_k", cfg.top_k),
)
st.session_state.cfg_threshold = st.slider(
    "Similarity threshold", min_value=0.0, max_value=1.0,
    value=st.session_state.get("cfg_threshold", DEFAULT_THRESHOLD),
)
st.session_state.cfg_mode = st.selectbox(
    "Retrieval mode", ["hybrid", "vector"],
    index=["hybrid", "vector"].index(
        st.session_state.get("cfg_mode", "hybrid")
    ),
)
st.session_state.cfg_chunk_size = st.number_input(
    "Chunk size (new uploads)", min_value=50, max_value=4000,
    value=st.session_state.get("cfg_chunk_size", cfg.chunk_size),
)
st.session_state.cfg_chunk_overlap = st.number_input(
    "Chunk overlap (new uploads)", min_value=0, max_value=1000,
    value=st.session_state.get("cfg_chunk_overlap", cfg.chunk_overlap),
)

st.subheader("Connections (values never shown)")
assert cfg.database_url is not None
st.text(f"DATABASE_URL: set ({'local' if is_local_database_url(cfg.database_url) else 'remote'})")
st.text(f"REDIS_URL: {'set' if cfg.redis_url else 'missing (postgres tool cache)'}")
st.text("NVIDIA_API_KEY: set")
st.text(f"NIM_CHAT_MODEL: {cfg.nim_chat_model}")
st.text(f"NIM_EMBED_MODEL: {cfg.nim_embed_model}")

st.subheader("Caches")
if st.button("Clear all caches"):
    try:
        counts = clear_caches(db_url, redis_url=cfg.redis_url)
        st.cache_resource.clear()
    except Exception as exc:  # noqa: BLE001 - mapped to UI error
        st.error(ui_helpers.friendly_error(exc))
    else:
        st.success(f"Caches cleared: {counts}")

st.subheader("Prompt versions")
for name in ("system", "answer", "query-rewrite", "router"):
    try:
        version, content = get_active(db_url, name)
    except Exception as exc:  # noqa: BLE001 - mapped to UI error
        st.error(ui_helpers.friendly_error(exc))
    else:
        with st.expander(f"{name} (v{version} active)"):
            st.code(content)

with st.expander("Add a prompt version"):
    pname = st.selectbox(
        "Template", ["system", "answer", "query-rewrite", "router"]
    )
    pcontent = st.text_area("Content (new version becomes active)")
    if st.button("Save new version") and pcontent.strip():
        try:
            new_version = add_version(db_url, pname, pcontent)
        except Exception as exc:  # noqa: BLE001 - mapped to UI error
            st.error(ui_helpers.friendly_error(exc))
        else:
            st.success(f"Saved {pname} v{new_version} (now active).")
            st.rerun()
