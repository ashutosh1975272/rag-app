"""Upload page: ingest PDF/TXT/MD files into the corpus."""

import streamlit as st

from rag import ingest as ingest_api
from rag import ui_helpers
from rag.nim_client import NimClient

st.set_page_config(page_title="Upload documents")


@st.cache_resource
def get_client(api_key, base_url, chat_model, embed_model):
    return NimClient(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embed_model=embed_model,
    )


cfg, error = ui_helpers.load_or_error()
if error:
    st.error(error)
    st.stop()
assert cfg is not None and cfg.database_url is not None

client = get_client(
    cfg.nvidia_api_key, cfg.nim_base_url, cfg.nim_chat_model, cfg.nim_embed_model
)

st.title("Upload documents")
st.caption("PDF, TXT, or MD. Re-uploading an identical file is deduplicated.")

uploaded = st.file_uploader(
    "Choose a file", type=["pdf", "txt", "md"], accept_multiple_files=False
)
if uploaded is not None:
    data = uploaded.getvalue()
    if st.button("Ingest", type="primary"):
        try:
            with st.spinner("Ingesting…"):
                result = ingest_api.ingest_file(
                    data,
                    uploaded.name,
                    database_url=cfg.database_url,
                    client=client,
                    chunk_size=st.session_state.get(
                        "cfg_chunk_size", cfg.chunk_size
                    ),
                    chunk_overlap=st.session_state.get(
                        "cfg_chunk_overlap", cfg.chunk_overlap
                    ),
                    embed_model=cfg.nim_embed_model,
                )
        except Exception as exc:  # noqa: BLE001 - mapped to UI error
            st.error(ui_helpers.friendly_error(exc))
        else:
            if result["deduped"]:
                st.info(
                    f"Already in the corpus as document "
                    f"#{result['document_id']} (0 new chunks)."
                )
            else:
                st.success(
                    f"Ingested document #{result['document_id']}: "
                    f"{result['chunks_added']} chunks. "
                    f"Corpus version {result['corpus_version']}."
                )
