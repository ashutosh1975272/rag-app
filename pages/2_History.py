"""History page: browse past conversations and export them."""

import streamlit as st

from rag import chat as chat_api
from rag import ui_helpers

st.set_page_config(page_title="History")

cfg, error = ui_helpers.load_or_error()
if error:
    st.error(error)
    st.stop()
assert cfg is not None and cfg.database_url is not None
db_url = cfg.database_url

st.title("History")

try:
    conversations = chat_api.list_conversations(db_url)
except Exception as exc:  # noqa: BLE001 - mapped to UI error
    st.error(ui_helpers.friendly_error(exc))
    st.stop()

if not conversations:
    st.info("No conversations yet. Ask something on the Chat page.")
    st.stop()

options = {f"{c['title']} ({c['message_count']})": c["id"] for c in conversations}
choice = st.selectbox("Conversation", list(options.keys()))
conv = chat_api.get_conversation(db_url, options[choice])

for msg in conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        badge = ui_helpers.badge_text(
            msg["route"], msg["latency_ms"], msg["cache_hit"]
        )
        if badge:
            st.caption(badge)

st.download_button(
    "Export Markdown",
    chat_api.export_markdown(db_url, conv["id"]),
    file_name=ui_helpers.export_filename(conv["title"]),
)
