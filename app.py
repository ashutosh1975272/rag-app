"""RAG chat app: conversations, cited answers, weather, feedback."""

import streamlit as st

from rag import chat as chat_api
from rag import ui_helpers
from rag.chat import DEFAULT_THRESHOLD
from rag.nim_client import NimClient

st.set_page_config(page_title="RAG Chat", layout="wide")


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

client = get_client(
    cfg.nvidia_api_key, cfg.nim_base_url, cfg.nim_chat_model, cfg.nim_embed_model
)
db_url = cfg.database_url
assert db_url is not None

if "conv_id" not in st.session_state:
    st.session_state.conv_id = None

with st.sidebar:
    st.header("Conversations")
    if st.button("New chat", use_container_width=True):
        try:
            conv = chat_api.create_conversation(db_url)
        except Exception as exc:  # noqa: BLE001 - mapped to friendly UI error
            st.error(ui_helpers.friendly_error(exc))
        else:
            st.session_state.conv_id = conv["id"]
            st.session_state.pop("last_weather", None)
            st.rerun()
    try:
        conversations = chat_api.list_conversations(db_url)
    except Exception as exc:  # noqa: BLE001 - mapped to friendly UI error
        st.error(ui_helpers.friendly_error(exc))
        conversations = []
    for conv in conversations:
        label = f"{conv['title']} ({conv['message_count']})"
        if st.button(
            label,
            key=f"conv-{conv['id']}",
            use_container_width=True,
            type=(
                "primary" if conv["id"] == st.session_state.conv_id else "secondary"
            ),
        ):
            st.session_state.conv_id = conv["id"]
            st.session_state.pop("last_weather", None)
            st.rerun()

if st.session_state.conv_id is None and conversations:
    st.session_state.conv_id = conversations[0]["id"]

st.title("RAG Chat")

if st.session_state.conv_id is None:
    st.info("Start a new chat from the sidebar, or just ask below.")
else:
    conv_id = st.session_state.conv_id
    try:
        conversation = chat_api.get_conversation(db_url, conv_id)
    except Exception as exc:  # noqa: BLE001 - mapped to friendly UI error
        st.error(ui_helpers.friendly_error(exc))
        st.session_state.conv_id = None
        st.stop()
    st.subheader(conversation["title"])
    for msg in conversation["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            badge = ui_helpers.badge_text(
                msg["route"], msg["latency_ms"], msg["cache_hit"]
            )
            if badge:
                st.caption(badge)
            if msg["role"] == "assistant" and msg["sources"]:
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"- {ui_helpers.format_citation(src)} "
                            f"(score {src['score']:.3f})"
                        )
            if msg["role"] == "assistant":
                choice = st.feedback("thumbs", key=f"fb-{msg['id']}")
                if choice == 1:
                    chat_api.set_feedback(db_url, msg["id"], 1)
                elif choice == 0:
                    chat_api.set_feedback(db_url, msg["id"], -1)
    with st.sidebar:
        st.divider()
        new_title = st.text_input("Rename chat", value=conversation["title"])
        if st.button("Rename", use_container_width=True) and new_title:
            chat_api.rename_conversation(db_url, conv_id, new_title)
            st.rerun()
        st.download_button(
            "Export Markdown",
            chat_api.export_markdown(db_url, conv_id),
            file_name=ui_helpers.export_filename(conversation["title"]),
            use_container_width=True,
        )
        if st.button("Delete chat", use_container_width=True):
            chat_api.delete_conversation(db_url, conv_id)
            st.session_state.conv_id = None
            st.rerun()

prompt = st.chat_input("Ask about your documents, or the weather…")
if prompt:
    top_k = st.session_state.get("cfg_top_k", cfg.top_k)
    threshold = st.session_state.get("cfg_threshold", DEFAULT_THRESHOLD)
    mode = st.session_state.get("cfg_mode", "hybrid")
    try:
        with st.spinner("Thinking…"):
            result = chat_api.ask_with_history(
                prompt,
                database_url=db_url,
                client=client,
                conversation_id=st.session_state.conv_id,
                top_k=top_k,
                mode=mode,
                threshold=threshold,
                use_cache=True,
                chat_model_name=cfg.nim_chat_model,
                embed_model_name=cfg.nim_embed_model,
                redis_url=cfg.redis_url,
            )
    except Exception as exc:  # noqa: BLE001 - mapped to friendly UI error
        st.error(ui_helpers.friendly_error(exc))
    else:
        st.session_state.conv_id = result["conversation_id"]
        if result.get("weather"):
            st.session_state.last_weather = result["weather"]
        st.rerun()

if st.session_state.get("last_weather"):
    card = st.session_state.last_weather
    st.subheader(f"Weather: {card['city']}, {card['country']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", f"{card['temp_c']}°C")
    col2.metric("Wind", f"{card['wind_kph']} kph")
    col3.metric("Condition", card["description"])
    if card.get("cached"):
        st.caption("served from cache")
