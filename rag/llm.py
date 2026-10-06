"""LCEL answer chain: prompt templates + chat model + string output."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


def build_answer_chain(
    chat_model, system_text: str, answer_text: str
) -> Runnable:
    """Build ``{context, question} -> answer string`` LCEL chain."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_text), ("human", answer_text)]
    )
    return prompt | chat_model | StrOutputParser()
