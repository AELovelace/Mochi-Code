# -*- coding: utf-8 -*-
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages:     Annotated[list, add_messages]
    intent:       str
    domain:       str
    confidence:   float
    rag_needed:   bool
    web_needed:   bool
    tools_needed: list[str]
    routing_note: str
    summary:      str
    rag_context:  str   # retrieved+compressed context from the Hybrid RAG pipeline
    web_query:    str
    web_results:  list[dict]
    research_brief: str
    context_override_mode: str
    context_override_query: str
    context_override_reason: str
