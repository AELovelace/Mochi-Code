# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1", module="pydantic")

from langchain_openai import ChatOpenAI

_llm_cache: dict[tuple, ChatOpenAI] = {}


def get_llm(address: str, streaming: bool = True, json_mode: bool = False,
            timeout: int = 120) -> ChatOpenAI:
    # `timeout` is the full request budget (connect + read). The researcher (Qwen on
    # llama.cpp) needs a generous window because the 35B model can still be warming up
    # on the very first call, so we give that endpoint a dedicated 120s wait.
    key = (address, streaming, json_mode, timeout)
    if key not in _llm_cache:
        kwargs: dict = {
            "base_url": address,
            "api_key": "not-needed",
            "model": "local-model",
            "streaming": streaming,
            "timeout": timeout,  # prevent infinite hang when SSE stream stalls mid-response
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        _llm_cache[key] = ChatOpenAI(**kwargs)
    return _llm_cache[key]


def rebuild_llms() -> None:
    _llm_cache.clear()
