from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional

import ollama

from gphotos.config import OLLAMA_ROUTER_MODEL

logger = logging.getLogger(__name__)

QueryIntent = Literal["search", "ask"]

_ROUTER_SYSTEM = """You classify one user message about their personal photo library.
The library is indexed only as text descriptions (what a vision model saw in each photo).

Choose exactly one intent:

- "search" — The user wants to FIND or LIST photos that match a topic: keywords, scenes, objects, people, places, colors, moods, activities described in short phrases. They want ranked photo results with links, not a prose essay. Examples: "beach sunset", "my dog", "food photos", "birthday party".

- "ask" — The user wants a WRITTEN ANSWER synthesized across photos: questions, summaries, comparisons, counts, "which countries", "what did I eat", "how many times", "tell me about my trips". Examples: "Which countries have I visited?", "What did I usually eat for breakfast?"

Reply with ONLY valid JSON on one line, no markdown, no other text:
{"intent":"search"}
or
{"intent":"ask"}
"""


def _parse_intent_json(text: str) -> Optional[QueryIntent]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        v = str(obj.get("intent", "")).lower()
        if v in ("search", "ask"):
            return v
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\"intent\"[^{}]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("intent", "")).lower()
            if v in ("search", "ask"):
                return v
        except json.JSONDecodeError:
            pass
    tl = text.lower()
    if '"intent":"search"' in tl.replace(" ", "") or '"intent": "search"' in tl:
        return "search"
    if '"intent":"ask"' in tl.replace(" ", "") or '"intent": "ask"' in tl:
        return "ask"
    return None


def resolve_query_intent(
    query: str,
    *,
    router_model: str = OLLAMA_ROUTER_MODEL,
) -> QueryIntent:
    """
    Use the router LLM (Ollama) to choose ``search`` vs ``ask``.

    Raises ``RuntimeError`` if the model fails or returns output that cannot be parsed as intent JSON.
    """
    try:
        resp = ollama.chat(
            model=router_model,
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0},
        )
        raw = resp.message.content
        text = raw if isinstance(raw, str) else str(raw or "")
        parsed = _parse_intent_json(text)
        if parsed:
            logger.debug("Router LLM intent=%s", parsed)
            return parsed
        raise RuntimeError(
            f"Router model returned unparseable JSON. Set OLLAMA_ROUTER_MODEL or --router-model. Raw output: {text[:500]!r}"
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Router LLM call failed ({e!r}). Is Ollama running and is model {router_model!r} pulled?"
        ) from e
