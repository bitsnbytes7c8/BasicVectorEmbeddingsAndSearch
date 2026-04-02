from __future__ import annotations

import ollama

from gphotos.chroma_search import search_similar
from gphotos.config import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_CHROMA_PATH,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
)


def answer_question_over_collection(
    question: str,
    *,
    top_k: int = 24,
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embed_model: str = OLLAMA_EMBED_MODEL,
    chat_model: str = OLLAMA_CHAT_MODEL,
) -> str:
    """
    Retrieve relevant photo descriptions from Chroma, then answer with a text-only Ollama model (RAG).
    """
    hits = search_similar(
        question,
        top_k=top_k,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embed_model=embed_model,
    )
    if not hits:
        return (
            "No indexed photos found in ChromaDB (empty collection or wrong path/name). "
            "Run `python run.py` first to index descriptions."
        )

    lines = []
    for h in hits:
        lines.append(f"[photo id={h.media_item_id}]\n{h.description}\n")
    context = "\n".join(lines)

    system = (
        "You answer questions using ONLY the retrieved photo descriptions below. "
        "Each block is what a vision model saw in one photo. "
        "If the context is insufficient, say what is missing. "
        "Be concise. For questions about places or activities, infer only what the descriptions support."
    )
    user = f"Question: {question}\n\n--- Retrieved photo descriptions ---\n{context}"

    resp = ollama.chat(
        model=chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.message.content
    return content if isinstance(content, str) else str(content or "")
