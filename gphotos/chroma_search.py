from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import chromadb

from gphotos.chroma_store import embed_text
from gphotos.config import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_CHROMA_PATH,
    OLLAMA_EMBED_MODEL,
)
from gphotos.photo_urls import google_photos_photo_url


@dataclass(frozen=True)
class SearchHit:
    """One result from semantic search over indexed descriptions."""

    rank: int
    media_item_id: str
    distance: Optional[float]
    description: str
    photo_url: str
    metadata: dict[str, Any]


def open_vision_collection(
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        return client.get_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Cannot open Chroma collection {collection_name!r} at {persist_directory!r}. "
            "Run `python run.py` first to create the index, or fix --chroma-path / --chroma-collection."
        ) from e


def search_similar(
    query: str,
    *,
    top_k: int = 10,
    persist_directory: str = DEFAULT_CHROMA_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embed_model: str = OLLAMA_EMBED_MODEL,
) -> list[SearchHit]:
    """
    Embed `query` with the same Ollama model used at index time and run vector search.
    """
    collection = open_vision_collection(persist_directory, collection_name)
    q_emb = embed_text(query, embed_model)
    raw = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    hits: list[SearchHit] = []
    for i, mid in enumerate(ids):
        doc = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) and metas[i] else {}
        dist = dists[i] if i < len(dists) else None
        hits.append(
            SearchHit(
                rank=i + 1,
                media_item_id=str(mid),
                distance=float(dist) if dist is not None else None,
                description=str(doc) if doc else "",
                photo_url=google_photos_photo_url(str(mid)),
                metadata=dict(meta) if isinstance(meta, dict) else {},
            )
        )
    return hits
