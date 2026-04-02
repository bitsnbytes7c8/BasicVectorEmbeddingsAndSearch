from __future__ import annotations

import logging
import threading
from typing import Any

import chromadb
import ollama

from gphotos.extract import PhotoRef
from gphotos.location_meta import location_fields_for_chroma

logger = logging.getLogger(__name__)

# Chroma persistent index: HNSW (default backend). Tune via collection metadata.
# See https://docs.trychroma.com/guides
DEFAULT_HNSW_METADATA: dict[str, Any] = {
    "hnsw:space": "cosine",
    "hnsw:M": 16,
    "hnsw:construction_ef": 100,
    "hnsw:search_ef": 100,
}


def embed_text(text: str, model: str) -> list[float]:
    """Ollama text embeddings; must match the model used when indexing."""
    resp = ollama.embeddings(model=model, prompt=text)
    emb = resp.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError(f"Unexpected embeddings response shape for model {model!r}")
    return emb


class VisionChromaStore:
    """
    Persists vision descriptions in ChromaDB: id = Google Photos mediaItemId,
    vector = embedding of the description text, document = raw description.
    When present in media metadata, latitude/longitude/location_name are stored as flat fields.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embed_model: str,
        hnsw_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._embed_model = embed_model
        self._lock = threading.Lock()
        meta = dict(hnsw_metadata or DEFAULT_HNSW_METADATA)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata=meta,
        )

    def upsert_description(self, photo: PhotoRef, description: str) -> None:
        """Upsert by mediaItemId; replaces prior row for the same id."""
        embedding = embed_text(description, self._embed_model)
        meta: dict[str, Any] = {"media_item_id": photo.media_item_id}
        md = photo.media_metadata or {}
        ct = md.get("creationTime")
        if ct:
            meta["creation_time"] = str(ct)
        meta.update(location_fields_for_chroma(md))

        with self._lock:
            self._collection.upsert(
                ids=[photo.media_item_id],
                embeddings=[embedding],
                documents=[description],
                metadatas=[meta],
            )
        logger.debug("Chroma upsert id=%s dim=%s", photo.media_item_id, len(embedding))
