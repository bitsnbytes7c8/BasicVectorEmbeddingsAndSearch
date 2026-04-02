#!/usr/bin/env python3
"""
Query indexed ChromaDB vision descriptions (separate from the ingest pipeline).

Plain text in — an LLM (Ollama) decides whether to run semantic search vs a RAG answer.
Override with --search / --ask to skip the router.

  python query_photos.py beach sunset
  python query_photos.py Which countries have I eaten cake in?
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from gphotos.chroma_search import search_similar
from gphotos.query_router import resolve_query_intent
from gphotos.rag_answer import answer_question_over_collection
from gphotos.config import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_CHROMA_PATH,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_ROUTER_MODEL,
)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    p = argparse.ArgumentParser(
        description=(
            "Query ChromaDB photo descriptions. Plain text: an Ollama model routes to "
            "ranked photo search or a written RAG answer."
        ),
    )
    p.add_argument(
        "query_parts",
        nargs="+",
        help="What to search or ask (words or quoted phrase).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--search",
        action="store_true",
        help="Force semantic search (ranked photos + URLs).",
    )
    mode.add_argument(
        "--ask",
        action="store_true",
        help="Force RAG text answer.",
    )
    p.add_argument(
        "--router-model",
        default=OLLAMA_ROUTER_MODEL,
        help="Ollama model for search vs ask routing (default OLLAMA_ROUTER_MODEL / same as chat).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="Chunks to retrieve (default: 10 for search, 24 for ask).",
    )
    p.add_argument("--chroma-path", default=DEFAULT_CHROMA_PATH)
    p.add_argument("--chroma-collection", default=CHROMA_COLLECTION_NAME)
    p.add_argument("--embed-model", default=OLLAMA_EMBED_MODEL)
    p.add_argument(
        "--chat-model",
        default=OLLAMA_CHAT_MODEL,
        help="Ollama text model for RAG answers only.",
    )
    args = p.parse_args(argv)

    query = " ".join(args.query_parts).strip()
    if not query:
        print("Empty query.", file=sys.stderr)
        return 2

    if args.search:
        intent = "search"
    elif args.ask:
        intent = "ask"
    else:
        try:
            intent = resolve_query_intent(query, router_model=args.router_model)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 1

    top_k = args.top_k
    if top_k is None:
        top_k = 10 if intent == "search" else 24

    logging.info("Query mode: %s", intent)

    if intent == "search":
        hits = search_similar(
            query,
            top_k=top_k,
            persist_directory=args.chroma_path,
            collection_name=args.chroma_collection,
            embed_model=args.embed_model,
        )
        if not hits:
            print("No results (empty index or no close matches).")
            return 1
        for h in hits:
            dist = f"{h.distance:.4f}" if h.distance is not None else "n/a"
            print(f"\n--- #{h.rank}  distance={dist} ---")
            print(f"mediaItemId: {h.media_item_id}")
            print(f"URL:         {h.photo_url}")
            print(f"Description: {h.description[:500]}{'…' if len(h.description) > 500 else ''}")
        return 0

    text = answer_question_over_collection(
        query,
        top_k=top_k,
        persist_directory=args.chroma_path,
        collection_name=args.chroma_collection,
        embed_model=args.embed_model,
        chat_model=args.chat_model,
    )
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
