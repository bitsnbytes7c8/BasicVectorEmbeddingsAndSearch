from __future__ import annotations

import argparse
import logging
from typing import Optional

from gphotos.auth import get_credentials
from gphotos.chroma_store import VisionChromaStore
from gphotos.config import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_CHROMA_PATH,
    DEFAULT_CLIENT_SECRET,
    DEFAULT_TOKEN_PATH,
    OLLAMA_EMBED_MODEL,
    OLLAMA_VISION_MODEL,
)
from gphotos.extract import PhotoRef
from gphotos.picker import build_picker_service, run_picker_and_collect_photo_refs
from gphotos.pipeline import run_producer_consumer

logger = logging.getLogger(__name__)

PLACEHOLDER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Describe what you see in the user's image. "
    "(Replace this default prompt in gphotos/main.py or pass --system-prompt.)"
)


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Google Photos Picker → vision descriptions → ChromaDB vector embeddings",
    )
    p.add_argument("--client-secret", default=DEFAULT_CLIENT_SECRET, help="OAuth client JSON path")
    p.add_argument("--token", default=DEFAULT_TOKEN_PATH, help="Saved OAuth token path")
    p.add_argument(
        "--system-prompt",
        default=PLACEHOLDER_SYSTEM_PROMPT,
        help="System prompt for llama3.2-vision",
    )
    p.add_argument("--user-prompt", default="Describe this image.", help="User message alongside the image")
    p.add_argument("--model", default=OLLAMA_VISION_MODEL, help="Ollama vision model name")
    p.add_argument(
        "--picker-no-autoclose",
        action="store_true",
        help="Do not append /autoclose to the picker URL (shows Done screen).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging.",
    )
    p.add_argument(
        "--chroma-path",
        default=DEFAULT_CHROMA_PATH,
        help="Chroma persistent storage directory (default from GPHOTOS_CHROMA_PATH or data/chroma_db).",
    )
    p.add_argument(
        "--chroma-collection",
        default=CHROMA_COLLECTION_NAME,
        help="Chroma collection name (default from GPHOTOS_CHROMA_COLLECTION or gphotos_vision).",
    )
    p.add_argument(
        "--embed-model",
        default=OLLAMA_EMBED_MODEL,
        help="Ollama embedding model for description text (default OLLAMA_EMBED_MODEL / nomic-embed-text).",
    )
    args = p.parse_args(argv)

    if args.verbose:
        logging.getLogger("gphotos").setLevel(logging.DEBUG)

    creds = get_credentials(args.client_secret, args.token)

    picker_svc = build_picker_service(creds)
    batch = run_picker_and_collect_photo_refs(
        picker_svc,
        autoclose=not args.picker_no_autoclose,
    )

    logger.info("Photos to process this run: %d", len(batch))
    if not batch:
        logger.info("No media selected or picker returned nothing; exiting.")
        return

    chroma = VisionChromaStore(
        args.chroma_path,
        args.chroma_collection,
        args.embed_model,
    )
    logger.info(
        "ChromaDB: path=%s collection=%s hnsw=cosine embed_model=%s",
        args.chroma_path,
        args.chroma_collection,
        args.embed_model,
    )

    def on_ok(photo: PhotoRef, text: str) -> None:
        logger.info("[%s] vision output: %s", photo.media_item_id, text[:500])
        try:
            chroma.upsert_description(photo, text)
        except Exception as e:
            logger.exception("[%s] ChromaDB upsert failed: %s", photo.media_item_id, e)

    def on_err(photo: PhotoRef, err: BaseException) -> None:
        logger.error("[%s] error: %s", photo.media_item_id, err)

    run_producer_consumer(
        batch,
        creds,
        args.token,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        on_result=on_ok,
        on_error=on_err,
        model=args.model,
    )


if __name__ == "__main__":
    main()
