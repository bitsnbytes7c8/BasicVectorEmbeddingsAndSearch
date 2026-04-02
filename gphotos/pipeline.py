from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable

from google.oauth2.credentials import Credentials

from gphotos.config import QUEUE_MAXSIZE
from gphotos.download import download_thumbnail_bytes
from gphotos.extract import PhotoRef
from gphotos.ollama_vision import run_vision

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    media_item_id: str
    base_url: str
    media_metadata: dict[str, Any]
    thumbnail_png: bytes


def run_producer_consumer(
    photos: list[PhotoRef],
    creds: Credentials,
    token_path: str | None,
    system_prompt: str,
    user_prompt: str,
    on_result: Callable[[PhotoRef, str], None],
    on_error: Callable[[PhotoRef, BaseException], None],
    model: str | None = None,
) -> None:
    """
    Producer downloads 448×448 thumbnails; consumer calls Ollama vision.
    `photos` should be pre-collected (API iterator is not thread-safe).
    """
    work: queue.Queue[WorkItem | None] = queue.Queue(maxsize=QUEUE_MAXSIZE)

    def producer() -> None:
        try:
            for p in photos:
                try:
                    thumb = download_thumbnail_bytes(p.base_url, creds, token_path)
                    work.put(
                        WorkItem(
                            media_item_id=p.media_item_id,
                            base_url=p.base_url,
                            media_metadata=p.media_metadata,
                            thumbnail_png=thumb,
                        )
                    )
                except BaseException as e:
                    logger.exception("Producer failed for %s", p.media_item_id)
                    on_error(p, e)
        finally:
            work.put(None)

    def consumer() -> None:
        while True:
            item = work.get()
            try:
                if item is None:
                    break
                pr = PhotoRef(
                    media_item_id=item.media_item_id,
                    base_url=item.base_url,
                    media_metadata=item.media_metadata,
                )
                try:
                    text = run_vision(
                        item.thumbnail_png,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        model=model,
                    )
                    on_result(pr, text)
                except BaseException as e:
                    logger.exception("Vision failed for %s", item.media_item_id)
                    on_error(pr, e)
            finally:
                work.task_done()

    t_prod = threading.Thread(target=producer, name="gphotos-producer", daemon=True)
    t_cons = threading.Thread(target=consumer, name="gphotos-consumer", daemon=True)
    t_prod.start()
    t_cons.start()
    t_prod.join()
    t_cons.join()
