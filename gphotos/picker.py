from __future__ import annotations

import logging
import time
import webbrowser
from typing import Any, Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from gphotos.extract import PhotoRef

logger = logging.getLogger(__name__)


def build_picker_service(creds: Credentials):
    return build("photospicker", "v1", credentials=creds, static_discovery=False)


def _parse_duration_seconds(s: Optional[str], default: float) -> float:
    if not s:
        return default
    s = str(s).strip()
    if s.endswith("s"):
        try:
            return float(s[:-1])
        except ValueError:
            return default
    try:
        return float(s)
    except ValueError:
        return default


def _picked_to_photo_ref(raw: dict[str, Any]) -> Optional[PhotoRef]:
    mid = raw.get("id")
    mf = raw.get("mediaFile") or {}
    base = mf.get("baseUrl")
    if not mid or not base:
        return None
    created = raw.get("createTime")
    meta: dict[str, Any] = {}
    if created:
        meta["creationTime"] = created
    if mf.get("mimeType"):
        meta["mimeType"] = mf["mimeType"]
    if raw.get("type"):
        meta["mediaType"] = raw["type"]
    loc = raw.get("location")
    if isinstance(loc, dict):
        meta["location"] = loc
    mf_loc = (mf.get("mediaFileMetadata") or {}).get("location")
    if isinstance(mf_loc, dict) and "location" not in meta:
        meta["location"] = mf_loc
    return PhotoRef(media_item_id=mid, base_url=base, media_metadata=meta)


def poll_until_picking_done(service, session_id: str, polling_config: dict[str, Any]) -> None:
    """Poll sessions.get until mediaItemsSet is true or timeout."""
    poll_interval = _parse_duration_seconds(polling_config.get("pollInterval"), 2.0)
    timeout_in = _parse_duration_seconds(polling_config.get("timeoutIn"), 0.0)
    start = time.monotonic()
    # timeoutIn 0 means "no limit" per API note; still cap wall time to avoid infinite hang.
    max_wall = timeout_in if timeout_in > 0 else 900.0

    while True:
        sess = service.sessions().get(sessionId=session_id).execute()
        if sess.get("mediaItemsSet"):
            logger.info("Picker session complete (mediaItemsSet=true).")
            return
        elapsed = time.monotonic() - start
        if elapsed >= max_wall:
            raise RuntimeError(
                f"Picker timed out after {elapsed:.0f}s waiting for user to finish (timeoutIn≈{max_wall}s)."
            )
        pc = sess.get("pollingConfig") or polling_config
        poll_interval = _parse_duration_seconds(pc.get("pollInterval"), poll_interval)
        logger.debug("Polling sessions.get; sleeping %ss", poll_interval)
        time.sleep(poll_interval)


def list_picked_media_items(service, session_id: str, page_size: int = 100) -> list[PhotoRef]:
    out: list[PhotoRef] = []
    page_token: Optional[str] = None
    while True:
        req = (
            service.mediaItems()
            .list(sessionId=session_id, pageSize=page_size, pageToken=page_token)
            .execute()
        )
        for raw in req.get("mediaItems", []) or []:
            pr = _picked_to_photo_ref(raw)
            if pr:
                out.append(pr)
        page_token = req.get("nextPageToken")
        if not page_token:
            break
    return out


def run_picker_and_collect_photo_refs(
    service,
    *,
    autoclose: bool = True,
) -> list[PhotoRef]:
    """
    Create a Picker session, open the picker in the browser, poll until done, list media.

    Caller must supply credentials that include photospicker.mediaitems.readonly.
    """
    try:
        session = service.sessions().create(body={}).execute()
    except HttpError as e:
        raise RuntimeError(f"sessions.create failed: {e}") from e

    session_id = session.get("id")
    picker_uri = session.get("pickerUri")
    polling_config = session.get("pollingConfig") or {}
    if not session_id or not picker_uri:
        raise RuntimeError(f"Invalid sessions.create response: {session!r}")

    open_uri = picker_uri.rstrip("/") + "/autoclose" if autoclose else picker_uri
    logger.info("Opening Google Photos picker in your browser. Select items, then finish in Photos.")
    logger.info("If nothing opens, visit: %s", open_uri)
    webbrowser.open(open_uri)

    poll_until_picking_done(service, session_id, polling_config)

    try:
        refs = list_picked_media_items(service, session_id)
    except HttpError as e:
        raise RuntimeError(f"mediaItems.list failed: {e}") from e
    finally:
        try:
            service.sessions().delete(sessionId=session_id).execute()
            logger.debug("Deleted picker session %s", session_id)
        except HttpError as e:
            logger.warning("sessions.delete failed (non-fatal): %s", e)

    logger.info("Picker returned %s media item(s).", len(refs))
    return refs
