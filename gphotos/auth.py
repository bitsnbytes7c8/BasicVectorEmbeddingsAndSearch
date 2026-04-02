from __future__ import annotations

import json
import logging
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from gphotos.config import DEFAULT_CLIENT_SECRET, DEFAULT_TOKEN_PATH, DEFAULT_OAUTH_SCOPES

logger = logging.getLogger(__name__)

SCOPES = list(DEFAULT_OAUTH_SCOPES)


def _require_desktop_oauth_client(path: str) -> None:
    """InstalledAppFlow + run_local_server need a Desktop client, not Web application."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "installed" in cfg:
        return
    if "web" in cfg:
        raise ValueError(
            f"{path!r} is a 'Web application' OAuth client. This app uses a local redirect "
            "and expects a 'Desktop app' client: APIs & Services → Credentials → "
            "Create Credentials → OAuth client ID → Application type: Desktop app, "
            "then download the JSON (it contains an 'installed' block) and replace your client file."
        )


def get_credentials(
    client_secrets_path: str | None = None,
    token_path: str | None = None,
) -> Credentials:
    """OAuth user credentials for Google Photos (readonly). Persists token to disk."""
    client_secrets_path = client_secrets_path or DEFAULT_CLIENT_SECRET
    token_path = token_path or DEFAULT_TOKEN_PATH

    # Load with scopes=None so granted scopes come from the file. Passing SCOPES here
    # would set credential.scopes to requested scopes even if the stored token was
    # issued for fewer scopes, leading to 403 "insufficient authentication scopes".
    creds: Credentials | None = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes=None)

    if creds and not creds.has_scopes(SCOPES):
        logger.info(
            "Saved token is missing required scope(s) %s; run browser sign-in again.",
            SCOPES,
        )
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            _require_desktop_oauth_client(client_secrets_path)
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, SCOPES)
            creds = flow.run_local_server(port=0)
        _save_credentials(creds, token_path)

    return creds


def _save_credentials(creds: Credentials, token_path: str) -> None:
    os.makedirs(os.path.dirname(token_path) or ".", exist_ok=True)
    with open(token_path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def ensure_fresh_credentials(creds: Credentials, token_path: str | None = None) -> Credentials:
    """Refresh if expired and persist token when refreshed."""
    token_path = token_path or DEFAULT_TOKEN_PATH
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_credentials(creds, token_path)
    return creds
