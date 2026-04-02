from __future__ import annotations

import urllib.error
import urllib.request
from io import BytesIO

from google.oauth2.credentials import Credentials
from PIL import Image, ImageOps

from gphotos.auth import ensure_fresh_credentials
from gphotos.config import THUMBNAIL_SIZE


def download_thumbnail_bytes(
    base_url: str,
    creds: Credentials,
    token_path: str | None = None,
    size: int = THUMBNAIL_SIZE,
) -> bytes:
    """Download a sized variant via Google Photos baseUrl (requires OAuth bearer)."""
    creds = ensure_fresh_credentials(creds, token_path)
    url = f"{base_url}=w{size}-h{size}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {creds.token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Thumbnail download failed: HTTP {e.code}") from e

    return fit_thumbnail_png(raw, size)


def fit_thumbnail_png(image_bytes: bytes, size: int = THUMBNAIL_SIZE) -> bytes:
    """Decode image, resize/cover to size×size, return PNG bytes."""
    im = Image.open(BytesIO(image_bytes))
    im = im.convert("RGB")
    out = ImageOps.fit(im, (size, size), Image.Resampling.LANCZOS)
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
