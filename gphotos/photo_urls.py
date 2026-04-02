"""Google Photos web links from API media item ids."""


def google_photos_photo_url(media_item_id: str) -> str:
    """Open a library item in the Google Photos web UI (user must be signed in)."""
    return f"https://photos.google.com/lr/photo/{media_item_id}"
