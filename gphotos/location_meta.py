from __future__ import annotations

from typing import Any

# Chroma metadata values must be str, int, float, or bool — no nested dicts.


def location_fields_for_chroma(media_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """
    Flatten GPS / place info from Google Photos-style mediaMetadata when present.

    Library API often omits location; Picker or future fields may include `location`
    with `latlng` or flat coordinates.
    """
    out: dict[str, Any] = {}
    if not media_metadata:
        return out

    loc = media_metadata.get("location")
    if isinstance(loc, dict):
        ll = loc.get("latlng") or loc.get("LatLng")
        if isinstance(ll, dict):
            lat, lng = ll.get("latitude"), ll.get("longitude")
            if isinstance(lat, (int, float)):
                out["latitude"] = float(lat)
            if isinstance(lng, (int, float)):
                out["longitude"] = float(lng)
        else:
            lat, lng = loc.get("latitude"), loc.get("longitude")
            if isinstance(lat, (int, float)):
                out["latitude"] = float(lat)
            if isinstance(lng, (int, float)):
                out["longitude"] = float(lng)

        for key in ("name", "address", "locationName", "displayName"):
            val = loc.get(key)
            if isinstance(val, str) and val.strip():
                out["location_name"] = val.strip()[:512]
                break

    # Rare: coordinates at top level of mediaMetadata
    if "latitude" not in out or "longitude" not in out:
        lat, lng = media_metadata.get("latitude"), media_metadata.get("longitude")
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            out["latitude"] = float(lat)
            out["longitude"] = float(lng)

    if "latitude" in out and "longitude" in out:
        out["has_location"] = True

    return out
