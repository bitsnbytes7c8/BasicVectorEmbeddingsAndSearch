from __future__ import annotations

import ollama

from gphotos.config import OLLAMA_VISION_MODEL

DEFAULT_USER_PROMPT = "Describe this image."


def run_vision(
    image_png_bytes: bytes,
    system_prompt: str,
    user_prompt: str = DEFAULT_USER_PROMPT,
    model: str | None = None,
) -> str:
    """Send a PNG to Ollama vision model; return assistant text."""
    model = model or OLLAMA_VISION_MODEL
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [image_png_bytes],
            },
        ],
    )
    content = resp.message.content
    return content if isinstance(content, str) else str(content or "")
