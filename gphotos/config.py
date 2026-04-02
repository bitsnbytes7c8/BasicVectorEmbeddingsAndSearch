import os

# Picker API: user selects photos/videos in Google Photos UI; your app receives those items only.
PHOTOS_PICKER_SCOPE = "https://www.googleapis.com/auth/photospicker.mediaitems.readonly"

DEFAULT_OAUTH_SCOPES = [PHOTOS_PICKER_SCOPE]

DEFAULT_CLIENT_SECRET = os.environ.get("GPHOTOS_CLIENT_SECRET", "client_secret.json")
DEFAULT_TOKEN_PATH = os.environ.get("GPHOTOS_TOKEN", "token.json")

THUMBNAIL_SIZE = 448
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "llama3.2-vision")

# Text embeddings for Chroma (vision descriptions); must match `ollama pull <model>`.
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

DEFAULT_CHROMA_PATH = os.environ.get("GPHOTOS_CHROMA_PATH", "data/chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("GPHOTOS_CHROMA_COLLECTION", "gphotos_vision")

QUEUE_MAXSIZE = int(os.environ.get("GPHOTOS_QUEUE_SIZE", "32"))

# Text-only LLM for Chroma RAG answers (`query_photos.py` ask path).
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")

# Classifies plain user input as search vs ask (`query_photos.py`). Defaults to chat model.
OLLAMA_ROUTER_MODEL = os.environ.get("OLLAMA_ROUTER_MODEL", OLLAMA_CHAT_MODEL)
