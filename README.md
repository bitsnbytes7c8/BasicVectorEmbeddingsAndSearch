# google_photos

Use the **[Google Photos Picker API](https://developers.google.com/photos/picker/guides/get-started-picker)** to choose photos or videos in the Google Photos UI, resize thumbnails to 448×448, run a local **Ollama** vision model (**llama3.2-vision** by default), and optionally store descriptions + embeddings in **ChromaDB**.

**Policy context:** Google [no longer allows](https://developers.google.com/photos/support/updates) bulk listing of a user’s full library via the Library API for typical apps. This project uses only the **Picker** flow so users explicitly select what to process.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dependencies

- **google-auth-oauthlib** — OAuth for Google APIs
- **google-api-python-client** — Google API client (Picker API uses dynamic discovery; network required once)
- **chromadb** — vector store for vision descriptions
- **ollama** — local LLM client
- **Pillow** — image resizing

## Google Cloud & Ollama

### 1. Google Cloud OAuth

1. In [Google Cloud Console](https://console.cloud.google.com/), create or select a project.
2. Enable **Google Photos Picker API** (APIs & Services → Library → Enable).
3. Create **OAuth 2.0 Client ID** of type **Desktop app** and save the JSON as `client_secret.json` (or set `GPHOTOS_CLIENT_SECRET`).
4. Under **OAuth consent screen** → **Scopes**, add:
   - `https://www.googleapis.com/auth/photospicker.mediaitems.readonly`

After changing scopes, **delete `token.json`** and sign in again.

### 2. Ollama

Install [Ollama](https://ollama.com/) and ensure it is running.

**Vision model** (describe images; override with `--model` / `OLLAMA_VISION_MODEL`):

```bash
ollama pull llama3.2-vision
```

**Embedding model** (ChromaDB text vectors; override with `--embed-model` / `OLLAMA_EMBED_MODEL`). **Pull this** before the first run unless you use `--no-chroma`:

```bash
ollama pull nomic-embed-text
```

If Ollama reports that `nomic-embed-text` is missing, run the command above while Ollama is running.

**Text / chat model** (default `llama3.2`): used to **route** plain `query_photos.py` input (search vs RAG) and to **answer** RAG questions. One pull covers both unless you set a different `OLLAMA_ROUTER_MODEL`.

```bash
ollama pull llama3.2
```

### 3. Run

```bash
python run.py
```

A browser opens the Google Photos picker; select items, finish in Photos, then the app downloads thumbnails, runs vision, and (by default) upserts into Chroma.

### 4. Query the index (separate from ingest)

Use the same Chroma path/collection and embedding model as indexing (defaults match `run.py`).

Pass **plain text**. An **Ollama LLM** (router) classifies the message as either:

- **search** — find ranked photos (keywords, scenes, “show me…”) → prints **`mediaItemId`**, **Google Photos** URL, description snippet.
- **ask** — answer a question across descriptions (RAG) → prints prose.

If the router model is missing, unreachable, or returns invalid JSON, the command **fails with an error** (no rule-based fallback).

```bash
python query_photos.py birthday cake indoors
python query_photos.py Which countries have I eaten cake in?
```

**Overrides:** `--search` or `--ask` skips the router and forces a mode. `--router-model` / `OLLAMA_ROUTER_MODEL` sets the routing model (defaults to the chat model).

Tune retrieval with `--top-k` (default **10** for search, **24** for ask). For RAG answers, `--chat-model` / `OLLAMA_CHAT_MODEL` selects the synthesis model.

## What it does

1. Creates a Picker **session**, opens **`pickerUri`** (with `/autoclose` unless `--picker-no-autoclose`).
2. Polls until you finish picking.
3. Lists selected media, runs download → Ollama vision → optional **ChromaDB** upsert (`mediaItemId` as id, HNSW/cosine, Ollama embeddings of the description). If the API provides **location** metadata, **latitude** / **longitude** / **location_name** / **has_location** are stored when possible.

## Command-line options

| Option | Description |
|--------|-------------|
| `--client-secret` | OAuth client JSON (default `client_secret.json` or `GPHOTOS_CLIENT_SECRET`) |
| `--token` | Saved OAuth token (default `token.json` or `GPHOTOS_TOKEN`) |
| `--system-prompt` | System prompt for the vision model |
| `--user-prompt` | User message with the image (default: `Describe this image.`) |
| `--model` | Ollama vision model (default `llama3.2-vision` or `OLLAMA_VISION_MODEL`) |
| `--picker-no-autoclose` | Don’t append `/autoclose` to the picker URL |
| `--no-chroma` | Don’t write to ChromaDB |
| `--chroma-path` | Chroma persist directory (default `data/chroma_db`) |
| `--chroma-collection` | Collection name (default `gphotos_vision`) |
| `--embed-model` | Ollama embedding model (default `nomic-embed-text`) |
| `-v`, `--verbose` | Debug logging |

### `query_photos.py` options

| Option | Description |
|--------|-------------|
| `query` (words) | Plain-text query or question |
| `--search` / `--ask` | Force semantic search or RAG (skip LLM router) |
| `--router-model` | Ollama model for routing (default `OLLAMA_ROUTER_MODEL`) |
| `--top-k` | Vectors to retrieve (defaults: 10 / 24 by mode) |
| `--chroma-path`, `--chroma-collection`, `--embed-model` | Same as `run.py` |
| `--chat-model` | Ollama model for RAG answers (`OLLAMA_CHAT_MODEL`) |

## Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `GPHOTOS_CLIENT_SECRET` | Path to OAuth client JSON |
| `GPHOTOS_TOKEN` | Path to OAuth token file |
| `OLLAMA_VISION_MODEL` | Default vision model |
| `OLLAMA_EMBED_MODEL` | Default embedding model for Chroma |
| `OLLAMA_CHAT_MODEL` | Text model for RAG answers (default `llama3.2`) |
| `OLLAMA_ROUTER_MODEL` | Text model for search vs ask routing (defaults to `OLLAMA_CHAT_MODEL`) |
| `GPHOTOS_CHROMA_PATH` | Chroma persist directory |
| `GPHOTOS_CHROMA_COLLECTION` | Chroma collection name |
| `GPHOTOS_QUEUE_SIZE` | Producer–consumer queue size (default `32`) |

---

_Add project-specific notes below as you build._
