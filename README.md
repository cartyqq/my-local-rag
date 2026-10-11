# Local RAG System with Ollama & Streamlit

A fully local Retrieval-Augmented Generation (RAG) application for chatting with your PDF documents, text files and system logs. The LLM, the embedding model and the vector database all run on your own machine, so your data never leaves it.

## Features

- **100% local:** no external API calls, no API keys. Inference runs through [Ollama](https://ollama.ai/).
- **Multi-format input:** the web UI analyzes `.pdf`, `.txt` and `.log` files.
- **Two interfaces:** a Streamlit web UI with chat history and source fragments, and a command-line interface tuned for log analysis.
- **Persistent vector storage:** chunking with LangChain, embeddings from `all-MiniLM-L6-v2`, storage in a local ChromaDB shared by both interfaces.
- **Transparent answers:** the web UI shows which fragments (file and page) were used to produce each answer.
- **Docker-ready:** deploy with Docker and Docker Compose, or run manually in a Miniconda environment. The embedding model is baked into the image, so the container works offline.

## How it works

```
 PDF / TXT / LOG
        │
        ▼
 RecursiveCharacterTextSplitter ──► chunks
        │
        ▼
 all-MiniLM-L6-v2 embeddings ──► ChromaDB (./chroma_db)
                                        │
 Your question ──► similarity search ◄──┘
        │                 │
        └──── question + top-k chunks ────► Ollama (/api/generate) ──► answer
```

## Tech Stack

| Component     | Technology                                     |
|---------------|------------------------------------------------|
| Language      | Python 3.10+                                   |
| LLM engine    | Ollama (default model: `qwen2.5:3b`)           |
| Orchestration | LangChain (`langchain-community`, `langchain-huggingface`, `langchain-chroma`) |
| Embeddings    | `sentence-transformers/all-MiniLM-L6-v2`       |
| Vector DB     | ChromaDB                                       |
| Frontend      | Streamlit                                      |
| Infrastructure| Docker, Docker Compose                         |

---

## Prerequisites

Make sure the following are installed on your host machine:

1. **[Ollama](https://ollama.ai/)**: the engine that runs the LLM locally.
2. **[Git](https://git-scm.com/)**: to clone this repository.
3. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** (recommended) **or** **[Miniconda](https://docs.anaconda.com/free/miniconda/)** / Python 3.10+ for a manual setup.

> **Hardware:** the default `qwen2.5:3b` model is a ~2 GB download and runs on a modern CPU, though a GPU makes responses much faster. Larger models need proportionally more RAM/VRAM.

---

## Installation & Setup

### 1. Install and start Ollama

Ollama runs on the host (not in a container) so it can use your GPU directly. Install it, then pull the default model:

```bash
ollama pull qwen2.5:3b
```

Make sure the server is running (the desktop app starts it automatically; otherwise run `ollama serve`) and check that it responds:

```bash
curl http://localhost:11434/api/tags
```

> **Linux + Docker:** by default Ollama listens only on `127.0.0.1`, which a container cannot reach. Start it with `OLLAMA_HOST=0.0.0.0 ollama serve`, or set that variable in the systemd service (`systemctl edit ollama`). If you do this, make sure port `11434` is not exposed to untrusted networks, since Ollama has no authentication. On Windows and macOS with Docker Desktop this is usually not needed.

### 2. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 3a. Run with Docker (recommended)

```bash
docker compose up --build
```

Then open **http://localhost:8501**.

The provided `docker-compose.yml` already takes care of the details:

- `OLLAMA_BASE_URL=http://host.docker.internal:11434` tells the app where to find Ollama on the host.
- `extra_hosts: host.docker.internal:host-gateway` makes that hostname work on Linux (it is harmless on Windows and macOS).
- `./chroma_db:/app/chroma_db` keeps the indexed documents between restarts.
- `HF_HUB_OFFLINE=1` prevents any contact with HuggingFace at runtime, because the embedding model is baked into the image.
- The port is published as `127.0.0.1:8501`, so the UI is reachable only from your own machine. To open it to your network, change it to `"8501:8501"`.

### 3b. Run manually with Miniconda

```bash
conda create -n rag python=3.10 -y
conda activate rag

pip install -r requirements.txt

streamlit run app.py
```

The app is available at **http://localhost:8501**. The embedding model (~90 MB) is downloaded from HuggingFace on the first run and cached afterwards.

---

## Usage

### Web UI

1. In the sidebar, upload a `.pdf`, `.txt` or `.log` file.
2. Click **Проиндексировать документ** ("Index document") and wait for the success message.
3. Ask questions in the chat. Expand **Использованные фрагменты** ("Used fragments") under an answer to see the exact source chunks.

Sidebar controls:

| Control | Purpose |
|---|---|
| Ollama model | Pick any model you have pulled in Ollama |
| top-k slider | How many chunks are sent to the LLM as context |
| Clear chat | Reset the conversation history |
| Clear database | Remove all indexed documents from ChromaDB |

Re-uploading a file with the same name replaces its previous chunks instead of duplicating them.

### Command line

`main.py` is a lightweight CLI tuned for log analysis. It reads plain-text files only (PDF is supported in the web UI):

```bash
# Index a log file
python main.py index app.log

# Ask a single question
python main.py ask "Why did the container crash?"

# Index and start an interactive session
python main.py chat --pdf app.log
```

> **Note:** the web UI and the CLI share the same ChromaDB collection (`rag_docs`) in `./chroma_db`, so documents indexed in one are visible in the other. Unlike the web UI, the CLI does not remove older chunks of a file when you index it again, so re-indexing the same file creates duplicates.

---

## Configuration

Settings live as constants at the top of `app.py` and `main.py`. The Ollama address can also be set with an environment variable.

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` (env) | `http://localhost:11434` | Address of the Ollama server, without a path. Set automatically in `docker-compose.yml` |
| `HF_HUB_OFFLINE` (env) | unset (`1` in Docker) | Set to `1` to forbid downloads from HuggingFace once the model is cached |
| `DEFAULT_MODEL` (UI) / `OLLAMA_MODEL` (CLI) | `qwen2.5:3b` | Model used for generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` (UI), `1500` / `300` (CLI) | Chunking parameters; larger chunks help with long tracebacks |
| `TOP_K` (CLI) / top-k slider (UI) | `4` | Number of retrieved chunks per question |
| `CHROMA_DIR` | `./chroma_db` | Where the vector database is stored |
| `COLLECTION_NAME` | `rag_docs` | ChromaDB collection used by both interfaces |

Example of pointing the app at a remote Ollama server:

```bash
OLLAMA_BASE_URL=http://192.168.1.50:11434 streamlit run app.py
```

---

## Project structure

```
.
├── app.py              # Streamlit web interface
├── main.py             # Command-line interface
├── requirements.txt    # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── chroma_db/          # Local vector database (created on first indexing)
└── README.md
```

---

## Troubleshooting

**"Ollama недоступна" / connection refused**
Ollama is not running or is unreachable. Start it with `ollama serve` and verify with `curl http://localhost:11434/api/tags`. Inside Docker, `localhost` points to the container itself, so the app uses `OLLAMA_BASE_URL` (`host.docker.internal`) instead. On Linux, also make sure Ollama listens on `0.0.0.0` (see the note in step 1).

**Model not found (404)**
Pull the model first: `ollama pull <model-name>`. The names in the sidebar dropdown come from `ollama list`.

**"Could not extract text from the file"**
The PDF is probably a scan without a text layer. Run OCR on it first (for example with `ocrmypdf`) and upload the result.

**Garbled characters in logs**
Text files are decoded as UTF-8. Convert logs in other encodings first, e.g. `iconv -f cp1251 -t utf-8 old.log > new.log`.

**Answers are vague or wrong**
Try a larger model, raise the top-k value, or ask a more specific question. The model is instructed to answer only from the retrieved fragments, so if the relevant part of the document was not retrieved, the answer will be incomplete.

**Slow first start**
The first question loads the LLM into memory, and when running outside Docker the first run also downloads the embedding model. Subsequent requests are faster.

**"database is locked" or slow indexing on Windows**
Bind mounts from the Windows file system into a container can be slow and sometimes break SQLite, which ChromaDB uses. Replace the bind mount in `docker-compose.yml` with a named volume:

```yaml
services:
  rag-app:
    volumes:
      - chroma_data:/app/chroma_db

volumes:
  chroma_data:
```
