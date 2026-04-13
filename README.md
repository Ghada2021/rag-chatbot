# 🤖 STAR Assurances — RAG Chatbot

A Retrieval-Augmented Generation chatbot built with **FastAPI**, **LangChain**, **Ollama** (on-premise LLM), and **ChromaDB**.

![Stack](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Stack](https://img.shields.io/badge/LangChain-121212?style=flat)
![Stack](https://img.shields.io/badge/Ollama-000?style=flat)
![Stack](https://img.shields.io/badge/ChromaDB-FF6F61?style=flat)

---

## Architecture

```
User ──▶ Chat UI (HTML/JS)
              │
              ▼
         FastAPI  /api/chat
              │
              ▼
     LangChain RAG Chain
       ┌───────┴───────┐
       │               │
   ChromaDB        Ollama (LLM)
   (retriever)     (generator)
       │
  HuggingFace
  Embeddings
  (local)
```

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Runtime |
| Ollama | latest | Local LLM server |

---

## Setup (5 minutes)

### 1. Install Ollama & pull a model

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (default: llama3.2 — change in app/config.py)
ollama pull llama3.2
```

### 2. Install Python dependencies

```bash
cd rag-chatbot
pip install -r requirements.txt
```

### 3. Start Ollama server

```bash
ollama serve        # runs on http://localhost:11434
```

### 4. Launch the chatbot

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

On first launch the server will automatically ingest the knowledge base and create the vector store. Subsequent launches will reuse the persisted store.

---

## Project Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── config.py        # all tuneable parameters
│   ├── ingestion.py     # load → split → embed → store
│   ├── rag_chain.py     # LangChain conversational RAG chain
│   └── main.py          # FastAPI app + routes
├── data/
│   └── knowledge_base.txt   # your knowledge base (edit this!)
├── static/
│   └── index.html       # chat UI
├── ingest.py            # standalone ingestion script
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serves the chat UI |
| `POST` | `/api/chat` | Send a question, get an answer + sources |
| `POST` | `/api/reingest` | Re-ingest the knowledge base after edits |
| `GET`  | `/api/health` | Health check |

### Example API call

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel est le prix de DAR ESSLAMA ?"}'
```

---

## Configuration

All settings live in `app/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K` | `3` | Number of chunks retrieved |

---

## Extending the Knowledge Base

1. Edit `data/knowledge_base.txt` — add new sections separated by `---`
2. Call the re-ingestion endpoint:
   ```bash
   curl -X POST http://localhost:8000/api/reingest
   ```
3. That's it — the chatbot now knows the new content.

You can also add more files by modifying `ingestion.py` to use `DirectoryLoader` instead of `TextLoader`.

---

## Tips

- **Larger models = better answers**: try `llama3.1:8b` or `mistral` if `llama3.2` feels limited.
- **GPU acceleration**: Ollama auto-detects NVIDIA GPUs. No config needed.
- **Production**: add `gunicorn` with `uvicorn` workers, put behind nginx, and add auth.
