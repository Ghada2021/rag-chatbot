"""
FastAPI application — serves the chatbot API and the web UI.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import CHROMA_PERSIST_DIR
from app.ingestion import ingest_documents, load_vectorstore
from app.rag_chain import build_rag_chain


# ---------------------------------------------------------------------------
# Lifespan: ingest docs + build chain once at startup
# ---------------------------------------------------------------------------
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain

    # If vector store already exists, just load it; otherwise ingest first
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("  Loading existing vector store …")
        vectorstore = load_vectorstore()
    else:
        print("  Ingesting knowledge base …")
        vectorstore = ingest_documents()

    rag_chain = build_rag_chain(vectorstore)
    print("🚀  RAG chain ready!")
    yield


app = FastAPI(
    title="STAR Assurances — RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (the chat UI)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the chat interface."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a question to the RAG chain and return the answer + sources."""
    result = rag_chain.invoke({"question": req.question})

    sources = [
        SourceChunk(
            content=doc.page_content,
            metadata=doc.metadata,
        )
        for doc in result.get("source_documents", [])
    ]

    return ChatResponse(answer=result["answer"], sources=sources)


@app.post("/api/reingest")
async def reingest():
    """Re-ingest the knowledge base (call after updating the text file)."""
    global rag_chain
    vectorstore = ingest_documents()
    rag_chain = build_rag_chain(vectorstore)
    return {"status": "ok", "message": "Knowledge base re-ingested successfully."}


@app.get("/api/health")
async def health():
    return {"status": "ok", "chain_loaded": rag_chain is not None}
