"""
Ingestion pipeline: load documents → split → embed → store in ChromaDB.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    KNOWLEDGE_BASE_PATH,
)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a sentence-transformer embedding model (runs locally)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def ingest_documents() -> Chroma:
    """
    Read the knowledge-base file, split it into chunks,
    embed each chunk and persist to ChromaDB.
    """
    # --- load -----------------------------------------------------------------
    loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding="utf-8")
    documents = loader.load()

    # --- split ----------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split knowledge base into {len(chunks)} chunks")

    # --- embed & store --------------------------------------------------------
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"  Vector store created at {CHROMA_PERSIST_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
