#!/usr/bin/env python3
"""
Run this script once to ingest the knowledge base into the vector store
BEFORE starting the FastAPI server (optional — the server will auto-ingest
on first launch if the vector store doesn't exist yet).

Usage:
    python ingest.py
"""

from app.ingestion import ingest_documents

if __name__ == "__main__":
    ingest_documents()
    print("\n  Done! You can now start the server with:")
    print("    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
