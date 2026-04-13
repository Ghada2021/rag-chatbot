"""
Configuration for the RAG chatbot.
Adjust these settings to match your environment.
"""

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"          # change to whichever model you pulled

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # runs locally via sentence-transformers

# ---------------------------------------------------------------------------
# ChromaDB (vector store)
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "star_assurances"

# ---------------------------------------------------------------------------
# RAG parameters
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between chunks
TOP_K = 3                 # number of chunks to retrieve

# ---------------------------------------------------------------------------
# Knowledge-base path
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_PATH = "./data/knowledge_base.txt"
