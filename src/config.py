import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "sample_docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "study_materials"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
