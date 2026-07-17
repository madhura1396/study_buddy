import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "sample_docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "study_materials"

EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

GOOGLE_CLIENT_SECRET_PATH = PROJECT_ROOT / "client_secret.json"
GOOGLE_TOKEN_PATH = PROJECT_ROOT / "token.json"
# drive.readonly: browse/search/download the user's existing Drive files.
# drive.file: create/update files this app itself creates (used to upload
# study docs as Google Docs) — deliberately not the broader "drive" scope,
# which would grant write access to every file in the user's Drive.
GOOGLE_DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]
