import os
from pathlib import Path

from dotenv import load_dotenv


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
ELASTIC_SEARCH_HOST = os.getenv("ELASTIC_SEARCH_HOST", None)
DEFAULT_INDEX_NAME = "documents-v1"
USE_DEVICE = "cpu"  # cpu or cuda
EMBEDDING_MODEL = "cl-nagoya/ruri-v3-30m"
EMBEDDING_DIM = 256
CHUNK_SIZE = 2048
NUM_CONTEXT_CHUNKS = 2
SUPPORTED_EXT = [".pdf", ".docx", ".pptx", ".txt"]
RRF_RANK_CONST = 60
RRF_TOP_K = 20
LLM = "gemini-2.5-flash-lite"
