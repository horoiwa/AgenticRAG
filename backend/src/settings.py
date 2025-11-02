import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
ELASTIC_SEARCH_HOST = os.getenv("ELASTIC_SEARCH_HOST", None)
INDEX_NAME = "documents"
USE_DEVICE = "cpu"  # cpu or cuda
EMBEDDING_MODEL = "cl-nagoya/ruri-v3-30m"
EMBEDDING_DIM = 256
CHUNK_SIZE = 2048
CONTEXT_SIZE = 5  # num chunks include as context
