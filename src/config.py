"""Central configuration for the naïve RAG application."""

from pathlib import Path


# Absolute path to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory containing source documents.
DATA_DIR = PROJECT_ROOT / "data"

# File types supported by the ingestion pipeline.
SUPPORTED_FILE_EXTENSIONS = {".pdf"}

# Chunking strategy used by the ingestion pipeline.
CHUNKING_STRATEGY = "fixed_size"

# Fixed-size chunking configuration.
FIXED_CHUNK_SIZE = 1000
FIXED_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100

# Recursive Chunking Strategy
RECURSIVE_CHUNK_SIZE = 1000
RECURSIVE_CHUNK_OVERLAP = 200

# Embedding configuration.
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"

# Number of documents used for the initial embedding test.
EMBEDDING_TEST_BATCH_SIZE = 5