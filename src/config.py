"""Central configuration for the naïve RAG application."""

from pathlib import Path


# Absolute path to the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory containing source documents.
DATA_DIR = PROJECT_ROOT / "data"

# File types supported by the ingestion pipeline.
SUPPORTED_FILE_EXTENSIONS = {".pdf"}