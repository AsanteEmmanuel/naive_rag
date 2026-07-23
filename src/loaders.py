"""Utilities for loading documents into the RAG pipeline."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from src.config import DATA_DIR


def find_pdf_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """
    Find all PDF files in the data directory.

    Args:
        data_dir: Directory containing the source documents.

    Returns:
        A sorted list of PDF file paths.

    Raises:
        FileNotFoundError: If the data directory does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data path is not a directory: {data_dir}")

    return sorted(data_dir.glob("*.pdf"))


def load_pdf(pdf_path: Path) -> list[Document]:
    """
    Load one PDF as LangChain Document objects.

    PyPDFLoader normally returns one Document per PDF page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Documents containing page text and metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the supplied path is not a PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {pdf_path.name}")

    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """
    Load every PDF in the data directory.

    Args:
        data_dir: Directory containing PDF files.

    Returns:
        A combined list of Documents from all PDFs.

    Raises:
        FileNotFoundError: If the directory has no PDF files.
    """
    pdf_files = find_pdf_files(data_dir)

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in {data_dir}. "
            "Add at least one PDF before running ingestion."
        )

    documents: list[Document] = []

    for pdf_path in pdf_files:
        pdf_documents = load_pdf(pdf_path)
        documents.extend(pdf_documents)

        print(f"Loaded: {pdf_path.name}")
        print(f"Pages: {len(pdf_documents)}")

    return documents