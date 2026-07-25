"""Vector database utilities for the naïve RAG application."""

from collections.abc import Sequence
from pathlib import Path
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
)


def create_vector_store(
    documents: Sequence[Document],
    embedding_model: Embeddings,
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
) -> Chroma:
    """
    Embed documents and store them in a persistent Chroma collection.

    Args:
        documents: Chunk-level documents to store.
        embedding_model: Model Chroma will use to create embeddings.
        collection_name: Name of the Chroma collection.
        persist_directory: Directory where Chroma stores its files.

    Returns:
        The populated Chroma vector store.

    Raises:
        ValueError: If no documents are supplied.
    """
    if not documents:
        raise ValueError(
            "At least one document is required to create "
            "the vector store."
        )

    return Chroma.from_documents(
        documents=list(documents),
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def load_vector_store(
    embedding_model: Embeddings,
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
) -> Chroma:
    """
    Load an existing persistent Chroma collection.

    Args:
        embedding_model: Model used to embed future search queries.
        collection_name: Name of the existing collection.
        persist_directory: Directory containing the database.

    Returns:
        A Chroma vector store connected to the existing collection.

    Raises:
        FileNotFoundError: If the persistence directory does not exist.
    """
    database_path = Path(persist_directory)

    if not database_path.exists():
        raise FileNotFoundError(
            f"Chroma database directory does not exist: "
            f"{database_path.resolve()}"
        )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory,
    )


def get_vector_count(
    vector_store: Chroma,
) -> int:
    """
    Return the number of stored records in the Chroma collection.

    Args:
        vector_store: Chroma vector store to inspect.

    Returns:
        Number of records in the underlying collection.
    """
    return vector_store._collection.count()


def delete_vector_store(
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
) -> None:
    """
    Delete the local persistent Chroma database directory.

    This is useful when rebuilding the complete index from scratch.

    Args:
        persist_directory: Directory containing the database files.
    """
    database_path = Path(persist_directory)

    if database_path.exists():
        shutil.rmtree(database_path)