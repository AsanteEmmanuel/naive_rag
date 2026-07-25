"""Embedding utilities for the naïve RAG application."""

import os
from collections.abc import Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)


def load_environment() -> None:
    """Load environment variables from the project's .env file."""
    load_dotenv()


def validate_openai_api_key() -> None:
    """
    Verify that an OpenAI API key is available.

    Raises:
        ValueError: If OPENAI_API_KEY is missing or empty.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Add it to the repository's .env file."
        )


def get_embedding_model(
    provider: str = EMBEDDING_PROVIDER,
    model_name: str = EMBEDDING_MODEL,
) -> Embeddings:
    """
    Create an embedding model for the selected provider.

    Args:
        provider: Name of the embedding provider.
        model_name: Name of the provider's embedding model.

    Returns:
        A LangChain-compatible embedding model.

    Raises:
        ValueError: If the provider is unsupported or configuration is missing.
    """
    load_environment()

    normalized_provider = provider.strip().lower()

    if normalized_provider == "openai":
        validate_openai_api_key()

        return OpenAIEmbeddings(
            model=model_name,
        )

    raise ValueError(
        f"Unsupported embedding provider: {provider!r}. "
        "Supported providers: openai."
    )


def extract_document_texts(
    documents: Sequence[Document],
) -> list[str]:
    """
    Extract nonempty text from LangChain documents.

    Args:
        documents: Documents whose text will be embedded.

    Returns:
        A list of cleaned document strings.

    Raises:
        ValueError: If a document contains empty text.
    """
    texts: list[str] = []

    for index, document in enumerate(documents):
        text = document.page_content.strip()

        if not text:
            raise ValueError(
                f"Document at index {index} contains no text."
            )

        texts.append(text)

    return texts


def embed_documents(
    documents: Sequence[Document],
    embedding_model: Embeddings,
) -> list[list[float]]:
    """
    Generate one embedding vector for each document.

    Args:
        documents: Chunk-level documents to embed.
        embedding_model: LangChain-compatible embedding model.

    Returns:
        A list containing one vector per document.

    Raises:
        ValueError: If no documents are supplied or the output is inconsistent.
    """
    if not documents:
        raise ValueError(
            "At least one document is required for embedding."
        )

    texts = extract_document_texts(documents)
    vectors = embedding_model.embed_documents(texts)

    if len(vectors) != len(documents):
        raise ValueError(
            "Embedding count does not match document count. "
            f"Documents: {len(documents)}, vectors: {len(vectors)}."
        )

    if not vectors or not vectors[0]:
        raise ValueError(
            "The embedding model returned empty vectors."
        )

    expected_dimension = len(vectors[0])

    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            raise ValueError(
                "Embedding dimensions are inconsistent. "
                f"Vector 0 has dimension {expected_dimension}, "
                f"but vector {index} has dimension {len(vector)}."
            )

    return vectors