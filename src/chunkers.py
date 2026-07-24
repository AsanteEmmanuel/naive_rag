"""Chunking strategies for the naïve RAG application."""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from src.config import (
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
)


ChunkingFunction = Callable[[list[Document]], list[Document]]


def validate_chunk_settings(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Validate common chunking parameters.

    Args:
        chunk_size: Target number of characters in each chunk.
        chunk_overlap: Number of characters shared by adjacent chunks.

    Raises:
        ValueError: If either setting is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )


def fixed_size_chunking(
    documents: list[Document],
    chunk_size: int = FIXED_CHUNK_SIZE,
    chunk_overlap: int = FIXED_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into fixed-size overlapping character windows.

    This strategy does not attempt to preserve paragraphs, sentences,
    or words. It serves as the naïve baseline for later comparisons.

    Args:
        documents: Page-level LangChain documents.
        chunk_size: Target number of characters in each chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        Chunk-level documents with inherited and additional metadata.
    """
    validate_chunk_settings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not documents:
        return []

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_index": chunk_index,
                "character_count": len(chunk.page_content),
                "chunking_strategy": "fixed_size",
            }
        )

    return chunks


CHUNKING_STRATEGIES: dict[str, ChunkingFunction] = {
    "fixed_size": fixed_size_chunking
}


def chunk_documents(
    documents: list[Document],
    strategy: str = "fixed_size",
) -> list[Document]:
    """
    Chunk documents using the selected strategy.

    Args:
        documents: Page-level LangChain documents.
        strategy: Name of the chunking strategy to use.

    Returns:
        Chunk-level LangChain documents.

    Raises:
        ValueError: If the requested strategy is not registered.
    """
    normalized_strategy = strategy.strip().lower()

    chunking_function = CHUNKING_STRATEGIES.get(
        normalized_strategy
    )

    if chunking_function is None:
        supported_strategies = ", ".join(
            sorted(CHUNKING_STRATEGIES)
        )

        raise ValueError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Supported strategies: {supported_strategies}."
        )

    return chunking_function(documents)