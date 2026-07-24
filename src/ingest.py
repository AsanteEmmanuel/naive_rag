"""Load PDF documents and split them into retrieval-sized chunks."""

from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from src.chunkers import chunk_documents
from src.config import (
    CHUNKING_STRATEGY,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
)
from src.loaders import load_documents


def get_source_name(document: Document) -> str:
    """
    Extract a readable filename from document metadata.

    Args:
        document: A LangChain document.

    Returns:
        The source filename or a fallback label.
    """
    source = document.metadata.get("source")

    if not source:
        return "unknown_source"

    return Path(str(source)).name


def print_source_summary(
    documents: list[Document],
    unit_name: str,
) -> None:
    """
    Print document counts grouped by source.

    Args:
        documents: Documents to summarize.
        unit_name: Description of what each document represents.
    """
    source_counts = Counter(
        get_source_name(document)
        for document in documents
    )

    print(f"\n{unit_name.capitalize()} by source:")

    for source_name, count in sorted(source_counts.items()):
        print(f"  {source_name}: {count}")


def print_chunk_statistics(
    chunks: list[Document],
) -> None:
    """
    Print descriptive statistics for generated chunks.

    Args:
        chunks: Chunk-level documents.
    """
    if not chunks:
        print("\nNo chunks were created.")
        return

    chunk_lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print("\nChunk-size statistics:")
    print(f"  Total chunks: {len(chunks)}")
    print(
        f"  Minimum: {min(chunk_lengths)} characters"
    )
    print(
        f"  Maximum: {max(chunk_lengths)} characters"
    )
    print(
        "  Average: "
        f"{sum(chunk_lengths) / len(chunk_lengths):.1f} "
        "characters"
    )


def preview_chunk(
    chunk: Document,
    preview_length: int = 500,
) -> None:
    """
    Print one chunk's metadata and content.

    Args:
        chunk: Chunk-level document to preview.
        preview_length: Maximum number of characters to print.
    """
    content = chunk.page_content.strip()

    print("\nSample chunk metadata:")
    print(chunk.metadata)

    print("\nSample chunk content:")
    print("-" * 60)
    print(content[:preview_length])

    if len(content) > preview_length:
        print("...")

    print("-" * 60)


def main() -> None:
    """Run document loading and chunking."""
    try:
        documents = load_documents()

        if not documents:
            raise ValueError(
                "No page-level documents were loaded."
            )

        print("\nDocument loading completed.")
        print(
            f"Total page documents: {len(documents)}"
        )

        print_source_summary(
            documents,
            unit_name="pages",
        )

        chunks = chunk_documents(
            documents,
            strategy=CHUNKING_STRATEGY,
        )

        if not chunks:
            raise ValueError(
                "The chunking process produced no chunks."
            )

    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as error:
        print(f"\nIngestion failed: {error}")
        raise SystemExit(1) from error

    except Exception as error:
        print(f"\nUnexpected ingestion error: {error}")
        raise SystemExit(1) from error

    print("\nChunking completed.")
    print(f"Strategy: {CHUNKING_STRATEGY}")

    if CHUNKING_STRATEGY == "fixed_size":
        print(
            f"Target chunk size: "
            f"{FIXED_CHUNK_SIZE} characters"
        )
        print(
            f"Chunk overlap: "
            f"{FIXED_CHUNK_OVERLAP} characters"
        )
        print(
            f"Minimum chunk size: "
            f"{MIN_CHUNK_SIZE} characters"
        )

    print_source_summary(
        chunks,
        unit_name="chunks",
    )

    print_chunk_statistics(chunks)
    preview_chunk(chunks[0])


if __name__ == "__main__":
    main()