"""Test the document embedding stage on a small sample."""

from src.chunkers import chunk_documents
from src.config import (
    CHUNKING_STRATEGY,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_TEST_BATCH_SIZE,
)
from src.embeddings import (
    embed_documents,
    get_embedding_model,
)
from src.loaders import load_documents


def main() -> None:
    """Load, chunk, and embed a small document sample."""
    try:
        documents = load_documents()

        if not documents:
            raise ValueError(
                "No page-level documents were loaded."
            )

        chunks = chunk_documents(
            documents,
            strategy=CHUNKING_STRATEGY,
        )

        if not chunks:
            raise ValueError(
                "No chunks were created."
            )

        sample_size = min(
            EMBEDDING_TEST_BATCH_SIZE,
            len(chunks),
        )

        sample_chunks = chunks[:sample_size]

        print("\nEmbedding test configuration:")
        print(f"  Provider: {EMBEDDING_PROVIDER}")
        print(f"  Model: {EMBEDDING_MODEL}")
        print(f"  Sample chunks: {sample_size}")

        embedding_model = get_embedding_model()

        vectors = embed_documents(
            documents=sample_chunks,
            embedding_model=embedding_model,
        )

    except ValueError as error:
        print(f"\nEmbedding test failed: {error}")
        raise SystemExit(1) from error

    except Exception as error:
        print(
            "\nUnexpected embedding error: "
            f"{type(error).__name__}: {error}"
        )
        raise SystemExit(1) from error

    print("\nEmbedding test completed successfully.")
    print(f"  Number of vectors: {len(vectors)}")
    print(
        f"  Embedding dimension: {len(vectors[0])}"
    )

    print("\nFirst vector preview:")
    print(vectors[0][:10])


if __name__ == "__main__":
    main()