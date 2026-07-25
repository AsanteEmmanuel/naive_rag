"""Build and reload a persistent Chroma vector database."""

from src.chunkers import chunk_documents
from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    CHUNKING_STRATEGY,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)
from src.embeddings import get_embedding_model
from src.loaders import load_documents
from src.vectordb import (
    create_vector_store,
    delete_vector_store,
    get_vector_count,
    load_vector_store,
)


def main() -> None:
    """Create, verify, and reload the Chroma vector store."""
    try:
        print("Loading documents...")
        documents = load_documents()

        if not documents:
            raise ValueError(
                "No page-level documents were loaded."
            )

        print("\nChunking documents...")
        chunks = chunk_documents(
            documents,
            strategy=CHUNKING_STRATEGY,
        )

        if not chunks:
            raise ValueError(
                "No chunks were created."
            )

        print("\nVector database configuration:")
        print(f"  Provider: {EMBEDDING_PROVIDER}")
        print(f"  Embedding model: {EMBEDDING_MODEL}")
        print(f"  Collection: {CHROMA_COLLECTION_NAME}")
        print(
            f"  Persistence directory: "
            f"{CHROMA_PERSIST_DIRECTORY}"
        )
        print(f"  Chunks to store: {len(chunks)}")

        embedding_model = get_embedding_model()

        print("\nRemoving any previous local vector database...")
        delete_vector_store()

        print("Creating vector database...")
        vector_store = create_vector_store(
            documents=chunks,
            embedding_model=embedding_model,
        )

        stored_count = get_vector_count(vector_store)

        print("\nInitial database verification:")
        print(f"  Expected records: {len(chunks)}")
        print(f"  Stored records: {stored_count}")

        if stored_count != len(chunks):
            raise ValueError(
                "Stored record count does not match chunk count. "
                f"Expected {len(chunks)}, found {stored_count}."
            )

        # Remove the active wrapper from memory before reloading.
        del vector_store

        print("\nReloading vector database from disk...")

        reloaded_store = load_vector_store(
            embedding_model=embedding_model,
        )

        reloaded_count = get_vector_count(reloaded_store)

        print("\nReloaded database verification:")
        print(f"  Reloaded records: {reloaded_count}")

        if reloaded_count != len(chunks):
            raise ValueError(
                "Reloaded record count does not match chunk count. "
                f"Expected {len(chunks)}, found {reloaded_count}."
            )

    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"\nVector database test failed: {error}")
        raise SystemExit(1) from error

    except Exception as error:
        print(
            "\nUnexpected vector database error: "
            f"{type(error).__name__}: {error}"
        )
        raise SystemExit(1) from error

    print("\nVector database test completed successfully.")
    print(f"  Collection: {CHROMA_COLLECTION_NAME}")
    print(f"  Stored chunks: {reloaded_count}")
    print(
        f"  Database location: "
        f"{CHROMA_PERSIST_DIRECTORY}"
    )


if __name__ == "__main__":
    main()