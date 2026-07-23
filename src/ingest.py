"""Entry point for loading and inspecting source documents."""

from langchain_core.documents import Document

from src.loaders import load_documents


def preview_document(document: Document, preview_length: int = 500) -> None:
    """
    Print a short preview of a LangChain Document.

    Args:
        document: Document to inspect.
        preview_length: Maximum number of characters to print.
    """
    content = document.page_content.strip()

    print("\nFirst document metadata:")
    print(document.metadata)

    print("\nFirst document content preview:")
    print("-" * 60)
    print(content[:preview_length])

    if len(content) > preview_length:
        print("...")

    print("-" * 60)


def main() -> None:
    """Load PDFs and print basic ingestion statistics."""
    try:
        documents = load_documents()
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"Ingestion failed: {error}")
        raise SystemExit(1) from error
    except Exception as error:
        print(f"Unexpected PDF loading error: {error}")
        raise SystemExit(1) from error

    print("\nDocument loading completed successfully.")
    print(f"Number of page documents loaded: {len(documents)}")

    nonempty_documents = [
        document
        for document in documents
        if document.page_content.strip()
    ]

    print(f"Documents containing extracted text: {len(nonempty_documents)}")

    if nonempty_documents:
        preview_document(nonempty_documents[0])
    else:
        print(
            "\nNo extractable text was found. "
            "The PDF may be scanned or image-based."
        )


if __name__ == "__main__":
    main()