# Naïve RAG System Architecture

## 1. Project Goal

This project builds a naïve Retrieval-Augmented Generation system from scratch.

The system will eventually:

1. Load source documents.
2. Divide the documents into retrieval-sized chunks.
3. convert the chunks into vector embeddings.
4. Store the embeddings in a vector database.
5. Retrieve relevant chunks for a user question.
6. Provide the retrieved context to a large language model.
7. Generate an answer grounded in the source documents.
8. Evaluate retrieval and answer quality.

---

## 2. Current Project Status

The offline indexing pipeline is complete.

```text
PDF Documents
      │
      ▼
Document Loading
      │
      ▼
432 Page Documents
      │
      ▼
Fixed-Size Chunking
      │
      ▼
1,872 Chunk Documents
      │
      ▼
OpenAI Embeddings
      │
      ▼
1,536-Dimensional Vectors
      │
      ▼
Persistent Chroma Database
```

The next milestone is the retrieval pipeline.

---

# 3. Repository Structure

```text
naive_rag/
│
├── data/
│   ├── 2026_Budget_Statement_and_Economic_Policy.pdf
│   ├── Manifesto_Abridged.pdf
│   └── SONA_Ghana_2026.pdf
│
├── docs/
│   └── architecture.md
│
├── chroma_db/
│   └── Generated Chroma database files
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── loaders.py
│   ├── chunkers.py
│   ├── embeddings.py
│   ├── vectordb.py
│   ├── ingest.py
│   ├── test_environment.py
│   ├── test_embeddings.py
│   └── test_vectordb.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 4. Current Architecture

## Offline Indexing Pipeline

```text
┌──────────────────────────────────────────────────────────────┐
│                     SOURCE DOCUMENTS                         │
│                                                              │
│  2026 Budget Statement                         304 pages      │
│  Manifesto Abridged                             72 pages      │
│  2026 State of the Nation Address               56 pages      │
│                                                              │
│  Total                                         432 pages      │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                         loaders.py                           │
│                                                              │
│  load_documents()                                            │
│  Reads the configured PDF files and converts every PDF page  │
│  into a LangChain Document with source and page metadata.     │
│                                                              │
│  Input:                                                      │
│      PDF files from the data directory                       │
│                                                              │
│  Output:                                                     │
│      List[Document] containing 432 page-level documents      │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                         chunkers.py                          │
│                                                              │
│  chunk_documents()                                           │
│  Acts as the public chunking interface and sends documents   │
│  to the chunking strategy selected in config.py.             │
│                                                              │
│  fixed_size_chunking()                                       │
│  Divides page documents into fixed-size, overlapping text    │
│  chunks and adds chunk-level metadata.                       │
│                                                              │
│  filter_small_chunks()                                       │
│  Removes chunks that contain fewer useful characters than    │
│  the configured minimum chunk size.                          │
│                                                              │
│  CHUNKING_STRATEGIES                                         │
│  Maps strategy names to their corresponding chunking         │
│  functions so new strategies can be added later.             │
│                                                              │
│  Input:                                                      │
│      432 page-level documents                                │
│                                                              │
│  Output:                                                     │
│      1,872 chunk-level documents                             │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        embeddings.py                         │
│                                                              │
│  load_environment()                                          │
│  Loads environment variables, including the OpenAI API key,  │
│  from the project's .env file.                               │
│                                                              │
│  validate_openai_api_key()                                   │
│  Checks that an OpenAI API key is available before an API     │
│  request is attempted.                                       │
│                                                              │
│  get_embedding_model()                                       │
│  Creates and returns the embedding provider and model        │
│  configured in config.py.                                    │
│                                                              │
│  extract_document_texts()                                    │
│  Extracts page_content strings from LangChain Documents.      │
│                                                              │
│  embed_documents()                                           │
│  Converts document text into numerical embedding vectors.    │
│                                                              │
│  Current provider:                                           │
│      OpenAI                                                  │
│                                                              │
│  Current model:                                              │
│      text-embedding-3-small                                  │
│                                                              │
│  Embedding dimension:                                        │
│      1,536                                                   │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                         vectordb.py                          │
│                                                              │
│  create_vector_store()                                       │
│  Uses Chroma to embed and store chunk documents in a         │
│  persistent vector database.                                │
│                                                              │
│  load_vector_store()                                         │
│  Reopens an existing Chroma collection from disk so it can   │
│  be searched without rebuilding the index.                  │
│                                                              │
│  get_vector_count()                                          │
│  Returns the number of records currently stored in the       │
│  Chroma collection.                                         │
│                                                              │
│  delete_vector_store()                                       │
│  Deletes the local Chroma persistence directory when a full  │
│  index rebuild is required.                                 │
│                                                              │
│  Input:                                                      │
│      Chunk documents and an embedding model                  │
│                                                              │
│  Output:                                                     │
│      Persistent Chroma vector store                          │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                         chroma_db/                           │
│                                                              │
│  Collection name:                                            │
│      ghana_government_documents                             │
│                                                              │
│  Stored records:                                             │
│      1,872                                                   │
│                                                              │
│  Each record contains:                                       │
│      • Original chunk text                                  │
│      • Embedding vector                                     │
│      • Document metadata                                    │
│                                                              │
│  Example metadata:                                           │
│      • source                                               │
│      • page                                                 │
│      • chunk_index                                          │
│      • character_count                                      │
│      • chunking_strategy                                    │
└──────────────────────────────────────────────────────────────┘
```

---

# 5. Configuration

## `config.py`

`config.py` is the central control panel for the project.

```text
┌──────────────────────────────────────────────────────────────┐
│                          config.py                           │
│                                                              │
│  Document configuration                                      │
│  ──────────────────────                                      │
│  DATA_DIRECTORY                                              │
│  Identifies the directory containing the source documents.   │
│                                                              │
│  Chunking configuration                                      │
│  ──────────────────────                                      │
│  CHUNKING_STRATEGY                                           │
│  Selects the chunking method used by chunk_documents().      │
│                                                              │
│  FIXED_CHUNK_SIZE                                            │
│  Sets the target character length of each fixed-size chunk.  │
│                                                              │
│  FIXED_CHUNK_OVERLAP                                         │
│  Sets the number of characters shared by neighboring chunks. │
│                                                              │
│  MIN_CHUNK_SIZE                                              │
│  Sets the minimum number of characters required to retain a  │
│  chunk.                                                      │
│                                                              │
│  Embedding configuration                                     │
│  ───────────────────────                                     │
│  EMBEDDING_PROVIDER                                          │
│  Selects the service used to generate embeddings.            │
│                                                              │
│  EMBEDDING_MODEL                                             │
│  Selects the embedding model used by the provider.           │
│                                                              │
│  Vector database configuration                               │
│  ─────────────────────────────                               │
│  CHROMA_COLLECTION_NAME                                      │
│  Sets the name of the persistent Chroma collection.          │
│                                                              │
│  CHROMA_PERSIST_DIRECTORY                                    │
│  Sets the local directory in which Chroma stores its data.   │
└──────────────────────────────────────────────────────────────┘
```

---

# 6. Pipeline Scripts

## `ingest.py`

```text
┌──────────────────────────────────────────────────────────────┐
│                          ingest.py                           │
│                                                              │
│  main()                                                      │
│  Runs the current ingestion workflow and prints document and │
│  chunk statistics for inspection.                           │
│                                                              │
│  Current workflow:                                           │
│                                                              │
│      Load PDFs                                               │
│          │                                                   │
│          ▼                                                   │
│      Chunk documents                                         │
│          │                                                   │
│          ▼                                                   │
│      Calculate chunk statistics                              │
│          │                                                   │
│          ▼                                                   │
│      Preview a sample chunk                                  │
│                                                              │
│  Planned workflow:                                           │
│                                                              │
│      Load PDFs                                               │
│          │                                                   │
│          ▼                                                   │
│      Chunk documents                                         │
│          │                                                   │
│          ▼                                                   │
│      Create embedding model                                  │
│          │                                                   │
│          ▼                                                   │
│      Build persistent vector store                           │
└──────────────────────────────────────────────────────────────┘
```

---

# 7. Test and Verification Scripts

## `test_environment.py`

```text
┌──────────────────────────────────────────────────────────────┐
│                  test_environment.py                         │
│                                                              │
│  Purpose:                                                    │
│  Confirms that the Python environment and required packages  │
│  are configured correctly.                                  │
│                                                              │
│  Checks:                                                     │
│      • Virtual environment                                  │
│      • Required imports                                     │
│      • Basic project setup                                  │
└──────────────────────────────────────────────────────────────┘
```

## `test_embeddings.py`

```text
┌──────────────────────────────────────────────────────────────┐
│                   test_embeddings.py                         │
│                                                              │
│  main()                                                      │
│  Tests the embedding layer on a small sample before running  │
│  a full indexing operation.                                 │
│                                                              │
│  Workflow:                                                   │
│      Load all PDFs                                           │
│          │                                                   │
│          ▼                                                   │
│      Chunk all documents                                     │
│          │                                                   │
│          ▼                                                   │
│      Select five chunks                                      │
│          │                                                   │
│          ▼                                                   │
│      Generate five embeddings                                │
│          │                                                   │
│          ▼                                                   │
│      Verify vector count and dimension                       │
│                                                              │
│  Verified result:                                            │
│      Five chunks produced five 1,536-dimensional vectors.    │
└──────────────────────────────────────────────────────────────┘
```

## `test_vectordb.py`

```text
┌──────────────────────────────────────────────────────────────┐
│                    test_vectordb.py                          │
│                                                              │
│  main()                                                      │
│  Tests creation, persistence, and reloading of the Chroma    │
│  vector database.                                           │
│                                                              │
│  Workflow:                                                   │
│      Load all PDFs                                           │
│          │                                                   │
│          ▼                                                   │
│      Create 1,872 chunks                                     │
│          │                                                   │
│          ▼                                                   │
│      Delete the previous local database                      │
│          │                                                   │
│          ▼                                                   │
│      Create a fresh Chroma database                          │
│          │                                                   │
│          ▼                                                   │
│      Verify stored record count                              │
│          │                                                   │
│          ▼                                                   │
│      Reload the database from disk                           │
│          │                                                   │
│          ▼                                                   │
│      Verify the reloaded record count                        │
│                                                              │
│  Verified result:                                            │
│      Expected records: 1,872                                 │
│      Stored records:   1,872                                 │
│      Reloaded records: 1,872                                 │
└──────────────────────────────────────────────────────────────┘
```

---

# 8. Data Transformations

The data changes form at each stage of the indexing pipeline.

```text
PDF File
   │
   │ Read by loaders.py
   ▼
Page-Level Document
   │
   │ Processed by chunkers.py
   ▼
Chunk-Level Document
   │
   │ Processed by the embedding model
   ▼
Embedding Vector
   │
   │ Stored by vectordb.py
   ▼
Chroma Record
```

## Page-Level Document

A page-level LangChain `Document` contains:

```python
Document(
    page_content="Text extracted from one PDF page...",
    metadata={
        "source": "document_name.pdf",
        "page": 0,
    },
)
```

## Chunk-Level Document

A chunk-level `Document` contains a smaller section of text and additional metadata:

```python
Document(
    page_content="A smaller overlapping section of page text...",
    metadata={
        "source": "document_name.pdf",
        "page": 0,
        "chunk_index": 0,
        "character_count": 1000,
        "chunking_strategy": "fixed_size",
    },
)
```

## Chroma Record

A stored Chroma record conceptually contains:

```text
Record
├── Document text
├── 1,536-dimensional embedding
└── Metadata
    ├── source
    ├── page
    ├── chunk_index
    ├── character_count
    └── chunking_strategy
```

---

# 9. Indexing Versus Retrieval

The work completed so far belongs to the indexing side of RAG.

## Indexing Pipeline

Indexing prepares the source documents for future search.

```text
Documents
    │
    ▼
Loading
    │
    ▼
Chunking
    │
    ▼
Embedding
    │
    ▼
Vector Storage
```

Indexing normally runs when:

* documents are added;
* documents are removed;
* documents are modified;
* the chunking configuration changes;
* the embedding model changes.

## Retrieval Pipeline

Retrieval happens when a user asks a question.

```text
User Question
      │
      ▼
Embed Question
      │
      ▼
Search Chroma
      │
      ▼
Return Top-K Chunks
```

The retrieval pipeline has not yet been implemented.

---

# 10. Planned Final RAG Architecture

```text
                        OFFLINE INDEXING

┌──────────────────┐
│  PDF Documents   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    loaders.py    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   chunkers.py    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  embeddings.py   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    vectordb.py   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Chroma Database │
└────────┬─────────┘
         │
         │
         │              ONLINE QUESTION-ANSWERING
         │
         │                 ┌──────────────────┐
         │                 │  User Question   │
         │                 └────────┬─────────┘
         │                          │
         │                          ▼
         │                 ┌──────────────────┐
         │                 │ Embed Question   │
         │                 └────────┬─────────┘
         │                          │
         └──────────────────────────┤
                                    ▼
                           ┌──────────────────┐
                           │    Retriever     │
                           │                  │
                           │ Search Chroma    │
                           │ Return Top-K     │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ Retrieved Context│
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ Prompt Template  │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │       LLM        │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ Grounded Answer  │
                           └──────────────────┘
```

---

# 11. Milestones

## Completed

* [x] Create the project structure.
* [x] Configure the Python virtual environment.
* [x] Load multiple PDF documents.
* [x] Convert PDF pages into LangChain Documents.
* [x] Implement fixed-size chunking.
* [x] Implement chunk overlap.
* [x] Remove undersized chunks.
* [x] Add chunk metadata.
* [x] Create a configurable embedding factory.
* [x] Generate and inspect sample embeddings.
* [x] Create a persistent Chroma database.
* [x] Store all 1,872 chunks.
* [x] Reload and verify the persisted database.
* [x] Create the architecture documentation.

## Next

* [ ] Implement similarity retrieval.
* [ ] Return the top-k most relevant chunks.
* [ ] Display similarity scores.
* [ ] Inspect retrieved text and metadata.
* [ ] Test retrieval using questions about the three documents.

## Later

* [ ] Add a prompt template.
* [ ] Configure a generation model.
* [ ] Build the complete RAG question-answering chain.
* [ ] Add source citations to generated answers.
* [ ] Add retrieval evaluation.
* [ ] Add answer-quality evaluation.
* [ ] Compare chunking strategies.
* [ ] Add recursive chunking.
* [ ] Add sentence-aware chunking.
* [ ] Add semantic chunking.
* [ ] Explore parent-child retrieval.
* [ ] Explore late chunking.
* [ ] Explore hybrid retrieval.

---

# 12. Design Principles

## Single Responsibility

Each module should have one primary responsibility.

```text
loaders.py       → document loading
chunkers.py      → document chunking
embeddings.py    → embedding configuration and generation
vectordb.py      → vector database operations
```

## Configuration Separation

Values likely to change should live in `config.py`, not be repeated throughout the codebase.

## Provider Abstraction

The pipeline should depend on general interfaces when practical, rather than being tightly coupled to one provider.

For example:

```python
embedding_model: Embeddings
```

is preferred over requiring every function to accept only:

```python
OpenAIEmbeddings
```

## Reproducible Indexing

The current ingestion approach rebuilds the complete database from scratch.

```text
Delete previous database
          │
          ▼
Load current documents
          │
          ▼
Create current chunks
          │
          ▼
Build a fresh vector index
```

This is simple and deterministic for the current learning project.

Incremental indexing can be added later.

## Test Each Layer Independently

Each major layer should be tested before it is added to the complete pipeline.

```text
Environment
    │
    ▼
Loading
    │
    ▼
Chunking
    │
    ▼
Embeddings
    │
    ▼
Vector storage
    │
    ▼
Retrieval
    │
    ▼
Generation
```

This makes bugs easier to identify and prevents several untested components from failing at the same time.

---

# 13. Current Project Summary

The project currently has a complete and verified offline indexing pipeline.

```text
3 PDF files
    │
    ▼
432 page-level documents
    │
    ▼
1,872 fixed-size overlapping chunks
    │
    ▼
OpenAI text-embedding-3-small
    │
    ▼
1,536-dimensional embeddings
    │
    ▼
1,872 persistent Chroma records
```

The next stage is to build the retrieval layer that searches these records and returns the most relevant chunks for a user question.
