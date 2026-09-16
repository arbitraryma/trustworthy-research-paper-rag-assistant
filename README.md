# Trustworthy Research Paper RAG Assistant

A Streamlit-based research-paper assistant that answers questions from uploaded PDFs using retrieval-augmented generation (RAG), page-level citations, reranking, and confidence-aware refusal.

## Features

- Upload research PDFs through a Streamlit interface
- Extract PDF text page by page
- Chunk documents while preserving source filename and page metadata
- Generate semantic embeddings using DashScope
- Retrieve candidate passages with FAISS
- Rerank FAISS candidates with `qwen3-rerank`
- Generate grounded answers with Qwen
- Display the retrieved evidence behind every answer
- Support multi-turn chat during a session
- Refuse questions when retrieved evidence is insufficient

## Architecture

```mermaid
flowchart TD
    A[Upload PDF] --> B[Extract text by page]
    B --> C[Chunk text with metadata]
    C --> D[DashScope embeddings]
    D --> E[FAISS vector index]

    Q[User question] --> QE[Query embedding]
    QE --> F[FAISS top-10 retrieval]
    F --> R[qwen3-rerank top-3]
    R --> G{Evidence score ≥ threshold?}
    G -->|Yes| L[Qwen grounded answer]
    G -->|No| X[Confidence-aware refusal]
    L --> CITE[Page-level citations]

