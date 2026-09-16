from pathlib import Path
import json
import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS


CHUNKS_PATH = Path("data/processed/chunks.jsonl")
VECTOR_STORE_DIR = Path("data/vector_store")


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunks created by ingest.py."""
    chunks = []

    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


def build_index() -> int:
    """Build and save a FAISS index. Returns the number of indexed chunks."""
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is missing from .env.")

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {CHUNKS_PATH}. Run 'python src/ingest.py' first."
        )

    chunks = load_chunks(CHUNKS_PATH)

    if not chunks:
        raise ValueError("No chunks were found.")

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]

    ids = [chunk["chunk_id"] for chunk in chunks]

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=api_key,
    )

    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))

    return len(chunks)


def main() -> None:
    chunk_count = build_index()

    print(f"Created a FAISS index from {chunk_count} chunks.")
    print(f"Saved vector store to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()