from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS


VECTOR_STORE_DIR = Path("data/vector_store")
TOP_K = 3


def main() -> None:
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is missing from your .env file.")

    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find '{VECTOR_STORE_DIR}'. "
            "Run 'python src/build_index.py' first."
        )

    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = input("Ask a question about your PDFs: ").strip()

    if not query:
        raise ValueError("Please provide a question.")

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=api_key,
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    results = vector_store.similarity_search_with_score(
        query,
        k=TOP_K,
    )

    print(f"\nQuestion: {query}")
    print(f"\nTop {TOP_K} retrieved passages:")

    for rank, (document, score) in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"Result {rank}")
        print(f"Source: {document.metadata['source']}")
        print(f"Page: {document.metadata['page']}")
        print(f"Chunk ID: {document.metadata['chunk_id']}")
        print(f"Distance score: {score:.4f}")
        print("\nPassage:")
        print(document.page_content[:800])

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()