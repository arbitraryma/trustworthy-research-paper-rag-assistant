from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS


VECTOR_STORE_DIR = Path("data/vector_store")
TOP_K = 3
MODEL_NAME = "qwen-plus"


def build_context(results: list[tuple]) -> str:
    """Format retrieved passages for the LLM prompt."""
    context_parts = []

    for rank, (document, _) in enumerate(results, start=1):
        source = document.metadata["source"]
        page = document.metadata["page"]

        context_parts.append(
            f"[Evidence {rank} | Source: {source} | Page: {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def main() -> None:
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is missing from your .env file.")

    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            "Vector store not found. Run 'python src/build_index.py' first."
        )

    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = input("Ask a question about your PDFs: ").strip()

    if not question:
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
        question,
        k=TOP_K,
    )

    context = build_context(results)

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )

    system_prompt = """
You are a reliable research-paper assistant.

Answer the user's question using ONLY the evidence provided below.
Do not use outside knowledge or invent details.

Rules:
1. If the evidence does not contain enough information, say:
   "I cannot answer this from the provided documents."
2. Cite every factual claim using this format:
   [Source: filename.pdf, p. N]
3. Keep the answer concise and academically precise.
"""

    user_prompt = f"""
Question:
{question}

Retrieved evidence:
{context}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for document, score in results:
        print(
            f"- {document.metadata['source']}, "
            f"p. {document.metadata['page']} "
            f"(distance: {score:.4f})"
        )


if __name__ == "__main__":
    main()