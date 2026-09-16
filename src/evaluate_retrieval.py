from pathlib import Path
import json
import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

from src.reranker import rerank_documents


VECTOR_STORE_DIR = Path("data/vector_store")
EVALUATION_PATH = Path("tests/evaluation_questions.json")

FAISS_CANDIDATES = 10
FINAL_TOP_K = 3


def load_evaluation_questions():
    with EVALUATION_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_hit(results, expected_source, expected_pages):
    return any(
        document.metadata["source"] == expected_source
        and document.metadata["page"] in expected_pages
        for document, _ in results
    )


def main():
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is missing from .env.")

    questions = load_evaluation_questions()

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=api_key,
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    faiss_hits = 0
    reranker_hits = 0

    print(
        f"Comparing FAISS Recall@{FINAL_TOP_K} "
        f"with Reranked Recall@{FINAL_TOP_K}\n"
    )

    for item in questions:
        faiss_top_10 = vector_store.similarity_search_with_score(
            item["question"],
            k=FAISS_CANDIDATES,
        )

        faiss_top_3 = faiss_top_10[:FINAL_TOP_K]

        reranked_top_3 = rerank_documents(
            query=item["question"],
            faiss_results=faiss_top_10,
            top_n=FINAL_TOP_K,
        )

        if item["id"] == "q2":
            print("\n--- q2 diagnostic ---")

            print("FAISS top-10 pages:")
            print(
                [
                    (document.metadata["source"], document.metadata["page"])
                    for document, _ in faiss_top_10
                ]
            )

            print("Reranked top-3 pages:")
            print(
                [
                    (document.metadata["source"], document.metadata["page"])
                    for document, _ in reranked_top_3
                ]
            )

            print("--- end diagnostic ---\n")

        faiss_hit = is_hit(
            faiss_top_3,
            item["expected_source"],
            item["expected_pages"],
        )

        reranker_hit = is_hit(
            reranked_top_3,
            item["expected_source"],
            item["expected_pages"],
        )

        faiss_hits += int(faiss_hit)
        reranker_hits += int(reranker_hit)

        print(f"Question: {item['id']} — {item['question']}")
        print(f"FAISS top-3: {'HIT' if faiss_hit else 'MISS'}")
        print(f"Reranked top-3: {'HIT' if reranker_hit else 'MISS'}\n")

    total = len(questions)

    print("=" * 60)
    print(f"FAISS Recall@{FINAL_TOP_K}: {faiss_hits / total:.1%}")
    print(f"Reranked Recall@{FINAL_TOP_K}: {reranker_hits / total:.1%}")
    print(f"Questions evaluated: {total}")


if __name__ == "__main__":
    main()