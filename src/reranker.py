from http import HTTPStatus
import os

import dashscope
from dotenv import load_dotenv


RERANK_MODEL = "qwen3-rerank"


def get_native_dashscope_url() -> str:
    """
    Convert the OpenAI-compatible endpoint in .env
    to DashScope's native API URL.
    """
    native_url = os.getenv("DASHSCOPE_NATIVE_BASE_URL")

    if native_url:
        return native_url

    compatible_url = os.getenv("DASHSCOPE_BASE_URL", "")

    if compatible_url:
        return compatible_url.replace(
            "/compatible-mode/v1",
            "/api/v1",
        )

    return "https://dashscope.aliyuncs.com/api/v1"


def rerank_texts(
    query: str,
    candidate_texts: list[str],
    top_n: int = 3,
) -> list[dict]:
    """Rerank text passages using DashScope qwen3-rerank."""
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is missing from .env.")

    dashscope.api_key = api_key
    dashscope.base_http_api_url = get_native_dashscope_url()

    response = dashscope.TextReRank.call(
        model=RERANK_MODEL,
        query=query,
        documents=candidate_texts,
        top_n=top_n,
        return_documents=False,
    )

    if response.status_code != HTTPStatus.OK:
        raise RuntimeError(
            f"Reranker request failed: "
            f"{response.code} - {response.message}"
        )

    return response.output["results"]


def rerank_documents(
    query: str,
    faiss_results: list[tuple],
    top_n: int = 3,
) -> list[tuple]:
    """
    Rerank FAISS candidate documents.

    Input:
        [(Document, faiss_distance), ...]

    Output:
        [(Document, reranker_relevance_score), ...]
    """
    candidate_texts = [
        document.page_content
        for document, _ in faiss_results
    ]

    ranked_items = rerank_texts(
        query=query,
        candidate_texts=candidate_texts,
        top_n=top_n,
    )

    reranked_results = []

    for item in ranked_items:
        original_index = item["index"]
        reranker_score = item["relevance_score"]

        document, _ = faiss_results[original_index]

        reranked_results.append(
            (document, reranker_score)
        )

    return reranked_results


def main():
    """Small standalone test of the reranker."""
    query = "What is a clustering method for multivariate time series?"

    candidate_texts = [
        "FCPCA is a fuzzy clustering method based on lagged cross-covariances.",
        "Quantum computing can solve some mathematical problems.",
        "VPCA is a competing clustering method used for comparison.",
        "This paper studies financial market returns.",
    ]

    results = rerank_texts(
        query=query,
        candidate_texts=candidate_texts,
        top_n=2,
    )

    print(f"Query: {query}\n")
    print("Reranked results:")

    for rank, item in enumerate(results, start=1):
        original_index = item["index"]
        score = item["relevance_score"]

        print(f"\nRank {rank}")
        print(f"Score: {score:.4f}")
        print(f"Text: {candidate_texts[original_index]}")


if __name__ == "__main__":
    main()