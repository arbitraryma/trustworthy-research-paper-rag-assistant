from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

from src.ingest import (
    RAW_PDFS_DIR,
    OUTPUT_PATH,
    extract_pages,
    chunk_pages,
    save_chunks,
)
from src.build_index import build_index
from src.reranker import rerank_documents


VECTOR_STORE_DIR = Path("data/vector_store")

FAISS_CANDIDATES = 10
FINAL_TOP_K = 3
MIN_RERANK_SCORE = 0.50

MODEL_NAME = "qwen-plus"


@st.cache_resource
def load_rag_resources():
    """Load the FAISS index and Qwen client once."""
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY is missing from .env.")

    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=api_key,
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    )

    return vector_store, client


def rebuild_index_from_pdfs():
    """Extract, chunk, embed, and index every PDF in data/raw_pdfs."""
    pdf_files = sorted(RAW_PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDFs are available to index.")

    all_pages = []

    for pdf_path in pdf_files:
        all_pages.extend(extract_pages(pdf_path))

    chunks = chunk_pages(all_pages)
    save_chunks(chunks, OUTPUT_PATH)

    indexed_chunk_count = build_index()

    return len(pdf_files), len(chunks), indexed_chunk_count


def build_context(results):
    """Format reranked evidence passages for the answering LLM."""
    context_parts = []

    for rank, (document, _) in enumerate(results, start=1):
        context_parts.append(
            f"[Evidence {rank} | "
            f"Source: {document.metadata['source']} | "
            f"Page: {document.metadata['page']}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def get_conversation_history(messages):
    """Keep the six most recent chat messages as follow-up context."""
    recent_messages = messages[-6:]

    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in recent_messages
    )


def generate_answer(question, results, client, conversation_history):
    """Generate an answer grounded only in reranked evidence."""
    context = build_context(results)

    system_prompt = """
You are a reliable research-paper assistant.

Use ONLY the retrieved evidence to answer factual questions.
Do not invent information or use outside knowledge.

If the evidence is insufficient, say:
"I cannot answer this from the provided documents."

Cite factual claims using:
[Source: filename.pdf, p. N]
"""

    user_prompt = f"""
Previous conversation:
{conversation_history}

New question:
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

    return response.choices[0].message.content


def has_sufficient_evidence(results) -> bool:
    """
    Refuse early when the strongest reranked evidence
    does not meet the relevance threshold.
    """
    if not results:
        return False

    best_reranker_score = float(results[0][1])

    return best_reranker_score >= MIN_RERANK_SCORE


def format_evidence(results):
    """Save evidence in simple form for Streamlit chat history."""
    return [
        {
            "source": document.metadata["source"],
            "page": document.metadata["page"],
            "score": float(score),
            "text": document.page_content,
        }
        for document, score in results
    ]


def show_evidence(evidence):
    """Display the passages supporting one answer."""
    with st.expander("View retrieved evidence"):
        for rank, item in enumerate(evidence, start=1):
            st.markdown(
                f"**Result {rank} — {item['source']}, p. {item['page']}**"
            )
            st.caption(
                f"Reranker relevance score: {item['score']:.4f}"
            )
            st.write(item["text"])
            st.divider()


def main():
    st.set_page_config(
        page_title="Research Paper RAG Assistant",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Research Paper RAG Assistant")
    st.caption(
        "Upload and query research papers. Answers are grounded in "
        "reranked evidence and include page-level citations."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Document library")

        uploaded_pdf = st.file_uploader(
            "Upload a research PDF",
            type=["pdf"],
        )

        if uploaded_pdf and st.button("Add PDF and rebuild index"):
            RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)

            safe_filename = Path(uploaded_pdf.name).name
            destination = RAW_PDFS_DIR / safe_filename

            with destination.open("wb") as file:
                file.write(uploaded_pdf.getbuffer())

            with st.spinner(
                "Extracting text, creating embeddings, and rebuilding FAISS..."
            ):
                pdf_count, chunk_count, _ = rebuild_index_from_pdfs()

            st.cache_resource.clear()
            st.session_state.messages = []

            st.success(
                f"Indexed {pdf_count} PDF(s) with {chunk_count} chunks. "
                "The conversation was cleared because the knowledge base changed."
            )

        st.divider()
        st.header("Chat controls")

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "FAISS retrieves 10 candidates; the reranker selects the best 3."
        )
        st.caption(
            f"Answers are refused below a reranker score of "
            f"{MIN_RERANK_SCORE:.2f}."
        )

    if not VECTOR_STORE_DIR.exists():
        st.info("Upload a PDF from the sidebar to create the first index.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                show_evidence(message["evidence"])

    question = st.chat_input("Ask a question about the research papers...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        try:
            vector_store, client = load_rag_resources()

            with st.chat_message("assistant"):
                with st.spinner(
                    "Retrieving candidates, reranking evidence, and generating an answer..."
                ):
                    # Stage 1: fast vector retrieval.
                    faiss_results = vector_store.similarity_search_with_score(
                        question,
                        k=FAISS_CANDIDATES,
                    )

                    # Stage 2: precise relevance reranking.
                    results = rerank_documents(
                        query=question,
                        faiss_results=faiss_results,
                        top_n=FINAL_TOP_K,
                    )

                    evidence = format_evidence(results)

                    if has_sufficient_evidence(results):
                        conversation_history = get_conversation_history(
                            st.session_state.messages[:-1]
                        )

                        answer = generate_answer(
                            question,
                            results,
                            client,
                            conversation_history,
                        )
                    else:
                        best_score = float(results[0][1])

                        answer = (
                            "I cannot answer this from the provided documents "
                            "because the retrieved evidence is not sufficiently "
                            "relevant.\n\n"
                            f"Top reranker relevance score: {best_score:.4f} "
                            f"(required: {MIN_RERANK_SCORE:.2f})."
                        )

                st.markdown(answer)
                show_evidence(evidence)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "evidence": evidence,
                }
            )

        except Exception as error:
            st.error(f"Something went wrong: {error}")


if __name__ == "__main__":
    main()