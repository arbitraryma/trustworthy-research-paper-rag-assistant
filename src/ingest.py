from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


RAW_PDFS_DIR = Path("data/raw_pdfs")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract non-empty text pages from one PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append(
                {
                    "source": pdf_path.name,
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split page text into chunks while preserving source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page_data in pages:
        text_chunks = splitter.split_text(page_data["text"])

        for chunk_index, chunk_text in enumerate(text_chunks):
            chunks.append(
                {
                    "chunk_id": (
                        f"{page_data['source']}"
                        f"_page_{page_data['page_number']}"
                        f"_chunk_{chunk_index}"
                    ),
                    "source": page_data["source"],
                    "page": page_data["page_number"],
                    "text": chunk_text,
                }
            )

    return chunks


def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Save chunks as JSON Lines: one dictionary per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    pdf_files = sorted(RAW_PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{RAW_PDFS_DIR}'. "
            "Please add PDF files and run again."
        )

    all_pages = []

    for pdf_path in pdf_files:
        pages = extract_pages(pdf_path)
        all_pages.extend(pages)
        print(f"Extracted {len(pages)} text pages from: {pdf_path.name}")

    chunks = chunk_pages(all_pages)
    save_chunks(chunks, OUTPUT_PATH)

    print(f"\nCreated {len(chunks)} chunks.")
    print(f"Saved to: {OUTPUT_PATH}")

    if chunks:
        first_chunk = chunks[0]

        print("\nExample metadata:")
        print(
            {
                "chunk_id": first_chunk["chunk_id"],
                "source": first_chunk["source"],
                "page": first_chunk["page"],
            }
        )

        print("\nExample text:")
        print(first_chunk["text"][:400])


if __name__ == "__main__":
    main()