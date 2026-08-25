from pathlib import Path
import pymupdf  # pip package name is PyMuPDF, import as pymupdf
from textwrap import shorten


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    pdf_path = project_root / "data" / "raw" / "test.pdf"

    if not pdf_path.exists():
        print(f"Place a small PDF at {pdf_path} and re-run this script.")
        raise SystemExit(1)

    doc = pymupdf.open(str(pdf_path))
    pages_text = []

    for page_no in range(doc.page_count):
        page = doc.load_page(page_no)
        text = page.get_text("text")
        pages_text.append({"page": page_no + 1, "text": text})

    # Simple chunking: split by double newlines and ignore very short chunks.
    chunks = []
    for page in pages_text:
        for part in page["text"].splitlines():
            section = "\n".join(part.strip().split())
            section = section.strip()
            if len(section) < 20:
                continue
            chunks.append({"page": page["page"], "chunk": section})

    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2], start=1):
        preview = shorten(chunk["chunk"], width=200, placeholder="...")
        print(f"Chunk {i}, page {chunk['page']}: {preview}\n")


if __name__ == "__main__":
    main()

# Run it from PowerShell (no activation needed):
# .\.venv\Scripts\python.exe .\app\ingestion\pdf_ingest_pymupdf.py
