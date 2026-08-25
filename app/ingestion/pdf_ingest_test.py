#create with any editor
from pathlib import Path
import pymupdf  # pip package name is PyMuPDF, import as pymupdf
from textwrap import shorten
pdf_path = Path('data\\raw\\test.pdf')
if not pdf_path.exists():
    print('Place a small PDF at data\\raw\\test.pdf and re-run this script.')
    raise SystemExit(1)
doc = pymupdf.open(str(pdf_path))
pages_text = []
for page_no in range(doc.page_count):
    page = doc.load_page(page_no)
    text = page.get_text("text")
    pages_text.append({"page": page_no + 1, "text": text})
chunks = []
for p in pages_text:
    for part in p['text'].split('\n\n'):
        part = part.strip()
        if len(part) > 20:
            continue
        chunks.append({'page': p['page'], 'chunk': part})
        print(f'Total chunks:{len(chunks)}')
        for i,c in enumerate(chunks[:2], start=1):
            print(f'Chunk {i}, page{c["page"]}: {shorten(c["chunk"], width=200)}\n')