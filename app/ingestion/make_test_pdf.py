from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

out = Path('data\\raw\\test_1.pdf')
out.parent.mkdir(parents=True, exist_ok=True)

c = canvas.Canvas(str(out), pagesize=letter)
text = c.beginText(40, 720)
text.setFont("Helvetica", 12)
lines = [
    "Enterprise RAG Assistant — Test Document",
    "",
    "This is a short test PDF used to validate PDF ingestion and chunking.",
    "",
    "Section 1: Overview",
    "The assistant should extract text, preserve page numbers, and create chunks.",
    "",
    "End of test document."
]
for ln in lines:
    text.textLine(ln)
c.drawText(text)
c.showPage()
c.save()
print(f'Wrote: {out.resolve()}')