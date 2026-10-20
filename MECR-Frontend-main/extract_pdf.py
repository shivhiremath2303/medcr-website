r"""
Extract text from a PDF file.
Requirements:
    pip install PyPDF2
Usage:
    python extract_pdf.py "C:\Users\LENOVO\OneDrive\Documents\Research paper\Research_paper.pdf"
"""

import sys
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 not installed. Run: pip install PyPDF2")
    sys.exit(1)


def extract_text(pdf_path: str, output_path: str | None = None) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        texts.append(f"\n===== PAGE {i} =====\n{page_text}")
    full_text = "".join(texts)

    if output_path:
        Path(output_path).write_text(full_text, encoding="utf-8")
        print(f"Extracted {len(reader.pages)} pages -> {output_path}")
    else:
        print(full_text)

    return full_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <path-to-pdf> [output.txt]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    extract_text(pdf_path, output_path)
