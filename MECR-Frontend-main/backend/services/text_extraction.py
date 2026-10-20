"""Text extraction for uploaded evidence and dataset files.

The web app accepts real legal files, so ingestion must do more than copy
bytes.  This module extracts text from:

* plain text / JSON / CSV / log files
* Word documents (``.docx``)
* PDFs with an embedded text layer (via ``pypdf``/PyMuPDF)
* scanned PDFs and images when an OCR backend is installed

Every extractor is optional at import time: a missing library produces a
structured ``error``/``needs_ocr`` result instead of crashing the API.  The
returned metadata is stored on the evidence item and surfaced in the UI, so a
reviewer can see whether the analysis used real text or needs OCR.
"""
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, Optional, Tuple


TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".jsonl", ".text", ".legal", ".rtf"}
DOCX_EXTENSIONS = {".docx"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="ignore").strip()


def _clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _extract_docx(payload: bytes) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method": "python-docx"}
    try:
        from docx import Document  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        return "", {**meta, "error": f"python-docx unavailable: {exc}"}
    try:
        document = Document(io.BytesIO(payload))
        blocks = [p.text for p in document.paragraphs if (p.text or "").strip()]
        for table in document.tables:
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    value = " ".join((cell.text or "").split())
                    if value:
                        cells.append(value)
                if cells:
                    blocks.append(" | ".join(cells))
        meta["paragraphs"] = len(document.paragraphs)
        meta["tables"] = len(document.tables)
        return _clean_text("\n".join(blocks)), meta
    except Exception as exc:
        return "", {**meta, "error": f"Could not read DOCX: {exc}"}


def _extract_pdf_pypdf(payload: bytes) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method": "pypdf"}
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover
        return "", {**meta, "error": f"pypdf unavailable: {exc}"}
    try:
        reader = PdfReader(io.BytesIO(payload))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        meta["pages"] = len(reader.pages)
        return _clean_text("\n\n".join(pages)), meta
    except Exception as exc:
        return "", {**meta, "error": f"Could not read PDF: {exc}"}


def _extract_pdf_mupdf(payload: bytes) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"method": "pymupdf"}
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover
        return "", {**meta, "error": f"PyMuPDF unavailable: {exc}"}
    try:
        document = fitz.open(stream=payload, filetype="pdf")
        pages = [page.get_text("text") or "" for page in document]
        meta["pages"] = len(pages)
        if not any(page.strip() for page in pages):
            # Render pages for the OCR fallback.
            images = []
            for page in document:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                images.append(pix.tobytes("png"))
            meta["_page_images"] = images
        return _clean_text("\n\n".join(pages)), meta
    except Exception as exc:
        return "", {**meta, "error": f"Could not read PDF with PyMuPDF: {exc}"}


def _ocr_backends():
    """Yield (name, callable) OCR backends in preference order.

    Callables accept raw image bytes and return extracted text.  The backends
    are optional; the first installed one wins.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        engine = RapidOCR()

        def _rapid(image_bytes: bytes) -> str:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore

            image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
            result, _ = engine(image)
            if not result:
                return ""
            return "\n".join(str(line[1]) for line in result if len(line) > 1)

        yield "rapidocr", _rapid
    except Exception:
        pass

    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        def _tesseract(image_bytes: bytes) -> str:
            return pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))

        yield "tesseract", _tesseract
    except Exception:
        pass

    try:
        import winocr  # type: ignore
        from PIL import Image  # type: ignore

        def _winocr(image_bytes: bytes) -> str:
            result = winocr.recognize_pil_sync(Image.open(io.BytesIO(image_bytes)))
            return getattr(result, "text", "") or ""

        yield "windows-ocr", _winocr
    except Exception:
        pass


def ocr_images(images: list[bytes]) -> Tuple[str, Dict[str, Any]]:
    """Run OCR over one or more images, returning text + metadata."""
    if not images:
        return "", {"method": "none", "needs_ocr": False}
    for name, func in _ocr_backends():
        try:
            chunks = [func(image) or "" for image in images]
            text = _clean_text("\n\n".join(chunks))
            if text:
                return text, {"method": f"ocr:{name}", "ocr_pages": len(images)}
        except Exception as exc:
            last_error = str(exc)
            continue
    return "", {
        "method": "none",
        "needs_ocr": True,
        "error": "No OCR backend is installed. Install rapidocr_onnxruntime or pytesseract+Tesseract for scanned files.",
    }


def _extract_image(payload: bytes) -> Tuple[str, Dict[str, Any]]:
    text, meta = ocr_images([payload])
    meta.setdefault("method", "ocr")
    return text, meta


def extract_bytes(filename: str, content_type: str, payload: bytes) -> Dict[str, Any]:
    """Extract text and metadata from an uploaded file.

    Returns a dict with ``text`` plus extraction metadata.  It never raises for
    malformed input; callers can store the metadata and show an actionable
    message in the UI.
    """
    name = (filename or "").lower()
    ext = os.path.splitext(name)[1]
    content_type = (content_type or "").lower()
    size = len(payload or b"")
    base: Dict[str, Any] = {
        "filename": filename or "",
        "extension": ext,
        "content_type": content_type,
        "bytes": size,
        "charCount": 0,
        "needs_ocr": False,
        "error": None,
    }

    if not payload:
        return {**base, "text": "", "method": "none", "error": "The uploaded file is empty."}

    # Plain text and structured text formats
    if ext in TEXT_EXTENSIONS or content_type.startswith("text/"):
        text = _clean_text(_decode_text(payload))
        method = "text"
        if ext in (".json", ".jsonl") and not text:
            method = "text"
        return {**base, "text": text, "method": method, "charCount": len(text)}

    if ext in DOCX_EXTENSIONS or content_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        text, meta = _extract_docx(payload)
        return {**base, **meta, "text": text, "charCount": len(text),
                "needs_ocr": False if text else bool(meta.get("error"))}

    if ext in PDF_EXTENSIONS or content_type == "application/pdf":
        text, meta = _extract_pdf_pypdf(payload)
        if not text.strip():
            mupdf_text, mupdf_meta = _extract_pdf_mupdf(payload)
            if mupdf_text.strip():
                text, meta = mupdf_text, {**meta, **mupdf_meta}
            else:
                page_images = mupdf_meta.get("_page_images") or []
                ocr_text, ocr_meta = ocr_images(page_images)
                meta = {**meta, **mupdf_meta, **ocr_meta}
                if ocr_text.strip():
                    text = ocr_text
                else:
                    meta.setdefault("needs_ocr", True)
                    meta.setdefault("error", "The PDF has no extractable text layer (scanned document).")
        meta.pop("_page_images", None)
        return {**base, **meta, "text": text, "charCount": len(text)}

    if ext in IMAGE_EXTENSIONS or content_type.startswith("image/"):
        text, meta = _extract_image(payload)
        return {**base, **meta, "text": text, "charCount": len(text),
                "needs_ocr": bool(meta.get("needs_ocr")) or not bool(text)}

    if ext in AUDIO_EXTENSIONS or content_type.startswith("audio/"):
        return {**base, "text": "", "method": "none", "needs_transcription": True,
                "error": "Audio transcription is not configured. Attach a transcript as text or .docx."}

    return {
        **base,
        "text": "",
        "method": "none",
        "error": f"Unsupported file type '{ext or content_type or 'unknown'}'. "
                 "Use PDF, DOCX, TXT, JSON, CSV, or an image.",
    }


def extract_path(path: str) -> Dict[str, Any]:
    """Extract text from a path (used by the dataset loader)."""
    try:
        with open(path, "rb") as handle:
            payload = handle.read()
    except OSError as exc:
        return {"text": "", "method": "none", "error": str(exc), "charCount": 0}
    import mimetypes

    content_type = mimetypes.guess_type(path)[0] or ""
    return extract_bytes(os.path.basename(path), content_type, payload)
