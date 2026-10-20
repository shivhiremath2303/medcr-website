"""Dataset service — handles zip extraction and multi-format dataset loading."""
import os
import json
import zipfile
import shutil
import mimetypes
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .text_extraction import extract_path


DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_store")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# Supported plain-text extensions for NER training (DOCX/PDF are parsed
# separately so binary Office/PDF files are never decoded as UTF-8).
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".rtf", ".text", ".legal"}

# Supported image file extensions (OCR when available, otherwise registered)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Supported document extensions
DOCUMENT_EXTENSIONS = {".pdf", ".docx"}


def ensure_dirs() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def extract_zip(zip_path: str, dest_dir: Optional[str] = None) -> str:
    """Extract a zip file and return the extraction directory."""
    ensure_dirs()
    if dest_dir is None:
        dest_dir = os.path.join(DATASET_DIR, Path(zip_path).stem)
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir


def list_datasets() -> List[Dict[str, Any]]:
    """List all extracted datasets."""
    ensure_dirs()
    out = []
    if not os.path.exists(DATASET_DIR):
        return out
    for name in sorted(os.listdir(DATASET_DIR)):
        p = os.path.join(DATASET_DIR, name)
        if os.path.isdir(p):
            # Count files by type
            text_files = 0
            image_files = 0
            json_files = 0
            document_files = 0
            other_files = 0
            for fp in Path(p).rglob("*"):
                if not fp.is_file():
                    continue
                ext = fp.suffix.lower()
                if ext in TEXT_EXTENSIONS:
                    text_files += 1
                elif ext in IMAGE_EXTENSIONS:
                    image_files += 1
                elif ext in (".json", ".jsonl"):
                    json_files += 1
                elif ext in DOCUMENT_EXTENSIONS:
                    document_files += 1
                else:
                    other_files += 1
            out.append({
                "name": name,
                "path": p,
                "files": text_files + image_files + json_files + document_files + other_files,
                "text_files": text_files,
                "image_files": image_files,
                "json_files": json_files,
                "document_files": document_files,
                "other_files": other_files,
            })
    return out


def load_dataset_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """Load a dataset folder — supports JSON/JSONL, text, PDF and DOCX.

    PDF/DOCX files are text-extracted before training.  Images are OCRed when
    an OCR backend is available; otherwise they are registered as evidence
    items so the caller can report that OCR is required.
    """
    results: List[Dict[str, Any]] = []
    folder = Path(folder_path)
    if not folder.exists():
        return results

    if folder.is_file():
        files = [folder]
    else:
        files = sorted(folder.rglob("*"))

    for file_path in files:
        if not file_path.is_file():
            continue

        try:
            ext = file_path.suffix.lower()

            # --- JSON files: structured NER data ---
            if ext == ".json":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        parsed = _parse_json_item(item, file_path.name)
                        if parsed:
                            results.append(parsed)
                elif isinstance(data, dict):
                    parsed = _parse_json_item(data, file_path.name)
                    if parsed:
                        results.append(parsed)

            # --- JSONL files: one record per line (large corpora) ---
            elif ext in (".jsonl", ".ndjson"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        parsed = _parse_json_item(item, file_path.name)
                        if parsed:
                            results.append(parsed)

            # --- Text files: plain text for NER ---
            elif ext in TEXT_EXTENSIONS:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if text:
                    results.append({
                        "text": text,
                        "entities": [],
                        "source": file_path.name,
                        "kind": "text",
                        "path": str(file_path),
                    })

            # --- Images: OCR when possible, otherwise register as evidence ---
            elif ext in IMAGE_EXTENSIONS:
                extracted = extract_path(str(file_path))
                if extracted.get("text"):
                    results.append({
                        "text": extracted["text"],
                        "entities": [],
                        "source": file_path.name,
                        "kind": "image_ocr",
                        "path": str(file_path),
                    })
                else:
                    results.append({
                        "text": "",
                        "entities": [],
                        "source": file_path.name,
                        "kind": "image",
                        "path": str(file_path),
                        "extraction": extracted,
                    })

            # --- PDF / DOCX documents: extract the text layer ---
            elif ext in DOCUMENT_EXTENSIONS:
                extracted = extract_path(str(file_path))
                if extracted.get("text"):
                    results.append({
                        "text": extracted["text"],
                        "entities": [],
                        "source": file_path.name,
                        "kind": "pdf" if ext == ".pdf" else "docx",
                        "path": str(file_path),
                        "extraction": extracted,
                    })
                else:
                    results.append({
                        "text": "",
                        "entities": [],
                        "source": file_path.name,
                        "kind": "pdf" if ext == ".pdf" else "docx",
                        "path": str(file_path),
                        "extraction": extracted,
                    })

            # --- Other files: skip but could be registered ---
            else:
                continue

        except Exception as e:
            # Skip malformed files but keep going
            continue

    return results


def _normalize_entities(raw_entities: Any) -> List[Dict[str, Any]]:
    """Coerce [start, end, label] tuples and dicts into one entity shape."""
    out: List[Dict[str, Any]] = []
    for ent in raw_entities or []:
        label: Any = "OTHER"
        if isinstance(ent, dict):
            start, end = ent.get("start"), ent.get("end")
            label = ent.get("label") or ent.get("type") or ent.get("labels") or "OTHER"
        elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
            start, end, label = ent[0], ent[1], ent[2]
        else:
            continue
        if isinstance(label, list):
            label = label[0] if label else "OTHER"
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            continue
        if end > start >= 0:
            out.append({"start": start, "end": end, "label": str(label)})
    return out


def _parse_json_item(item: Any, source_name: str) -> Optional[Dict[str, Any]]:
    """Parse a single JSON item into {text, entities, source, kind} format."""
    # Format 3: (text, entity_dict) tuple stored as list
    if isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[1], dict):
        text = item[0]
        entities = _normalize_entities(item[1].get("entities"))
        if isinstance(text, str) and text.strip():
            return {"text": text, "entities": entities, "source": source_name, "kind": "json"}
        return None

    if not isinstance(item, dict):
        return None

    # Format 1: Label Studio / annotation tool
    if "data" in item and "annotations" in item:
        text = item["data"].get("text", "")
        entities = []
        for ann in item.get("annotations", []):
            for result in ann.get("result", []):
                val = result.get("value", {})
                if "start" in val and "end" in val and "labels" in val:
                    entities.append({
                        "start": val["start"],
                        "end": val["end"],
                        "label": val["labels"][0] if isinstance(val["labels"], list) else val["labels"],
                    })
        if text:
            return {"text": text, "entities": _normalize_entities(entities),
                    "source": source_name, "kind": "json"}

    # Format 2: Direct {text, entities}
    if "text" in item:
        entities = item.get("entities", [])
        if isinstance(entities, list):
            return {"text": item["text"], "entities": _normalize_entities(entities),
                    "source": source_name, "kind": "json"}

    return None


def save_model_metadata(model_type: str, metrics: Dict[str, Any], model_path: str) -> None:
    """Save training metadata for model versioning."""
    ensure_dirs()
    meta_file = os.path.join(MODELS_DIR, "models_metadata.json")
    metadata = {}
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    if model_type not in metadata:
        metadata[model_type] = []

    entry = {
        **metrics,
        "model_path": model_path,
        "trained_at": __import__("datetime").datetime.now().isoformat(),
    }
    # Retraining writes to the same artifact path. Replace the previous record
    # for that path instead of appending a duplicate version that would also
    # be marked active alongside the fresh one.
    versions = metadata[model_type]
    for index, existing in enumerate(versions):
        if existing.get("model_path") == model_path:
            entry["active"] = bool(existing.get("active"))
            if existing.get("activated_at"):
                entry["activated_at"] = existing["activated_at"]
            versions[index] = entry
            break
    else:
        versions.append(entry)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def list_model_metadata() -> Dict[str, Any]:
    """List all saved model metadata."""
    ensure_dirs()
    meta_file = os.path.join(MODELS_DIR, "models_metadata.json")
    if not os.path.exists(meta_file):
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_jsonl_records(folder_path: str) -> List[Dict[str, Any]]:
    """Read every ``.jsonl`` record in a dataset folder verbatim.

    Unlike ``load_dataset_from_folder`` this keeps records that are not NER
    items (e.g. NLI pairs with premise/hypothesis/label).
    """
    records: List[Dict[str, Any]] = []
    folder = Path(folder_path)
    if not folder.exists():
        return records
    if folder.is_file():
        files = [folder]
    else:
        files = sorted(folder.rglob("*.jsonl"))
    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    return records
