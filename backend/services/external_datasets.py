"""Download and materialise large public legal datasets for training.

Everything is cached under ``services/dataset_store/<name>/`` so the rest of the
app can treat these exactly like an uploaded ZIP dataset.

Sources
-------
* legal_ner_large  -> AjayMukundS/Indian_Legal_NER_Dataset (MIT)
                      13 legal entity types, ~8k annotated Indian judgments.
* legal_nli_large  -> stanfordnlp/snli (CC-BY-SA-4.0)
                      balanced 3-class NLI (entailment / neutral / contradiction).
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

from .dataset_service import DATASET_DIR, ensure_dirs

NER_DATASET_ID = "AjayMukundS/Indian_Legal_NER_Dataset"
NLI_DATASET_ID = "stanfordnlp/snli"

BUILDERS = {"legal_ner_large", "legal_nli_large"}


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_legal_ner(max_records: int = 30000, force: bool = False) -> Dict[str, Any]:
    """Materialise the Indian legal NER corpus (text + character-span entities)."""
    from datasets import get_dataset_split_names, load_dataset

    out_dir = os.path.join(DATASET_DIR, "legal_ner_large")
    data_path = os.path.join(out_dir, "data.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(data_path) and not force:
        return json.load(open(meta_path, encoding="utf-8"))

    ensure_dirs()
    os.makedirs(out_dir, exist_ok=True)

    try:
        splits = get_dataset_split_names(NER_DATASET_ID)
    except Exception:
        splits = ["train"]
    ordered = [s for s in ("train", "dev", "validation") if s in splits] or splits[:1]

    rows: List[Dict[str, Any]] = []
    for split in ordered:
        ds = load_dataset(NER_DATASET_ID, split=split, streaming=True)
        for row in ds:
            text = (row.get("text") or "").strip()
            ents = row.get("entities") or []
            if len(text) < 40 or not isinstance(ents, list):
                continue
            clean_ents = []
            for e in ents:
                try:
                    s, en = int(e["start"]), int(e["end"])
                except Exception:
                    continue
                if 0 <= s < en <= len(text):
                    clean_ents.append({"start": s, "end": en, "label": e.get("label", "OTHER")})
            if not clean_ents:
                continue
            rows.append({
                "text": text,
                "entities": clean_ents,
                "source": f"hf:{NER_DATASET_ID}:{split}",
            })
            if len(rows) >= max_records:
                break
        if len(rows) >= max_records:
            break

    if not rows:
        raise RuntimeError("Could not download the legal NER dataset")

    _write_jsonl(data_path, rows)
    meta = {
        "name": "legal_ner_large",
        "source": NER_DATASET_ID,
        "license": "MIT",
        "task": "legal named-entity recognition",
        "records": len(rows),
        "splits": ordered,
        "entity_types": sorted({e["label"] for r in rows for e in r["entities"]}),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_legal_nli(
    train_per_class: int = 5000,
    test_per_class: int = 1000,
    force: bool = False,
) -> Dict[str, Any]:
    """Materialise a balanced SNLI subset for 3-class contradiction detection."""
    from datasets import load_dataset

    out_dir = os.path.join(DATASET_DIR, "legal_nli_large")
    train_path = os.path.join(out_dir, "train.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(train_path) and os.path.exists(test_path) and not force:
        return json.load(open(meta_path, encoding="utf-8"))

    ensure_dirs()
    os.makedirs(out_dir, exist_ok=True)

    # SNLI label -> our label (0 consistent / 1 contradiction / 2 neutral)
    label_map = {0: 0, 2: 1, 1: 2}
    wanted = {"train": train_per_class, "validation": test_per_class}

    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, per_class in wanted.items():
        ds = load_dataset(NLI_DATASET_ID, split=split, streaming=True)
        counts = {0: 0, 1: 0, 2: 0}
        rows: List[Dict[str, Any]] = []
        for row in ds:
            snli_label = row.get("label")
            if snli_label not in label_map:
                continue
            our_label = label_map[snli_label]
            if counts[our_label] >= per_class:
                continue
            premise = (row.get("premise") or "").strip()
            hypothesis = (row.get("hypothesis") or "").strip()
            if not premise or not hypothesis:
                continue
            rows.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "label": our_label,
                "source": f"hf:{NLI_DATASET_ID}:{split}",
            })
            counts[our_label] += 1
            if all(counts[k] >= per_class for k in counts):
                break
        random.Random(42).shuffle(rows)
        result[split] = rows

    _write_jsonl(train_path, result["train"])
    _write_jsonl(test_path, result["validation"])
    meta = {
        "name": "legal_nli_large",
        "source": NLI_DATASET_ID,
        "license": "CC-BY-SA-4.0",
        "task": "natural language inference / contradiction detection",
        "labels": {"0": "consistent", "1": "contradiction", "2": "neutral"},
        "train_records": len(result["train"]),
        "test_records": len(result["validation"]),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def build_dataset(name: str, force: bool = False, **kwargs) -> Dict[str, Any]:
    if name == "legal_ner_large":
        return build_legal_ner(force=force, **kwargs)
    if name == "legal_nli_large":
        return build_legal_nli(force=force, **kwargs)
    raise ValueError(f"Unknown external dataset: {name}")


def build_all(force: bool = False) -> Dict[str, Any]:
    return {
        "legal_ner_large": build_legal_ner(force=force),
        "legal_nli_large": build_legal_nli(force=force),
    }


def load_nli_split(split: str) -> List[Dict[str, Any]]:
    path = os.path.join(DATASET_DIR, "legal_nli_large", f"{split}.jsonl")
    if not os.path.exists(path):
        return []
    return _read_jsonl(path)
