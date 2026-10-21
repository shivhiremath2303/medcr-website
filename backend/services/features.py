"""Shared entity/feature utilities used by BOTH training and inference.

Keeping feature construction in one place guarantees that a model trained here
sees exactly the same inputs when the web app runs it in production.
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Canonical entity taxonomy
# ---------------------------------------------------------------------------
# The Indian legal NER corpus uses: ORG, PROVISION, WITNESS, GPE, LAWYER,
# PRECEDENT, OTHER_PERSON, RESPONDENT, CASE_NUMBER, STATUTE, DATE, PETITIONER,
# COURT, JUDGE.  We keep those and add a few app-level types so the UI can show
# DEFENDANT / EVIDENCE / PLACE as well.
ENTITY_TYPES: List[str] = [
    "COURT",
    "JUDGE",
    "PETITIONER",
    "RESPONDENT",
    "DEFENDANT",
    "WITNESS",
    "LAWYER",
    "DATE",
    "PLACE",
    "EVIDENCE",
    "STATUTE",
    "PROVISION",
    "PRECEDENT",
    "CASE_NUMBER",
    "ORG",
    "PERSON",
    "OTHER",
]
# ``EVENT`` is an app-level display type from the research specification.  It
# is deliberately kept outside ``ENTITY_TYPES`` because that list defines the
# R-GCN one-hot width; changing it would invalidate every deployed checkpoint.
EXTRA_ENTITY_TYPES: List[str] = ["EVENT"]
ENTITY_TYPE_INDEX: Dict[str, int] = {t: i for i, t in enumerate(ENTITY_TYPES)}

_LABEL_MAP = {
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "LOCATION": "PLACE",
    "PLACE": "PLACE",
    "PER": "PERSON",
    "PERSON": "PERSON",
    "OTHER_PERSON": "PERSON",
    "DEFENDANT": "DEFENDANT",
    "ACCUSED": "DEFENDANT",
    "PETITIONER": "PETITIONER",
    "APPELLANT": "PETITIONER",
    "RESPONDENT": "RESPONDENT",
    "WITNESS": "WITNESS",
    "LAWYER": "LAWYER",
    "ADVOCATE": "LAWYER",
    "COURT": "COURT",
    "JUDGE": "JUDGE",
    "DATE": "DATE",
    "TIME": "DATE",
    "EVIDENCE": "EVIDENCE",
    "EXHIBIT": "EVIDENCE",
    "EVENT": "EVENT",
    "INCIDENT": "EVENT",
    "OFFENCE": "EVENT",
    "OFFENSE": "EVENT",
    "STATUTE": "STATUTE",
    "PROVISION": "PROVISION",
    "PRECEDENT": "PRECEDENT",
    "CASE_NUMBER": "CASE_NUMBER",
    "MISC": "OTHER",
    "OTHER": "OTHER",
}


def canonical_type(label: str) -> str:
    """Map an arbitrary NER label (possibly BIO-prefixed) to our taxonomy."""
    if not label:
        return "OTHER"
    lab = label
    if lab.startswith("B-") or lab.startswith("I-"):
        lab = lab[2:]
    lab = lab.upper().strip()
    mapped = _LABEL_MAP.get(lab)
    if mapped:
        return mapped
    if lab in ENTITY_TYPE_INDEX or lab in EXTRA_ENTITY_TYPES:
        return lab
    return "OTHER"


# ---------------------------------------------------------------------------
# Relation categories (R-GCN edge types)
# ---------------------------------------------------------------------------
RELATION_GENERIC = 0
RELATION_PARTY = 1
RELATION_FACTUAL = 2
RELATION_AUTHORITY = 3
RELATION_ORG = 4
NUM_RELATIONS = 5

_PARTY_TYPES = {"PETITIONER", "RESPONDENT", "DEFENDANT", "WITNESS", "LAWYER", "JUDGE", "PERSON"}
_FACTUAL_TYPES = {"DATE", "PLACE", "EVIDENCE", "CASE_NUMBER", "EVENT"}
_AUTHORITY_TYPES = {"STATUTE", "PROVISION", "PRECEDENT", "COURT"}
_ORG_TYPES = {"ORG", "OTHER"}


def relation_for(type_a: str, type_b: str) -> int:
    a, b = canonical_type(type_a), canonical_type(type_b)
    pair = {a, b}
    if pair <= _PARTY_TYPES or (a in _PARTY_TYPES and b in _PARTY_TYPES):
        return RELATION_PARTY
    if pair <= _FACTUAL_TYPES or (a in _FACTUAL_TYPES and b in _FACTUAL_TYPES):
        return RELATION_FACTUAL
    if pair <= _AUTHORITY_TYPES or (a in _AUTHORITY_TYPES and b in _AUTHORITY_TYPES):
        return RELATION_AUTHORITY
    if pair <= _ORG_TYPES or (a in _ORG_TYPES and b in _ORG_TYPES):
        return RELATION_ORG
    return RELATION_GENERIC


# ---------------------------------------------------------------------------
# Document-level features (XGBoost)
# ---------------------------------------------------------------------------
DOC_FEATURES: List[str] = [
    "n_chars",
    "n_words",
    "n_sentences",
    "avg_sentence_len",
    "max_sentence_len",
    "avg_word_len",
    "unique_word_ratio",
    "digit_ratio",
    "upper_ratio",
    "punct_ratio",
    "newline_ratio",
    "repeat_line_ratio",
    "legal_kw_count",
    "legal_kw_ratio",
    "date_entities",
    "court_entities",
    "judge_entities",
    "party_entities",
    "lawyer_entities",
    "statute_entities",
    "evidence_entities",
    "entity_count",
    "entity_density",
    "type_diversity",
    "entities_per_sentence",
    "ends_complete",
    "redaction_signals",
    "placeholder_ratio",
    "boilerplate_signals",
    "short_line_ratio",
]

_LEGAL_KEYWORDS = (
    "court", "judge", "justice", "petitioner", "respondent", "appellant",
    "accused", "witness", "evidence", "exhibit", "section", "article",
    "statute", "act", "provision", "counsel", "advocate", "judgment",
    "order", "appeal", "petition", "hearing", "testimony", "affidavit",
)
_LEGAL_KW_RE = re.compile(r"\b(" + "|".join(_LEGAL_KEYWORDS) + r")\b", re.I)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def document_features(text: str, entities: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Compute the exact feature vector used by the XGBoost anomaly model."""
    text = text or ""
    n_chars = len(text)
    words = re.findall(r"[A-Za-z0-9']+", text)
    n_words = len(words)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sentences = max(1, len(sentences))
    sent_lens = [len(s.split()) for s in sentences] or [0]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen = set()
    repeated = 0
    for ln in lines:
        key = ln.lower()
        if key in seen:
            repeated += 1
        seen.add(key)

    kw = _LEGAL_KW_RE.findall(text)

    counts: Dict[str, int] = {}
    for ent in entities or []:
        t = canonical_type(ent.get("type") or ent.get("label") or "OTHER")
        counts[t] = counts.get(t, 0) + 1

    party_types = ("PETITIONER", "RESPONDENT", "DEFENDANT", "WITNESS", "PERSON")
    party_count = sum(counts.get(t, 0) for t in party_types)

    stripped = text.rstrip()
    ends_complete = 1.0 if stripped and stripped[-1] in '.!?"\')]' else 0.0
    redaction_signals = len(re.findall(r"\[redacted\]|\bredacted\b|\bxxx+\b|_{3,}", text, re.I))
    boilerplate_signals = len(re.findall(
        r"in the high court of judicature|civil appellate jurisdiction|coram:|case no\.\s*_{3,}",
        text, re.I,
    ))

    return {
        "n_chars": float(n_chars),
        "n_words": float(n_words),
        "n_sentences": float(len(sentences)),
        "avg_sentence_len": float(sum(sent_lens) / n_sentences),
        "max_sentence_len": float(max(sent_lens)),
        "avg_word_len": float(sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "unique_word_ratio": _safe_div(len({w.lower() for w in words}), n_words),
        "digit_ratio": _safe_div(sum(ch.isdigit() for ch in text), n_chars),
        "upper_ratio": _safe_div(sum(ch.isupper() for ch in text), n_chars),
        "punct_ratio": _safe_div(sum(not ch.isalnum() and not ch.isspace() for ch in text), n_chars),
        "newline_ratio": _safe_div(text.count("\n"), n_chars),
        "repeat_line_ratio": _safe_div(repeated, len(lines)),
        "legal_kw_count": float(len(kw)),
        "legal_kw_ratio": _safe_div(len(kw), n_words),
        "date_entities": float(counts.get("DATE", 0)),
        "court_entities": float(counts.get("COURT", 0)),
        "judge_entities": float(counts.get("JUDGE", 0)),
        "party_entities": float(party_count),
        "lawyer_entities": float(counts.get("LAWYER", 0)),
        "statute_entities": float(counts.get("STATUTE", 0) + counts.get("PROVISION", 0)),
        "evidence_entities": float(counts.get("EVIDENCE", 0)),
        "entity_count": float(len(entities or [])),
        "entity_density": _safe_div(len(entities or []), n_chars),
        "type_diversity": float(len(counts)),
        "entities_per_sentence": _safe_div(len(entities or []), n_sentences),
        "ends_complete": ends_complete,
        "redaction_signals": float(redaction_signals),
        "placeholder_ratio": _safe_div(len(re.findall(r"_{3,}", text)), max(1, n_words) / 1000.0),
        "boilerplate_signals": float(boilerplate_signals),
        "short_line_ratio": _safe_div(sum(1 for ln in lines if len(ln) < 20), len(lines)),
    }


def feature_row(text: str, entities: Sequence[Dict[str, Any]]) -> List[float]:
    feats = document_features(text, entities)
    return [feats[name] for name in DOC_FEATURES]


# ---------------------------------------------------------------------------
# Graph construction (R-GCN)
# ---------------------------------------------------------------------------
NODE_FEATURES: List[str] = [
    *[f"type_{t}" for t in ENTITY_TYPES],
    "position_norm",
    "log_len",
    "degree_norm",
    "is_first",
    "is_last",
    "type_freq_norm",
    "graph_size_norm",
    "edges_per_node_norm",
    "missing_key_norm",
    "type_entropy",
]
IN_CHANNELS = len(NODE_FEATURES)
MAX_GRAPH_NODES = 48
KEY_ENTITY_TYPES = ("COURT", "JUDGE", "DATE", "PETITIONER", "RESPONDENT")


def entity_nodes(entities: Sequence[Dict[str, Any]], text_len: int) -> List[Dict[str, Any]]:
    """Normalise + cap an entity list for graph construction."""
    cleaned: List[Dict[str, Any]] = []
    for ent in entities or []:
        etype = canonical_type(ent.get("type") or ent.get("label") or "OTHER")
        span = str(ent.get("text", "") or "")
        cleaned.append({
            "text": span,
            "type": etype,
            "start": int(ent.get("start", 0) or 0),
            "end": int(ent.get("end", 0) or 0),
        })
    cleaned.sort(key=lambda e: e["start"])
    if len(cleaned) > MAX_GRAPH_NODES:
        # Keep an even sample across the document so both ends stay represented.
        step = len(cleaned) / MAX_GRAPH_NODES
        cleaned = [cleaned[int(i * step)] for i in range(MAX_GRAPH_NODES)]
    return cleaned


def graph_tensors(entities: Sequence[Dict[str, Any]], text_len: int = 0):
    """Build (x, edge_index, edge_type) tensors compatible with the R-GCN."""
    import torch

    nodes = entity_nodes(entities, text_len)
    n = len(nodes)
    if n == 0:
        return None

    text_len = max(1, int(text_len or (max((e["end"] for e in nodes), default=1))))
    type_counts: Dict[str, int] = {}
    degrees = [0] * n
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_types: List[int] = []

    for i in range(n):
        type_counts[nodes[i]["type"]] = type_counts.get(nodes[i]["type"], 0) + 1

    for i in range(n):
        for j in range(i + 1, n):
            rel = relation_for(nodes[i]["type"], nodes[j]["type"])
            edge_src.extend((i, j))
            edge_dst.extend((j, i))
            edge_types.extend((rel, rel))
            degrees[i] += 1
            degrees[j] += 1

    max_degree = max(degrees) if degrees else 1
    possible_edges = max(1, n * (n - 1) / 2)
    graph_size_norm = min(1.0, n / MAX_GRAPH_NODES)
    edges_per_node_norm = min(1.0, (len(edge_types) / 2) / possible_edges)
    missing_key_norm = sum(1 for t in KEY_ENTITY_TYPES if t not in type_counts) / len(KEY_ENTITY_TYPES)
    probs = [c / n for c in type_counts.values()]
    type_entropy = 0.0
    for p in probs:
        if p > 0:
            type_entropy -= p * math.log(p)
    type_entropy = type_entropy / math.log(len(ENTITY_TYPES)) if len(ENTITY_TYPES) > 1 else 0.0

    rows: List[List[float]] = []
    for i, node in enumerate(nodes):
        onehot = [0.0] * len(ENTITY_TYPES)
        onehot[ENTITY_TYPE_INDEX.get(node["type"], len(ENTITY_TYPES) - 1)] = 1.0
        rows.append([
            *onehot,
            min(1.0, max(0.0, node["start"] / text_len)),
            math.log1p(max(0, node["end"] - node["start"])),
            _safe_div(degrees[i], max_degree),
            1.0 if i == 0 else 0.0,
            1.0 if i == n - 1 else 0.0,
            _safe_div(type_counts.get(node["type"], 0), n),
            graph_size_norm,
            edges_per_node_norm,
            missing_key_norm,
            type_entropy,
        ])

    x = torch.tensor(rows, dtype=torch.float)
    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    return x, edge_index, edge_type, nodes


# ---------------------------------------------------------------------------
# Realistic document-integrity corruptions
# ---------------------------------------------------------------------------
CORRUPTION_KINDS = ["truncation", "redaction", "entity_drop", "duplication", "boilerplate"]


def corrupt_document(
    text: str,
    entities: Sequence[Dict[str, Any]],
    rng: random.Random,
    kind: str | None = None,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Return (corrupted_text, corrupted_entities, kind).

    These simulate real evidence-handling problems: missing pages, redactions,
    metadata loss, duplicated scans and canned headers.
    """
    kind = kind or rng.choice(CORRUPTION_KINDS)
    ents = [
        {
            "text": str(e.get("text", "")),
            "type": canonical_type(e.get("type") or e.get("label") or "OTHER"),
            "start": int(e.get("start", 0) or 0),
            "end": int(e.get("end", 0) or 0),
        }
        for e in (entities or [])
    ]
    ents.sort(key=lambda e: e["start"])

    if kind == "truncation":
        keep = rng.uniform(0.25, 0.6)
        cut = max(60, int(len(text) * keep))
        new_text = text[:cut]
        new_ents = [e for e in ents if e["end"] <= cut]
        return new_text, new_ents, kind

    if kind == "redaction":
        if not ents:
            cut = max(60, int(len(text) * 0.5))
            return text[:cut] + "\n[REDACTED]\n", [], kind
        keep_idx = set(range(len(ents)))
        drop_n = max(1, int(len(ents) * rng.uniform(0.45, 0.8)))
        dropped = set(rng.sample(sorted(keep_idx), min(drop_n, len(ents))))
        pieces: List[str] = []
        cursor = 0
        for i, e in enumerate(ents):
            if i not in dropped:
                continue
            s, en = e["start"], e["end"]
            if s < cursor or s >= len(text):
                continue
            pieces.append(text[cursor:s])
            pieces.append("[REDACTED]")
            cursor = min(len(text), en)
        pieces.append(text[cursor:])
        new_text = "".join(pieces)
        new_ents = [e for i, e in enumerate(ents) if i not in dropped]
        return new_text, new_ents, kind

    if kind == "entity_drop":
        # Simulate lost metadata: most entity mentions disappear.
        keep_ratio = rng.uniform(0.1, 0.4)
        new_ents = [e for e in ents if rng.random() < keep_ratio]
        return text, new_ents, kind

    if kind == "duplication":
        # Simulate a duplicated scan / repeated page.
        paras = [p for p in text.split("\n\n") if p.strip()] or [text]
        block = paras[: max(1, len(paras) // 3)]
        extra = "\n\n".join(block)
        new_text = text + "\n\n" + extra + "\n\n" + extra
        offset = len(text) + 2
        dup = [dict(e, start=e["start"] + offset, end=e["end"] + offset) for e in ents]
        dup2 = [dict(e, start=e["start"] + offset + len(extra) + 2,
                     end=e["end"] + offset + len(extra) + 2) for e in ents]
        return new_text, ents + dup + dup2, kind

    # boilerplate header injection
    header = (
        "IN THE HIGH COURT OF JUDICATURE\n"
        "CIVIL APPELLATE JURISDICTION\n"
        "CASE NO. ______ OF 20__\n"
        "CORAM: HON'BLE MR. JUSTICE ______\n"
    )
    repeat = rng.randint(2, 4)
    prefix = (header + "\n") * repeat
    offset = len(prefix)
    new_ents = [dict(e, start=e["start"] + offset, end=e["end"] + offset) for e in ents]
    return prefix + text, new_ents, kind


# ---------------------------------------------------------------------------
# Lightweight regex entity extractor (used when no NER model is available)
# ---------------------------------------------------------------------------
_REGEX_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("CASE_NUMBER", re.compile(r"\b(?:[A-Z][A-Za-z]*\s+)?(?:No\.?|Nos\.?)\s*[A-Za-z0-9./-]+")),
    ("DATE", re.compile(r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4})\b", re.I)),
    ("DATE", re.compile(r"\b(?:1[5-9]\d{2}|20\d{2})\b")),
    ("DATE", re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")),
    ("EVENT", re.compile(
        r"\b(?:incident|occurrence|murder|culpable homicide|theft|robbery|dacoity|"
        r"assault|grievous hurt|accident|fire|fraud|cheating|conspiracy|abduction|"
        r"kidnapping|rape|dowry death|suicide|attack|altercation|scuffle)\b",
        re.I,
    )),
    ("COURT", re.compile(r"\b(?:Supreme Court|High Court(?: of [A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?)?|District Court|Sessions Court|Magistrate(?:'s)? Court|Tribunal)\b")),
    ("JUDGE", re.compile(r"\b(?:Justice|Hon'ble Mr\. Justice|Hon'ble Mrs\. Justice)\s+[A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+){0,2}")),
    ("STATUTE", re.compile(r"\b(?:Section|Sec\.|S\.|Article|Art\.)\s*\d+[A-Za-z]?(?:\s+of\s+[A-Z][A-Za-z ]+)?")),
    ("LAWYER", re.compile(r"\b(?:Sr\.\s+)?(?:Advocate|Adv\.|Counsel|Public Prosecutor)(?![A-Za-z])(?:[ \t]+[A-Z][A-Za-z.]+){0,3}")),
    ("EVIDENCE", re.compile(
        r"\b(?:Exhibit|Ex\.|Ext\.|EV-\d+|P-\d+|FIR|First Information Report|"
        r"charge[- ]?sheet|post[- ]?mortem(?: report)?|seizure memo|mahazar|"
        r"CCTV(?: footage)?|receipt|affidavit|witness statement|case diary|"
        r"medical report|forensic report|call records?|photograph)\b",
        re.I,
    )),
]


def extract_entities_regex(text: str) -> List[Dict[str, Any]]:
    """Best-effort entity extraction for documents without gold annotations."""
    out: List[Dict[str, Any]] = []
    taken: List[Tuple[int, int]] = []
    for etype, pattern in _REGEX_PATTERNS:
        for m in pattern.finditer(text or ""):
            s, e = m.span()
            if e - s < 2:
                continue
            if any(not (e <= ts or s >= te) for ts, te in taken):
                continue
            taken.append((s, e))
            out.append({"text": m.group(0).strip(), "type": etype, "start": s, "end": e,
                        "confidence": 0.86, "source": "regex"})
    out.sort(key=lambda x: x["start"])
    return out


def merge_entities(primary: Sequence[Dict[str, Any]],
                   secondary: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine a model's predictions with rule-based findings.

    Primary (model) spans win; secondary spans are only added when they do not
    overlap an existing entity. This keeps the NER model authoritative while
    still capturing highly structured items (dates, case numbers, exhibits)
    that the model may miss.
    """
    merged: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []

    for ent in list(primary or []) + list(secondary or []):
        try:
            start = int(ent.get("start", 0) or 0)
            end = int(ent.get("end", 0) or 0)
        except Exception:
            continue
        if end <= start:
            continue
        if any(not (end <= s or start >= e) for s, e in occupied):
            continue
        occupied.append((start, end))
        merged.append({
            "text": ent.get("text", ""),
            "type": canonical_type(ent.get("type") or ent.get("label") or "OTHER"),
            "start": start,
            "end": end,
            "confidence": ent.get("confidence", 0.9),
        })

    merged.sort(key=lambda e: e["start"])
    return merged
