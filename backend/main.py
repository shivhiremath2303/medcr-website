"""MECR Backend — FastAPI, aligned with the research paper.

"Multi-Defendant & Multi-Evidence Conflict Resolver"

Pipeline (4 modules, 5 models):
  1. Data Ingestion & Pre-processing        → SpaCy (clean / tokenize / BIO / vectorize)
  2. Deep-Learning NER                      → BiLSTM-CRF
  3. Evidence Cross-Referencing / Conflict  → LegalBERT (contradictions)
                                             → XGBoost   (anomalies)
                                             → R-GCN     (relationships)
  4. Explainable Insights & Reporting       → rule reasoning + report

Run:  cd backend && pip install -r requirements.txt && python -m uvicorn main:app --reload --port 8000
"""
import asyncio
import hashlib
import json
import os
import re
import uuid
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel

from services.dataset_service import (
    extract_zip,
    list_datasets,
    load_dataset_from_folder,
    list_model_metadata,
    save_model_metadata,
    MODELS_DIR,
    DATASET_DIR,
)
from services.text_extraction import extract_bytes as extract_uploaded_text
from services.rules import (
    annotate_findings,
    evidence_person_ids,
    inject_case_party_entities,
    integrity_signals,
    manipulation_anomalies,
    missing_information,
    party_entity_type,
    party_summary,
)

# ---------------------------------------------------------------------------
# Model Manager — loads active trained models from models_metadata.json
# When you "Activate" a model in the admin panel, it gets used here.
#
# Heavy ML libraries (torch, transformers, xgboost, torch_geometric) and the
# training service are imported lazily inside the methods that need them so
# FastAPI can start without loading the whole dependency tree.
# ---------------------------------------------------------------------------


class ModelManager:
    """Loads and caches trained models. Only the 'active' model per type is used."""

    def __init__(self):
        self._bilstm_crf = None
        self._xgboost = None
        self._legalbert = None
        self._rgcn = None
        self._word2idx = None
        self._tag2idx = None
        self._idx2tag = None

    def _load_metadata(self) -> dict:
        meta_file = os.path.join(MODELS_DIR, "models_metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _resolve_model_path(self, path: Optional[str]) -> Optional[str]:
        """Return a usable model path, tolerating moved checkouts.

        Activation metadata stores absolute paths from the machine that trained
        the model.  If that path no longer exists (repo moved, or the app is
        deployed elsewhere), fall back to the same filename inside MODELS_DIR.
        """
        if not path:
            return None
        if os.path.exists(path):
            return path
        name = os.path.basename(path.rstrip("\\/"))
        candidate = os.path.join(MODELS_DIR, name)
        return candidate if os.path.exists(candidate) else path

    def _get_active_path(self, model_type: str) -> Optional[str]:
        metadata = self._load_metadata()
        for mtype, versions in metadata.items():
            for v in versions:
                if v.get("active") and mtype == model_type:
                    return self._resolve_model_path(v.get("model_path"))
        return None

    # ----- BiLSTM-CRF NER -----
    def get_bilstm_crf(self):
        import torch

        if self._bilstm_crf is not None:
            return self._bilstm_crf, self._word2idx, self._tag2idx, self._idx2tag
        path = self._get_active_path("bilstm_crf")
        if not path or not os.path.exists(path):
            return None, None, None, None
        try:
            entry = self._get_active_entry("bilstm_crf") or {}
            dataset = entry.get("dataset", "")
            w2i_path = path.replace(".pth", "_word2idx.json")
            t2i_path = path.replace(".pth", "_tag2idx.json")
            if not os.path.exists(w2i_path):
                w2i_path = os.path.join(MODELS_DIR, f"word2idx_{dataset}.json")
            if not os.path.exists(t2i_path):
                t2i_path = os.path.join(MODELS_DIR, f"tag2idx_{dataset}.json")
            with open(w2i_path, "r", encoding="utf-8") as f:
                self._word2idx = json.load(f)
            with open(t2i_path, "r", encoding="utf-8") as f:
                self._tag2idx = json.load(f)
            self._idx2tag = {v: k for k, v in self._tag2idx.items()}

            from services.training_service import BiLSTM_CRF
            model = BiLSTM_CRF(len(self._word2idx), self._tag2idx)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            self._bilstm_crf = model
            return self._bilstm_crf, self._word2idx, self._tag2idx, self._idx2tag
        except Exception as e:
            print(f"[ModelManager] Failed to load BiLSTM-CRF: {e}")
            return None, None, None, None

    # ----- XGBoost Anomaly -----
    def get_xgboost(self):
        import xgboost as xgb

        if self._xgboost is not None:
            return self._xgboost
        path = self._get_active_path("xgboost")
        if not path or not os.path.exists(path):
            return None
        try:
            model = xgb.XGBClassifier()
            model.load_model(path)
            self._xgboost = model
            return self._xgboost
        except Exception as e:
            print(f"[ModelManager] Failed to load XGBoost: {e}")
            return None

    # ----- LegalBERT Contradiction -----
    def get_legalbert(self):
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        if self._legalbert is not None:
            return self._legalbert
        path = self._get_active_path("legalbert")
        if not path or not os.path.exists(path):
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.eval()
            self._legalbert = (model, tokenizer)
            return self._legalbert
        except Exception as e:
            print(f"[ModelManager] Failed to load LegalBERT: {e}")
            return None

    # ----- R-GCN -----
    def get_rgcn(self):
        import torch

        if self._rgcn is not None:
            return self._rgcn
        path = self._get_active_path("rgcn")
        if not path or not os.path.exists(path):
            return None
        try:
            from services.training_service import LegalRGCNModel
            meta = self._get_active_entry("rgcn") or {}
            in_channels = int(meta.get("in_channels", 23))
            num_relations = int(meta.get("num_relations", 5))
            hidden_channels = int(meta.get("hidden_channels", 64))
            model = LegalRGCNModel(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=2,
                num_relations=num_relations,
            )
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            self._rgcn = model
            return self._rgcn
        except Exception as e:
            print(f"[ModelManager] Failed to load R-GCN: {e}")
            return None

    def _get_active_entry(self, model_type: str) -> Optional[dict]:
        metadata = self._load_metadata()
        for v in metadata.get(model_type, []):
            if v.get("active"):
                return v
        return None

    def reset(self) -> None:
        """Forget cached models so newly activated weights are picked up."""
        self._bilstm_crf = None
        self._xgboost = None
        self._legalbert = None
        self._rgcn = None
        self._word2idx = None
        self._tag2idx = None
        self._idx2tag = None


model_manager = ModelManager()


# ----- BiLSTM-CRF NER inference -----
def run_bilstm_crf_ner(text: str):
    """Run BiLSTM-CRF NER over long documents using overlapping windows."""
    import torch

    model, word2idx, tag2idx, idx2tag = model_manager.get_bilstm_crf()
    if model is None:
        return None
    from services.features import canonical_type

    import spacy
    nlp = spacy.blank("en")
    doc = nlp(text[:20000])
    tokens = doc[:]
    if len(tokens) == 0:
        return []

    window = 128
    stride = window - 24
    spans = []  # (start_token, end_token, type)

    for offset in range(0, len(tokens), stride):
        chunk = tokens[offset:offset + window]
        if not chunk:
            continue
        x = torch.tensor([[word2idx.get(t.text, word2idx.get("UNK", 1)) for t in chunk]],
                         dtype=torch.long)
        mask = (x != word2idx.get("PAD", 0)).to(torch.bool)
        with torch.no_grad():
            try:
                pred = model.predict(x, mask=mask)[0]
            except Exception:
                continue

        local_tags = [idx2tag.get(i, "O") for i in pred]
        cur_start, cur_type = None, None
        for i, tag in enumerate(local_tags):
            if tag.startswith("B-"):
                if cur_start is not None:
                    spans.append((offset + cur_start, offset + i - 1, cur_type))
                cur_start, cur_type = i, tag[2:]
            elif tag.startswith("I-") and cur_start is not None and tag[2:] == cur_type:
                continue
            else:
                if cur_start is not None:
                    spans.append((offset + cur_start, offset + i - 1, cur_type))
                cur_start, cur_type = None, None
        if cur_start is not None and len(chunk) > 0:
            spans.append((offset + cur_start, offset + len(chunk) - 1, cur_type))
        if offset + window >= len(tokens):
            break

    # Deduplicate overlapping windows, preferring the longest span.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    chosen = []
    for span in spans:
        if chosen and span[0] <= chosen[-1][1]:
            prev = chosen[-1]
            if (span[1] - span[0]) > (prev[1] - prev[0]):
                chosen[-1] = span
            continue
        chosen.append(span)

    entities = []
    for start_tok, end_tok, etype in chosen:
        if start_tok >= len(tokens) or end_tok >= len(tokens):
            continue
        start_char = tokens[start_tok].idx
        end_char = tokens[end_tok].idx + len(tokens[end_tok].text)
        if end_char <= start_char:
            continue
        entities.append({
            "text": text[start_char:end_char],
            "type": canonical_type(etype),
            "start": int(start_char),
            "end": int(end_char),
            "confidence": 0.9,
        })
    return entities


# ----- LegalBERT contradiction -----
_CLAIM_PREFIX_RE = re.compile(
    r"^\s*(?:(?:witness statement recorded on oath|witness statement|"
    r"the witness (?:stated|recorded|affirmed|deposed)[^:.]*)\s*[:\-]\s*|"
    r"scanned case diary\s*-\s*ocr partial\s*[:\-]?\s*|"
    r"scanned case diary\s*[:\-]?\s*)",
    re.I,
)


def _normalize_claim(text: str) -> str:
    """Strip meta prefixes so the NLI model compares the facts themselves.

    A quoted note such as ``Witness statement recorded on oath: <sentence>``
    otherwise shifts the input far away from the sentence pairs LegalBERT was
    trained on, and the model defaults to 'neutral'.
    """
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _CLAIM_PREFIX_RE.sub("", cleaned, count=1).strip()
    return cleaned


def run_legalbert_contradiction(statement_a: str, statement_b: str):
    """Run LegalBERT contradiction detection. Returns (is_contradiction, confidence, explanation)."""
    import torch

    result = model_manager.get_legalbert()
    if result is None:
        return None
    model, tokenizer = result

    claim_a = _normalize_claim(statement_a)
    claim_b = _normalize_claim(statement_b)
    inputs = tokenizer(claim_a, claim_b, return_tensors="pt",
                       truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits, dim=-1).item())
        confidence = float(probs[0][pred].item())

    default_names = {0: "consistent", 1: "contradiction", 2: "neutral"}
    id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
    name = str(id2label.get(pred, default_names.get(pred, "neutral"))).lower()
    if name.isdigit():
        name = default_names.get(int(name), "neutral")

    is_contradiction = "contradict" in name
    explanation = (
        f"LegalBERT classified the pair as '{name}' "
        f"(confidence {confidence:.2f})."
    )
    return is_contradiction, confidence, explanation


# ----- XGBoost Anomaly -----
def run_xgboost_anomaly(text: str, entities: Optional[list] = None):
    """Run XGBoost document-integrity anomaly detection. Returns (is_anomaly, score)."""
    model = model_manager.get_xgboost()
    if model is None:
        return None
    try:
        from services.features import feature_row, DOC_FEATURES
        import pandas as pd
        row = feature_row(text or "", entities or [])
        if len(row) != len(DOC_FEATURES):
            return None
        X = pd.DataFrame([row], columns=DOC_FEATURES)
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        return pred, proba
    except Exception as exc:
        print(f"[XGBoost] inference failed: {exc}")
        return None


# ----- R-GCN -----
def run_rgcn_graph(entities: list, text_len: int = 0):
    """Run R-GCN on the entity graph. Returns a React-Flow compatible graph."""
    import torch

    model = model_manager.get_rgcn()
    if model is None or len(entities) < 1:
        return None

    try:
        from services.features import graph_tensors, relation_for, canonical_type
        built = graph_tensors(entities, text_len)
    except Exception as exc:
        print(f"[R-GCN] graph build failed: {exc}")
        return None
    if built is None:
        return None

    x, edge_index, edge_type, nodes = built
    batch_vec = torch.zeros(x.size(0), dtype=torch.long)

    with torch.no_grad():
        try:
            out = model(x, edge_index, edge_type, batch_vec)
        except Exception as exc:
            print(f"[R-GCN] forward failed: {exc}")
            return None
        probs = torch.softmax(out, dim=-1)
        pred = int(out.argmax(dim=1).item())
        confidence = float(probs[0][pred].item())

    relation_names = {0: "related", 1: "party", 2: "factual", 3: "authority", 4: "organization"}

    vis_nodes = []
    for i, node in enumerate(nodes):
        vis_nodes.append({
            "id": f"ent-{i}",
            "data": {"label": f"{node['text']} ({node['type']})"},
            "position": {"x": (i % 6) * 200, "y": (i // 6) * 130},
            "type": "default",
        })

    vis_edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            rel = relation_for(nodes[i]["type"], nodes[j]["type"])
            vis_edges.append({
                "id": f"e-{i}-{j}",
                "source": f"ent-{i}",
                "target": f"ent-{j}",
                "label": relation_names.get(rel, "related"),
                "style": {"stroke": "#A8384A" if pred == 1 else "#3B5B8C"},
            })

    return {"nodes": vis_nodes, "edges": vis_edges, "prediction": pred,
            "confidence": confidence}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="MECR Backend", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",  # Vite dev ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _no_store_api(request: Request, call_next):
    """Never let a browser reuse one account's case data for another account."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    """URL-safe identifier (case numbers contain '/' and spaces)."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
DATA_FILE = os.path.join(BASE_DIR, "mecr_data.json")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _read_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return default
    return default


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# In-memory stores (persisted to disk)
# ---------------------------------------------------------------------------
USERS: Dict[str, dict] = _read_json(USERS_FILE, {})
CASES: Dict[str, dict] = {}
EVIDENCE: Dict[str, dict] = {}
ANALYSES: Dict[str, dict] = {}  # caseId -> {"results": {...}, "status": ...}

# ---------------------------------------------------------------------------
# Mock session store (token -> email). In production replace with real JWT.
# ---------------------------------------------------------------------------
ACTIVE_SESSIONS: Dict[str, str] = {}


def _first_user_is_admin() -> bool:
    return not any(u.get("is_admin") for u in USERS.values())


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------
def get_current_email(authorization: str | None = Header(None)) -> str | None:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1]
    return ACTIVE_SESSIONS.get(token)


def get_current_user(email: str | None = Depends(get_current_email)) -> dict | None:
    if not email:
        return None
    return USERS.get(email)


def require_admin(user: dict | None = Depends(get_current_user)) -> dict:
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def require_user(user: dict | None = Depends(get_current_user)) -> dict:
    """Any signed-in account. Case data is scoped to this user."""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _save_users() -> None:
    _write_json(USERS_FILE, USERS)


def _save_data() -> None:
    _write_json(DATA_FILE, {"cases": CASES, "evidence": EVIDENCE})


def _first_admin_email() -> Optional[str]:
    return next((email for email, user in USERS.items() if user.get("is_admin")), None)


def _claim_unowned_cases(email: str) -> int:
    """Give legacy/ownerless cases to the administrator account.

    Runs once after an upgrade (and on a fresh install when the first admin
    registers) so demo data is not visible to later regular users.
    """
    claimed = 0
    for case in CASES.values():
        if not case.get("owner"):
            case["owner"] = email
            claimed += 1
    if claimed:
        _save_data()
    return claimed


def _load_data() -> None:
    data = _read_json(DATA_FILE, {})
    CASES.update(data.get("cases", {}))
    EVIDENCE.update(data.get("evidence", {}))
    default_owner = _first_admin_email()
    if default_owner:
        _claim_unowned_cases(default_owner)


def _owned_case(case_id: str, user: dict) -> dict:
    """Return the case only when it belongs to this user (404 otherwise)."""
    case = CASES.get(case_id)
    if not case or case.get("owner") != user["email"]:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


def _owned_evidence(evidence_id: str, user: dict) -> dict:
    ev = EVIDENCE.get(evidence_id)
    case = CASES.get((ev or {}).get("caseId"))
    if not ev or not case or case.get("owner") != user["email"]:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return ev


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    salt = salt or uuid.uuid4().hex
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
    return digest, salt


def _verify_password(password: str, digest: str, salt: str) -> bool:
    return _hash_password(password, salt)[0] == digest


def _password_errors(password: str) -> List[str]:
    errors: List[str] = []
    if len(password) < 8:
        errors.append("at least 8 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("an uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("a lowercase letter")
    if not re.search(r"\d", password):
        errors.append("a number")
    if not re.search(r"[^A-Za-z0-9]", password):
        errors.append("a special character")
    return errors


# ---------------------------------------------------------------------------
# Demo / mock analysis content (the layer your trained models will replace)
# ---------------------------------------------------------------------------
PIPELINE_STAGES = [
    "Cleaning & BIO-tagging text",
    "Running BiLSTM-CRF entity extraction",
    "Building knowledge graph (R-GCN)",
    "Cross-referencing claims (LegalBERT)",
    "Detecting anomalies (XGBoost)",
    "Generating explainable report",
]

LEGAL_TEXT = (
    "In The High Court Of Kerala At Ernakulam, Crl MC No. 1622 of 2006, before Justice Menon, "
    "the petitioner T.R. Ajayan, S/O O. Raman, represented by Adv. R. Krishnan, submitted a "
    "forensic report referencing CCTV footage timestamped 22:14 as contradicting the statement "
    "given by respondent K. Balan, who claimed presence at a separate location supported by a "
    "fuel receipt dated the same evening."
)

AUDIO_TRANSCRIPT = (
    "Witness S. Nair states that on the evening of the incident K. Balan was near the market "
    "until about 22:05. The witness does not mention the fuel station and cannot confirm the "
    "22:20 sighting claimed in the receipt."
)

# (substring, entity type) — offsets computed from the source text, non-overlapping
_ENTITY_SPECS = [
    ("High Court Of Kerala At Ernakulam", "COURT"),
    ("2006", "DATE"),
    ("Justice Menon", "JUDGE"),
    ("T.R. Ajayan", "DEFENDANT"),
    ("O. Raman", "WITNESS"),
    ("Adv. R. Krishnan", "LAWYER"),
    ("CCTV footage", "EVIDENCE"),
    ("22:14", "DATE"),
    ("contradicting the statement", "EVENT"),
    ("K. Balan", "DEFENDANT"),
    ("separate location", "PLACE"),
    ("fuel receipt", "EVIDENCE"),
]

_AUDIO_ENTITY_SPECS = [
    ("S. Nair", "WITNESS"),
    ("K. Balan", "DEFENDANT"),
    ("market", "PLACE"),
    ("22:05", "DATE"),
    ("fuel station", "EVIDENCE"),
    ("22:20", "DATE"),
]


def _entities_from(text: str, specs) -> List[dict]:
    out: List[dict] = []
    cursor = 0
    for needle, etype in specs:
        start = text.find(needle, cursor)
        if start == -1:
            continue
        end = start + len(needle)
        out.append({"text": needle, "type": etype, "start": start, "end": end,
                    "confidence": round(0.86 + (len(needle) % 12) / 100, 2)})
        cursor = end
    return out


IMAGE_ANOMALIES = [
    {"x": 12, "y": 30, "w": 35, "h": 8, "note": "Signature block misaligned by 3px — possible splice", "confidence": 0.93},
    {"x": 55, "y": 62, "w": 28, "h": 10, "note": "Font mismatch in date field", "confidence": 0.71},
]


def _document_for(ev: Optional[dict]) -> dict:
    """Return the document view for an exhibit.

    Canned demo text is only used for the seeded demonstration case (marked
    ``demo``).  An arbitrary uploaded PDF/image with no text layer returns an
    empty document plus extraction metadata instead of silently analysing
    unrelated sample text.
    """
    ev = ev or {}
    raw = (ev.get("rawText") or "").strip()
    kind = ev.get("kind", "text")
    extraction = ev.get("extraction") or {}
    if raw:
        entities = ev.get("entities") or []
        if not entities:
            from services.features import extract_entities_regex
            entities = extract_entities_regex(raw)
        return {
            "kind": kind,
            "imageUrl": ev.get("imageUrl"),
            "rawText": raw,
            "transcript": raw if kind == "audio" else None,
            "entities": entities,
            "anomalies": [],
            "extraction": extraction,
            "charCount": len(raw),
        }

    # Seeded demonstration exhibits keep their canned narrative so the demo
    # still works on a fresh install.  Real uploads never take this branch.
    if ev.get("demo"):
        if kind == "audio":
            return {"kind": "audio", "imageUrl": None, "rawText": AUDIO_TRANSCRIPT,
                    "transcript": AUDIO_TRANSCRIPT,
                    "entities": _entities_from(AUDIO_TRANSCRIPT, _AUDIO_ENTITY_SPECS),
                    "anomalies": [], "extraction": {"method": "demo"}, "charCount": len(AUDIO_TRANSCRIPT)}
        if kind == "image":
            return {"kind": "image", "imageUrl": None, "rawText": LEGAL_TEXT,
                    "entities": _entities_from(LEGAL_TEXT, _ENTITY_SPECS),
                    "anomalies": IMAGE_ANOMALIES, "extraction": {"method": "demo"},
                    "charCount": len(LEGAL_TEXT)}
        return {"kind": "pdf", "imageUrl": None, "rawText": LEGAL_TEXT,
                "entities": _entities_from(LEGAL_TEXT, _ENTITY_SPECS),
                "anomalies": [], "extraction": {"method": "demo"}, "charCount": len(LEGAL_TEXT)}

    return {
        "kind": kind,
        "imageUrl": ev.get("imageUrl"),
        "rawText": "",
        "transcript": None,
        "entities": [],
        "anomalies": [],
        "extraction": extraction or {
            "method": "none",
            "needs_ocr": kind in {"image", "pdf"},
            "error": "No text was extracted from this file.",
        },
        "charCount": 0,
    }


def _conflicts_for(case_id: str) -> List[dict]:
    return [
        {
            "id": "c1",
            "type": "Location contradiction",
            "severity": "material",
            "model": "LegalBERT",
            "confidence": 0.87,
            "left": {"source": "CCTV footage [EV-0114]", "claim": "Places K. Balan at the scene at 22:14"},
            "right": {"source": "Fuel receipt [EV-0119]", "claim": "Places K. Balan 40km away at 22:20"},
            "explanation": "LegalBERT scored the two statements as semantically contradictory (0.87). "
                           "The CCTV timestamp and the receipt time differ by 6 minutes yet imply locations "
                           "40 km apart, which is not physically possible.",
        },
        {
            "id": "c2",
            "type": "Timeline gap",
            "severity": "low",
            "model": "XGBoost",
            "confidence": 0.62,
            "left": {"source": "Testimony [T-0027]", "claim": "Describes a 90-minute unaccounted window"},
            "right": {"source": "— missing evidence —", "claim": "No corroborating record found"},
            "explanation": "No document covers 21:20–22:50. XGBoost flagged this as a missing-information "
                           "anomaly (0.62) because the entity timeline has an unexplained gap.",
        },
    ]


def _anomalies_for(case_id: str) -> List[dict]:
    return [
        {"id": "a1", "entity": "T.R. Ajayan (testimony) vs CCTV footage EV-0114", "type": "contradiction",
         "score": 0.81, "label": "High", "model": "LegalBERT",
         "explanation": "Statements describe conflicting locations for the same 22:14 timestamp.",
         "sources": ["Testimony T-0027", "EV-0114"]},
        {"id": "a2", "entity": "K. Balan (testimony) vs fuel receipt EV-0119", "type": "contradiction",
         "score": 0.44, "label": "Low", "model": "LegalBERT",
         "explanation": "Weak semantic mismatch; could be explained by travel time between the two points.",
         "sources": ["Testimony T-0029", "EV-0119"]},
        {"id": "a3", "entity": "Timeline window 22:14–22:20", "type": "missing_information",
         "score": 0.62, "label": "Medium", "model": "XGBoost",
         "explanation": "No corroborating record exists for a 6-minute window central to the alibi.",
         "sources": ["EV-0114", "EV-0119"]},
        {"id": "a4", "entity": "Signature block on EV-0119", "type": "manipulation",
         "score": 0.93, "label": "High", "model": "XGBoost",
         "explanation": "Pixel-level alignment and font analysis indicate the signature block was spliced.",
         "sources": ["EV-0119"]},
    ]


_ENTITY_CLASS = {
    "PETITIONER": "party", "RESPONDENT": "party", "DEFENDANT": "party",
    "WITNESS": "party", "JUDGE": "party", "LAWYER": "party",
    "PERSON": "party", "ORG": "party",
    "COURT": "authority", "STATUTE": "authority", "PROVISION": "authority",
    "PRECEDENT": "authority", "CASE_NUMBER": "authority",
    "DATE": "fact", "PLACE": "fact", "EVIDENCE": "fact",
}
_CLASS_ORDER = ["party", "authority", "fact"]
_CLASS_X = {"document": 90, "party": 350, "authority": 610, "fact": 870}
_CLASS_STYLE = {
    "document": {"label": "Documents", "tone": "#3B5B8C"},
    "party": {"label": "Parties & people", "tone": "#B98A2E"},
    "authority": {"label": "Court & law", "tone": "#3F7D6B"},
    "fact": {"label": "Facts & evidence", "tone": "#8C5A9E"},
}
_CLASS_RANK = {"party": 0, "authority": 1, "fact": 2}


def _short(value, limit: int) -> str:
    value = str(value or "").strip()
    return value if len(value) <= limit else value[: limit - 1] + "\u2026"


def _conflict_label(conf: dict) -> str:
    raw = str(conf.get("type", "Conflict"))
    for noise in ("detected", "inconsistency", "Conflict"):
        raw = raw.replace(noise, " ")
    raw = raw.strip() or "Conflict"
    return raw[:26]


def _build_case_graph(documents: Dict[str, dict],
                      doc_entities: Dict[str, List[dict]],
                      all_entities: List[dict],
                      conflicts: List[dict],
                      anomalies: Optional[List[dict]] = None,
                      rgcn: Optional[dict] = None,
                      max_docs: int = 10,
                      max_per_class: int = 12,
                      case: Optional[dict] = None,
                      person_ids_by_evidence: Optional[Dict[str, List[str]]] = None,
                      party_rows: Optional[List[dict]] = None) -> dict:
    """Layered, readable graph: documents on the left, entities grouped by class.

    Mention edges are thinned to the most informative entities per document so
    the diagram stays legible, and cross-document conflicts become explicit
    red links so a reviewer can see *which* exhibits disagree.
    """
    anomalies = anomalies or []

    flagged = set()
    for item in anomalies:
        for src in (item.get("sources") or []):
            flagged.add(str(src))

    # --- Documents (skip byte-identical exhibits) ---
    doc_nodes: List[dict] = []
    doc_ids: List[str] = []
    seen_fp = set()
    for doc_id, doc in documents.items():
        raw = (doc.get("rawText") or "").strip()
        fp = raw[:600]
        if fp and fp in seen_fp:
            continue
        seen_fp.add(fp)
        src = doc.get("source", doc_id)
        doc_ids.append(doc_id)
        doc_nodes.append({
            "id": f"doc-{doc_id}",
            "type": "documentNode",
            "data": {
                "label": _short(src, 32),
                "docType": "image" if re.search(r"\.(jpe?g|png|tiff?|bmp|gif)$", src, re.I) else "text",
                "entityCount": len(doc_entities.get(doc_id, [])),
                "excerpt": _short(raw.replace("\n", " "), 96),
                "flagged": src in flagged,
            },
        })
        if len(doc_nodes) >= max_docs:
            break

    # --- Aggregate entity mentions across documents ---
    mentions: Dict[Tuple[str, str], dict] = {}
    for doc_id in doc_ids:
        for ent in doc_entities.get(doc_id, []) or []:
            text = str(ent.get("text", "")).strip()
            etype = str(ent.get("type", "OTHER"))
            if not text:
                continue
            key = (text, etype)
            rec = mentions.setdefault(key, {"text": text, "type": etype,
                                            "docs": [], "count": 0})
            if doc_id not in rec["docs"]:
                rec["docs"].append(doc_id)
            rec["count"] += 1

    groups: Dict[str, List[dict]] = {c: [] for c in _CLASS_ORDER}
    for rec in mentions.values():
        groups[_ENTITY_CLASS.get(rec["type"], "fact")].append(rec)
    for cls in groups:
        groups[cls].sort(key=lambda r: (-len(r["docs"]), -r["count"], r["text"]))

    entity_nodes: List[dict] = []
    id_map: Dict[Tuple[str, str], str] = {}
    for cls in _CLASS_ORDER:
        for rec in groups[cls][:max_per_class]:
            nid = f"ent-{len(entity_nodes)}"
            id_map[(rec["text"], rec["type"])] = nid
            entity_nodes.append({
                "id": nid,
                "type": "entityNode",
                "data": {
                    "label": _short(rec["text"], 28),
                    "entityType": rec["type"],
                    "entityClass": cls,
                    "mentions": rec["count"],
                    "docCount": len(rec["docs"]),
                    "docIds": [f"doc-{d}" for d in rec["docs"]],
                },
            })

    # --- Layout: one column per semantic class, vertically centred ---
    def place(nodes: List[dict], x: int, gap: int) -> None:
        if not nodes:
            return
        total = (len(nodes) - 1) * gap
        for i, node in enumerate(nodes):
            node["position"] = {"x": x, "y": round(i * gap - total / 2)}

    place(doc_nodes, _CLASS_X["document"], 120)
    for cls in _CLASS_ORDER:
        place([n for n in entity_nodes if n["data"]["entityClass"] == cls],
              _CLASS_X[cls], 60)

    # --- Mention edges (thinned to the most informative entities) ---
    edges: List[dict] = []
    for doc_id in doc_ids:
        did = f"doc-{doc_id}"
        ents = doc_entities.get(doc_id, []) or []
        picked = sorted(
            ents,
            key=lambda e: (_CLASS_RANK.get(_ENTITY_CLASS.get(str(e.get("type", "")), "fact"), 3),
                           str(e.get("text", ""))),
        )[:12]
        used = set()
        for ent in picked:
            key = (str(ent.get("text", "")).strip(), str(ent.get("type", "OTHER")))
            tid = id_map.get(key)
            if not tid or tid in used:
                continue
            used.add(tid)
            linked = bool(ent.get("personId"))
            inferred = bool(ent.get("inferred"))
            edges.append({
                "id": f"m-{doc_id}-{tid}",
                "source": did,
                "target": tid,
                "type": "linkEdge" if linked else "mentionEdge",
                "data": {
                    "relation": "case-party" if inferred else "linked-evidence" if linked else "mentions",
                    "inferred": inferred,
                    "personId": ent.get("personId"),
                },
            })

    # --- Conflict edges between the exhibits that disagree ---
    src_to_node = {documents[d].get("source", d): f"doc-{d}" for d in doc_ids}
    seen_pairs = set()
    conflict_count = 0
    for idx, conf in enumerate(conflicts):
        left = conf.get("left") or {}
        right = conf.get("right") or {}
        na = src_to_node.get(left.get("source"))
        nb = src_to_node.get(right.get("source"))
        if not na or not nb or na == nb:
            continue
        pair = tuple(sorted((na, nb)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        conflict_count += 1
        edges.append({
            "id": f"c-{idx}-{na}-{nb}",
            "source": na,
            "target": nb,
            "type": "conflictEdge",
            "data": {
                "relation": "contradicts",
                "label": _conflict_label(conf),
                "severity": conf.get("severity", "low"),
                "model": conf.get("model", ""),
                "confidence": conf.get("confidence"),
                "leftClaim": _short((left.get("claim") or ""), 160),
                "rightClaim": _short((right.get("claim") or ""), 160),
                "explanation": _short(conf.get("explanation"), 220),
                "parties": conf.get("parties") or [],
            },
        })

    rgcn = rgcn or {}
    mention_count = len([e for e in edges if e["type"] in {"mentionEdge", "linkEdge"}])
    party_link_count = len([e for e in edges if e["type"] == "linkEdge"])
    base_meta = {
        "documents": len(doc_nodes),
        "entities": len(entity_nodes),
        "mentions": mention_count,
        "partyLinks": party_link_count,
        "conflicts": conflict_count,
        "parties": len(party_rows or []),
    }
    if rgcn:
        meta = {
            **base_meta,
            "model": "R-GCN",
            "integrity": "Suspicious" if rgcn.get("prediction") == 1 else "Intact",
            "confidence": round(float(rgcn.get("confidence", 0.0)), 3),
        }
    else:
        meta = {
            **base_meta,
            "model": "Entity graph",
            "integrity": "unscored",
            "confidence": 0.0,
        }
    legend = [
        {"kind": "document", **_CLASS_STYLE["document"]},
        {"kind": "party", **_CLASS_STYLE["party"]},
        {"kind": "authority", **_CLASS_STYLE["authority"]},
        {"kind": "fact", **_CLASS_STYLE["fact"]},
    ]
    return {"nodes": doc_nodes + entity_nodes, "edges": edges,
            "meta": meta, "legend": legend}


def _graph_for(case_id: str) -> dict:
    """Static fallback graph when no model or evidence is available."""
    nodes = [
        {"id": "doc-EV-0114", "type": "documentNode",
         "position": {"x": 0, "y": -120},
         "data": {"label": "EV-0114 CCTV footage", "docType": "image",
                  "entityCount": 3, "excerpt": "Camera 2 shows the corridor at 22:14.",
                  "flagged": False}},
        {"id": "doc-EV-0121", "type": "documentNode",
         "position": {"x": 0, "y": 90},
         "data": {"label": "EV-0121 Call records", "docType": "text",
                  "entityCount": 3, "excerpt": "Call log places the caller at 22:05.",
                  "flagged": False}},
        {"id": "ent-ev1", "type": "entityNode",
         "position": {"x": 470, "y": -80},
         "data": {"label": "T.R. Ajayan", "entityType": "PETITIONER",
                  "entityClass": "party", "mentions": 2, "docCount": 2,
                  "docIds": ["doc-EV-0114", "doc-EV-0121"]}},
        {"id": "ent-ev2", "type": "entityNode",
         "position": {"x": 940, "y": -80},
         "data": {"label": "High Court of Kerala", "entityType": "COURT",
                  "entityClass": "authority", "mentions": 1, "docCount": 1,
                  "docIds": ["doc-EV-0114"]}},
        {"id": "ent-ev3", "type": "entityNode",
         "position": {"x": 1410, "y": -80},
         "data": {"label": "22:14", "entityType": "DATE",
                  "entityClass": "fact", "mentions": 1, "docCount": 1,
                  "docIds": ["doc-EV-0114"]}},
        {"id": "ent-ev4", "type": "entityNode",
         "position": {"x": 1410, "y": 40},
         "data": {"label": "22:05", "entityType": "DATE",
                  "entityClass": "fact", "mentions": 1, "docCount": 1,
                  "docIds": ["doc-EV-0121"]}},
    ]
    edges = [
        {"id": "m1", "source": "doc-EV-0114", "target": "ent-ev1", "type": "mentionEdge", "data": {"relation": "mentions"}},
        {"id": "m2", "source": "doc-EV-0121", "target": "ent-ev1", "type": "mentionEdge", "data": {"relation": "mentions"}},
        {"id": "m3", "source": "doc-EV-0114", "target": "ent-ev2", "type": "mentionEdge", "data": {"relation": "mentions"}},
        {"id": "m4", "source": "doc-EV-0114", "target": "ent-ev3", "type": "mentionEdge", "data": {"relation": "mentions"}},
        {"id": "m5", "source": "doc-EV-0121", "target": "ent-ev4", "type": "mentionEdge", "data": {"relation": "mentions"}},
        {"id": "c1", "source": "doc-EV-0114", "target": "doc-EV-0121", "type": "conflictEdge",
         "data": {"relation": "contradicts", "label": "Timeline", "severity": "low", "model": "Entity consistency"}},
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {"documents": 2, "entities": 4, "mentions": 5, "conflicts": 1,
                 "model": "Entity graph", "integrity": "Intact", "confidence": 0.0},
        "legend": [
            {"kind": "document", **_CLASS_STYLE["document"]},
            {"kind": "party", **_CLASS_STYLE["party"]},
            {"kind": "authority", **_CLASS_STYLE["authority"]},
            {"kind": "fact", **_CLASS_STYLE["fact"]},
        ],
    }


def _invalidate_results(case_id: str) -> None:
    if case_id in ANALYSES:
        ANALYSES[case_id].pop("results", None)


_ABBREVIATIONS = r"(?:No|Nos|Adv|Sr|Smt|Shri|Mr|Mrs|Ms|Dr|Hon|Crl|MC|Cri|IPC|Art|Sec|S|P|W|v|Vs|Ltd|Co|Occu|Aged|Res|Govt|Dept|Distt|Ors|Anr|Petnr|Respt|S/o|D/o|W/o|R/o)"


def _split_sentences(text: str) -> List[str]:
    """Sentence splitter that does not break on legal abbreviations or initials."""
    protected = text or ""
    # Protect initials such as "T.R." or "S. Nair" before splitting.
    protected = re.sub(r"\b(?:[A-Z]\.)+",
                       lambda m: m.group(0).replace(".", "<DOT>"), protected)
    protected = re.sub(r"\b(" + _ABBREVIATIONS + r")\.", r"\1<DOT>", protected)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    out = []
    for part in parts:
        cleaned = part.replace("<DOT>", ".").strip()
        if len(cleaned) >= 40:
            out.append(cleaned)
    return out


def _claim_snippets(text: str, entities: List[dict], max_n: int = 3) -> List[str]:
    """Return substantive sentences that mention a factual legal entity.

    Headers such as ``Criwp1904-2021.Doc`` are full of entities but carry no
    claim, so candidates are scored on length, entity richness and the
    presence of ordinary lower-case prose. This keeps the excerpts shown for a
    contradiction readable and useful for a reviewer.
    """
    normalized = re.sub(r"\s+", " ", text or "").strip()
    sentences = _split_sentences(normalized)
    factual = {"DATE", "PLACE", "EVIDENCE", "PERSON", "PETITIONER", "RESPONDENT",
               "WITNESS", "DEFENDANT", "CASE_NUMBER", "COURT", "STATUTE", "PROVISION"}
    header_re = re.compile(r"^\s*\S+\.(?:docx?|pdf|txt|jpe?g|png)\b", re.I)
    bad_re = re.compile(
        r"^(?:Criwp|Crl|WP|W\.P|\$~|\*|In The|Before|Coram|Judgment|Order|Memo|"
        r"List|Hon|Present|Cause Title)|Coram:|having (?:his|her) address|"
        r"Aged \d+|Occu\.|\.\.\.Petitioner|\.\.\.Respondent",
        re.I,
    )

    def clean(sentence: str) -> str:
        return re.sub(r"\s+", " ", sentence).strip()

    scored: List[Tuple[float, str]] = []
    seen = set()
    for ent in entities or []:
        if str(ent.get("type", "")) not in factual:
            continue
        needle = str(ent.get("text", "")).strip()
        if len(needle) < 3:
            continue
        for sentence in sentences:
            if needle not in sentence:
                continue
            candidate = clean(sentence)
            if candidate in seen:
                continue
            words = candidate.split()
            lower_words = [w for w in words if w[:1].islower()]
            if len(words) < 8 or len(lower_words) < 4:
                continue
            if header_re.match(candidate) and len(words) < 14:
                continue
            if bad_re.search(candidate):
                continue
            # Prefer sentences that carry several facts, then longer prose.
            entity_hits = sum(1 for e in entities or []
                              if str(e.get("text", "")).strip() in candidate)
            score = entity_hits * 2.0 + min(len(words), 45) / 15.0
            if 60 <= len(candidate) <= 420:
                score += 1.0
            scored.append((score, candidate))
            seen.add(candidate)
            break

    scored.sort(key=lambda item: -item[0])
    out = [sentence for _, sentence in scored[:max_n]]
    if not out:
        out = [clean(s) for s in sentences
               if len(clean(s).split()) >= 10
               and not bad_re.search(clean(s))
               and not header_re.match(clean(s))][:max_n]
    return out


_SUBJECT_TYPES = {"PERSON", "PETITIONER", "RESPONDENT", "DEFENDANT", "WITNESS"}

_MONTH_NUMBERS = {
    name.lower(): i for i, name in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], 1)
}


def _date_key(value: str) -> str:
    """Return a comparable YYYY-MM-DD key for common legal date formats.

    ``27/06/1967``, ``27.06.1967`` and ``27th June, 1967`` all collapse to
    ``1967-06-27`` so a formatting difference is not reported as a conflict.
    Unrecognised values are compared verbatim.
    """
    text = value.strip()
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", text)
    if match:
        day, month, year = (int(g) for g in match.groups())
        if year < 100:
            year += 1900 if year >= 50 else 2000
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    match = re.fullmatch(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", text)
    if match:
        year, month, day = (int(g) for g in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    match = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})",
                         text, re.I)
    if match and match.group(2).lower() in _MONTH_NUMBERS:
        day = int(match.group(1))
        month = _MONTH_NUMBERS[match.group(2).lower()]
        year = int(match.group(3))
        if 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return text


def _date_groups(entities: List[dict]) -> Dict[str, set]:
    """Split DATE mentions into comparable shapes (time / year / full date)."""
    groups: Dict[str, set] = {}
    for ent in entities or []:
        if ent.get("type") != "DATE":
            continue
        value = str(ent.get("text", "")).strip()
        if not value:
            continue
        if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", value):
            shape = "time"
        elif re.fullmatch(r"(?:1[5-9]|20)\d{2}", value):
            shape = "year"
        else:
            shape = "date"
        groups.setdefault(shape, set()).add(value)
        if shape == "date":
            # A full date also carries a year; index it so it can be compared
            # with documents that only mention years.
            year = re.search(r"\b(1[89]\d{2}|20\d{2})\b", value)
            if year:
                groups.setdefault("year", set()).add(year.group(1))
    return groups


def _entity_inconsistencies(documents: Dict[str, dict],
                            doc_entities: Dict[str, List[dict]],
                            max_conflicts: int = 3) -> List[dict]:
    """Surface facts that cannot both be true across two exhibits.

    LegalBERT judges entailment between sentences. This complementary check
    compares structured entities, so conflicting times or places recorded for
    the same party are surfaced even when the wording is not a direct negation.
    It is deliberately conservative: both documents must mention a shared
    subject *and* hold differing values of the same fact shape.
    """
    unique: List[Tuple[str, dict]] = []
    seen_texts = set()
    for doc_id, doc in documents.items():
        raw = (doc.get("rawText") or "").strip()
        if not raw or raw[:600] in seen_texts:
            continue
        seen_texts.add(raw[:600])
        unique.append((doc_id, doc))

    out: List[dict] = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            if len(out) >= max_conflicts:
                return out
            id_a, doc_a = unique[i]
            id_b, doc_b = unique[j]
            ents_a = doc_entities.get(id_a, []) or []
            ents_b = doc_entities.get(id_b, []) or []

            subjects_a = {str(e.get("text", "")).lower(): str(e.get("text", ""))
                          for e in ents_a if e.get("type") in _SUBJECT_TYPES}
            subjects_b = {str(e.get("text", "")).lower(): str(e.get("text", ""))
                          for e in ents_b if e.get("type") in _SUBJECT_TYPES}
            shared = sorted(set(subjects_a) & set(subjects_b))
            subject = subjects_a[shared[0]] if shared else "Case record"

            groups_a: Dict[str, set] = {}
            groups_b: Dict[str, set] = {}
            for shape, values in _date_groups(ents_a).items():
                groups_a[f"time:{shape}"] = values
            for shape, values in _date_groups(ents_b).items():
                groups_b[f"time:{shape}"] = values
            places_a = {str(e.get("text", "")).strip()
                        for e in ents_a if e.get("type") == "PLACE"
                        and str(e.get("text", "")).strip()}
            places_b = {str(e.get("text", "")).strip()
                        for e in ents_b if e.get("type") == "PLACE"
                        and str(e.get("text", "")).strip()}
            if places_a:
                groups_a["place"] = places_a
            if places_b:
                groups_b["place"] = places_b

            comparable = set(groups_a) & set(groups_b)
            if not shared:
                # Without a shared party, overlapping date sets are still
                # strong evidence that both exhibits describe the same
                # timeline; a differing year/date is then an inconsistency.
                dates_a = {_date_key(v) for k, vals in groups_a.items()
                           if k.startswith("time:") for v in vals}
                dates_b = {_date_key(v) for k, vals in groups_b.items()
                           if k.startswith("time:") for v in vals}
                overlap = len(dates_a & dates_b)
                same_timeline = (overlap >= 2 and dates_a and dates_b
                                 and overlap / max(1, min(len(dates_a), len(dates_b))) >= 0.5)
                if not same_timeline:
                    comparable = {g for g in comparable if g == "time:time"}

            flagged_groups = set()
            for group in sorted(comparable):
                if group == "time:year" and "time:date" in flagged_groups:
                    # The year is already implied by the full-date conflict.
                    continue
                values_a, values_b = groups_a[group], groups_b[group]
                if group.startswith("time:"):
                    # Compare by calendar value so date-format variants
                    # (27/06/1967 vs 27.06.1967) are not false conflicts.
                    map_a, map_b = {}, {}
                    for value in values_a:
                        map_a.setdefault(_date_key(value), value)
                    for value in values_b:
                        map_b.setdefault(_date_key(value), value)
                    only_a = [map_a[k] for k in sorted(set(map_a) - set(map_b))]
                    only_b = [map_b[k] for k in sorted(set(map_b) - set(map_a))]
                else:
                    only_a = sorted(values_a - values_b)
                    only_b = sorted(values_b - values_a)
                if not only_a or not only_b:
                    continue
                flagged_groups.add(group)
                kind = group.split(":", 1)[0]
                conflict_type = "Timeline inconsistency" if kind == "time" else "Location inconsistency"
                label = group.split(":", 1)[1] if ":" in group else "place"
                subject_note = "" if subject == "Case record" else f" for {subject}"
                out.append({
                    "id": f"r{len(out) + 1}",
                    "type": conflict_type,
                    "severity": "low",
                    "model": "Entity consistency",
                    "confidence": 0.62,
                    "left": {"source": doc_a.get("source", id_a), "evidenceId": id_a,
                             "claim": f"{subject} — {label}: {', '.join(only_a[:3])}"},
                    "right": {"source": doc_b.get("source", id_b), "evidenceId": id_b,
                              "claim": f"{subject} — {label}: {', '.join(only_b[:3])}"},
                    "explanation": (
                        f"The exhibits record different {label} values"
                        f"{subject_note}. "
                        "This is a structured factual inconsistency: check the originals to "
                        "decide which record is authoritative."
                    ),
                })
                if len(out) >= max_conflicts:
                    return out
    return out


_REVIEW_STATUSES = {"pending", "confirmed", "dismissed"}


def _apply_reviews(case: dict, conflicts: List[dict], anomalies: List[dict],
                   report: Optional[dict] = None) -> dict:
    """Merge persisted human-review decisions into live findings."""
    reviews = case.setdefault("reviews", {})
    for item in list(conflicts) + list(anomalies):
        record = reviews.get(str(item.get("id"))) or {}
        item["review"] = {
            "status": record.get("status", "pending"),
            "reviewer": record.get("reviewer"),
            "reviewedAt": record.get("reviewedAt"),
            "note": record.get("note", ""),
        }
    summary = {"confirmed": 0, "dismissed": 0, "pending": 0}
    for item in list(conflicts) + list(anomalies):
        status = (item.get("review") or {}).get("status", "pending")
        summary[status] = summary.get(status, 0) + 1
    summary["total"] = len(conflicts) + len(anomalies)
    if report is not None:
        report["reviewSummary"] = summary
        report["reviews"] = {
            key: value for key, value in reviews.items()
            if any(str(item.get("id")) == key for item in list(conflicts) + list(anomalies))
        }
    return summary


def _active_model_summary() -> Dict[str, dict]:
    """Small, report-friendly provenance block for active models."""
    metadata = model_manager._load_metadata()
    out: Dict[str, dict] = {}
    for model_type, versions in metadata.items():
        for entry in versions or []:
            if not entry.get("active"):
                continue
            out[model_type] = {
                "dataset": entry.get("dataset"),
                "accuracy": entry.get("accuracy"),
                "entity_f1": entry.get("entity_f1"),
                "f1": entry.get("f1"),
                "roc_auc": entry.get("roc_auc"),
                "samples": entry.get("samples"),
                "evaluation": entry.get("evaluation"),
                "cv": entry.get("cv"),
                "version": entry.get("version") or entry.get("createdAt"),
            }
    return out


def _ensure_results(case_id: str) -> dict:
    """Build analysis results using the active models + explainable rules."""
    cached = ANALYSES.get(case_id, {}).get("results")
    if cached:
        return cached

    case = CASES.get(case_id, {"id": case_id})
    evidence = [EVIDENCE[e] for e in case.get("evidence", []) if e in EVIDENCE]
    people = case.get("people") or []
    documents: Dict[str, dict] = {}
    for ev in evidence:
        doc = _document_for(ev)
        doc["source"] = ev.get("name") or ev["id"]
        documents[ev["id"]] = doc
    is_demo = any(ev.get("demo") for ev in evidence)

    # ---- 1. NER: BiLSTM-CRF over every document -------------------------
    bilstm_active = model_manager.get_bilstm_crf()[0] is not None
    doc_entities: Dict[str, List[dict]] = {}
    for doc_id, doc in documents.items():
        entities = list(doc.get("entities", []))
        if doc.get("rawText"):
            from services.features import extract_entities_regex, merge_entities
            supplements = extract_entities_regex(doc["rawText"])
            if bilstm_active:
                ner_entities = run_bilstm_crf_ner(doc["rawText"])
                if ner_entities:
                    supplements = merge_entities(ner_entities, supplements)
            # Gold annotations stay first, then model output, then regex.
            entities = merge_entities(entities, supplements)
        doc_entities[doc_id] = entities
        doc["entities"] = entities

    # ---- 1b. Party grounding: attach case people to exhibits -------------
    text_by_id = {eid: (doc.get("rawText") or "") for eid, doc in documents.items()}
    person_ids_by_evidence = evidence_person_ids(evidence, people, text_by_id)
    person_ids_by_evidence = inject_case_party_entities(
        case, evidence, documents, doc_entities, person_ids_by_evidence
    )
    all_entities: List[dict] = []
    seen = set()
    for doc_id, entities in doc_entities.items():
        documents[doc_id]["entities"] = entities
        for ent in entities:
            key = (ent.get("text", ""), ent.get("type", ""))
            if key not in seen:
                seen.add(key)
                all_entities.append(ent)

    # ---- 2. Contradictions: LegalBERT over cross-document claims --------
    conflicts: List[dict] = []
    legalbert_active = model_manager.get_legalbert() is not None
    legalbert_called = False
    if evidence and legalbert_active:
        # Skip byte-identical documents (e.g. the same exhibit attached twice)
        # so the model never "compares" a text with itself.
        seen_texts = set()
        doc_ids: List[str] = []
        for doc_id in documents:
            raw = (documents[doc_id].get("rawText") or "").strip()
            if not raw:
                continue
            fingerprint = raw[:600]
            if fingerprint in seen_texts:
                continue
            seen_texts.add(fingerprint)
            doc_ids.append(doc_id)
        calls = 0
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                if calls >= 10:
                    break
                id_a, id_b = doc_ids[i], doc_ids[j]
                snippets_a = _claim_snippets(documents[id_a].get("rawText", ""),
                                             doc_entities.get(id_a, []), max_n=2)
                snippets_b = _claim_snippets(documents[id_b].get("rawText", ""),
                                             doc_entities.get(id_b, []), max_n=2)
                for claim_a in snippets_a:
                    for claim_b in snippets_b:
                        if calls >= 10:
                            break
                        calls += 1
                        result = run_legalbert_contradiction(claim_a, claim_b)
                        if not result:
                            continue
                        legalbert_called = True
                        is_contra, conf, explanation = result
                        if not is_contra or conf < 0.6:
                            continue
                        conflicts.append({
                            "id": f"c{len(conflicts) + 1}",
                            "type": "Contradiction detected",
                            "category": "Contradiction",
                            "subtype": "semantic_contradiction",
                            "severity": "material" if conf >= 0.75 else "low",
                            "model": "LegalBERT",
                            "confidence": round(conf, 3),
                            "left": {"source": documents[id_a].get("source", id_a),
                                     "evidenceId": id_a, "claim": claim_a[:240]},
                            "right": {"source": documents[id_b].get("source", id_b),
                                      "evidenceId": id_b, "claim": claim_b[:240]},
                            "explanation": explanation,
                        })
        conflicts.extend(_entity_inconsistencies(documents, doc_entities))

    # Canned demo findings are only a last resort for the seeded demo case.
    if is_demo and not legalbert_called and evidence and not conflicts:
        conflicts = _conflicts_for(case_id)
    for conflict in conflicts:
        conflict.setdefault("category", "Contradiction")
        conflict.setdefault("subtype", "structured_inconsistency")

    # ---- 3. Anomalies: XGBoost + transparent rules ----------------------
    model_anomalies: List[dict] = []
    xgb_active = model_manager.get_xgboost() is not None
    xgb_called = False
    if evidence and xgb_active:
        for doc_id, doc in documents.items():
            raw = doc.get("rawText") or ""
            if not raw.strip():
                continue
            result = run_xgboost_anomaly(raw, doc_entities.get(doc_id, []))
            if not result:
                continue
            xgb_called = True
            pred, proba = result
            if pred != 1:
                continue
            model_anomalies.append({
                "id": f"a{len(model_anomalies) + 1}",
                "entity": doc.get("source", doc_id),
                "type": "manipulation",
                "category": "Possible manipulation",
                "subtype": "document_integrity",
                "score": round(proba, 3),
                "label": "High" if proba >= 0.8 else "Medium" if proba >= 0.55 else "Low",
                "model": "XGBoost",
                "explanation": (
                    f"XGBoost flagged this document as damaged or tampered "
                    f"(integrity risk {proba:.2f}) based on length, entity coverage, "
                    f"repetition and legal-metadata signals."
                ),
                "sources": [doc.get("source", doc_id)],
                "evidenceIds": [doc_id],
            })

    # Rule-based manipulation checks enrich XGBoost findings and catch cases
    # the statistical model is unsure about.
    rule_anomalies = manipulation_anomalies(evidence, documents, model_anomalies)
    anomalies: List[dict] = list(model_anomalies) + list(rule_anomalies)

    # Missing-information rules always run: this is the third anomaly family
    # promised by the research report.
    missing_anomalies = missing_information(
        case, evidence, documents, doc_entities, person_ids_by_evidence
    )
    anomalies.extend(missing_anomalies)

    # Preserve the demo narrative if no statistical model is deployed.
    if is_demo and not xgb_called:
        existing_keys = {(a.get("entity"), a.get("type")) for a in anomalies}
        for item in _anomalies_for(case_id):
            if (item.get("entity"), item.get("type")) not in existing_keys:
                item.setdefault("category", "Missing information" if item.get("type") == "missing_information"
                                else "Possible manipulation" if item.get("type") == "manipulation"
                                else "Contradiction")
                anomalies.append(item)

    # ---- 4. R-GCN structural integrity ----------------------------------
    graph: dict = {"nodes": [], "edges": [], "meta": {}, "legend": []}
    rgcn_active = model_manager.get_rgcn() is not None
    rgcn_result = None
    if evidence and all_entities and rgcn_active:
        total_len = sum(len(d.get("rawText") or "") for d in documents.values())
        rgcn_result = run_rgcn_graph(all_entities, total_len)
        if rgcn_result is not None and rgcn_result.get("prediction") == 1:
            anomalies.append({
                "id": f"a{len(anomalies) + 1}",
                "entity": "Case entity graph",
                "type": "manipulation",
                "category": "Possible manipulation",
                "subtype": "graph_integrity",
                "score": round(float(rgcn_result.get("confidence", 0.0)), 3),
                "label": "High" if rgcn_result.get("confidence", 0) >= 0.8 else "Medium",
                "model": "R-GCN",
                "explanation": (
                    "R-GCN classified the case entity graph as structurally "
                    "incomplete or tampered (missing parties, judge, court or dates)."
                ),
                "sources": [d.get("source", k) for k, d in list(documents.items())[:5]],
                "evidenceIds": list(documents.keys())[:5],
            })

    # ---- 5. Party attribution, summary and graph ------------------------
    person_ids_by_evidence = annotate_findings(
        case, evidence, documents, conflicts, anomalies, person_ids_by_evidence
    )
    party_rows = party_summary(case, evidence, person_ids_by_evidence,
                               conflicts, anomalies, documents)
    if evidence:
        graph = _build_case_graph(
            documents, doc_entities, all_entities, conflicts, anomalies,
            rgcn_result, case=case, person_ids_by_evidence=person_ids_by_evidence,
            party_rows=party_rows,
        )
        if not graph["nodes"]:
            graph = _graph_for(case_id)

    # ---- 6. Explainable report (Module 4) --------------------------------
    review_summary = _apply_reviews(case, conflicts, anomalies)
    from services.report_service import build_explainable_report
    report = build_explainable_report(
        case=case,
        evidence=evidence,
        documents=documents,
        doc_entities=doc_entities,
        all_entities=all_entities,
        conflicts=conflicts,
        anomalies=anomalies,
        party_rows=party_rows,
        graph=graph,
        active_models=_active_model_summary(),
        review_summary=review_summary,
        reviews=case.get("reviews") or {},
        pipeline_stages=PIPELINE_STAGES,
        rgcn_result=rgcn_result,
    )

    results = {"conflicts": conflicts, "anomalies": anomalies, "graph": graph,
               "documents": documents, "entities": all_entities,
               "parties": party_rows, "report": report}
    ANALYSES.setdefault(case_id, {})["results"] = results
    if case_id in CASES:
        CASES[case_id]["conflicts"] = len(conflicts)
        CASES[case_id]["anomalies"] = len(anomalies)
        _save_data()
    return results


# ---------------------------------------------------------------------------
# Seed demo data so the app is usable immediately
# ---------------------------------------------------------------------------
_DEMO_SEED = [
    ("Crl MC 1622/2006", "High Court of Kerala", "302, 34, 120B", "active"),
    ("Crl A 0451/2011", "High Court of Delhi", "420, 468", "active"),
    ("Crl P 0902/2014", "High Court of Madras", "379, 411", "archived"),
]


def _seed_demo() -> None:
    if CASES:
        return
    owner = _first_admin_email()
    for number, court, ipc, status in _DEMO_SEED:
        cid = _slug(number)
        CASES[cid] = {
            "id": cid,
            "caseNumber": number,
            "court": court,
            "ipcSections": ipc,
            "defendants": [],
            "people": [],
            "evidence": [],
            "status": status,
            "conflicts": 0,
            "createdAt": _now(),
            "owner": owner,
        }
    cid = _slug("Crl MC 1622/2006")
    CASES[cid]["defendants"] = [
        {"name": "T.R. Ajayan", "role": "Defendant"},
        {"name": "K. Balan", "role": "Defendant"},
        {"name": "S. Nair", "role": "Witness"},
    ]
    CASES[cid]["people"] = [
        {"id": "p1", "name": "T.R. Ajayan", "role": "Defendant"},
        {"id": "p2", "name": "K. Balan", "role": "Defendant"},
        {"id": "p3", "name": "S. Nair", "role": "Witness"},
    ]
    demo_evidence = [
        ("EV-0114", "CCTV footage.jpg", "image", "image/jpeg", "p2"),
        ("EV-0119", "Fuel receipt.jpg", "image", "image/jpeg", "p2"),
        ("EV-0121", "Call records.pdf", "pdf", "application/pdf", "p1"),
        ("EV-0122", "Witness statement.mp3", "audio", "audio/mpeg", "p3"),
    ]
    for eid, name, kind, typ, link in demo_evidence:
        EVIDENCE[eid] = {"id": eid, "name": name, "kind": kind, "type": typ,
                         "linkedTo": link, "caseId": cid, "uploadedAt": _now(),
                         "demo": True}
        CASES[cid]["evidence"].append(eid)
    _ensure_results(cid)
    _save_data()


def _mark_demo_evidence() -> None:
    """Mark the seeded demonstration exhibits so only they use canned text."""
    demo_case_id = _slug("Crl MC 1622/2006")
    demo_case = CASES.get(demo_case_id)
    if not demo_case:
        return
    changed = False
    for eid in demo_case.get("evidence", []):
        ev = EVIDENCE.get(eid)
        if ev and not (ev.get("rawText") or "").strip() and not ev.get("demo"):
            ev["demo"] = True
            changed = True
    if changed:
        _save_data()


_load_data()
_seed_demo()
_mark_demo_evidence()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class LoginIn(BaseModel):
    email: str
    password: str


class RegisterIn(BaseModel):
    name: str
    email: str
    password: str


class VerifyIn(BaseModel):
    email: str
    token: str


class EmailIn(BaseModel):
    email: str


class ResetIn(BaseModel):
    token: str
    password: str


class DefendantIn(BaseModel):
    name: str
    role: str


class CaseIn(BaseModel):
    caseId: str
    ipcSections: str
    court: Optional[str] = None
    defendants: List[DefendantIn]


class LinkIn(BaseModel):
    evidenceId: str
    personId: str


class EvidenceTextIn(BaseModel):
    name: str
    text: str
    kind: str = "text"
    entities: Optional[List[dict]] = None


class ReviewIn(BaseModel):
    status: str
    note: Optional[str] = ""


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/api/v1/auth/register")
async def register(body: RegisterIn):
    email = body.email.strip().lower()
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Full name is required")
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Enter a valid email address")
    if email in USERS:
        raise HTTPException(409, "An account with this email already exists. Please sign in.")
    errors = _password_errors(body.password)
    if errors:
        raise HTTPException(400, "Password must contain " + ", ".join(errors) + ".")

    digest, salt = _hash_password(body.password)
    verify_token = uuid.uuid4().hex[:8].upper()
    is_admin = _first_user_is_admin()
    USERS[email] = {"name": name, "email": email, "password_hash": digest, "salt": salt,
                    "verified": False, "verify_token": verify_token, "reset_token": None,
                    "is_admin": is_admin, "createdAt": _now()}
    _save_users()
    if is_admin:
        # First administrator on a fresh install inherits the seeded demo cases.
        _claim_unowned_cases(email)
    return {"user": {"name": name, "email": email, "is_admin": is_admin}, "verification_required": True,
            "dev_verify_token": verify_token, "message": "Account created. Please verify your email."}


@app.post("/api/v1/auth/login")
async def login(body: LoginIn):
    email = body.email.strip().lower()
    if not email or not body.password:
        raise HTTPException(400, "Email and password are required")
    user = USERS.get(email)
    if not user:
        raise HTTPException(401, "No account found for this email. Please register first.")
    if not _verify_password(body.password, user["password_hash"], user["salt"]):
        raise HTTPException(401, "Incorrect password. Please try again.")
    if not user.get("verified", False):
        raise HTTPException(403, "Please verify your email before signing in.")
    token = f"mock.jwt.{uuid.uuid4().hex}"
    ACTIVE_SESSIONS[token] = email
    return {
        "access_token": token,
        "user": {
            "name": user["name"],
            "email": email,
            "is_admin": user.get("is_admin", False),
        },
    }


@app.post("/api/v1/auth/verify-email")
async def verify_email(body: VerifyIn):
    email = body.email.strip().lower()
    user = USERS.get(email)
    if not user:
        raise HTTPException(404, "No account found for this email.")
    if user.get("verified"):
        raise HTTPException(400, "This email is already verified. Please sign in.")
    if body.token.strip().upper() != (user.get("verify_token") or ""):
        raise HTTPException(400, "Invalid verification code.")
    user["verified"] = True
    user["verify_token"] = None
    _save_users()
    token = f"mock.jwt.{uuid.uuid4().hex}"
    ACTIVE_SESSIONS[token] = email
    return {
        "access_token": token,
        "user": {
            "name": user["name"],
            "email": email,
            "is_admin": user.get("is_admin", False),
        },
    }


@app.post("/api/v1/auth/resend-verification")
async def resend_verification(body: EmailIn):
    email = body.email.strip().lower()
    user = USERS.get(email)
    if not user:
        raise HTTPException(404, "No account found for this email.")
    if user.get("verified"):
        raise HTTPException(400, "This email is already verified.")
    token = uuid.uuid4().hex[:8].upper()
    user["verify_token"] = token
    _save_users()
    return {"dev_verify_token": token, "message": "A new verification code was sent."}


@app.post("/api/v1/auth/forgot-password")
async def forgot_password(body: EmailIn):
    email = body.email.strip().lower()
    user = USERS.get(email)
    if not user:
        return {"message": "If that email is registered, a reset code has been sent."}
    token = uuid.uuid4().hex[:8].upper()
    user["reset_token"] = token
    _save_users()
    return {"message": "Reset code sent.", "dev_reset_token": token}


@app.post("/api/v1/auth/reset-password")
async def reset_password(body: ResetIn):
    token = body.token.strip().upper()
    target = next((u for u in USERS.values() if token and u.get("reset_token") == token), None)
    if not target:
        raise HTTPException(400, "Invalid or expired reset code.")
    errors = _password_errors(body.password)
    if errors:
        raise HTTPException(400, "Password must contain " + ", ".join(errors) + ".")
    digest, salt = _hash_password(body.password)
    target["password_hash"] = digest
    target["salt"] = salt
    target["reset_token"] = None
    _save_users()
    return {"message": "Password updated. You can now sign in."}


# ---------------------------------------------------------------------------
# Admin management (Admin only)
# ---------------------------------------------------------------------------
class AdminUserIn(BaseModel):
    name: str
    email: str
    password: str
    is_admin: bool = False


class AdminPasswordIn(BaseModel):
    password: str


@app.get("/api/v1/admin/me")
async def admin_me(user: dict = Depends(require_admin)):
    return {
        "name": user["name"],
        "email": user["email"],
        "is_admin": user.get("is_admin", False),
    }


@app.get("/api/v1/admin/users")
async def admin_list_users(user: dict = Depends(require_admin)):
    out = []
    for email, u in USERS.items():
        out.append({
            "email": email,
            "name": u["name"],
            "is_admin": u.get("is_admin", False),
            "verified": u.get("verified", False),
            "createdAt": u.get("createdAt"),
        })
    return out


@app.post("/api/v1/admin/users/{email}/password")
async def admin_reset_password(email: str, body: AdminPasswordIn,
                               user: dict = Depends(require_admin)):
    """Set a new password for any account and clear its active sessions."""
    email = email.strip().lower()
    target = USERS.get(email)
    if not target:
        raise HTTPException(404, "User not found")
    errors = _password_errors(body.password)
    if errors:
        raise HTTPException(400, "Password must contain " + ", ".join(errors) + ".")
    digest, salt = _hash_password(body.password)
    target["password_hash"] = digest
    target["salt"] = salt
    target["reset_token"] = None
    _save_users()
    for token, owner in list(ACTIVE_SESSIONS.items()):
        if owner == email:
            ACTIVE_SESSIONS.pop(token, None)
    return {"message": f"Password updated for {email}.", "email": email}


@app.post("/api/v1/admin/users")
async def admin_create_user(body: AdminUserIn, user: dict = Depends(require_admin)):
    email = body.email.strip().lower()
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Full name is required")
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Enter a valid email address")
    if email in USERS:
        raise HTTPException(409, "An account with this email already exists.")
    errors = _password_errors(body.password)
    if errors:
        raise HTTPException(400, "Password must contain " + ", ".join(errors) + ".")

    digest, salt = _hash_password(body.password)
    USERS[email] = {
        "name": name,
        "email": email,
        "password_hash": digest,
        "salt": salt,
        "verified": True,  # admin-created accounts are pre-verified
        "verify_token": None,
        "reset_token": None,
        "is_admin": body.is_admin,
        "createdAt": _now(),
    }
    _save_users()
    return {
        "email": email,
        "name": name,
        "is_admin": body.is_admin,
        "message": "User created.",
    }


@app.put("/api/v1/admin/users/{email}/role")
async def admin_toggle_role(email: str, body: dict, user: dict = Depends(require_admin)):
    email = email.strip().lower()
    target = USERS.get(email)
    if not target:
        raise HTTPException(404, "User not found")
    if email == user["email"]:
        raise HTTPException(400, "Cannot change your own admin role.")
    target["is_admin"] = bool(body.get("is_admin", False))
    _save_users()
    return {"email": email, "is_admin": target["is_admin"]}


@app.delete("/api/v1/admin/users/{email}")
async def admin_delete_user(email: str, user: dict = Depends(require_admin)):
    email = email.strip().lower()
    if email == user["email"]:
        raise HTTPException(400, "Cannot delete your own account.")
    if email not in USERS:
        raise HTTPException(404, "User not found")
    # Prevent deleting the last admin
    if USERS[email].get("is_admin") and sum(1 for u in USERS.values() if u.get("is_admin")) <= 1:
        raise HTTPException(400, "Cannot delete the last admin account.")
    removed_cases = [cid for cid, case in CASES.items() if case.get("owner") == email]
    for cid in removed_cases:
        CASES.pop(cid, None)
        ANALYSES.pop(cid, None)
    removed_evidence = [eid for eid, ev in EVIDENCE.items() if ev.get("caseId") in removed_cases]
    for eid in removed_evidence:
        ev = EVIDENCE.pop(eid, None)
        if ev and ev.get("filePath"):
            try:
                os.remove(ev["filePath"])
            except OSError:
                pass
    for token, owner in list(ACTIVE_SESSIONS.items()):
        if owner == email:
            ACTIVE_SESSIONS.pop(token, None)
    del USERS[email]
    _save_users()
    _save_data()
    return {"deleted": email, "cases_deleted": len(removed_cases)}


# ---------------------------------------------------------------------------
# Admin — Dataset Management
# ---------------------------------------------------------------------------
@app.post("/api/v1/admin/dataset/upload")
async def admin_upload_dataset(file: UploadFile = File(...), user: dict = Depends(require_admin)):
    """Upload a zip dataset. Extracts and stores it for training."""
    ensure_dirs_func = lambda: None  # placeholder
    from services.dataset_service import ensure_dirs
    ensure_dirs()
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only .zip files are supported")
    zip_path = os.path.join(DATASET_DIR, file.filename)
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    extract_dir = extract_zip(zip_path)
    return {
        "filename": file.filename,
        "dataset_name": Path(extract_dir).name,
        "path": extract_dir,
        "message": "Dataset uploaded and extracted.",
    }


@app.get("/api/v1/admin/datasets")
async def admin_list_datasets(user: dict = Depends(require_admin)):
    return list_datasets()


@app.get("/api/v1/admin/datasets/catalog")
async def admin_dataset_catalog(user: dict = Depends(require_admin)):
    """List public datasets that can be downloaded for training."""
    from services.external_datasets import BUILDERS
    existing = {d["name"]: d for d in list_datasets()}
    catalog = []
    for name in sorted(BUILDERS):
        info = existing.get(name, {})
        catalog.append({
            "name": name,
            "built": bool(info),
            "files": info.get("files", 0),
            "json_files": info.get("json_files", 0),
        })
    return catalog


@app.post("/api/v1/admin/dataset/fetch/{dataset_name}")
async def admin_fetch_dataset(dataset_name: str, user: dict = Depends(require_admin)):
    """Download + materialise a public legal dataset from Hugging Face."""
    from starlette.concurrency import run_in_threadpool
    from services.external_datasets import BUILDERS, build_dataset
    if dataset_name not in BUILDERS:
        raise HTTPException(400, f"Unknown dataset. Use one of: {sorted(BUILDERS)}")
    try:
        meta = await run_in_threadpool(build_dataset, dataset_name)
    except Exception as exc:
        raise HTTPException(500, f"Download failed: {exc}")
    return {"message": f"{dataset_name} ready", "dataset": meta}


@app.get("/api/v1/admin/dataset/download/{dataset_name}")
async def admin_download_dataset(dataset_name: str, user: dict = Depends(require_admin)):
    """Download a dataset ZIP file. Supports 'sample' for the built-in MECR sample dataset."""
    from services.dataset_service import DATASET_DIR
    import zipfile
    import io

    if dataset_name == "sample":
        zip_path = os.path.join(DATASET_DIR, "sample_mecr_dataset.zip")
        if not os.path.exists(zip_path):
            raise HTTPException(404, "Sample dataset not found")
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="sample_mecr_dataset.zip",
        )

    # For uploaded datasets, zip them on the fly
    folder_path = os.path.join(DATASET_DIR, dataset_name)
    if not os.path.exists(folder_path):
        raise HTTPException(404, f"Dataset '{dataset_name}' not found")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zf.write(file_path, arcname)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{dataset_name}.zip"'},
    )


# ---------------------------------------------------------------------------
# Admin — Model Training (SSE progress)
# ---------------------------------------------------------------------------
TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}


class TrainRequest(BaseModel):
    dataset_name: str
    epochs: Optional[int] = None


def _sse_progress(job_id: str, message: str, progress: int, status: str = "running"):
    TRAINING_JOBS[job_id] = {"message": message, "progress": progress, "status": status}
    yield f"data: {json.dumps({'message': message, 'progress': progress, 'status': status})}\n\n"


@app.post("/api/v1/admin/train/{model_type}")
async def admin_train_model(model_type: str, body: TrainRequest, user: dict = Depends(require_admin)):
    """Start training a model. Returns job_id for status polling."""
    import uuid as _uuid

    valid_types = ["bilstm_crf", "xgboost", "legalbert", "rgcn"]
    if model_type not in valid_types:
        raise HTTPException(400, f"Invalid model type. Use one of: {valid_types}")

    available = [d["name"] for d in list_datasets()]
    if body.dataset_name not in available:
        raise HTTPException(404, f"Dataset '{body.dataset_name}' not found. Upload it first.")

    # Pick a compatible source automatically:
    #   LegalBERT needs premise/hypothesis pairs, the other three need NER text.
    dataset_name = body.dataset_name
    if model_type == "legalbert":
        from services.dataset_service import DATASET_DIR, load_jsonl_records
        records = load_jsonl_records(os.path.join(DATASET_DIR, dataset_name))
        if not any("premise" in r and "hypothesis" in r for r in records) and "legal_claims_large" in available:
            dataset_name = "legal_claims_large"
    elif dataset_name == "legal_claims_large" and "inlegalner_large" in available:
        dataset_name = "inlegalner_large"

    job_id = f"{model_type}_{dataset_name}_{_uuid.uuid4().hex[:8]}"
    TRAINING_JOBS[job_id] = {"message": "Starting...", "progress": 0, "status": "queued"}

    async def run_training():
        try:
            TRAINING_JOBS[job_id]["status"] = "running"

            if model_type == "bilstm_crf":
                from services.training_service import train_bilstm_crf

                epochs = body.epochs or 5
                result = train_bilstm_crf(dataset_name, epochs=epochs)
            elif model_type == "xgboost":
                from services.training_service import train_xgboost_anomaly

                result = train_xgboost_anomaly(dataset_name, max_docs=10000)
            elif model_type == "legalbert":
                from services.training_service import train_legalbert_contradiction

                epochs = body.epochs or 1
                result = train_legalbert_contradiction(
                    dataset_name, epochs=epochs, max_train=18000, max_test=2100
                )
            elif model_type == "rgcn":
                from services.training_service import train_rgcn

                epochs = body.epochs or 40
                result = train_rgcn(dataset_name, epochs=epochs, max_docs=8000)

            TRAINING_JOBS[job_id] = {
                "message": "Training complete",
                "progress": 100,
                "status": "completed",
                "result": result,
            }
            yield f"data: {json.dumps({'message': 'Training complete', 'progress': 100, 'status': 'completed', 'result': result})}\n\n"
        except Exception as e:
            TRAINING_JOBS[job_id] = {"message": str(e), "progress": 0, "status": "failed"}
            yield f"data: {json.dumps({'message': str(e), 'progress': 0, 'status': 'failed'})}\n\n"

    return StreamingResponse(run_training(), media_type="text/event-stream")


@app.get("/api/v1/admin/training/status/{job_id}")
async def admin_training_status(job_id: str, user: dict = Depends(require_admin)):
    if job_id not in TRAINING_JOBS:
        raise HTTPException(404, "Job not found")
    return TRAINING_JOBS[job_id]


# ---------------------------------------------------------------------------
# Admin — Model Metadata & Deployment
# ---------------------------------------------------------------------------
@app.get("/api/v1/admin/models")
async def admin_list_models(user: dict = Depends(require_admin)):
    return list_model_metadata()


class ActivateModel(BaseModel):
    model_type: str
    model_path: str


@app.post("/api/v1/admin/models/activate")
async def admin_activate_model(body: ActivateModel, user: dict = Depends(require_admin)):
    """Mark a trained model as the active production model."""
    meta_file = os.path.join(MODELS_DIR, "models_metadata.json")
    metadata = {}
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    # Only replace the model of the same type — the other three stay deployed.
    versions = metadata.get(body.model_type, [])
    matches = [m for m in versions if m.get("model_path") == body.model_path]
    if not matches:
        raise HTTPException(404, f"No {body.model_type} model registered at that path")
    target = max(matches, key=lambda m: str(m.get("trained_at", "")))
    for m in versions:
        m["active"] = False
    target["active"] = True
    target["activated_at"] = datetime.now(timezone.utc).isoformat()

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Pick up the new weights and drop cached analysis results.
    model_manager.reset()
    for cid in list(ANALYSES.keys()):
        ANALYSES[cid].pop("results", None)

    return {
        "message": f"Activated {body.model_type} model",
        "active_path": body.model_path,
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@app.get("/api/v1/cases")
async def list_cases(user: dict = Depends(require_user)):
    out = []
    for c in CASES.values():
        if c.get("owner") != user["email"]:
            continue
        results = ANALYSES.get(c["id"], {}).get("results")
        conflicts = len(results["conflicts"]) if results else c.get("conflicts", 0)
        anomalies = len(results["anomalies"]) if results else 0
        defendant_names = [p["name"] for p in c.get("people", []) if p["role"] == "Defendant"]
        out.append({
            "id": c["id"],
            "caseNumber": c.get("caseNumber", c["id"]),
            "court": c["court"],
            "ipcSections": c.get("ipcSections", ""),
            "defendants": ", ".join(defendant_names) or "—",
            "status": c.get("status", "active"),
            "conflicts": conflicts,
            "anomalies": anomalies,
            "evidence": len(c.get("evidence", [])),
        })
    return out


@app.post("/api/v1/cases")
async def create_case(body: CaseIn, user: dict = Depends(require_user)):
    number = body.caseId.strip()
    if not number:
        raise HTTPException(400, "Case ID is required")
    cid = _slug(number)
    base, n = cid, 1
    while cid in CASES:
        n += 1
        cid = f"{base}-{n}"
    people = []
    for i, d in enumerate(body.defendants):
        people.append({"id": f"p{i + 1}", "name": d.name, "role": d.role})
    case = {
        "id": cid,
        "caseNumber": number,
        "court": body.court or "—",
        "ipcSections": body.ipcSections,
        "defendants": [d.model_dump() for d in body.defendants],
        "people": people,
        "evidence": [],
        "status": "active",
        "conflicts": 0,
        "createdAt": _now(),
        "owner": user["email"],
    }
    CASES[cid] = case
    _save_data()
    return case


@app.get("/api/v1/cases/{case_id}")
async def get_case(case_id: str, user: dict = Depends(require_user)):
    case = _owned_case(case_id, user)
    results = ANALYSES.get(case_id, {}).get("results")
    evidence = [EVIDENCE[e] for e in case.get("evidence", []) if e in EVIDENCE]
    return {
        **case,
        "evidenceDetail": evidence,
        "analysed": bool(results),
        "conflicts": len(results["conflicts"]) if results else 0,
        "anomalies": len(results["anomalies"]) if results else 0,
    }


@app.delete("/api/v1/cases/{case_id}")
async def delete_case(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    CASES.pop(case_id, None)
    for eid in [e for e, v in list(EVIDENCE.items()) if v.get("caseId") == case_id]:
        ev = EVIDENCE.pop(eid, None)
        if ev and ev.get("filePath"):
            try:
                os.remove(ev["filePath"])
            except OSError:
                pass
    ANALYSES.pop(case_id, None)
    _save_data()
    return {"deleted": case_id}


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
def _kind_of(content_type: str, filename: str) -> str:
    ct = (content_type or "").lower()
    fn = (filename or "").lower()
    if ct.startswith("audio") or fn.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac")):
        return "audio"
    if ct.startswith("image") or fn.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")):
        return "image"
    if fn.endswith(".docx"):
        return "docx"
    if (ct.startswith("text/") or fn.endswith(
            (".txt", ".md", ".csv", ".log", ".json", ".jsonl", ".text", ".legal"))
    ):
        return "text"
    return "pdf"


def _next_evidence_ids(count: int = 1) -> List[str]:
    """Allocate unique evidence IDs even when earlier IDs were deleted.

    Using ``len(EVIDENCE)`` breaks as soon as a case is removed and leaves a
    gap, because two new items can then resolve to the same key and overwrite
    each other. Deriving the next number from the highest existing ID avoids
    that class of collisions.
    """
    highest = 99
    for key in EVIDENCE:
        match = re.fullmatch(r"EV-(\d+)", str(key))
        if match:
            highest = max(highest, int(match.group(1)))
    return [f"EV-{highest + 1 + i}" for i in range(max(1, count))]


@app.post("/api/v1/cases/{case_id}/evidence")
async def upload_evidence(case_id: str, files: List[UploadFile] = File(...),
                          user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    out = []
    ids = _next_evidence_ids(len(files))
    for i, f in enumerate(files):
        eid = ids[i]
        kind = _kind_of(f.content_type, f.filename)
        payload = b""
        try:
            payload = await f.read()
        except Exception:
            payload = b""
        extracted = extract_uploaded_text(f.filename or f"{eid}.bin",
                                          f.content_type or "", payload)
        raw_text = (extracted.get("text") or "").strip()
        saved_path = None
        if payload:
            suffix = Path(f.filename or "").suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                suffix = ""
            saved_path = os.path.join(UPLOADS_DIR, f"{eid}{suffix}")
            try:
                with open(saved_path, "wb") as handle:
                    handle.write(payload)
            except OSError:
                saved_path = None
        EVIDENCE[eid] = {
            "id": eid,
            "name": f.filename,
            "kind": kind,
            "type": f.content_type or "unknown",
            "linkedTo": None,
            "caseId": case_id,
            "uploadedAt": _now(),
            "rawText": raw_text,
            "charCount": len(raw_text),
            "size": len(payload),
            "extraction": extracted,
            "filePath": saved_path,
            "fileUrl": f"/api/v1/evidence/{eid}/file" if saved_path else None,
        }
        CASES[case_id]["evidence"].append(eid)
        out.append(EVIDENCE[eid])
    _invalidate_results(case_id)
    _save_data()
    return out


@app.post("/api/v1/cases/{case_id}/evidence/text")
async def add_text_evidence(case_id: str, body: EvidenceTextIn,
                            user: dict = Depends(require_user)):
    """Attach a plain-text document (judgment, statement, note) to a case."""
    _owned_case(case_id, user)
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, "Document text is required")
    eid = _next_evidence_ids(1)[0]
    EVIDENCE[eid] = {
        "id": eid,
        "name": body.name.strip() or f"{eid}.txt",
        "kind": body.kind or "text",
        "type": "text/plain",
        "linkedTo": None,
        "caseId": case_id,
        "uploadedAt": _now(),
        "rawText": text,
        "charCount": len(text),
        "entities": body.entities or [],
        "extraction": {"method": "text", "charCount": len(text), "needs_ocr": False,
                       "error": None},
    }
    CASES[case_id]["evidence"].append(eid)
    _invalidate_results(case_id)
    _save_data()
    return EVIDENCE[eid]


@app.get("/api/v1/cases/{case_id}/evidence")
async def list_evidence(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    return [EVIDENCE[e] for e in CASES[case_id].get("evidence", []) if e in EVIDENCE]


@app.post("/api/v1/evidence/{evidence_id}/link")
async def link_evidence(evidence_id: str, body: LinkIn,
                        user: dict = Depends(require_user)):
    ev = _owned_evidence(evidence_id, user)
    ev["linkedTo"] = body.personId
    if ev.get("caseId"):
        _invalidate_results(ev["caseId"])
    _save_data()
    return {"evidenceId": evidence_id, "personId": body.personId, "status": "linked"}


@app.delete("/api/v1/evidence/{evidence_id}")
async def delete_evidence(evidence_id: str, user: dict = Depends(require_user)):
    _owned_evidence(evidence_id, user)
    ev = EVIDENCE.pop(evidence_id, None)
    if ev and ev.get("caseId") in CASES:
        CASES[ev["caseId"]]["evidence"] = [e for e in CASES[ev["caseId"]]["evidence"] if e != evidence_id]
        _invalidate_results(ev["caseId"])
    if ev and ev.get("filePath"):
        try:
            os.remove(ev["filePath"])
        except OSError:
            pass
    _save_data()
    return {"deleted": evidence_id}


@app.get("/api/v1/evidence/{evidence_id}/document")
async def get_document(evidence_id: str, user: dict = Depends(require_user)):
    ev = _owned_evidence(evidence_id, user)
    return _document_for(ev)


@app.get("/api/v1/evidence/{evidence_id}/file")
async def get_evidence_file(evidence_id: str, user: dict = Depends(require_user)):
    """Serve the original exhibit to its owner (used for image previews)."""
    ev = _owned_evidence(evidence_id, user)
    path = ev.get("filePath")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Original file is not available")
    return FileResponse(path, filename=ev.get("name") or os.path.basename(path),
                        media_type=ev.get("type") or "application/octet-stream")


# ---------------------------------------------------------------------------
# Forgery detection — secondary experimental screening feature
# Model lives entirely on the backend; the .joblib file is never exposed.
# ---------------------------------------------------------------------------

MAX_FORGERY_UPLOAD_BYTES = 25 * 1024 * 1024


def _run_forgery_inference(image_bytes: bytes) -> dict:
    """Lazy-load the inference module and run predict_document.

    Wrapped in a helper so startup is not blocked by model loading.
    Any failure returns a structured error dict instead of crashing.
    """
    try:
        import inference as _inference  # noqa: PLC0415
        return _inference.predict_document(image_bytes)
    except FileNotFoundError:
        return {
            "result": None,
            "error_code": "model_missing",
            "error": "Forgery model is not available on this server. Please contact the administrator.",
        }
    except ImportError:
        return {
            "result": None,
            "error_code": "dependency_missing",
            "error": "Forgery screening dependencies are not installed on this server.",
        }
    except ValueError as exc:
        detail = str(exc) or "Invalid or unreadable image file."
        if not detail.lower().startswith("invalid or unreadable"):
            detail = f"Invalid or unreadable image file: {detail}"
        return {
            "result": None,
            "error_code": "invalid_image",
            "error": detail,
        }
    except Exception:  # noqa: BLE001
        # Do not leak model internals or stack traces to the client.
        return {
            "result": None,
            "error_code": "prediction_failed",
            "error": "The forgery model could not process this image. Please try another image.",
        }


def _forgery_http_error(result: dict) -> HTTPException:
    """Map an internal forgery failure to a safe HTTP response."""
    code = result.get("error_code")
    status = {
        "invalid_image": 422,
        "model_missing": 503,
        "dependency_missing": 503,
    }.get(code, 500)
    return HTTPException(status_code=status, detail=result.get("error"))


@app.post("/api/v1/forgery/check")
async def forgery_check_upload(
    file: UploadFile = File(...),
    user: dict = Depends(require_user),
):
    """Accept a directly-uploaded image and run forgery screening.

    This endpoint is the primary entry-point for the standalone forgery
    screening tool.  The model and bundle stay server-side at all times.
    """
    ALLOWED_IMAGE_TYPES = {
        "image/jpeg", "image/jpg", "image/png", "image/tiff",
        "image/bmp", "image/webp", "image/gif",
    }
    ct = (file.content_type or "").lower()
    fn = (file.filename or "").lower()
    is_image_ct = ct in ALLOWED_IMAGE_TYPES
    is_image_ext = fn.endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".gif"))

    if not is_image_ct and not is_image_ext:
        raise HTTPException(
            status_code=422,
            detail=(
                "Only image files are supported for forgery screening "
                "(JPEG, PNG, TIFF, BMP, WebP)."
            ),
        )

    try:
        payload = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file.")

    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(payload) > MAX_FORGERY_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is too large for forgery screening (maximum 25 MB).",
        )

    result = _run_forgery_inference(payload)
    if result.get("error"):
        raise _forgery_http_error(result)

    return {
        "filename": file.filename,
        **result,
    }


@app.get("/api/v1/evidence/{evidence_id}/forgery-check")
async def forgery_check_evidence(
    evidence_id: str,
    user: dict = Depends(require_user),
):
    """Run forgery screening against a previously-uploaded evidence image.

    Reads the saved file from UPLOADS_DIR — the model file is never sent
    to the client.
    """
    ev = _owned_evidence(evidence_id, user)
    if ev.get("kind") != "image":
        raise HTTPException(
            status_code=422,
            detail="Forgery screening is only available for image evidence.",
        )

    path = ev.get("filePath")
    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Original image file is not available for screening.",
        )

    try:
        with open(path, "rb") as fh:
            payload = fh.read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read evidence file: {exc}")

    result = _run_forgery_inference(payload)
    if result.get("error"):
        raise _forgery_http_error(result)

    return {
        "evidenceId": evidence_id,
        "filename": ev.get("name"),
        **result,
    }


# ---------------------------------------------------------------------------
# AI processing — SSE stream of pipeline stages
# ---------------------------------------------------------------------------
@app.post("/api/v1/analysis/run/{case_id}")
async def run_analysis(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)

    async def event_generator():
        # A manual re-run must recompute, not replay the cached findings.
        ANALYSES.setdefault(case_id, {}).pop("results", None)
        for stage in PIPELINE_STAGES:
            ANALYSES.setdefault(case_id, {})["stage"] = stage
            ANALYSES[case_id]["status"] = "running"
            yield f"data: {stage}\n\n"
            await asyncio.sleep(1.5)
        _ensure_results(case_id)
        ANALYSES[case_id]["status"] = "complete"
        yield "data: __complete__\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Analysis results (Module 3)
# ---------------------------------------------------------------------------
@app.get("/api/v1/cases/{case_id}/conflicts")
async def get_conflicts(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    return _ensure_results(case_id)["conflicts"]


@app.get("/api/v1/cases/{case_id}/anomalies")
async def get_anomalies(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    return _ensure_results(case_id)["anomalies"]


@app.get("/api/v1/cases/{case_id}/parties")
async def get_parties(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    return _ensure_results(case_id).get("parties", [])


@app.post("/api/v1/cases/{case_id}/review/{finding_id}")
async def review_finding(case_id: str, finding_id: str, body: ReviewIn,
                         user: dict = Depends(require_user)):
    """Record a human reviewer's confirm/dismiss decision on a finding."""
    case = _owned_case(case_id, user)
    status = (body.status or "").strip().lower()
    if status not in _REVIEW_STATUSES:
        raise HTTPException(400, "Review status must be pending, confirmed or dismissed")
    results = _ensure_results(case_id)
    findings = list(results.get("conflicts", [])) + list(results.get("anomalies", []))
    finding = next((item for item in findings if str(item.get("id")) == finding_id), None)
    if not finding:
        raise HTTPException(404, "Finding not found")

    reviews = case.setdefault("reviews", {})
    if status == "pending":
        reviews.pop(finding_id, None)
    else:
        reviews[finding_id] = {
            "status": status,
            "note": (body.note or "").strip(),
            "reviewer": user["email"],
            "reviewedAt": _now(),
        }
    summary = _apply_reviews(case, results.get("conflicts", []),
                             results.get("anomalies", []), results.get("report"))
    _save_data()
    return {"findingId": finding_id, "review": finding.get("review"),
            "reviewSummary": summary}


@app.get("/api/v1/cases/{case_id}/graph")
async def get_graph(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    return _ensure_results(case_id)["graph"]


@app.get("/api/v1/cases/{case_id}/entities")
async def get_entities(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    return _ensure_results(case_id)["entities"]


# ---------------------------------------------------------------------------
# Report (Module 4 — Explainable Insights & Reporting)
# ---------------------------------------------------------------------------
@app.get("/api/v1/report/{case_id}")
async def report_meta(case_id: str, user: dict = Depends(require_user)):
    case = _owned_case(case_id, user)
    number = case.get("caseNumber", case_id)
    slug_name = _slug(number)
    return {
        "url": f"/api/v1/report/{case_id}/content",
        "pdfUrl": f"/api/v1/report/{case_id}/pdf",
        "filename": f"{slug_name}-conflict-report.json",
        "pdfFilename": f"{slug_name}-conflict-report.pdf",
    }


@app.get("/api/v1/report/{case_id}/content")
async def report_content(case_id: str, user: dict = Depends(require_user)):
    _owned_case(case_id, user)
    return _ensure_results(case_id)["report"]


@app.post("/api/v1/report/{case_id}/regenerate")
async def report_regenerate(case_id: str, user: dict = Depends(require_user)):
    """Force re-run of analysis and fresh explainable report assembly."""
    _owned_case(case_id, user)
    ANALYSES.setdefault(case_id, {}).pop("results", None)
    results = _ensure_results(case_id)
    return results["report"]


@app.get("/api/v1/report/{case_id}/pdf")
async def report_pdf(case_id: str, user: dict = Depends(require_user)):
    """Generate and return a downloadable, publication-grade PDF report."""
    case = _owned_case(case_id, user)
    results = _ensure_results(case_id)
    report = results.get("report") or {}
    from services.report_service import render_report_pdf
    try:
        pdf_bytes = render_report_pdf(report)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF: {str(exc)}")
    number = case.get("caseNumber", case_id)
    filename = f"{_slug(number)}-conflict-report.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )
