"""Training service — the 4 MECR models, trained on large public legal corpora.

Models
------
1. BiLSTM-CRF  — legal NER (InLegalNER, 11k judgments, official splits)
2. XGBoost     — document-integrity anomaly detection (real judgments + realistic damage)
3. LegalBERT   — 3-class contradiction detection (legal claim pairs, legal-bert-small)
4. R-GCN       — entity-graph anomaly detection (same corpus, real node features)

All feature engineering lives in ``services.features`` so training and inference
use byte-for-byte identical inputs.
"""
import os
import json
import random
import re
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import xgboost as xgb
import spacy
from torch_geometric.nn import RGCNConv, global_mean_pool, global_max_pool

from .dataset_service import (
    MODELS_DIR,
    DATASET_DIR,
    list_datasets,
    load_dataset_from_folder,
    load_jsonl_records,
    save_model_metadata,
)
from .features import (
    DOC_FEATURES,
    NODE_FEATURES,
    IN_CHANNELS,
    NUM_RELATIONS,
    document_features,
    feature_row,
    graph_tensors,
    corrupt_document,
    canonical_type,
)

# Keep CPU work predictable and avoid oversubscribing the machine.
torch.set_num_threads(max(1, min(12, os.cpu_count() or 4)))

LEGALBERT_CANDIDATES = [
    "nlpaueb/legal-bert-small-uncased",
    "nlpaueb/legal-bert-base-uncased",
]

# Labels for the contradiction classifier (index order matters).
LABEL2ID = {"consistent": 0, "contradiction": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dataset_folder(dataset_name: str) -> str:
    folder = os.path.abspath(os.path.join(DATASET_DIR, dataset_name))
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    return folder


def _ner_records(dataset_name: str, file_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return text+entity records from a dataset folder (or one split file)."""
    folder = _dataset_folder(dataset_name)
    target = os.path.join(folder, file_name) if file_name else folder
    raw = load_dataset_from_folder(target)
    out: List[Dict[str, Any]] = []
    for item in raw:
        # Keep the text verbatim: character offsets in the annotations are
        # relative to the original string, so stripping would shift them.
        text = item.get("text") or ""
        if not text.strip():
            continue
        ents = []
        for e in item.get("entities") or []:
            try:
                s, en = int(e.get("start", 0)), int(e.get("end", 0))
            except Exception:
                continue
            if 0 <= s < en <= len(text):
                ents.append({
                    "start": s,
                    "end": en,
                    "label": e.get("label", e.get("type", "OTHER")),
                    "text": text[s:en],
                })
        out.append({"text": text, "entities": ents, "source": item.get("source", "")})
    return out


def _train_records(dataset_name: str) -> List[Dict[str, Any]]:
    """Prefer the official train split so evaluation stays leakage-free."""
    folder = _dataset_folder(dataset_name)
    if os.path.exists(os.path.join(folder, "train.jsonl")):
        return _ner_records(dataset_name, "train.jsonl")
    return _ner_records(dataset_name)


# ============================================================================
# 1. BiLSTM-CRF NER
# ============================================================================

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=192, hidden_dim=384,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        # Two unrelated PyPI projects both expose a CRF class, with different
        # constructor/forward APIs.  Prefer kmkurn's `pytorch-crf` (module
        # `torchcrf`) and fall back to the `TorchCRF` package when only that
        # is installed, so the NER model loads in either environment.
        try:
            from torchcrf import CRF
            self.crf = CRF(len(tag_to_ix), batch_first=True)
            self._crf_api = "pytorch_crf"
        except ImportError:  # pragma: no cover - depends on local install
            from TorchCRF import CRF
            self.crf = CRF(len(tag_to_ix), use_gpu=torch.cuda.is_available())
            self._crf_api = "torchcrf"

    # Checkpoints trained with kmkurn's `pytorch-crf` store transition
    # parameters under different names than the `TorchCRF` package.  The
    # tensors are shape-identical, so renaming is enough to keep a trained
    # model fully usable regardless of which package is installed.
    _CRF_KEY_RENAMES = {
        "torchcrf": {
            "crf.start_transitions": "crf.start_trans",
            "crf.end_transitions": "crf.end_trans",
            "crf.transitions": "crf.trans_matrix",
        },
        "pytorch_crf": {
            "crf.start_trans": "crf.start_transitions",
            "crf.end_trans": "crf.end_transitions",
            "crf.trans_matrix": "crf.transitions",
        },
    }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        renames = self._CRF_KEY_RENAMES.get(getattr(self, "_crf_api", ""), {})
        if renames:
            state_dict = {renames.get(k, k): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        return self.hidden2tag(self.dropout(lstm_out))

    def loss(self, x, tags, mask):
        emissions = self.forward(x)
        if self._crf_api == "pytorch_crf":
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        # TorchCRF: forward() returns the negative log-likelihood tensor.
        return self.crf(emissions, tags, mask.bool()).mean()

    def predict(self, x, mask):
        emissions = self.forward(x)
        if self._crf_api == "pytorch_crf":
            return self.crf.decode(emissions, mask=mask)
        return self.crf.viterbi_decode(emissions, mask.bool())


class LegalNERDataset(Dataset):
    def __init__(self, sentences, tags, word2idx, tag2idx, max_len=128):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        seq = self.sentences[idx][:self.max_len]
        tag_seq = self.tags[idx][:self.max_len]
        x = [self.word2idx.get(w, self.word2idx.get("UNK", 1)) for w in seq]
        y = [self.tag2idx.get(t, self.tag2idx.get("O", 0)) for t in tag_seq]
        pad = self.max_len - len(x)
        x = x + [self.word2idx.get("PAD", 0)] * pad
        y = y + [self.tag2idx.get("PAD", 0)] * pad
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def _spans_from_tags(tags: List[str]) -> set:
    spans = set()
    start = None
    current = None
    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if start is not None:
                spans.add((start, i - 1, current))
            start, current = i, tag[2:]
        elif tag.startswith("I-") and start is not None and tag[2:] == current:
            continue
        else:
            if start is not None:
                spans.add((start, i - 1, current))
            start, current = None, None
    if start is not None:
        spans.add((start, len(tags) - 1, current))
    return spans


def _entity_prf(true_tags: List[str], pred_tags: List[str]) -> Tuple[float, float, float]:
    start = 0
    tp = fp = fn = 0
    while start < len(true_tags):
        end = start
        while end < len(true_tags) and true_tags[end] != "PAD":
            end += 1
        t = true_tags[start:end]
        p = pred_tags[start:end]
        ts, ps = _spans_from_tags(t), _spans_from_tags(p)
        tp += len(ts & ps)
        fp += len(ps - ts)
        fn += len(ts - ps)
        start = end + 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _pick_windows(n_tokens: int, tags: List[str], max_len: int = 128,
                  max_windows: int = 3) -> List[Tuple[int, int]]:
    """Choose up to N windows that cover the head, densest and tail regions.

    Legal judgments put parties in the preamble, statutes/evidence in the body
    and the outcome at the end; training on just the first 128 tokens misses
    most of that, so we sample three complementary windows.
    """
    if n_tokens <= max_len:
        return [(0, n_tokens)]

    starts = [0]
    best_start, best_score = 0, -1
    for start in range(0, n_tokens - max_len + 1, max_len // 2):
        score = sum(1 for t in tags[start:start + max_len] if t != "O")
        if score > best_score:
            best_score, best_start = score, start
    starts.append(best_start)
    starts.append(n_tokens - max_len)

    windows: List[Tuple[int, int]] = []
    seen = set()
    for start in starts:
        start = max(0, min(start, n_tokens - max_len))
        if start in seen:
            continue
        seen.add(start)
        windows.append((start, start + max_len))
        if len(windows) >= max_windows:
            break
    return windows


def _encode_ner(records: List[Dict[str, Any]], nlp, max_len: int = 128,
                max_windows: int = 3) -> Tuple[List[List[str]], List[List[str]]]:
    """Turn annotated documents into (token windows, BIO tag windows)."""
    sentences: List[List[str]] = []
    tag_seqs: List[List[str]] = []

    for rec in records:
        text = rec["text"][:12000]
        doc = nlp(text)
        if not len(doc):
            continue
        tags = ["O"] * len(doc)
        for ent in rec["entities"]:
            label = canonical_type(ent["label"])
            if label == "OTHER":
                continue
            span = doc.char_span(ent["start"], ent["end"], label=label, alignment_mode="expand")
            if span is None:
                continue
            for i, token in enumerate(span):
                tags[token.i] = f"{'B' if i == 0 else 'I'}-{label}"
        tokens = [t.text for t in doc]
        for start, end in _pick_windows(len(tokens), tags, max_len=max_len,
                                        max_windows=max_windows):
            sentences.append(tokens[start:end])
            tag_seqs.append(tags[start:end])
    return sentences, tag_seqs


def train_bilstm_crf(dataset_name: str, epochs: int = 8, max_len: int = 128) -> Dict[str, Any]:
    """Train BiLSTM-CRF NER on an annotated legal dataset."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    folder = _dataset_folder(dataset_name)
    has_official_split = (os.path.exists(os.path.join(folder, "train.jsonl"))
                          and os.path.exists(os.path.join(folder, "test.jsonl")))
    nlp = spacy.blank("en")

    if has_official_split:
        train_records = _ner_records(dataset_name, "train.jsonl")
        test_records = _ner_records(dataset_name, "test.jsonl")
        if not train_records:
            raise ValueError("No valid training records found")
        X_train, y_train = _encode_ner(train_records, nlp, max_len=max_len)
        X_test, y_test = _encode_ner(test_records, nlp, max_len=max_len)
        if len(X_test) > 3000:
            X_test, y_test = X_test[:3000], y_test[:3000]
    else:
        records = _ner_records(dataset_name)
        if not records:
            raise ValueError("No valid text records found in dataset")
        X_all, y_all = _encode_ner(records, nlp, max_len=max_len)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.1, random_state=42
        )
        if len(X_test) > 2500:
            X_test, y_test = X_test[:2500], y_test[:2500]

    if not X_train:
        raise ValueError("No training windows produced from the dataset")

    # Vocabulary with a frequency floor keeps the embedding table sane.
    from collections import Counter
    word_counts = Counter(w for sent in X_train for w in sent)
    vocab_words = [w for w, c in word_counts.items() if c >= 2]
    vocab_words.sort()
    word2idx = {"PAD": 0, "UNK": 1}
    for w in vocab_words:
        word2idx.setdefault(w, len(word2idx))

    tag_set = sorted({t for seq in y_train for t in seq} | {"PAD", "O"})
    tag2idx = {t: i for i, t in enumerate(tag_set)}
    idx2tag = {v: k for k, v in tag2idx.items()}

    train_loader = DataLoader(LegalNERDataset(X_train, y_train, word2idx, tag2idx, max_len),
                              batch_size=32, shuffle=True)
    test_loader = DataLoader(LegalNERDataset(X_test, y_test, word2idx, tag2idx, max_len),
                             batch_size=32, shuffle=False)

    device = _device()
    model = BiLSTM_CRF(len(word2idx), tag2idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    best_f1 = -1.0
    best_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            mask = (x_batch != word2idx["PAD"]).to(torch.bool)
            model.zero_grad()
            loss = model.loss(x_batch, y_batch, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                mask = (x_batch != word2idx["PAD"]).to(torch.bool)
                preds = model.predict(x_batch, mask=mask)
                for row, gold in zip(preds, y_batch):
                    preds_all.extend(row)
                    labels_all.extend(gold[:len(row)].tolist())

        true_tags = [idx2tag[i] for i in labels_all if idx2tag.get(i) != "PAD"]
        pred_tags = [idx2tag[i] for i in preds_all[:len(true_tags)]]
        acc = accuracy_score(true_tags, pred_tags) * 100 if true_tags else 0.0
        _, _, ent_f1 = _entity_prf(true_tags, pred_tags)
        avg_loss = total_loss / max(1, len(train_loader))
        history.append({"epoch": epoch + 1, "loss": round(avg_loss, 4),
                        "accuracy": round(acc, 2), "entity_f1": round(ent_f1, 4)})
        print(f"BiLSTM-CRF epoch {epoch+1}/{epochs} loss={avg_loss:.4f} "
              f"token_acc={acc:.2f}% entity_f1={ent_f1:.4f}")
        if ent_f1 > best_f1:
            best_f1 = ent_f1
            best_acc = acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = os.path.join(MODELS_DIR, f"bilstm_crf_{dataset_name}.pth")
    torch.save(model.state_dict(), model_path)
    with open(model_path.replace(".pth", "_word2idx.json"), "w", encoding="utf-8") as f:
        json.dump(word2idx, f)
    with open(model_path.replace(".pth", "_tag2idx.json"), "w", encoding="utf-8") as f:
        json.dump(tag2idx, f)
    # Also keep the short aliases used by older checkpoints.
    with open(os.path.join(MODELS_DIR, f"word2idx_{dataset_name}.json"), "w", encoding="utf-8") as f:
        json.dump(word2idx, f)
    with open(os.path.join(MODELS_DIR, f"tag2idx_{dataset_name}.json"), "w", encoding="utf-8") as f:
        json.dump(tag2idx, f)

    save_model_metadata("bilstm_crf", {
        "dataset": dataset_name,
        "epochs": epochs,
        "accuracy": round(best_acc, 2),
        "entity_f1": round(best_f1, 4),
        "vocab_size": len(word2idx),
        "num_tags": len(tag2idx),
        "samples": len(X_train) + len(X_test),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "evaluation": "official test.jsonl (held-out documents)" if has_official_split else "random 10% split",
        "max_len": max_len,
    }, model_path)

    return {
        "model": "BiLSTM-CRF NER",
        "dataset": dataset_name,
        "epochs": epochs,
        "accuracy": round(best_acc, 2),
        "entity_f1": round(best_f1, 4),
        "loss_history": history,
        "samples": len(X_train) + len(X_test),
        "evaluation": "official test.jsonl (held-out documents)" if has_official_split else "random 10% split",
        "vocab_size": len(word2idx),
        "num_tags": len(tag2idx),
    }


# ============================================================================
# 2. XGBoost anomaly detection
# ============================================================================

def _xgboost_rows(records: List[Dict[str, Any]], max_docs: int, seed: int = 42):
    rng = random.Random(seed)
    rows, labels, kinds = [], [], []
    docs = records[:max_docs]
    for rec in docs:
        text, ents = rec["text"], rec["entities"]
        rows.append(feature_row(text, ents))
        labels.append(0)
        kinds.append("clean")
        ctext, cents, kind = corrupt_document(text, ents, rng)
        rows.append(feature_row(ctext, cents))
        labels.append(1)
        kinds.append(kind)
    return rows, labels, kinds


def train_xgboost_anomaly(dataset_name: str, max_docs: int = 8000) -> Dict[str, Any]:
    """Train XGBoost to detect damaged / tampered legal documents."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        classification_report,
    )

    records = _train_records(dataset_name)
    if not records:
        raise ValueError("No valid text records found in dataset")

    X, y, kinds = _xgboost_rows(records, max_docs)
    if len(set(y)) < 2:
        raise ValueError("Need both clean and anomalous samples to train")

    # Fit on a train split; keep a validation split for early stopping and a
    # final held-out test split for the reported metrics.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15, random_state=42, stratify=y_tmp
    )

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=1.5,
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=8,
    )
    model.fit(
        pd.DataFrame(X_train, columns=DOC_FEATURES), y_train,
        eval_set=[(pd.DataFrame(X_val, columns=DOC_FEATURES), y_val)],
        verbose=False,
    )

    y_pred = model.predict(pd.DataFrame(X_test, columns=DOC_FEATURES))
    y_proba = model.predict_proba(pd.DataFrame(X_test, columns=DOC_FEATURES))[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    model_path = os.path.join(MODELS_DIR, f"xgboost_anomaly_{dataset_name}.json")
    model.save_model(model_path)

    save_model_metadata("xgboost", {
        "dataset": dataset_name,
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "roc_auc": round(auc, 4),
        "samples": len(X),
        "features": DOC_FEATURES,
        "anomaly_definition": "real judgment + simulated document damage (truncation/redaction/entity loss/duplication/boilerplate)",
    }, model_path)

    return {
        "model": "XGBoost Anomaly",
        "dataset": dataset_name,
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "roc_auc": round(auc, 4),
        "samples": len(X),
        "features": DOC_FEATURES,
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "corruption_mix": {k: kinds.count(k) for k in set(kinds)},
    }


# ============================================================================
# 3. LegalBERT contradiction detection
# ============================================================================

def _first_legalbert() -> str:
    from transformers import AutoConfig
    for candidate in LEGALBERT_CANDIDATES:
        try:
            AutoConfig.from_pretrained(candidate)
            return candidate
        except Exception:
            continue
    return "nlpaueb/legal-bert-base-uncased"


def _legacy_nli_pairs(dataset_name: str):
    records = _ner_records(dataset_name)
    premises, hypotheses, labels = [], [], []
    for rec in records:
        text = rec["text"]
        if not text:
            continue
        premises.append(text[:1000])
        hypotheses.append(text[:1000])
        labels.append(0)
        premises.append(text[:1000])
        hypotheses.append(("It is not true that " + text)[:1000])
        labels.append(1)
    return premises, hypotheses, labels


def _active_legalbert_checkpoint() -> Optional[str]:
    """Return the deployed LegalBERT directory so retraining can warm-start."""
    meta_file = os.path.join(MODELS_DIR, "models_metadata.json")
    try:
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    for entry in (meta.get("legalbert") or []):
        if not entry.get("active"):
            continue
        path = entry.get("model_path")
        if path and os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
    return None


def train_legalbert_contradiction(dataset_name: str, epochs: int = 2,
                                  max_len: int = 128, max_train: int = 6000,
                                  max_test: int = 1500) -> Dict[str, Any]:
    """Fine-tune LegalBERT-small on balanced 3-class NLI data."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from datasets import Dataset as HFDataset
    from sklearn.metrics import accuracy_score, f1_score

    premises: List[str] = []
    hypotheses: List[str] = []
    labels: List[int] = []

    nli_records = load_jsonl_records(_dataset_folder(dataset_name))
    nli_records = [r for r in nli_records if "premise" in r and "hypothesis" in r]

    folder = _dataset_folder(dataset_name)
    train_file = os.path.join(folder, "train.jsonl")
    test_file = os.path.join(folder, "test.jsonl")
    if os.path.exists(train_file) and os.path.exists(test_file):
        # Honour the official split files when they are present.
        train_recs = [r for r in load_jsonl_records(train_file)
                      if "premise" in r and "hypothesis" in r]
        test_recs = [r for r in load_jsonl_records(test_file)
                     if "premise" in r and "hypothesis" in r]
        random.Random(42).shuffle(train_recs)
        for r in train_recs[:max_train]:
            premises.append(r["premise"])
            hypotheses.append(r["hypothesis"])
            labels.append(int(r.get("label", 0)))
        for r in test_recs[:max_test]:
            premises.append(r["premise"])
            hypotheses.append(r["hypothesis"])
            labels.append(int(r.get("label", 0)))
        n_train = min(len(train_recs), max_train)
    elif nli_records:
        rng = random.Random(42)
        rng.shuffle(nli_records)
        # A 90/10 split keeps a genuine held-out set.
        cut = int(len(nli_records) * 0.9)
        train_recs, test_recs = nli_records[:cut], nli_records[cut:]
        for r in train_recs[:max_train]:
            premises.append(r["premise"])
            hypotheses.append(r["hypothesis"])
            labels.append(int(r.get("label", 0)))
        for r in test_recs[:max_test]:
            premises.append(r["premise"])
            hypotheses.append(r["hypothesis"])
            labels.append(int(r.get("label", 0)))
        n_train = min(len(train_recs), max_train)
    else:
        premises, hypotheses, labels = _legacy_nli_pairs(dataset_name)
        n_train = int(len(premises) * 0.9)

    if not premises:
        raise ValueError("No usable text pairs found in dataset")

    dataset = HFDataset.from_dict({
        "premise": premises,
        "hypothesis": hypotheses,
        "label": labels,
        "_is_train": [1] * n_train + [0] * (len(premises) - n_train),
    })
    train_ds = dataset.filter(lambda x: x["_is_train"] == 1).remove_columns(["_is_train"])
    test_ds = dataset.filter(lambda x: x["_is_train"] == 0).remove_columns(["_is_train"])
    if len(test_ds) == 0:
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, test_ds = split["train"], split["test"]

    base_model = _active_legalbert_checkpoint() or _first_legalbert()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def tokenize_function(examples):
        return tokenizer(examples["premise"], examples["hypothesis"],
                         truncation=True, padding="max_length", max_length=max_len)

    train_tok = train_ds.map(tokenize_function, batched=True)
    test_tok = test_ds.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, f"legalbert_{dataset_name}_tmp"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    def compute_metrics(eval_pred):
        logits, labels_eval = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels_eval, predictions)),
            "f1": float(f1_score(labels_eval, predictions, average="macro")),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_results = trainer.evaluate()

    model_dir = os.path.join(MODELS_DIR, f"legalbert_contradiction_{dataset_name}")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    acc = float(eval_results.get("eval_accuracy", 0.0))
    f1 = float(eval_results.get("eval_f1", 0.0))
    save_model_metadata("legalbert", {
        "dataset": dataset_name,
        "base_model": base_model,
        "epochs": epochs,
        "accuracy": round(acc * 100, 2),
        "f1": round(f1, 4),
        "samples": len(train_ds) + len(test_ds),
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "classes": 3,
        "labels": LABEL2ID,
    }, model_dir)

    return {
        "model": "LegalBERT Contradiction",
        "dataset": dataset_name,
        "base_model": base_model,
        "epochs": epochs,
        "accuracy": round(acc * 100, 2),
        "f1": round(f1, 4),
        "samples": len(train_ds) + len(test_ds),
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "eval_results": eval_results,
    }


# ============================================================================
# 4. R-GCN entity-graph anomaly detection
# ============================================================================

class LegalRGCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations, num_bases=2)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations, num_bases=2)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.fc = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_type, batch_vector):
        x = self.conv1(x, edge_index, edge_type)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        x = self.norm2(x)
        x = F.relu(x)
        # Concatenating mean and max pooling keeps both the average graph
        # profile and the most extreme node (e.g. a lone surviving entity).
        mean_pool = global_mean_pool(x, batch_vector)
        max_pool = global_max_pool(x, batch_vector)
        return self.fc(torch.cat([mean_pool, max_pool], dim=1))


def _rgcn_graphs(records: List[Dict[str, Any]], max_docs: int):
    from torch_geometric.data import Data

    rng = random.Random(42)
    graphs = []
    kinds = []
    # Boilerplate headers barely change the entity graph, so R-GCN is trained
    # on the corruptions that actually damage entity structure.
    structure_kinds = ["truncation", "redaction", "entity_drop", "duplication"]
    for rec in records[:max_docs]:
        text, ents = rec["text"], rec["entities"]

        normal = graph_tensors(ents, len(text))
        if normal is not None:
            x, ei, et, _ = normal
            graphs.append(Data(x=x, edge_index=ei, edge_type=et,
                               y=torch.tensor([0], dtype=torch.long)))
            kinds.append("clean")

        # Two different damage variants per document doubles the signal.
        for _ in range(2):
            ctext, cents, kind = corrupt_document(text, ents, rng,
                                                 kind=rng.choice(structure_kinds))
            corrupt = graph_tensors(cents, len(ctext))
            if corrupt is None:
                # An emptied graph is itself a strong "metadata wiped" signal.
                x = torch.zeros(1, IN_CHANNELS)
                ei = torch.empty((2, 0), dtype=torch.long)
                et = torch.empty((0,), dtype=torch.long)
                graphs.append(Data(x=x, edge_index=ei, edge_type=et,
                                   y=torch.tensor([1], dtype=torch.long)))
            else:
                x, ei, et, _ = corrupt
                graphs.append(Data(x=x, edge_index=ei, edge_type=et,
                                   y=torch.tensor([1], dtype=torch.long)))
            kinds.append(kind)
    return graphs, kinds


def train_rgcn(dataset_name: str, epochs: int = 30, max_docs: int = 6000) -> Dict[str, Any]:
    """Train R-GCN to classify entity graphs as clean vs damaged/tampered."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    records = _train_records(dataset_name)
    if not records:
        raise ValueError("No valid text records found in dataset")

    graphs, kinds = _rgcn_graphs(records, max_docs)
    if not graphs:
        raise ValueError("Could not build any entity graphs from the dataset")

    rng = random.Random(42)
    rng.shuffle(graphs)
    cut = int(len(graphs) * 0.8)
    train_ds, test_ds = graphs[:cut], graphs[cut:]

    train_loader = PyGDataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = PyGDataLoader(test_ds, batch_size=64, shuffle=False)

    device = _device()
    model = LegalRGCNModel(
        in_channels=IN_CHANNELS,
        hidden_channels=96,
        out_channels=2,
        num_relations=NUM_RELATIONS,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    hidden_channels = 96

    # Balanced class weights
    n_pos = sum(int(g.y.item()) for g in train_ds)
    n_neg = len(train_ds) - n_pos
    weights = torch.tensor([1.0, max(1.0, n_neg / max(1, n_pos))], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": round(total_loss / max(1, len(train_loader)), 4)})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"R-GCN epoch {epoch+1}/{epochs} loss={history[-1]['loss']}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            y_pred.extend(out.argmax(dim=1).cpu().tolist())
            y_true.extend(batch.y.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    model_path = os.path.join(MODELS_DIR, f"rgcn_{dataset_name}.pth")
    torch.save(model.state_dict(), model_path)

    save_model_metadata("rgcn", {
        "dataset": dataset_name,
        "epochs": epochs,
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "samples": len(graphs),
        "train_graphs": len(train_ds),
        "test_graphs": len(test_ds),
        "in_channels": IN_CHANNELS,
        "num_relations": NUM_RELATIONS,
        "hidden_channels": hidden_channels,
        "node_features": NODE_FEATURES,
        "anomaly_definition": "entity-graph damage: removed metadata, truncation, duplicated scans, boilerplate",
    }, model_path)

    return {
        "model": "R-GCN Conflict Graph",
        "dataset": dataset_name,
        "epochs": epochs,
        "accuracy": round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "samples": len(graphs),
        "train_graphs": len(train_ds),
        "test_graphs": len(test_ds),
        "history": history,
    }


# ============================================================================
# 5-fold cross-validation for the BiLSTM-CRF (the report's Table II claim)
# ============================================================================
def _save_bilstm_cv(cv: Dict[str, Any]) -> None:
    """Attach CV results to the currently deployed BiLSTM metadata entry."""
    meta_file = os.path.join(MODELS_DIR, "models_metadata.json")
    try:
        with open(meta_file, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:
        return
    for entry in metadata.get("bilstm_crf", []) or []:
        if entry.get("active"):
            entry["cv"] = cv
            break
    else:
        return
    with open(meta_file, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def cross_validate_bilstm(
    dataset_name: str = "inlegalner_large",
    folds: int = 5,
    epochs: int = 2,
    max_len: int = 128,
    max_samples: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run k-fold CV on a sampled set of training windows.

    The deployed checkpoint is not overwritten.  The mean/std token accuracy and
    entity F1 are written to the active model's metadata so the admin panel and
    report can show a cross-validated estimate alongside the held-out test
    result.  A cap keeps the evaluation practical on CPU; the cap is recorded
    in the result so the number is never presented as full-corpus CV.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score

    folder = _dataset_folder(dataset_name)
    nlp = spacy.blank("en")
    if os.path.exists(os.path.join(folder, "train.jsonl")):
        records = _ner_records(dataset_name, "train.jsonl")
    else:
        records = _ner_records(dataset_name)
    if not records:
        raise ValueError("No training records found for cross-validation")

    X_all, y_all = _encode_ner(records, nlp, max_len=max_len, max_windows=1)
    if not X_all:
        raise ValueError("No windows produced for cross-validation")
    rng = random.Random(seed)
    indices = list(range(len(X_all)))
    rng.shuffle(indices)
    indices = indices[:max_samples]
    X = [X_all[i] for i in indices]
    y = [y_all[i] for i in indices]

    # Tag vocabulary is shared across folds; words are fold-specific to avoid
    # validation vocabulary leakage.
    tag_set = sorted({t for seq in y for t in seq} | {"PAD", "O"})
    tag2idx = {t: i for i, t in enumerate(tag_set)}
    idx2tag = {v: k for k, v in tag2idx.items()}
    device = _device()
    fold_results: List[Dict[str, Any]] = []

    kfold = KFold(n_splits=max(2, folds), shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        from collections import Counter
        word_counts = Counter(w for sent in X_train for w in sent)
        vocab_words = sorted(w for w, count in word_counts.items() if count >= 1)
        word2idx = {"PAD": 0, "UNK": 1}
        for word in vocab_words:
            word2idx.setdefault(word, len(word2idx))

        train_loader = DataLoader(
            LegalNERDataset(X_train, y_train, word2idx, tag2idx, max_len),
            batch_size=32, shuffle=True,
        )
        test_loader = DataLoader(
            LegalNERDataset(X_test, y_test, word2idx, tag2idx, max_len),
            batch_size=32, shuffle=False,
        )
        model = BiLSTM_CRF(len(word2idx), tag2idx).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(max(1, epochs)):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                mask = (x_batch != word2idx["PAD"]).to(torch.bool)
                model.zero_grad()
                loss = model.loss(x_batch, y_batch, mask=mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                mask = (x_batch != word2idx["PAD"]).to(torch.bool)
                preds = model.predict(x_batch, mask=mask)
                for row, gold in zip(preds, y_batch):
                    preds_all.extend(row)
                    labels_all.extend(gold[:len(row)].tolist())
        true_tags = [idx2tag[i] for i in labels_all if idx2tag.get(i) != "PAD"]
        pred_tags = [idx2tag[i] for i in preds_all[:len(true_tags)]]
        acc = accuracy_score(true_tags, pred_tags) * 100 if true_tags else 0.0
        _, _, ent_f1 = _entity_prf(true_tags, pred_tags)
        fold_results.append({
            "fold": fold,
            "accuracy": round(acc, 2),
            "entity_f1": round(ent_f1, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        })
        print(f"BiLSTM-CRF CV fold {fold}/{folds}: token_acc={acc:.2f}% entity_f1={ent_f1:.4f}")

    accuracies = [r["accuracy"] for r in fold_results]
    f1s = [r["entity_f1"] for r in fold_results]

    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    def _std(values):
        if not values:
            return 0.0
        mean = _mean(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5

    cv = {
        "folds": folds,
        "epochs": epochs,
        "samples": len(X),
        "requested_samples": len(X_all),
        "accuracy_mean": round(_mean(accuracies), 2),
        "accuracy_std": round(_std(accuracies), 2),
        "entity_f1_mean": round(_mean(f1s), 4),
        "entity_f1_std": round(_std(f1s), 4),
        "per_fold": fold_results,
        "max_len": max_len,
        "evaluation": (
            f"{folds}-fold CV on {len(X)} sampled training windows"
            + ("" if len(X) >= len(X_all) else f" (cap {max_samples} of {len(X_all)})")
        ),
    }
    _save_bilstm_cv(cv)
    return {"model": "BiLSTM-CRF 5-fold CV", "dataset": dataset_name, **cv}
