"""Build app-ready datasets from the local InLegalNER corpus.

Produces:
  dataset_store/inlegalner_large/{train,dev,test}.jsonl
  dataset_store/legal_claims_large/{train,test}.jsonl

The NER files use the official InLegalNER train/dev/test splits so the reported
entity F1 is measured on genuinely unseen judgments.  The claim-pair files are
derived from those same splits (train pairs from train docs, test pairs from
test docs) and teach LegalBERT three legal-domain relations:

  0 consistent     two statements of the same fact, possibly re-framed
  1 contradiction  a date/amount/section/outcome/negation was altered
  2 neutral        statements from unrelated documents

Run from the backend folder:
    python scripts/build_from_local.py
"""
import collections
import json
import os
import random
import re
import sys

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND)

from services.dataset_service import DATASET_DIR  # noqa: E402
from services.features import canonical_type, extract_entities_regex  # noqa: E402

SOURCE = r"C:\Users\LENOVO\OneDrive\Documents\Dataset\InLegalNER"
OUT_NER = os.path.join(DATASET_DIR, "inlegalner_large")
OUT_NLI = os.path.join(DATASET_DIR, "legal_claims_large")

SPLIT_FILES = {
    "train": ["NER_TRAIN_JUDGEMENT.json", "NER_TRAIN_PREAMBLE.json"],
    "dev": ["NER_DEV_JUDGEMENT.json", "NER_DEV_PREAMBLE.json"],
    "test": ["NER_TEST_JUDGEMENT.json", "NER_TEST_PREAMBLE.json"],
}

ABBREV = {
    "no", "nos", "vs", "v", "adv", "sr", "smt", "mr", "mrs", "ms", "dr",
    "hon", "j", "jj", "crl", "cr", "writ", "pet", "resp", "s", "t", "r",
    "p", "pp", "sec", "scc", "air", "sc", "hc", "ib", "ibid", "etc",
    "co", "ltd", "pvt", "govt", "dist", "tal", "sta", "viz",
}


# ---------------------------------------------------------------------------
# NER loading
# ---------------------------------------------------------------------------

def load_split(split: str):
    """Return [{text, entities:[{start,end,label}]}] for one official split."""
    docs = []
    for name in SPLIT_FILES[split]:
        path = os.path.join(SOURCE, f"NER_{split.upper()}", name)
        with open(path, encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        for rec in data:
            text = (rec.get("data") or {}).get("text") or ""
            if not text.strip():
                continue
            ents = []
            for ann in rec.get("annotations") or []:
                for result in ann.get("result") or []:
                    val = result.get("value") or {}
                    try:
                        start, end = int(val.get("start")), int(val.get("end"))
                    except (TypeError, ValueError):
                        continue
                    labels = val.get("labels") or []
                    if not labels:
                        continue
                    label = canonical_type(labels[0])
                    if label == "OTHER" or end <= start or end > len(text):
                        continue
                    # Skip whitespace-only spans.
                    if not text[start:end].strip():
                        continue
                    ents.append({"start": start, "end": end, "label": label})
            # Drop overlaps, keeping the longest span.
            ents.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
            clean = []
            for ent in ents:
                if clean and ent["start"] < clean[-1]["end"]:
                    continue
                clean.append(ent)
            docs.append({"text": text, "entities": clean, "source": name})
    return docs


def write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Sentence / pair generation
# ---------------------------------------------------------------------------

ABBREV_RE = re.compile(r"\b(" + "|".join(sorted(ABBREV)) + r")\.$", re.I)


def split_sentences(text: str):
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text.replace("\n", " "))
    out = []
    for frag in raw:
        frag = re.sub(r"\s+", " ", frag).strip()
        if not frag:
            continue
        if out and ABBREV_RE.search(out[-1]):
            out[-1] = out[-1] + " " + frag
        else:
            out.append(frag)
    return [s for s in out if 35 <= len(s) <= 420]


MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def alter_date(value: str, rng: random.Random):
    # HH:MM
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        h = (h + rng.choice([-3, -2, 2, 3])) % 24
        mi = (mi + rng.choice([-10, -5, 5, 10])) % 60
        return f"{h:02d}:{mi:02d}"

    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    m = re.fullmatch(r"(\d{1,2})([/.-])(\d{1,2})\2(\d{2,4})", value)
    if m:
        d, sep, mo, y = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        d2 = rng.choice([x for x in range(1, 29) if x != d])
        return f"{d2:02d}{sep}{mo:02d}{sep}{y}"

    # YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        d2 = rng.choice([x for x in range(1, 29) if x != d])
        return f"{y}-{mo:02d}-{d2:02d}"

    # 12 January 2020 / 12th January 2020
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})", value)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        new_month = rng.choice([x for x in MONTHS if x.lower() != month.lower()])
        return f"{day} {new_month} {year}"

    # January 12, 2020
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", value)
    if m:
        month, day, year = m.group(1), m.group(2), m.group(3)
        new_month = rng.choice([x for x in MONTHS if x.lower() != month.lower()])
        return f"{new_month} {day}, {year}"

    # Bare year
    if re.fullmatch(r"(19|20)\d{2}", value):
        year = int(value) + rng.choice([-3, -2, -1, 1, 2, 3])
        return str(year)

    return None


def alter_number(value: str, rng: random.Random):
    matches = list(re.finditer(r"\d+", value))
    if not matches:
        return None
    m = matches[-1]
    old = m.group(0)
    if len(old) >= 4:
        new = str(rng.randint(1000, 9999))
    elif len(old) >= 2:
        new = str(rng.randint(10, 99))
    else:
        new = str(rng.randint(1, 9))
    if new == old:
        new = str(int(old) + 1)
    return value[: m.start()] + new + value[m.end():]


OUTCOME_SWAPS = [
    ("allowed", "dismissed"), ("dismissed", "allowed"),
    ("accepted", "rejected"), ("rejected", "accepted"),
    ("granted", "refused"), ("refused", "granted"),
    ("acquitted", "convicted"), ("convicted", "acquitted"),
    ("upheld", "set aside"), ("set aside", "upheld"),
    ("affirmed", "reversed"), ("reversed", "affirmed"),
]

NEGATION_RE = re.compile(
    r"\b(is|was|were|are|has|have|had|shall|will|can|could|may|might|did|does|do)\b",
    re.I,
)


def contradict(sentence: str, rng: random.Random):
    """Return a contradictory rewrite of the sentence, or None."""
    ents = extract_entities_regex(sentence)

    # 1. Altered date/clock value.
    for ent in ents:
        if ent.get("type") == "DATE":
            value = sentence[ent["start"]:ent["end"]]
            new_value = alter_date(value, rng)
            if new_value and new_value != value:
                return (sentence[: ent["start"]] + new_value
                        + sentence[ent["end"]:])

    # 2. Altered statute / case number.
    for ent in ents:
        if ent.get("type") in ("STATUTE", "CASE_NUMBER", "PROVISION"):
            value = sentence[ent["start"]:ent["end"]]
            new_value = alter_number(value, rng)
            if new_value and new_value != value:
                return (sentence[: ent["start"]] + new_value
                        + sentence[ent["end"]:])

    # 3. Flipped legal outcome.
    for old, new in OUTCOME_SWAPS:
        match = re.search(r"\b" + re.escape(old) + r"\b", sentence, re.I)
        if match:
            repl = new
            if match.group(0)[0].isupper():
                repl = new[0].upper() + new[1:]
            return sentence[: match.start()] + repl + sentence[match.end():]

    # 4. Negation of the first auxiliary verb.
    match = NEGATION_RE.search(sentence)
    if match:
        tail = sentence[match.end(): match.end() + 6].lower()
        if not tail.startswith(" not") and not tail.startswith("n't"):
            return sentence[: match.end()] + " not" + sentence[match.end():]

    return None


CONSISTENT_PREFIXES = [
    "The record states that ",
    "According to the case file, ",
    "It is recorded in the judgment that ",
    "The document notes that ",
]

CONSISTENT_SUFFIXES = [
    ", as recorded in the case file.",
    ", according to the judgment.",
    ", as noted in the record.",
]


def consistent_pair(sentence: str, rng: random.Random):
    roll = rng.random()
    if roll < 0.55:
        prefix = rng.choice(CONSISTENT_PREFIXES)
        rest = sentence[0].lower() + sentence[1:] if sentence else sentence
        paraphrased = prefix + rest
    elif roll < 0.9:
        paraphrased = sentence.rstrip(".") + rng.choice(CONSISTENT_SUFFIXES)
    else:
        paraphrased = sentence
    if rng.random() < 0.5:
        return sentence, paraphrased
    return paraphrased, sentence


SHARED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z]{5,}\b")


def neutral_pair(sentence_a: str, sentence_b: str):
    return sentence_a, sentence_b


def build_nli_pairs(docs, split: str, per_class: int, seed: int):
    rng = random.Random(seed)
    contradictions, consistents, neutrals = [], [], []

    all_sentences = [split_sentences(doc["text"]) for doc in docs]

    for i, sentences in enumerate(all_sentences):
        if not sentences:
            continue
        sentence = rng.choice(sentences)

        contra = contradict(sentence, rng)
        if contra and contra != sentence:
            contradictions.append({"premise": sentence, "hypothesis": contra,
                                   "label": 1, "source": split})

        p, h = consistent_pair(sentence, rng)
        consistents.append({"premise": p, "hypothesis": h,
                            "label": 0, "source": split})

        if len(all_sentences) > 2:
            j = rng.randrange(len(all_sentences))
            if j != i and all_sentences[j]:
                a, b = sentence, rng.choice(all_sentences[j])
                # Keep neutral pairs from sharing distinctive names.
                names_a = set(SHARED_TOKEN_RE.findall(a))
                names_b = set(SHARED_TOKEN_RE.findall(b))
                if not (names_a & names_b):
                    neutrals.append({"premise": a, "hypothesis": b,
                                     "label": 2, "source": split})

    rng.shuffle(contradictions)
    rng.shuffle(consistents)
    rng.shuffle(neutrals)
    rows = (contradictions[:per_class] + consistents[:per_class]
            + neutrals[:per_class])
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------

def main():
    random.seed(42)
    doc_counts = {}
    for split in ("train", "dev", "test"):
        docs = load_split(split)
        doc_counts[split] = len(docs)

        path = os.path.join(OUT_NER, f"{split}.jsonl")
        write_jsonl(path, docs)

        label_counts = collections.Counter(
            e["label"] for d in docs for e in d["entities"]
        )
        print(f"[NER] {split}: {len(docs)} docs, {sum(label_counts.values())} entities, "
              f"{len(label_counts)} labels")
        print("      ", label_counts.most_common())

        if split in ("train", "test"):
            per_class = 6000 if split == "train" else 700
            rows = build_nli_pairs(docs, split, per_class, seed=42 if split == "train" else 7)
            nli_path = os.path.join(OUT_NLI, f"{split}.jsonl")
            write_jsonl(nli_path, rows)
            dist = collections.Counter(r["label"] for r in rows)
            print(f"[NLI] {split}: {len(rows)} pairs -> {dict(sorted(dist.items()))}")

    with open(os.path.join(OUT_NER, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "name": "inlegalner_large",
            "description": "Indian legal NER corpus (InLegalNER) with official train/dev/test splits",
            "source": "InLegalNER, IIT Kharagpur / Kaggle",
            "license": "CC BY-NC-SA 4.0 (research use)",
            "labels": ["COURT", "JUDGE", "PETITIONER", "RESPONDENT", "WITNESS",
                       "LAWYER", "DATE", "PLACE", "STATUTE", "PROVISION",
                       "PRECEDENT", "CASE_NUMBER", "ORG", "PERSON"],
            "documents": doc_counts,
            "format": "jsonl with {text, entities:[{start,end,label}]}",
        }, f, indent=2)

    with open(os.path.join(OUT_NLI, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "name": "legal_claims_large",
            "description": "Legal claim pairs derived from InLegalNER judgments",
            "labels": {"0": "consistent", "1": "contradiction", "2": "neutral"},
            "generation": "date/section/outcome/negation edits; re-framed restatements; cross-document unrelated sentences",
            "format": "jsonl with {premise, hypothesis, label}",
        }, f, indent=2)

    print("done")


if __name__ == "__main__":
    main()
