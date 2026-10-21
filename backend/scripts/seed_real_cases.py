"""Seed the app with real InLegalNER judgments so every model can be tested.

Creates two cases, each with:
  * a clean judgment extract (real text),
  * a witness note that contradicts a date/time in the judgment,
  * a damaged/truncated scan (document-integrity anomaly).

Run the backend first, then:
    python scripts/seed_real_cases.py
"""
import json
import os
import re
import sys

import requests

BASE = "http://localhost:8000/api/v1"
BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_FILE = os.path.join(BACKEND, "services", "dataset_store", "inlegalner_large", "test.jsonl")

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")
DATE_RE = re.compile(
    r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b"
    rf"|\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})[\s,]+(\d{{4}})\b"
    r"|\b(\d{1,2}):(\d{2})\b",
    re.I,
)
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
AMOUNT_RE = re.compile(r"\b(?:Rs\.?|INR)\s*([\d,]{4,})", re.I)


def login():
    r = requests.post(f"{BASE}/auth/login",
                      json={"email": "admin@mecr.com", "password": "Admin@1234"},
                      timeout=60)
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def load_candidates():
    """Pick real, prose judgment extracts with a checkable fact (date/amount)."""
    verbs = ("is", "was", "were", "has", "had", "directed", "ordered", "held",
             "submitted", "observed", "notified", "issued", "convicted",
             "acquitted", "argued", "contended", "stated", "recorded", "filed")
    rows = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text") or ""
            if row.get("source") != "NER_TEST_JUDGEMENT.json":
                continue  # skip party/header preambles
            if not (260 <= len(text) <= 900):
                continue
            if len(row.get("entities") or []) < 3:
                continue
            if not DATE_RE.search(text) and not AMOUNT_RE.search(text):
                continue
            if len(re.findall(r"\bPW\s*\d+", text)) >= 4:
                continue  # witness lists are not a claim
            lower = [w for w in text.split() if w[:1].islower()]
            if len(lower) < 10:
                continue
            if not any(re.search(rf"\b{verb}\b", text, re.I) for verb in verbs):
                continue
            rows.append(row)

    def score(row):
        text = row["text"]
        lower = sum(1 for w in text.split() if w[:1].islower())
        facts = len(DATE_RE.findall(text)) + len(AMOUNT_RE.findall(text))
        return lower + facts * 6

    rows.sort(key=lambda r: -score(r))
    return rows


ABBREVIATIONS = (r"(?:No|Nos|Adv|Sr|Smt|Shri|Mr|Mrs|Ms|Dr|Hon|Crl|MC|Cri|IPC|"
                 r"Art|Sec|S|P|W|v|Vs|Ltd|Co|Doc)")


def split_sentences(text):
    """Whitespace-normalised splitter that respects legal abbreviations."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    flat = re.sub(r"\b(?:[A-Z]\.)+",
                  lambda m: m.group(0).replace(".", "<DOT>"), flat)
    flat = re.sub(r"\b(" + ABBREVIATIONS + r")\.", r"\1<DOT>", flat, flags=re.I)
    out = []
    for part in re.split(r"(?<=[.!?])\s+", flat):
        cleaned = part.replace("<DOT>", ".").strip()
        if len(cleaned) >= 60 and len(cleaned.split()) >= 8:
            out.append(cleaned)
    return out


def date_sentence(text):
    sentences = split_sentences(text)
    bad = re.compile(
        r"^(?:Criwp|Crl|WP|W\.P|\$~|\*|In The|Before|Coram|Judgment|Order|Memo|"
        r"List|Hon|Present)|Coram:|having (?:his|her) address|Aged \d+|Occu\.",
        re.I,
    )

    def usable(sentence):
        lower = [w for w in sentence.split() if w[:1].islower()]
        return (60 <= len(sentence) <= 480 and len(lower) >= 6
                and not bad.search(sentence))

    candidates = [s for s in sentences if usable(s)]
    for sentence in candidates:
        if re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})[\s,]+(\d{{4}})",
                     sentence, re.I):
            return sentence
    for sentence in candidates:
        if YEAR_RE.search(sentence):
            return sentence
    for sentence in candidates:
        if DATE_RE.search(sentence):
            return sentence
    return None


def alter_date_in(sentence):
    m = DATE_RE.search(sentence)
    if m:
        if m.group(1):  # DD/MM/YYYY
            d, mo, y = m.group(1), m.group(2), m.group(3)
            try:
                new_year = str(int(y) - 1)
            except ValueError:
                new_year = y
            replacement = f"{d}/{mo}/{new_year}"
        elif m.group(4):  # DD Month YYYY
            replacement = f"{m.group(4)} {m.group(5)} {int(m.group(6)) - 1}"
        else:  # HH:MM
            hour = (int(m.group(7)) + 2) % 24
            replacement = f"{hour:02d}:{m.group(8)}"
        return sentence[:m.start()] + replacement + sentence[m.end():]
    y = YEAR_RE.search(sentence)
    if y:
        return sentence[:y.start()] + str(int(y.group(1)) - 1) + sentence[y.end():]
    return None


def _sentence_around(body, match):
    start = body.rfind(".", 0, match.start())
    start = 0 if start < 0 else start + 1
    end = body.find(".", match.end())
    end = len(body) if end < 0 else end + 1
    snippet = body[start:end].strip()
    if len(snippet) < 60:
        snippet = body[max(0, match.start() - 90):match.end() + 160].strip()
    return snippet


def alter_amount_in(text):
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    if len(digits) < 4:
        return None
    return text[:match.start(1)] + str(int(digits) + 111111) + text[match.end(1):]


def contradiction_note(text):
    """Build a witness note that disagrees with one fact in the judgment."""
    body = re.sub(r"\s+", " ", text or "").strip()
    sentence = date_sentence(text)
    if sentence:
        altered = alter_date_in(sentence)
        if altered and altered != sentence:
            return ("Witness statement recorded on oath: " + altered
                    + " The witness affirmed this account under cross-examination.")
    altered = alter_amount_in(body)
    if altered and altered != body:
        return ("Witness statement recorded on oath: " + altered
                + " The witness affirmed this account under cross-examination.")
    # Last resort: quote the clause around the first year/date and shift it.
    match = YEAR_RE.search(body) or DATE_RE.search(body)
    if match:
        snippet = _sentence_around(body, match)
        altered = alter_date_in(snippet) or snippet
        return ("Witness statement recorded on oath: " + altered
                + " The witness affirmed this account under cross-examination.")
    return None


def damaged_version(text):
    """A partially redacted, mid-sentence scan — distinct from the clean text."""
    body = re.sub(r"\s+", " ", text).strip()
    cut = max(140, int(len(body) * 0.42))
    damaged = body[:cut].rstrip().rstrip(".!?,;:")
    return ("SCANNED CASE DIARY - OCR PARTIAL\n\n" + damaged + "\n\n[REDACTED]")


def entity_text(doc, label):
    text = doc.get("text") or ""
    for ent in doc.get("entities") or []:
        if ent.get("label") != label:
            continue
        if ent.get("text"):
            return ent["text"].strip()
        try:
            value = text[int(ent["start"]):int(ent["end"])].strip()
        except (KeyError, TypeError, ValueError):
            continue
        if value:
            return value
    return None


def gold_entities(doc):
    """Convert InLegalNER offset annotations into span records."""
    text = doc.get("text") or ""
    lead = len(text) - len(text.lstrip())
    out = []
    for ent in doc.get("entities") or []:
        try:
            start, end = int(ent["start"]) - lead, int(ent["end"]) - lead
        except (KeyError, TypeError, ValueError):
            continue
        if start < 0 or end <= start:
            continue
        span = text[start + lead:end + lead].strip()
        if not span:
            continue
        out.append({"text": span, "type": ent.get("label") or "OTHER",
                    "start": start, "end": end})
    return out


def party_names(doc):
    pet = entity_text(doc, "PETITIONER")
    resp = entity_text(doc, "RESPONDENT")
    people = []
    if pet:
        people.append({"name": pet, "role": "Petitioner"})
    if resp:
        people.append({"name": resp, "role": "Defendant"})
    if not people:
        people.append({"name": "Party as named in judgment", "role": "Defendant"})
    return people


def court_of(doc):
    court = entity_text(doc, "COURT")
    return court or "High Court (see judgment)"


def case_number_of(doc):
    text = re.sub(r"\s+", " ", doc.get("text") or "")
    pattern = (r"\b(?:Crl\.?\s*A|W\.P|WP|Crl\.?\s*M\.?C|C\.A|R\.S\.A|Cri\.?A|"
               r"O\.S\.A|A\.S|C\.M\.A|Cri\.?M\.?C)\.?\s*(?:No\.?)?\s*[\w./-]{2,24}")
    match = re.search(pattern, text, re.I)
    if match:
        return match.group(0).strip()[:44]
    return None


def existing_cases(headers):
    r = requests.get(f"{BASE}/cases", headers=headers, timeout=60)
    r.raise_for_status()
    return {c.get("caseNumber"): c.get("id") for c in r.json()}


def main() -> int:
    headers = login()
    rows = load_candidates()
    if len(rows) < 2:
        print("not enough candidate judgments")
        return 1

    plans = []
    used_numbers = set()
    for index, doc in enumerate(rows[:2]):
        number = case_number_of(doc) or f"IN-REAL-{index + 1:02d}"
        if number in used_numbers:
            number = f"{number}-B"
        used_numbers.add(number)
        extra = rows[2 + index] if len(rows) > 2 + index else None
        plans.append((number, doc, extra))
    have = existing_cases(headers)
    created = []

    # Remove the earlier preamble-based demo cases so the dashboard stays clean.
    for stale in ("WP 3009/2005", "WP 4921/2021"):
        if stale in have:
            requests.delete(f"{BASE}/cases/{have[stale]}", headers=headers, timeout=60)
            print(f"removed stale case {stale}")

    for number, doc, extra in plans:
        text = doc["text"]
        note = contradiction_note(text)
        damaged = damaged_version(text)

        if number in have:
            # Re-seed cleanly so the demo always reflects the latest generator.
            requests.delete(f"{BASE}/cases/{have[number]}", headers=headers, timeout=60)
            print(f"replaced existing case {number}")

        r = requests.post(f"{BASE}/cases", headers=headers, json={
            "caseId": number,
            "ipcSections": "See judgment",
            "court": court_of(doc),
            "defendants": party_names(doc),
        }, timeout=60)
        r.raise_for_status()
        case = r.json()
        cid = case["id"]
        created.append(cid)
        print(f"created {number} -> {cid} ({len(text)} chars, {len(doc.get('entities', []))} entities)")

        docs = [{"name": "Judgment extract.txt", "text": text,
                 "entities": gold_entities(doc)}]
        if extra:
            docs.append({"name": "Related judgment extract.txt", "text": extra["text"],
                         "entities": gold_entities(extra)})
        if note:
            docs.append({"name": "Witness note.txt", "text": note, "entities": []})
        docs.append({"name": "Case diary (damaged scan).txt", "text": damaged,
                     "entities": []})

        for item in docs:
            er = requests.post(f"{BASE}/cases/{cid}/evidence/text", headers=headers,
                               json={"name": item["name"], "text": item["text"],
                                     "kind": "text", "entities": item["entities"]},
                               timeout=60)
            er.raise_for_status()
            print("  +", item["name"], len(item["text"]), "chars",
                  f"({len(item['entities'])} gold entities)")

    for cid in created:
        print(f"\n=== analysis {cid} ===")
        r = requests.get(f"{BASE}/cases/{cid}/conflicts", headers=headers, timeout=900)
        r.raise_for_status()
        conflicts = r.json()
        anomalies = requests.get(f"{BASE}/cases/{cid}/anomalies", headers=headers,
                                 timeout=300).json()
        entities = requests.get(f"{BASE}/cases/{cid}/entities", headers=headers,
                                timeout=300).json()
        graph = requests.get(f"{BASE}/cases/{cid}/graph", headers=headers,
                             timeout=300).json()
        print(f"conflicts={len(conflicts)} anomalies={len(anomalies)} entities={len(entities)}")
        print("graph:", graph.get("meta"))
        for c in conflicts[:5]:
            print("  conflict:", c.get("type"), "|", c.get("model"), "|",
                  str((c.get("left") or {}).get("claim", ""))[:80])
        for a in anomalies[:5]:
            print("  anomaly:", a.get("type"), "|", a.get("model"), "|", a.get("score"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
