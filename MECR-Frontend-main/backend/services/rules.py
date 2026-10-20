"""Explainable rule reasoning for the MECR pipeline.

The research report describes three anomaly families — contradictions,
missing information and possible manipulation — discovered with rule
reasoning.  The deep models supply semantic/statistical signals; this module
adds transparent, auditable checks and ties every finding to a person and an
exhibit.

Keeping the rules here (rather than in ``main.py``) makes them unit-testable
and ensures the API and the report use exactly the same explanations.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _norm(value).lower()).strip()


def _short(value: Any, limit: int = 90) -> str:
    text = _norm(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _person_label(person: dict) -> str:
    role = _norm(person.get("role")) or "Person"
    name = _norm(person.get("name")) or "Unnamed"
    return f"{name} ({role})"


def _evidence_source(ev: dict) -> str:
    return _norm(ev.get("name")) or _norm(ev.get("id")) or "Exhibit"


def _source_evidence_id(source: Any, evidence: Sequence[dict]) -> Optional[str]:
    """Map a report source string (name or ``EV-...``) to an evidence id."""
    text = _norm(source)
    if not text:
        return None
    match = re.search(r"\bEV[- ]?(\d+)\b", text, re.I)
    if match:
        wanted = f"EV-{int(match.group(1))}"
        if any(_norm(ev.get("id")) == wanted for ev in evidence):
            return wanted
    for ev in evidence:
        if _norm(ev.get("name")) == text:
            return _norm(ev.get("id"))
    for ev in evidence:
        name = _norm(ev.get("name"))
        if name and (name.lower() in text.lower() or text.lower() in name.lower()):
            return _norm(ev.get("id"))
    return None


def _text_mentions_person(text: str, person: dict) -> bool:
    name = _norm(person.get("name"))
    if not name:
        return False
    # Generic placeholders used by the public demo data should not produce
    # dozens of false mentions.
    if name.lower() in {"party as named in judgment", "unnamed", "unknown"}:
        return False
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in name.split()) + r"\b"
    return bool(re.search(pattern, text or "", re.I))


def evidence_person_ids(
    evidence: Sequence[dict],
    people: Sequence[dict],
    document_text: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """Return evidence-id -> person-ids using explicit links + name mentions."""
    document_text = document_text or {}
    out: Dict[str, List[str]] = {}
    for ev in evidence:
        eid = _norm(ev.get("id"))
        linked = _norm(ev.get("linkedTo"))
        ids: List[str] = []
        if linked and any(_norm(p.get("id")) == linked for p in people):
            ids.append(linked)
        text = document_text.get(eid, "") or _norm(ev.get("rawText"))
        for person in people:
            pid = _norm(person.get("id"))
            if pid and pid not in ids and _text_mentions_person(text, person):
                ids.append(pid)
        out[eid] = ids
    return out


def _party_lookup(people: Sequence[dict]) -> Dict[str, dict]:
    return {_norm(p.get("id")): p for p in people}


def party_entity_type(person: dict) -> str:
    """Map a case role to the NER display type used in the graph."""
    role = _norm(person.get("role")).lower()
    if role in {"defendant", "accused"}:
        return "DEFENDANT"
    if role in {"petitioner", "appellant", "complainant", "plaintiff"}:
        return "PETITIONER"
    if role in {"respondent"}:
        return "RESPONDENT"
    if role in {"witness"}:
        return "WITNESS"
    if role in {"judge", "justice"}:
        return "JUDGE"
    if role in {"lawyer", "advocate", "counsel"}:
        return "LAWYER"
    return "PERSON"


def inject_case_party_entities(
    case: dict,
    evidence: Sequence[dict],
    documents: Dict[str, dict],
    doc_entities: Dict[str, List[dict]],
    person_ids_by_evidence: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Add case people to each document's entity list.

    The public InLegalNER model has ``PETITIONER``/``RESPONDENT`` labels but
    never ``DEFENDANT`` or an explicit evidence-to-person edge.  The case file
    already knows the parties, so we merge them in.  In a one-defendant case an
    unlinked exhibit is treated as belonging to that defendant and marked
    ``inferred``; in a multi-defendant case we only add explicit links and
    name mentions.
    """
    people = case.get("people") or []
    if not people:
        return person_ids_by_evidence
    lookup = _party_lookup(people)
    defendants = [p for p in people if _norm(p.get("role")).lower() in {"defendant", "accused"}]
    single_defendant = len(defendants) == 1

    for ev in evidence:
        eid = _norm(ev.get("id"))
        doc = documents.get(eid) or {}
        raw = _norm(doc.get("rawText"))
        entities = doc_entities.setdefault(eid, [])
        existing = {(_norm(e.get("text")).lower(), _norm(e.get("type"))) for e in entities}
        pids = list(person_ids_by_evidence.get(eid, []))
        if single_defendant and _norm(defendants[0].get("id")) not in pids:
            pids.append(_norm(defendants[0].get("id")))
            person_ids_by_evidence[eid] = pids
        for pid in pids:
            person = lookup.get(pid)
            if not person:
                continue
            name = _norm(person.get("name"))
            etype = party_entity_type(person)
            if not name or (name.lower(), etype) in existing:
                continue
            start = raw.lower().find(name.lower()) if raw else -1
            inferred = start < 0
            entities.append({
                "text": name,
                "type": etype,
                "start": max(0, start),
                "end": max(0, start) + len(name) if start >= 0 else 0,
                "confidence": 1.0 if not inferred else 0.88,
                "source": "case_party_inferred" if inferred else "case_party",
                "inferred": inferred,
                "personId": pid,
            })
            existing.add((name.lower(), etype))

    for eid in doc_entities:
        doc_entities[eid].sort(key=lambda e: (int(e.get("start", 0) or 0), str(e.get("text", ""))))
    return person_ids_by_evidence


def _parties_from_evidence_ids(
    evidence_ids: Iterable[str],
    evidence_by_id: Dict[str, dict],
    people: Sequence[dict],
    person_ids_by_evidence: Dict[str, List[str]],
    claims: str = "",
) -> List[str]:
    """Return person ids involved in the given exhibits/claims."""
    lookup = _party_lookup(people)
    found: List[str] = []
    for eid in evidence_ids:
        for pid in person_ids_by_evidence.get(eid, []):
            if pid in lookup and pid not in found:
                found.append(pid)
    for person in people:
        pid = _norm(person.get("id"))
        if pid not in found and _text_mentions_person(claims, person):
            found.append(pid)
    return found


def _finding_ids(conflict: dict) -> List[str]:
    out = []
    for side in ("left", "right"):
        value = _norm((conflict.get(side) or {}).get("evidenceId"))
        if value:
            out.append(value)
    return out


def annotate_findings(
    case: dict,
    evidence: Sequence[dict],
    documents: Dict[str, dict],
    conflicts: List[dict],
    anomalies: List[dict],
    person_ids_by_evidence: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[str]]:
    """Attach ``partyIds``/``parties`` to conflicts and anomalies in place.

    Returns evidence-id -> person-ids for reuse by the party summary/graph.
    """
    people = case.get("people") or []
    evidence_by_id = {_norm(ev.get("id")): ev for ev in evidence}
    text_by_id = {eid: _norm((doc or {}).get("rawText")) for eid, doc in documents.items()}
    person_ids = person_ids_by_evidence or evidence_person_ids(evidence, people, text_by_id)
    lookup = _party_lookup(people)

    for conflict in conflicts:
        ids = _finding_ids(conflict)
        if not ids:
            ids = [x for x in (
                _source_evidence_id((conflict.get("left") or {}).get("source"), evidence),
                _source_evidence_id((conflict.get("right") or {}).get("source"), evidence),
            ) if x]
        claims = " ".join([
            _norm((conflict.get("left") or {}).get("claim")),
            _norm((conflict.get("right") or {}).get("claim")),
            _norm(conflict.get("explanation")),
        ])
        party_ids = _parties_from_evidence_ids(ids, evidence_by_id, people, person_ids, claims)
        conflict["partyIds"] = party_ids
        conflict["parties"] = [lookup[pid].get("name") for pid in party_ids if pid in lookup]

    for anomaly in anomalies:
        ids: List[str] = []
        for value in (anomaly.get("evidenceIds") or []):
            if _norm(value):
                ids.append(_norm(value))
        for source in (anomaly.get("sources") or []):
            found = _source_evidence_id(source, evidence)
            if found:
                ids.append(found)
        ids = list(dict.fromkeys(ids))
        claims = " ".join([
            _norm(anomaly.get("entity")),
            _norm(anomaly.get("explanation")),
            " ".join(_norm(s) for s in (anomaly.get("sources") or [])),
        ])
        party_ids = _parties_from_evidence_ids(ids, evidence_by_id, people, person_ids, claims)
        anomaly["evidenceIds"] = ids
        anomaly["partyIds"] = party_ids
        anomaly["parties"] = [lookup[pid].get("name") for pid in party_ids if pid in lookup]

    return person_ids


# ---------------------------------------------------------------------------
# Missing-information rules
# ---------------------------------------------------------------------------
_JUDGMENT_RE = re.compile(
    r"\b(judgment|judgement|order|decree|final order|appellate|revision|petition)\b", re.I
)
_MISSING_HINTS_RE = re.compile(
    r"\b(unaccounted|not accounted|missing record|no corroborat|no record|gap of|"
    r"whereabouts unknown|not produced|not traceable|untraced|absent from the record)\b",
    re.I,
)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def missing_information(
    case: dict,
    evidence: Sequence[dict],
    documents: Dict[str, dict],
    doc_entities: Dict[str, List[dict]],
    person_ids_by_evidence: Dict[str, List[str]],
) -> List[dict]:
    """Rule-based detection of missing documents, metadata and coverage.

    The rules are deliberately conservative.  They produce an explainable
    finding with the exact missing item rather than a vague low-confidence
    anomaly, and cap their output so a sparse case is readable.
    """
    out: List[dict] = []
    if not evidence:
        return out
    people = case.get("people") or []
    lookup = _party_lookup(people)
    defendants = [p for p in people if _norm(p.get("role")).lower() in {"defendant", "accused"}]

    # 1. Files that yielded no text at all (scanned PDF without OCR, audio
    #    without a transcript, unsupported format).
    for ev in evidence:
        eid = _norm(ev.get("id"))
        raw = _norm((documents.get(eid) or {}).get("rawText"))
        if raw:
            continue
        extraction = ev.get("extraction") or {}
        reason = extraction.get("error") or "No extractable text was found."
        out.append({
            "id": f"mi{len(out) + 1}",
            "entity": _evidence_source(ev),
            "type": "missing_information",
            "category": "Missing information",
            "subtype": "unreadable_evidence",
            "score": 0.92,
            "label": "High",
            "model": "Rule engine",
            "explanation": (
                f"No text could be extracted from this exhibit, so the models could not "
                f"analyse its contents. {reason}"
            ),
            "sources": [_evidence_source(ev)],
            "evidenceIds": [eid],
            "missing": ["extractable text / OCR transcript"],
        })
        if len(out) >= 3:
            break

    # 2. Per-defendant evidence coverage.  Only meaningful for a genuinely
    #    multi-defendant case: in a one-defendant case every exhibit belongs to
    #    the same case file, so an explicit link is not required.
    if len(defendants) >= 2:
        for person in defendants:
            pid = _norm(person.get("id"))
            linked = [
                _norm(ev.get("id")) for ev in evidence
                if pid in (person_ids_by_evidence.get(_norm(ev.get("id")), []))
            ]
            if linked:
                continue
            out.append({
                "id": f"mi{len(out) + 1}",
                "entity": _person_label(person),
                "type": "missing_information",
                "category": "Missing information",
                "subtype": "missing_defendant_evidence",
                "score": 0.84,
                "label": "High",
                "model": "Rule engine",
                "explanation": (
                    f"{person.get('name')} is a named defendant, but no exhibit is linked "
                    "to or mentions this person. Link at least one statement, exhibit or "
                    "witness note to this defendant before relying on the case graph."
                ),
                "sources": [_evidence_source(ev) for ev in evidence[:4]],
                "evidenceIds": [],
                "partyIds": [pid],
                "parties": [person.get("name")],
                "missing": ["evidence / statement linked to this defendant"],
            })
            if len(out) >= 5:
                break

    # 3. Judgment-like documents that are missing core legal metadata.  A
    #    single missing type is normal in an extract; two or more suggests an
    #    incomplete scan or page loss.
    judgment_docs = 0
    for ev in evidence:
        eid = _norm(ev.get("id"))
        doc = documents.get(eid) or {}
        raw = _norm(doc.get("rawText"))
        if not raw or not _JUDGMENT_RE.search(raw) and not _JUDGMENT_RE.search(_evidence_source(ev)):
            continue
        judgment_docs += 1
        if judgment_docs > 3:
            break
        types = {_norm(e.get("type")).upper() for e in (doc_entities.get(eid) or [])}
        party_types = {"PETITIONER", "RESPONDENT", "DEFENDANT", "PERSON", "WITNESS"}
        checks = {
            "court": "COURT" in types,
            "date": "DATE" in types,
            "party": bool(types & party_types),
        }
        absent = [name for name, present in checks.items() if not present]
        if len(absent) < 2:
            continue
        out.append({
            "id": f"mi{len(out) + 1}",
            "entity": _evidence_source(ev),
            "type": "missing_information",
            "category": "Missing information",
            "subtype": "incomplete_legal_metadata",
            "score": 0.72,
            "label": "Medium",
            "model": "Rule engine",
            "explanation": (
                "This judgment/order is missing core legal metadata ("
                + ", ".join(absent)
                + "). Check whether pages or header fields were lost during scanning."
            ),
            "sources": [_evidence_source(ev)],
            "evidenceIds": [eid],
            "missing": absent,
        })
        if len(out) >= 6:
            break

    # 4. Explicit timeline gaps.  We only flag text that actually says a
    #    period is unaccounted/missing; a sparse date set is not an anomaly.
    for ev in evidence:
        eid = _norm(ev.get("id"))
        raw = _norm((documents.get(eid) or {}).get("rawText"))
        if not raw or not _MISSING_HINTS_RE.search(raw):
            continue
        times = _TIME_RE.findall(raw)
        if len(times) < 2:
            continue
        out.append({
            "id": f"mi{len(out) + 1}",
            "entity": f"Timeline window {times[0]}–{times[-1]}",
            "type": "missing_information",
            "category": "Missing information",
            "subtype": "timeline_gap",
            "score": 0.68,
            "label": "Medium",
            "model": "Rule engine",
            "explanation": (
                f"The document refers to an unaccounted period between {times[0]} and "
                f"{times[-1]}, but no corroborating record covers that window."
            ),
            "sources": [_evidence_source(ev)],
            "evidenceIds": [eid],
            "missing": [f"record covering {times[0]}–{times[-1]}"],
        })
        if len(out) >= 7:
            break

    # 5. Case-level metadata (court / sections) — advisory only, and never for
    #    records that intentionally say "see judgment".
    court = _norm(case.get("court"))
    sections = _norm(case.get("ipcSections"))
    if (not court or court == "—") or (not sections or sections == "—"):
        missing = []
        if not court or court == "—":
            missing.append("court")
        if not sections or sections == "—":
            missing.append("IPC/BNS sections")
        out.append({
            "id": f"mi{len(out) + 1}",
            "entity": "Case metadata",
            "type": "missing_information",
            "category": "Missing information",
            "subtype": "missing_case_metadata",
            "score": 0.55,
            "label": "Low",
            "model": "Rule engine",
            "explanation": "The case record is missing " + " and ".join(missing) + ".",
            "sources": [],
            "evidenceIds": [],
            "missing": missing,
        })

    return out


# ---------------------------------------------------------------------------
# Manipulation / integrity rules (complements XGBoost)
# ---------------------------------------------------------------------------
_REDACTION_RE = re.compile(r"\[redacted\]|\bredacted\b|\bxxx+\b|_{4,}", re.I)
_DUPLICATE_LINE_RE = re.compile(r"^\s*(\S.{8,})\s*$", re.M)


def integrity_signals(text: str) -> Dict[str, Any]:
    """Return transparent document-integrity signals with a score 0..1."""
    text = text or ""
    stripped = text.rstrip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    redactions = len(_REDACTION_RE.findall(text))
    duplicate_lines = 0
    seen = set()
    for line in lines:
        key = line.lower()
        if key in seen and len(key) > 12:
            duplicate_lines += 1
        seen.add(key)
    abrupt_end = bool(stripped) and stripped[-1] not in '.!?"\')]'
    placeholder = len(re.findall(r"_{3,}", text))
    partial_ocr = bool(re.search(r"OCRed? partial|OCR partial|partial OCR|illegible", text, re.I))
    score = 0.0
    reasons = []
    if redactions >= 2:
        score += min(0.45, 0.12 * redactions)
        reasons.append(f"{redactions} redaction/placeholder spans")
    if duplicate_lines >= 2:
        score += min(0.35, 0.08 * duplicate_lines)
        reasons.append(f"{duplicate_lines} repeated lines (possible duplicated scan)")
    if abrupt_end and len(stripped) > 120:
        score += 0.22
        reasons.append("text ends abruptly (possible missing page)")
    if placeholder >= 3:
        score += 0.2
        reasons.append("underscore placeholders in a legal field")
    if partial_ocr:
        score += 0.2
        reasons.append("the scan itself is marked as partially OCRed/illegible")
    return {
        "score": round(min(0.98, score), 3),
        "reasons": reasons,
        "redactions": redactions,
        "duplicateLines": duplicate_lines,
        "abruptEnd": abrupt_end,
        "partialOcr": partial_ocr,
    }


def manipulation_anomalies(
    evidence: Sequence[dict],
    documents: Dict[str, dict],
    existing: Sequence[dict],
    limit: int = 4,
) -> List[dict]:
    """Create rule-based manipulation findings, merging XGBoost duplicates.

    ``existing`` is mutated: if XGBoost already flagged the same exhibit, its
    explanation is enriched with the concrete rule signals instead of adding a
    second card for the same document.
    """
    out: List[dict] = []
    by_source = {_norm(a.get("entity")): a for a in existing}
    for ev in evidence:
        eid = _norm(ev.get("id"))
        raw = _norm((documents.get(eid) or {}).get("rawText"))
        if not raw:
            continue
        signals = integrity_signals(raw)
        if signals["score"] < 0.3 or not signals["reasons"]:
            continue
        source = _evidence_source(ev)
        existing_item = by_source.get(source)
        if existing_item is not None:
            existing_item.setdefault("evidenceIds", []).append(eid)
            existing_item["ruleSignals"] = signals["reasons"]
            existing_item["explanation"] = (
                str(existing_item.get("explanation") or "").rstrip()
                + " Rule checks also found: " + "; ".join(signals["reasons"]) + "."
            )
            continue
        out.append({
            "id": f"a{len(existing) + len(out) + 1}",
            "entity": source,
            "type": "manipulation",
            "category": "Possible manipulation",
            "subtype": "document_integrity",
            "score": signals["score"],
            "label": "High" if signals["score"] >= 0.75 else "Medium" if signals["score"] >= 0.5 else "Low",
            "model": "Rule engine",
            "explanation": (
                "Integrity rules found evidence-handling problems: "
                + "; ".join(signals["reasons"])
                + ". Check the original against the court record."
            ),
            "sources": [source],
            "evidenceIds": [eid],
            "ruleSignals": signals["reasons"],
        })
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Party / defendant summary
# ---------------------------------------------------------------------------
def party_summary(
    case: dict,
    evidence: Sequence[dict],
    person_ids_by_evidence: Dict[str, List[str]],
    conflicts: Sequence[dict],
    anomalies: Sequence[dict],
    documents: Dict[str, dict],
) -> List[dict]:
    """Build a per-person evidence/conflict/anomaly summary for the UI/report."""
    people = case.get("people") or []
    if not people:
        return []
    evidence_by_id = {_norm(ev.get("id")): ev for ev in evidence}
    summary: List[dict] = []
    defendants = [p for p in people if _norm(p.get("role")).lower() in {"defendant", "accused"}]
    single_defendant = len(defendants) == 1

    for person in people:
        pid = _norm(person.get("id"))
        evidence_ids: List[str] = []
        inferred: List[str] = []
        for ev in evidence:
            eid = _norm(ev.get("id"))
            text = (documents.get(eid) or {}).get("rawText") or ev.get("rawText") or ""
            explicit = _norm(ev.get("linkedTo")) == pid or _text_mentions_person(text, person)
            if explicit:
                evidence_ids.append(eid)
            elif single_defendant and pid == _norm(defendants[0].get("id")):
                # In a one-defendant case an unlinked exhibit still belongs to
                # that case file; mark it as inferred rather than pretending it
                # was manually linked.
                evidence_ids.append(eid)
                inferred.append(eid)

        party_conflicts = [c for c in conflicts if pid in (c.get("partyIds") or [])]
        party_anomalies = [a for a in anomalies if pid in (a.get("partyIds") or [])]
        missing = []
        for item in party_anomalies:
            if item.get("type") == "missing_information":
                missing.append(_short(item.get("entity"), 80))
        summary.append({
            "id": pid,
            "name": person.get("name"),
            "role": person.get("role"),
            "label": _person_label(person),
            "evidenceIds": list(dict.fromkeys(evidence_ids)),
            "inferredEvidenceIds": inferred,
            "evidenceCount": len(set(evidence_ids)),
            "linkedEvidenceCount": len(set(evidence_ids) - set(inferred)),
            "conflicts": len(party_conflicts),
            "anomalies": len(party_anomalies),
            "missing": list(dict.fromkeys(missing)),
            "documents": [
                {
                    "id": eid,
                    "name": _evidence_source(evidence_by_id.get(eid, {})),
                    "kind": (documents.get(eid) or {}).get("kind", "text"),
                    "hasText": bool(_norm((documents.get(eid) or {}).get("rawText"))),
                    "inferred": eid in inferred,
                }
                for eid in list(dict.fromkeys(evidence_ids))
            ],
        })
    return summary
