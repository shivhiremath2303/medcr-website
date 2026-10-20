"""Explainable conflict-resolution report service for MECR.

Assembles grounded, verifiable legal conflict reports from actual analysis
pipeline outputs:
- Case and Evidence Summary
- Legal Entities (BiLSTM-CRF + Grounding)
- Contradictions and Conflicts (LegalBERT + Rules)
- Anomalies & Manipulation (XGBoost + Rules + Missing Info)
- Entity & Evidence Relationship Graph (R-GCN)
- Defendant-wise coverage and timeline analysis
- Traceable source evidence citations and offsets
- Human review status and recommendations
- Programmatic JSON, structured HTML, and downloadable PDF (ReportLab)
"""
from __future__ import annotations

import html
import io
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


def _clean_str(val: Any) -> str:
    return re.sub(r"\s+", " ", str(val or "")).strip()


def build_explainable_report(
    case: dict,
    evidence: Sequence[dict],
    documents: Dict[str, dict],
    doc_entities: Dict[str, List[dict]],
    all_entities: List[dict],
    conflicts: List[dict],
    anomalies: List[dict],
    party_rows: List[dict],
    graph: dict,
    active_models: Dict[str, dict],
    review_summary: dict,
    reviews: Dict[str, dict],
    pipeline_stages: Sequence[str],
    rgcn_result: Optional[dict] = None,
) -> dict:
    """Assemble a complete, evidence-grounded conflict-resolution report."""
    now_iso = datetime.now(timezone.utc).isoformat()
    case_num = case.get("caseNumber") or case.get("id") or "Unknown Case"
    court = case.get("court") or "Unspecified Court"
    ipc = case.get("ipcSections") or "Not specified"

    # Evidence inventory & readability
    evidence_inventory: List[dict] = []
    unreadable_evidence: List[dict] = []
    for ev in evidence:
        eid = ev.get("id", "")
        name = ev.get("name") or eid
        kind = ev.get("kind") or "unknown"
        char_count = int(ev.get("charCount") or len(ev.get("rawText") or ""))
        extraction = ev.get("extraction") or {}
        is_ocr = extraction.get("ocrRequired", False)
        status = "processed" if char_count > 0 else ("ocr_required" if is_ocr else "empty_or_unreadable")
        item = {
            "evidenceId": eid,
            "name": name,
            "kind": kind,
            "charCount": char_count,
            "status": status,
            "ocrRequired": is_ocr,
            "linkedTo": ev.get("linkedTo"),
            "uploadedAt": ev.get("uploadedAt"),
        }
        evidence_inventory.append(item)
        if status != "processed":
            unreadable_evidence.append(item)

    # Entity classification breakdown
    entities_by_type: Dict[str, List[dict]] = {}
    for ent in all_entities:
        t = ent.get("type") or "OTHER"
        entities_by_type.setdefault(t, []).append(ent)

    # Grounded conflict explanations
    grounded_conflicts: List[dict] = []
    for c in conflicts:
        cid = str(c.get("id", ""))
        rev = reviews.get(cid) or (c.get("review") or {})
        left = c.get("left") or {}
        right = c.get("right") or {}
        grounded_conflicts.append({
            "id": cid,
            "category": c.get("category", "Contradiction"),
            "subtype": c.get("subtype", "semantic_contradiction"),
            "type": c.get("type", "Contradiction detected"),
            "severity": c.get("severity", "material"),
            "model": c.get("model", "Rule engine"),
            "confidence": c.get("confidence"),
            "explanation": c.get("explanation", ""),
            "grounding": {
                "statementA": {
                    "source": left.get("source"),
                    "evidenceId": left.get("evidenceId"),
                    "claim": left.get("claim"),
                    "offset": left.get("offset"),
                },
                "statementB": {
                    "source": right.get("source"),
                    "evidenceId": right.get("evidenceId"),
                    "claim": right.get("claim"),
                    "offset": right.get("offset"),
                },
            },
            "recommendation": (
                f"Verify the factual basis in {left.get('source') or 'Source A'} and "
                f"{right.get('source') or 'Source B'}. Determine which document takes legal precedence."
            ),
            "review": {
                "status": rev.get("status", "pending"),
                "reviewer": rev.get("reviewer"),
                "reviewedAt": rev.get("reviewedAt"),
                "note": rev.get("note", ""),
            },
        })

    # Grounded anomaly explanations
    grounded_anomalies: List[dict] = []
    for a in anomalies:
        aid = str(a.get("id", ""))
        rev = reviews.get(aid) or (a.get("review") or {})
        grounded_anomalies.append({
            "id": aid,
            "category": a.get("category", "Possible manipulation"),
            "subtype": a.get("subtype", "document_integrity"),
            "type": a.get("type", "manipulation"),
            "entity": a.get("entity", ""),
            "label": a.get("label", "Medium"),
            "score": a.get("score"),
            "model": a.get("model", "Rule engine"),
            "explanation": a.get("explanation", ""),
            "sources": a.get("sources", []),
            "evidenceIds": a.get("evidenceIds", []),
            "review": {
                "status": rev.get("status", "pending"),
                "reviewer": rev.get("reviewer"),
                "reviewedAt": rev.get("reviewedAt"),
                "note": rev.get("note", ""),
            },
        })

    # Defendant-wise summary
    defendants_summary = [
        p for p in party_rows
        if str(p.get("role", "")).lower() in {"defendant", "accused"}
    ]

    missing_count = sum(1 for a in anomalies if a.get("category") == "Missing information")
    manipulation_count = sum(1 for a in anomalies if a.get("category") == "Possible manipulation")

    summary_text = (
        f"{len(evidence)} evidence documents analysed for Case {case_num}. "
        f"{len(all_entities)} distinct legal entities extracted. "
        f"{len(conflicts)} potential contradictions and {len(anomalies)} anomalies identified "
        f"({missing_count} missing-information signals, {manipulation_count} integrity/manipulation alerts). "
        f"{review_summary.get('confirmed', 0)} confirmed, {review_summary.get('dismissed', 0)} dismissed, "
        f"{review_summary.get('pending', 0)} pending human verification."
    )

    actionable_recommendations = [
        "Conduct thorough human legal review of all detected contradictions before relying on them in pleadings.",
        "Verify source document chain of custody and provenance for exhibits flagged with high integrity risk.",
        "Obtain clear scanned or original copies for any documents marked as unreadable or requiring OCR.",
        "Address identified missing information items (witness statements, post-mortem, FIR or seizure records).",
    ]

    report = {
        "title": f"Explainable Conflict-Resolution Report: {case_num}",
        "generatedAt": now_iso,
        "case": {
            "id": case.get("id"),
            "caseNumber": case_num,
            "court": court,
            "ipcSections": ipc,
            "status": case.get("status", "active"),
            "peopleCount": len(case.get("people") or []),
            "evidenceCount": len(evidence),
        },
        "executiveSummary": summary_text,
        "summary": summary_text,
        "stats": {
            "documents": len(evidence),
            "entities": len(all_entities),
            "conflicts": len(conflicts),
            "anomalies": len(anomalies),
            "missingInformation": missing_count,
            "possibleManipulation": manipulation_count,
            "parties": len(party_rows),
            "defendants": len(defendants_summary),
            "graph_nodes": len(graph.get("nodes", [])),
            "graph_edges": len(graph.get("edges", [])),
            "unreadableEvidence": len(unreadable_evidence),
        },
        "reviewSummary": review_summary,
        "evidenceInventory": evidence_inventory,
        "unreadableEvidence": unreadable_evidence,
        "extractedEntities": {
            "total": len(all_entities),
            "byType": {k: len(v) for k, v in entities_by_type.items()},
            "sample": all_entities[:100],
        },
        "entities": all_entities,
        "conflicts": grounded_conflicts,
        "anomalies": grounded_anomalies,
        "partySummary": party_rows,
        "parties": party_rows,
        "defendantSummary": defendants_summary,
        "relationshipGraphSummary": {
            "totalNodes": len(graph.get("nodes", [])),
            "totalEdges": len(graph.get("edges", [])),
            "rgcnStructuralIntegrity": rgcn_result,
            "nodeTypes": list({n.get("type", "node") for n in graph.get("nodes", [])}),
            "edgeLabels": list({e.get("label", "related") for e in graph.get("edges", [])}),
        },
        "models": active_models,
        "methodology": {
            "pipeline": list(pipeline_stages),
            "ner": "BiLSTM-CRF sequence tagger trained on InLegalNER with rule-based fallback and party grounding",
            "contradiction": "LegalBERT 3-class cross-document natural language inference (NLI)",
            "anomaly": "XGBoost document feature anomaly classifier + R-GCN structural integrity + rule engine",
            "grounding": "Strict cross-reference against uploaded text excerpts, entity offsets, and exhibit identifiers",
        },
        "limitations": [
            "This report is a decision-support aid and does not constitute a definitive legal finding or court ruling.",
            "All model-predicted contradictions and anomaly scores reflect statistical and semantic patterns and require independent human verification.",
            "Documents lacking OCR or high-quality text extraction cannot be fully verified for contradiction or completeness.",
            "Entity relationships reflect co-occurrence and explicit references in provided records.",
        ],
        "recommendations": actionable_recommendations,
        "legalDisclaimer": (
            "NOTICE: This document is an AI-generated decision-support analysis prepared by the Multi-Evidence "
            "& Multi-Defendant Conflict Resolver (MECR). The findings, confidence scores, and integrity indicators "
            "must be independently reviewed and verified by qualified legal counsel prior to formal filing or judicial presentation."
        ),
    }
    return report


def render_report_pdf(report: dict) -> bytes:
    """Render a clean, professional PDF conflict report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
    except ImportError:
        raise RuntimeError("ReportLab library is required to render PDF reports.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0F172A'),
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#94A3B8'),
        textTransform='uppercase',
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
    )
    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold',
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748B'),
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor('#475569'),
    )

    story: List[Any] = []

    case_info = report.get("case", {})
    case_no = case_info.get("caseNumber") or "Case"
    court = case_info.get("court") or ""
    ipc = case_info.get("ipcSections") or ""
    generated_at = report.get("generatedAt") or ""

    # Header
    story.append(Paragraph("MECR EXPLAINABLE CONFLICT-RESOLUTION REPORT", subtitle_style))
    story.append(Paragraph(html.escape(case_no), title_style))
    meta_line = f"<b>Court:</b> {html.escape(court)} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Sections:</b> IPC/BNS {html.escape(ipc)} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Generated:</b> {html.escape(generated_at[:19])} UTC"
    story.append(Paragraph(meta_line, body_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceBefore=4, spaceAfter=12))

    # Executive summary
    story.append(Paragraph("Executive Summary", h2_style))
    summary_text = report.get("executiveSummary") or report.get("summary") or ""
    story.append(Paragraph(html.escape(summary_text), body_style))
    story.append(Spacer(1, 10))

    # Metrics Overview Table
    stats = report.get("stats", {})
    metrics_data = [
        [
            Paragraph(f"<b>Evidence Files</b><br/>{stats.get('documents', 0)}", body_style),
            Paragraph(f"<b>Extracted Entities</b><br/>{stats.get('entities', 0)}", body_style),
            Paragraph(f"<b>Contradictions</b><br/>{stats.get('conflicts', 0)}", body_style),
            Paragraph(f"<b>Anomalies</b><br/>{stats.get('anomalies', 0)}", body_style),
            Paragraph(f"<b>Human Reviews</b><br/>{report.get('reviewSummary', {}).get('confirmed', 0)} conf / {report.get('reviewSummary', {}).get('pending', 0)} pend", body_style),
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[105, 105, 105, 105, 110])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 14))

    # Evidence Inventory Table
    inventory = report.get("evidenceInventory") or []
    if inventory:
        story.append(Paragraph("Scope of Analysis & Evidence Inventory", h2_style))
        inv_data = [["Evidence ID", "File / Source Name", "Kind", "Characters", "Status"]]
        for item in inventory[:15]:
            inv_data.append([
                html.escape(item.get("evidenceId", "")),
                html.escape(item.get("name", "")[:32]),
                html.escape(item.get("kind", "")),
                str(item.get("charCount", 0)),
                html.escape(item.get("status", "")),
            ])
        inv_table = Table(inv_data, colWidths=[70, 220, 60, 70, 110])
        inv_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ]))
        story.append(inv_table)
        story.append(Spacer(1, 14))

    # Conflicts Section
    conflicts = report.get("conflicts") or []
    story.append(Paragraph(f"Detected Contradictions & Conflicts ({len(conflicts)})", h2_style))
    if not conflicts:
        story.append(Paragraph("<i>No direct contradictions or cross-document conflicts detected.</i>", caption_style))
    else:
        for c in conflicts[:20]:
            cid = c.get("id", "")
            cat = c.get("category", "Contradiction")
            sev = c.get("severity", "material")
            model = c.get("model", "Model")
            conf = c.get("confidence")
            conf_str = f"({conf:.2f})" if isinstance(conf, (int, float)) else ""
            expl = c.get("explanation", "")
            g = c.get("grounding", {})
            st_a = g.get("statementA", {})
            st_b = g.get("statementB", {})
            rev = c.get("review", {})

            conflict_rows = [
                [
                    Paragraph(f"<b>[{cid}] {html.escape(cat)}</b> (Severity: {html.escape(sev)}, Model: {html.escape(model)} {conf_str})", bold_body),
                    Paragraph(f"Status: <b>{html.escape(rev.get('status', 'pending')).upper()}</b>", bold_body),
                ],
                [
                    Paragraph(f"<b>Source A ({html.escape(str(st_a.get('source') or ''))}):</b> {html.escape(str(st_a.get('claim') or ''))}", body_style),
                    Paragraph(f"<b>Source B ({html.escape(str(st_b.get('source') or ''))}):</b> {html.escape(str(st_b.get('claim') or ''))}", body_style),
                ],
                [
                    Paragraph(f"<b>Grounded Explanation:</b> {html.escape(expl)}", caption_style),
                    Paragraph(f"<b>Recommendation:</b> {html.escape(c.get('recommendation', ''))}", caption_style),
                ]
            ]
            c_table = Table(conflict_rows, colWidths=[265, 265])
            c_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FEF2F2')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#FCA5A5')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FEE2E2')),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(c_table)
            story.append(Spacer(1, 8))

    story.append(Spacer(1, 10))

    # Anomalies Section
    anomalies = report.get("anomalies") or []
    story.append(Paragraph(f"Anomalies & Missing Information ({len(anomalies)})", h2_style))
    if not anomalies:
        story.append(Paragraph("<i>No anomalies or missing information items detected.</i>", caption_style))
    else:
        for a in anomalies[:20]:
            aid = a.get("id", "")
            cat = a.get("category", "Anomaly")
            model = a.get("model", "Rule")
            label = a.get("label", "Medium")
            expl = a.get("explanation", "")
            sources = ", ".join(str(s) for s in a.get("sources", []))
            rev = a.get("review", {})

            a_rows = [
                [
                    Paragraph(f"<b>[{aid}] {html.escape(cat)}: {html.escape(a.get('entity', ''))}</b> (Risk: {html.escape(label)}, {html.escape(model)})", bold_body),
                    Paragraph(f"Status: <b>{html.escape(rev.get('status', 'pending')).upper()}</b>", bold_body),
                ],
                [
                    Paragraph(f"<b>Details & Grounding:</b> {html.escape(expl)}", body_style),
                    Paragraph(f"<b>Sources:</b> {html.escape(sources or 'Case records')}", caption_style),
                ]
            ]
            a_table = Table(a_rows, colWidths=[360, 170])
            a_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFFBEB')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#FDE68A')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FEF3C7')),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(a_table)
            story.append(Spacer(1, 6))

    story.append(Spacer(1, 10))

    # Defendant Summary Section
    defendants = report.get("defendantSummary") or []
    if defendants:
        story.append(Paragraph(f"Defendant-Wise Evidence Coverage ({len(defendants)})", h2_style))
        def_data = [["Defendant / Accused", "Role", "Linked Exhibits", "Contradictions", "Integrity Flags"]]
        for d in defendants:
            def_data.append([
                html.escape(d.get("name", "")),
                html.escape(d.get("role", "")),
                str(len(d.get("evidenceIds", []))),
                str(d.get("conflicts", 0)),
                str(d.get("anomalies", 0)),
            ])
        d_table = Table(def_data, colWidths=[140, 100, 100, 90, 100])
        d_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ]))
        story.append(d_table)
        story.append(Spacer(1, 12))

    # Recommendations & Limitations
    story.append(Paragraph("Recommendations & Limitations", h2_style))
    recs = report.get("recommendations", [])
    for idx, r in enumerate(recs, 1):
        story.append(Paragraph(f"{idx}. {html.escape(r)}", body_style))
    story.append(Spacer(1, 8))

    limits = report.get("limitations", [])
    for l in limits:
        story.append(Paragraph(f"• <i>{html.escape(l)}</i>", caption_style))
    story.append(Spacer(1, 14))

    # Legal Disclaimer Banner
    disclaimer_box = [
        [Paragraph(f"<b>LEGAL NOTICE &amp; DISCLAIMER:</b> {html.escape(report.get('legalDisclaimer', ''))}", disclaimer_style)]
    ]
    d_box = Table(disclaimer_box, colWidths=[530])
    d_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F1F5F9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94A3B8')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(d_box)

    doc.build(story)
    return buffer.getvalue()
