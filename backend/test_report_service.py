import unittest
from services.report_service import build_explainable_report, render_report_pdf


class TestReportService(unittest.TestCase):
    def setUp(self):
        self.sample_case = {
            "id": "crl-mc-1622-2006",
            "caseNumber": "Crl MC 1622/2006",
            "court": "High Court of Kerala",
            "ipcSections": "302, 34, 120B",
            "status": "active",
            "people": [
                {"name": "Anil Kumar", "role": "Defendant"},
                {"name": "Suresh Nair", "role": "Witness"},
            ],
            "reviews": {
                "c1": {"status": "confirmed", "reviewer": "advocate@mecr.internal"},
                "a1": {"status": "pending", "reviewer": None},
            }
        }
        self.evidence = [
            {
                "id": "EV-100",
                "name": "FIR_01.pdf",
                "kind": "pdf",
                "rawText": "First Information Report filed at 10:30 PM regarding incident at MG Road.",
                "charCount": 72,
                "status": "processed",
            },
            {
                "id": "EV-101",
                "name": "Witness_Statement_Suresh.txt",
                "kind": "text",
                "rawText": "Witness Suresh states incident occurred at 11:45 PM near the junction.",
                "charCount": 68,
                "status": "processed",
            },
            {
                "id": "EV-102",
                "name": "Scanned_Medical_Memo.pdf",
                "kind": "pdf",
                "rawText": "",
                "charCount": 0,
                "extraction": {"ocrRequired": True},
            }
        ]
        self.documents = {
            "EV-100": {"source": "FIR_01.pdf", "rawText": self.evidence[0]["rawText"]},
            "EV-101": {"source": "Witness_Statement_Suresh.txt", "rawText": self.evidence[1]["rawText"]},
            "EV-102": {"source": "Scanned_Medical_Memo.pdf", "rawText": ""},
        }
        self.doc_entities = {
            "EV-100": [{"text": "MG Road", "type": "LOCATION"}, {"text": "10:30 PM", "type": "TIME"}],
            "EV-101": [{"text": "Suresh", "type": "PERSON"}, {"text": "11:45 PM", "type": "TIME"}],
            "EV-102": [],
        }
        self.all_entities = [
            {"text": "MG Road", "type": "LOCATION"},
            {"text": "10:30 PM", "type": "TIME"},
            {"text": "Suresh", "type": "PERSON"},
            {"text": "11:45 PM", "type": "TIME"},
        ]
        self.conflicts = [
            {
                "id": "c1",
                "type": "Contradiction detected",
                "category": "Contradiction",
                "subtype": "timeline_conflict",
                "severity": "material",
                "model": "LegalBERT",
                "confidence": 0.88,
                "left": {"source": "FIR_01.pdf", "evidenceId": "EV-100", "claim": "Incident at 10:30 PM"},
                "right": {"source": "Witness_Statement_Suresh.txt", "evidenceId": "EV-101", "claim": "Incident at 11:45 PM"},
                "explanation": "Significant 75-minute discrepancy between FIR time and eyewitness testimony.",
            }
        ]
        self.anomalies = [
            {
                "id": "a1",
                "entity": "FIR_01.pdf",
                "type": "manipulation",
                "category": "Possible manipulation",
                "subtype": "document_integrity",
                "score": 0.72,
                "label": "Medium",
                "model": "XGBoost",
                "explanation": "Timing anomaly flagged based on filing gap.",
                "sources": ["FIR_01.pdf"],
                "evidenceIds": ["EV-100"],
            },
            {
                "id": "a2",
                "entity": "Post-mortem Report",
                "type": "missing_information",
                "category": "Missing information",
                "subtype": "required_evidence",
                "label": "High",
                "model": "Rule engine",
                "explanation": "No post-mortem report found for Section 302 homicide charge.",
                "sources": [],
                "evidenceIds": [],
            }
        ]
        self.party_rows = [
            {
                "name": "Anil Kumar",
                "role": "Defendant",
                "evidenceIds": ["EV-100"],
                "conflicts": 1,
                "anomalies": 0,
            }
        ]
        self.graph = {
            "nodes": [{"id": "n1", "type": "PERSON"}, {"id": "n2", "type": "EVIDENCE"}],
            "edges": [{"source": "n1", "target": "n2", "label": "MENTIONED_IN"}],
        }
        self.active_models = {
            "ner": {"dataset": "InLegalNER", "f1": 0.89},
            "contradiction": {"dataset": "LegalBERT NLI", "accuracy": 0.85},
            "anomaly": {"dataset": "XGBoost Anomaly", "roc_auc": 0.88},
        }
        self.review_summary = {"total": 2, "confirmed": 1, "dismissed": 0, "pending": 1}

    def test_build_explainable_report_structure(self):
        report = build_explainable_report(
            case=self.sample_case,
            evidence=self.evidence,
            documents=self.documents,
            doc_entities=self.doc_entities,
            all_entities=self.all_entities,
            conflicts=self.conflicts,
            anomalies=self.anomalies,
            party_rows=self.party_rows,
            graph=self.graph,
            active_models=self.active_models,
            review_summary=self.review_summary,
            reviews=self.sample_case["reviews"],
            pipeline_stages=["ner", "contradiction", "anomaly", "reporting"],
        )

        self.assertIn("title", report)
        self.assertIn("case", report)
        self.assertEqual(report["case"]["caseNumber"], "Crl MC 1622/2006")
        self.assertEqual(report["stats"]["documents"], 3)
        self.assertEqual(report["stats"]["conflicts"], 1)
        self.assertEqual(report["stats"]["anomalies"], 2)
        self.assertEqual(report["stats"]["unreadableEvidence"], 1)

        # Check evidence inventory
        self.assertEqual(len(report["evidenceInventory"]), 3)
        unreadable = report["unreadableEvidence"]
        self.assertEqual(len(unreadable), 1)
        self.assertEqual(unreadable[0]["evidenceId"], "EV-102")
        self.assertTrue(unreadable[0]["ocrRequired"])

        # Check grounding in conflicts
        conf = report["conflicts"][0]
        self.assertEqual(conf["id"], "c1")
        self.assertEqual(conf["grounding"]["statementA"]["source"], "FIR_01.pdf")
        self.assertEqual(conf["grounding"]["statementB"]["source"], "Witness_Statement_Suresh.txt")
        self.assertEqual(conf["review"]["status"], "confirmed")

        # Check defendant summary
        self.assertEqual(len(report["defendantSummary"]), 1)
        self.assertEqual(report["defendantSummary"][0]["name"], "Anil Kumar")

        # Check legal disclaimer exists and is clear
        self.assertIn("legalDisclaimer", report)
        self.assertIn("decision-support", report["legalDisclaimer"])

    def test_render_report_pdf(self):
        report = build_explainable_report(
            case=self.sample_case,
            evidence=self.evidence,
            documents=self.documents,
            doc_entities=self.doc_entities,
            all_entities=self.all_entities,
            conflicts=self.conflicts,
            anomalies=self.anomalies,
            party_rows=self.party_rows,
            graph=self.graph,
            active_models=self.active_models,
            review_summary=self.review_summary,
            reviews=self.sample_case["reviews"],
            pipeline_stages=["ner", "contradiction", "anomaly", "reporting"],
        )

        pdf_bytes = render_report_pdf(report)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(len(pdf_bytes) > 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_empty_analysis_handling(self):
        empty_case = {"id": "empty-case", "caseNumber": "Empty 001/2026"}
        report = build_explainable_report(
            case=empty_case,
            evidence=[],
            documents={},
            doc_entities={},
            all_entities=[],
            conflicts=[],
            anomalies=[],
            party_rows=[],
            graph={"nodes": [], "edges": []},
            active_models={},
            review_summary={"total": 0, "confirmed": 0, "dismissed": 0, "pending": 0},
            reviews={},
            pipeline_stages=[],
        )
        self.assertEqual(report["stats"]["documents"], 0)
        self.assertEqual(report["stats"]["conflicts"], 0)
        self.assertEqual(report["stats"]["anomalies"], 0)

        # PDF rendering should still cleanly produce a valid PDF
        pdf_bytes = render_report_pdf(report)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))


if __name__ == "__main__":
    unittest.main()
