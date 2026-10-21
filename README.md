# MECR — Multi-Evidence Conflict Resolver

React + Vite + Tailwind front end for the MECR workbench, backed by the FastAPI
service in `backend/`. The analysis pipeline runs four trained legal NLP models
(BiLSTM-CRF NER, XGBoost anomaly detection, LegalBERT contradiction detection
and an R-GCN knowledge graph), and the app ships with an admin console and real
judgment-based demo cases.

## Run it

Requirements: **Node.js 18+** and **Python 3.11+**. The commands below are for
Windows PowerShell; on macOS/Linux use `source .venv/bin/activate` and
`.venv/bin/python` instead of the `.venv\Scripts\...` paths.

You need **two terminals**: one for the backend, one for the frontend.

### 1. First-time setup (from the repository root)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
npm install
```

`npm install` installs the frontend toolchain (React, Vite, Tailwind, React
Flow); `pip install` pulls in FastAPI, PyTorch, scikit-learn, reportlab and the
other model/reporting libraries.

### 2. Backend (terminal 1)

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --port 8000
```

Wait for `Application startup complete`. The four models are pre-warmed in the
background — the log prints `[warmup] models preloaded` when they are ready
(usually within a few seconds). The API is usable immediately; only the very
first analysis is slightly slower until warm-up finishes.

- API docs: `http://127.0.0.1:8000/docs`

### 3. Frontend (terminal 2, repository root)

```powershell
npm run dev
```

Visit **`http://127.0.0.1:5173`** and sign in with the seeded administrator:

```
admin@mecr.com / Admin@1234
```

Vite is configured with `host: '127.0.0.1'` and `strictPort: true`, so it fails
fast instead of silently switching ports.

### Ports already in use

If Uvicorn reports `[WinError 10048]`, something is already on port `8000`. If
`http://127.0.0.1:8000/docs` opens, the backend is already running — leave it
and just start the frontend. To restart it, press `Ctrl+C` in the backend
terminal first.

If port `5173` is busy, run the frontend on `5174` and visit
`http://127.0.0.1:5174`:

```powershell
npm run dev -- --port 5174
```

The frontend talks to `http://127.0.0.1:8000/api/v1` by default. To point it at
a different backend, create `.env` in the repository root:

```bash
VITE_API_URL=http://127.0.0.1:8001/api/v1
```

If you move the backend to another port, pass `--port 8001` to Uvicorn **and**
update `VITE_API_URL` to match.

> **Sessions are in memory.** Restarting the backend signs everyone out; the app
> detects the expired token and redirects to the login page automatically.

### Forgery screening (experimental, secondary)

The backend loads `backend/forgery_detector_bundle.joblib` through
`backend/inference.py` and exposes it only through authenticated API routes.
The model file is never sent to the browser.

- Standalone screen: **Forgery Screen** in the sidebar, backed by
  `POST /api/v1/forgery/check`.
- Uploaded exhibit screen: the **Screen** action on an image row in a case,
  backed by `GET /api/v1/evidence/{evidence_id}/forgery-check`.

Results are one of **Likely Authentic**, **Potentially Tampered**, or
**Review Required**. This is an experimental decision-support signal
(held-out accuracy 73.89%, ROC-AUC 0.7954 on the model's own test split), not
proof of authenticity or forgery. See `FORGERY_INTEGRATION.md` for the full
integration and test notes.

Seeded administrator: `admin@mecr.com` / `Admin@1234`. On a fresh install the
first registered account becomes an administrator; later accounts are regular
users and must verify their email (the development verification code is returned
by the API). The `/admin` page is only linked and rendered for administrators.
Administrators can create accounts, change roles, reset passwords and delete
users from **User Management**.

### Per-user case privacy

Case data is private to the signed-in account. The case list, case details,
evidence, analysis results, knowledge graph and report endpoints all check the
case owner, so another user's case (or evidence ID) returns `404` rather than
revealing that it exists. The seeded demo cases belong to the administrator
account; a new regular account starts with an empty dashboard and only sees
cases it creates. Deleting an account in **User Management** also deletes that
account's cases and invalidates its sessions.

## Where each requirement lives

| # | Requirement | File |
|---|---|---|
| 1 | Project setup | this whole repo (Vite + Tailwind) |
| 2 | Login / Register | `src/pages/Login.jsx`, `Register.jsx` |
| 3 | Dashboard | `src/pages/Dashboard.jsx` |
| 4 | Case creation form | `src/pages/CaseCreate.jsx` |
| 5 | Multi-modal upload | `src/pages/CaseUpload.jsx` |
| 6 | Entity tagging dropdown | `src/pages/CaseUpload.jsx` (link-to-person select) |
| 7 | AI processing status | `src/pages/CaseUpload.jsx` (spinner + stage text) |
| 8 | Conflict matrix | `src/pages/CaseAnalysis.jsx` → `ConflictMatrix` |
| 9 | Credibility scoring | `src/pages/CaseAnalysis.jsx` → `Credibility` |
| 10 | Split-screen viewer | `src/pages/CaseAnalysis.jsx` → `SplitViewer` |
| 11 | Knowledge graph | `src/pages/CaseAnalysis.jsx` → `KnowledgeGraph` (React Flow) |
| 12 | Spatial anomaly overlay | `src/pages/CaseAnalysis.jsx` → `SplitViewer` (bounding boxes) |
| 13 | Report export | `src/pages/CaseAnalysis.jsx` → `ReportExport` |

## Wiring to your FastAPI backend

1. Set `VITE_API_URL` in a `.env` file (defaults to `http://localhost:8000/api/v1`).
2. In `src/lib/api.js`, each function has the real `fetch()` call commented
   above the mock implementation — uncomment and delete the mock body.
3. Auth token is stored in `localStorage` under `mecr_token` and already
   read by `authHeaders()` — attach it to your real fetch calls' headers.
4. `getGraph()` should return `{ nodes, edges }` in React Flow's shape —
   see `MOCK_GRAPH` in `api.js` for the exact structure your Neo4j layer
   needs to be mapped into.
5. Anomaly boxes in the split-screen viewer expect `{ x, y, w, h, note }`
   as **percentages** of the image, so they scale with the document regardless
   of resolution — have your forgery-detection endpoint return coordinates
   in that form (or normalize them client-side).

## Known gaps

- Audio exhibits are transcribed for analysis, but there is no dedicated
  waveform/transcript viewer in the UI yet.
- Forgery screening is an experimental decision-support signal (see above), not
  proof of authenticity or tampering.

## Training the four models on large public legal datasets

Two corpora power the models. They live as `.jsonl` under
`backend/services/dataset_store/`, so they behave exactly like an uploaded ZIP.

| Dataset | Source | Size | Used by |
|---|---|---|---|
| `inlegalner_large` | InLegalNER (Indian legal NER corpus) with the **official train/dev/test splits** | 10,995 train + 1,074 dev + 4,501 test paragraphs, 14 entity types | BiLSTM-CRF, XGBoost, R-GCN |
| `legal_claims_large` | Legal claim pairs derived from InLegalNER judgments | 18,000 train + 2,100 test balanced pairs (consistent / contradiction / neutral) | LegalBERT |
| `legal_ner_large` | [AjayMukundS/Indian_Legal_NER_Dataset](https://huggingface.co/datasets/AjayMukundS/Indian_Legal_NER_Dataset) (MIT) | 9,642 annotated Indian judgments, 14 entity types | legacy / alternative corpus |
| `legal_nli_large` | [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli) (CC-BY-SA-4.0) | 15,000 train + 3,000 test balanced pairs | legacy / alternative corpus |

`inlegalner_large` and `legal_claims_large` are built from the local InLegalNER
release with:

```bash
cd backend
python scripts/build_from_local.py
```

The two downloadable corpora can also be fetched from Hugging Face
(Admin -> Datasets -> *Download Large Public Datasets*):

```bash
python scripts/build_datasets.py          # download + prepare both public corpora
```

### Command line

```bash
cd backend
python scripts/train_all.py               # train all four models on the local corpora
python scripts/activate_best.py           # deploy the best checkpoint of each
```

Train a single model:

```bash
python scripts/train_all.py --models rgcn --epochs 40
python scripts/train_all.py --models legalbert --epochs 1
```

BiLSTM-CRF training now samples up to three windows per judgment (head, densest
entity region, tail) and evaluates on the **official held-out test split**, so
the reported entity F1 is measured on documents the model has never seen.

### What each model learns

- **BiLSTM-CRF** - token-level legal NER (COURT, JUDGE, PETITIONER, RESPONDENT,
  WITNESS, LAWYER, DATE, PLACE, EVIDENCE, STATUTE, PROVISION, PRECEDENT, CASE_NUMBER, ORG).
- **XGBoost** - document-integrity anomalies. Trained on real judgments plus realistic
  damage (missing pages, redactions, lost entity metadata, duplicated scans, boilerplate headers).
- **LegalBERT** - 3-class contradiction detection (consistent / contradiction / neutral),
  fine-tuned from `nlpaueb/legal-bert-small-uncased`.
- **R-GCN** - graph-level integrity classification over the entity co-occurrence graph
  (mean + max pooling, 5 relation types, 27 node features).

Feature engineering is shared by training and inference in
`backend/services/features.py`, so the deployed models see exactly the inputs
they were trained on.

### Activating a model

Admin -> **Deployed Models** -> **Activate**. Activation now:

- only replaces the model of the same type (the other three stay deployed),
- clears the in-memory analysis cache so the new weights take effect immediately,
- reloads the model on the next request (no backend restart needed).

### Knowledge graph tab

The graph is a layered, interactive view rather than a dense entity mesh:

- **Documents** sit in the left column; each card shows the exhibit name, document
  type, linked-entity count and a short excerpt.
- Entities are grouped into three semantic columns: **parties & people**,
  **court & law**, and **facts & evidence**, each colour-coded by entity type.
- Thin grey **mention links** connect a document to the entities it contains
  (toggleable with *Entity links*).
- **Conflict links** are red dashed arrows between the exhibits that disagree;
  the conflict type (e.g. *Timeline*) is listed directly below the graph.
- Page scrolling over the canvas is never hijacked: zoom with the controls,
  minimap or trackpad pinch, and pan by dragging the background.
- Click any node to dim everything unrelated and trace its neighbours; the
  minimap, zoom controls and legend keep large cases navigable.
- The header shows the R-GCN integrity verdict and the document/entity/conflict
  counts.
- A **Models run** strip above the tabs summarises each engine's output for the
  case (NER entities, LegalBERT contradictions, XGBoost anomalies, R-GCN
  integrity), so it is clear which models were active.

## Measured results (held-out test splits)

| Model | Before (10-doc sample) | After (large public corpora) | Metric |
|---|---|---|---|
| BiLSTM-CRF NER | 26.2% token accuracy | **95.1% token accuracy**, entity F1 **0.715** | 10,995 train judgments, official 4,501-doc test split |
| XGBoost anomaly | 50.0% (single class) | **91.5% accuracy**, AUC **0.966**, F1 **0.912** | 21,990 balanced samples (real judgments + simulated damage) |
| R-GCN graph integrity | 100% (10-doc sample) | **78.3% accuracy**, F1 **0.867** | 6,937 entity-graph samples |
| LegalBERT contradiction | 67.6% on synthetic pairs | **98.0% accuracy**, macro F1 **0.980** | 6,000 train / 1,000 held-out balanced NLI pairs |

Numbers come from `scripts/train_all.py` and are stored in
`backend/services/models/models_metadata.json`, which the admin panel displays.

> Note: LegalBERT inputs are normalised before inference (meta prefixes such as
> "Witness statement recorded on oath:" are stripped) so the model compares the
> underlying facts rather than the quotation wrapper. Contradictions are also
> cross-checked by a structural entity-consistency pass, which catches changed
> dates, years and places even when the NLI model is uncertain.
