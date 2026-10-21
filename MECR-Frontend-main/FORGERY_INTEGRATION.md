# Forgery Detection Integration

This document describes the optional, experimental document-authenticity screen
added to the MECR workbench. The feature is secondary to the legal
conflict-resolution pipeline and does not change existing case, evidence, user,
or analysis behaviour.

## What was integrated

| Item | Location |
|---|---|
| Trained model bundle | `backend/forgery_detector_bundle.joblib` |
| Inference code | `backend/inference.py` (provided `predict_document` logic) |
| Backend routes | `backend/main.py` |
| Standalone screen UI | `src/pages/ForgeryCheck.jsx` (`/forgery-check`) |
| Per-exhibit screen action | `src/pages/CaseDetail.jsx` |
| API client | `src/lib/api.js` |
| Dependencies | `backend/requirements.txt` |

The model is loaded lazily on the first prediction request and stays entirely
on the server. The `.joblib` file is not served by any endpoint and is not part
of the frontend build.

## API

Both routes require a signed-in user.

### `POST /api/v1/forgery/check`

Multipart upload with one `file` field. Images only (JPEG, PNG, TIFF, BMP,
WebP, GIF). Example:

```bash
curl -X POST http://localhost:8000/api/v1/forgery/check \
  -H "Authorization: Bearer <token>" \
  -F "file=@document.jpg"
```

### `GET /api/v1/evidence/{evidence_id}/forgery-check`

Runs the same model against the stored original file for an image exhibit. It
uses the normal evidence ownership checks, so another user's exhibit returns
`404`. Evidence without a stored image returns `404`; non-image evidence
returns `422`.

### Response

```json
{
  "filename": "document.jpg",
  "result": "Likely Authentic",
  "raw_result": "LIKELY AUTHENTIC",
  "tampered_score": 0.2933,
  "low_threshold": 0.35,
  "high_threshold": 0.65,
  "disclaimer": "This is an experimental, secondary screening feature. Output is not guaranteed proof of authenticity or tampering."
}
```

Display labels are exactly:

- `Likely Authentic` — score at or below the low threshold
- `Review Required` — score inside the review band
- `Potentially Tampered` — score at or above the high threshold

The UI shows the score, both thresholds, and an explicit note that the result
is experimental and not legal proof.

## Error handling

| Case | Response |
|---|---|
| No/invalid token | `401 Not authenticated` |
| Non-image file | `422`, clear message |
| Corrupt or unreadable image | `422 Invalid or unreadable image file` |
| Empty upload | `422 Uploaded file is empty` |
| Model file missing | `503 Forgery model is not available on this server` |
| Missing Python dependency | `503 Forgery screening dependencies are not installed` |
| Unexpected prediction failure | `500` with a generic message (no stack trace or model internals) |

## Dependencies

Installed from the bundle's `requirements.txt` and recorded in
`backend/requirements.txt`:

```text
numpy
Pillow
scikit-image
scikit-learn
joblib
```

The deployed environment currently uses scikit-learn 1.9.0 while the bundle was
pickled with 1.6.1. Loading emits an `InconsistentVersionWarning`; predictions
still run and were verified. If exact training-time parity is required, run the
backend in an environment with scikit-learn 1.6.1 and re-validate.

## Testing performed

Tests were run through the running website (`http://127.0.0.1:5173`) and the
authenticated API (`http://127.0.0.1:8000`):

| Input | Result |
|---|---|
| Clean generated document (`authentic_document.jpg`) | **Likely Authentic** (tamper score 0.2933) |
| Same document with an edited date and spliced signature (`tampered_document.jpg`) | **Review Required** (tamper score 0.3533) |
| InLegalNER authentic sample from the training archive | **Likely Authentic** (0.3000) |
| Real forged sample from the training archive | **Review Required** (below the 0.65 high threshold) |
| High-frequency stress input | **Potentially Tampered** (0.8067) |
| Invalid image bytes | `422` with a readable message |
| Non-image file | `422` with a readable message |
| Request without a token | `401` |

The real forged sample landing in **Review Required** is expected model
behaviour, not a UI failure: the supplied bundle's high threshold is 0.65 and
most inputs score between 0.28 and 0.63. The review band is doing its job by
declining to make a confident claim. The model and thresholds were not altered.

## Important limitations

- Held-out accuracy reported by the training run: **73.89%**.
- ROC-AUC reported by the training run: **0.7954**.
- The model was trained on CASIA-style image manipulation data, not on the
  specific courts, scanners, stamps, or languages in this deployment.
- A low score does not prove authenticity; a high score does not prove
  tampering. Originals, chain of custody, and a qualified forensic examiner
  remain necessary.
- Scanned documents without OCR are still screened as images, but the text
  pipeline cannot verify their contents.
