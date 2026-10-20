"""Activate the best trained checkpoint for each model type.

Keeps the metadata file tidy (one entry per model/dataset pair) and guarantees
exactly one active version per model type.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.dataset_service import MODELS_DIR  # noqa: E402

PREFERRED = {
    "bilstm_crf": ["inlegalner_large"],
    "xgboost": ["inlegalner_large"],
    "legalbert": ["legal_claims_large"],
    "rgcn": ["inlegalner_large"],
}


def score(entry: dict) -> float:
    return float(entry.get("accuracy") or 0)


def main() -> None:
    path = os.path.join(MODELS_DIR, "models_metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    cleaned = {}
    for model_type, versions in metadata.items():
        # Keep the newest entry per dataset AND per artefact path: a retrain
        # overwrites the same file, so an older row would describe stale
        # metrics for the current weights.
        best_by_key = {}
        for entry in versions:
            key = (entry.get("dataset", "unknown"), entry.get("model_path", ""))
            if key not in best_by_key or str(entry.get("trained_at", "")) >= str(best_by_key[key].get("trained_at", "")):
                best_by_key[key] = entry
        cleaned[model_type] = list(best_by_key.values())

        for entry in cleaned[model_type]:
            entry["active"] = False

        preferred = PREFERRED.get(model_type, [])
        candidates = [e for e in cleaned[model_type] if e.get("dataset") in preferred]
        pool = candidates or cleaned[model_type]
        if pool:
            best = max(pool, key=score)
            best["active"] = True
            print(f"{model_type}: active -> {best.get('dataset')} ({best.get('accuracy')}%)")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)
    print("metadata cleaned")


if __name__ == "__main__":
    main()
