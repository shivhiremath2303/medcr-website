"""Train all four MECR models on the large public legal corpora.

Usage
-----
    python scripts/train_all.py
    python scripts/train_all.py --models xgboost,rgcn
    python scripts/train_all.py --models bilstm_crf --epochs 10
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS = ["xgboost", "rgcn", "bilstm_crf", "legalbert"]
DEFAULT_EPOCHS = {"xgboost": None, "rgcn": 40, "bilstm_crf": 5, "legalbert": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS),
                        help="comma separated subset of: " + ", ".join(MODELS))
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for every model")
    parser.add_argument("--ner-dataset", default="inlegalner_large")
    parser.add_argument("--nli-dataset", default="legal_claims_large")
    parser.add_argument("--xgb-max-docs", type=int, default=10000)
    parser.add_argument("--rgcn-max-docs", type=int, default=8000)
    parser.add_argument("--bert-max-train", type=int, default=18000)
    parser.add_argument("--bert-max-test", type=int, default=2100)
    args = parser.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in selected if m not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {unknown}")

    from services.training_service import (
        train_bilstm_crf,
        train_legalbert_contradiction,
        train_rgcn,
        train_xgboost_anomaly,
    )

    started = time.time()
    summary = {}

    for model in selected:
        epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[model]
        print(f"\n===== {model.upper()} =====", flush=True)
        t0 = time.time()
        if model == "xgboost":
            summary[model] = train_xgboost_anomaly(args.ner_dataset, max_docs=args.xgb_max_docs)
        elif model == "rgcn":
            summary[model] = train_rgcn(args.ner_dataset, epochs=epochs or 40,
                                        max_docs=args.rgcn_max_docs)
        elif model == "bilstm_crf":
            summary[model] = train_bilstm_crf(args.ner_dataset, epochs=epochs or 5)
        elif model == "legalbert":
            summary[model] = train_legalbert_contradiction(
                args.nli_dataset, epochs=epochs or 1,
                max_train=args.bert_max_train, max_test=args.bert_max_test,
            )
        print(f"{model} finished in {(time.time() - t0) / 60:.1f} min", flush=True)

    printable = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("report", "history", "loss_history", "eval_results")}
        for k, v in summary.items()
    }
    print("\n===== SUMMARY =====")
    print(json.dumps(printable, indent=2))
    print(f"Total: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
