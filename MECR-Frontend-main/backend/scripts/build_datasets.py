"""Download the public training corpora into services/dataset_store/.

Usage
-----
    python scripts/build_datasets.py
    python scripts/build_datasets.py --force
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.external_datasets import build_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()

    result = build_all(force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
