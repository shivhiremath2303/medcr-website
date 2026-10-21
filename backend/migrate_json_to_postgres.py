"""Safely import MECR JSON records into PostgreSQL.

Run from the backend directory using DATABASE_URL.
Existing database rows are never overwritten by this migration.
"""

import json
import os
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = BASE_DIR / "users.json"
DATA_FILE = BASE_DIR / "mecr_data.json"


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required migration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    users = read_json(USERS_FILE)
    data = read_json(DATA_FILE)
    cases = data.get("cases", {})
    evidence = data.get("evidence", {})

    if not all(isinstance(x, dict) for x in (cases, evidence)):
        raise ValueError("Cases and evidence must be JSON objects.")

    if not users:
        raise ValueError("No users found; refusing to migrate an empty user file.")

    for email, user in users.items():
        if (
            not isinstance(user, dict)
            or not user.get("password_hash")
            or not user.get("salt")
        ):
            raise ValueError(f"User record is incomplete for {email!r}.")

    print(
        f"Source records: users={len(users)}, "
        f"cases={len(cases)}, evidence={len(evidence)}"
    )

    with psycopg.connect(database_url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mecr_users (
                    email TEXT PRIMARY KEY,
                    data JSONB NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mecr_cases (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS mecr_evidence (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL
                )
            """)

            for email, user in users.items():
                normalized_email = email.strip().lower()
                if not normalized_email:
                    raise ValueError("Empty user email encountered.")
                cur.execute(
                    """INSERT INTO mecr_users (email, data)
                       VALUES (%s, %s)
                       ON CONFLICT (email) DO NOTHING""",
                    (normalized_email, Jsonb(user)),
                )

            for case_id, case in cases.items():
                cur.execute(
                    """INSERT INTO mecr_cases (id, data)
                       VALUES (%s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                    (str(case_id), Jsonb(case)),
                )

            for evidence_id, item in evidence.items():
                cur.execute(
                    """INSERT INTO mecr_evidence (id, data)
                       VALUES (%s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                    (str(evidence_id), Jsonb(item)),
                )

            counts = {}
            for table in ("mecr_users", "mecr_cases", "mecr_evidence"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = cur.fetchone()[0]

        # psycopg commits the transaction on successful context exit.

    print("PostgreSQL record counts:", counts)
    print("Migration insert pass completed.")
    print("Existing rows were preserved; conflicts were not overwritten.")


if __name__ == "__main__":
    main()
