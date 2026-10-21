"""PostgreSQL persistence adapter for MECR's existing JSON-shaped records.

The API can keep its current dictionaries and response schemas while records
are loaded from and saved to PostgreSQL. DATABASE_URL must be configured in
Render's environment. This adapter intentionally does not store uploaded file
bytes; files need persistent disk/object storage separately.
"""
from __future__ import annotations

import os
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


class DatabaseNotConfigured(RuntimeError):
    pass


def _connect():
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise DatabaseNotConfigured("DATABASE_URL is not configured")
    # Render PostgreSQL URLs commonly use postgres://; psycopg accepts
    # postgresql://, so normalize the legacy scheme.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return psycopg.connect(url, connect_timeout=10)


def is_configured() -> bool:
    return bool(os.getenv("DATABASE_URL", "").strip())


def load_all() -> dict[str, dict[str, dict[str, Any]]]:
    with _connect() as conn:
        users = {row[0]: row[1] for row in conn.execute("SELECT email, data FROM mecr_users").fetchall()}
        cases = {row[0]: row[1] for row in conn.execute("SELECT id, data FROM mecr_cases").fetchall()}
        evidence = {row[0]: row[1] for row in conn.execute("SELECT id, data FROM mecr_evidence").fetchall()}
    return {"users": users, "cases": cases, "evidence": evidence}


def _save_table(conn, table: str, key_column: str, records: dict[str, dict]) -> None:
    # Table and column names are internal constants supplied only by this module.
    for key, value in records.items():
        conn.execute(
            f"INSERT INTO {table} ({key_column}, data) VALUES (%s, %s) "
            f"ON CONFLICT ({key_column}) DO UPDATE SET data = EXCLUDED.data",
            (key, Jsonb(value)),
        )
    # Remove records deleted from the in-memory store so deletion semantics
    # remain consistent with the existing API. Run all table updates atomically.
    if records:
        conn.execute(
            f"DELETE FROM {table} WHERE {key_column} <> ALL(%s)",
            (list(records.keys()),),
        )
    else:
        conn.execute(f"DELETE FROM {table}")


def save_users(users: dict[str, dict]) -> None:
    with _connect() as conn:
        _save_table(conn, "mecr_users", "email", users)


def save_data(cases: dict[str, dict], evidence: dict[str, dict]) -> None:
    with _connect() as conn:
        _save_table(conn, "mecr_cases", "id", cases)
        _save_table(conn, "mecr_evidence", "id", evidence)
