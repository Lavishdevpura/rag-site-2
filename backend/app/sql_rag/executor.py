"""
sql_rag/executor.py — Executes a validated, read-only SQL SELECT via
SQLAlchemy and returns the results. Adapted from the standalone Sql Query
Generator project (2026-08-17), simplified since validators.
validate_read_only() already guarantees only SELECT reaches this point —
no DDL/DML commit branch is needed here.

Recommended (out of this code's control): connect with a database user
that has SELECT-only grants, so a bug upstream in generation/validation
still can't write, rather than relying solely on the application-layer
checks in validators.py.
"""
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from sql_rag.config import MAX_RESULT_ROWS, SQL_DATABASE_URL

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        if not SQL_DATABASE_URL:
            raise RuntimeError("SQL_DATABASE_URL is not configured.")
        _engine = create_engine(SQL_DATABASE_URL)
    return _engine


def execute_sql(sql: str) -> tuple[bool, list[dict[str, Any]], str]:
    """Execute a validated read-only SQL string. Returns (success, rows, message)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            if not result.returns_rows:
                # validate_read_only() should make this unreachable — kept
                # as a safety fallback rather than assuming that gate can
                # never be bypassed.
                return False, [], "Blocked: statement returned no rows (not a SELECT)."

            raw_rows = result.fetchmany(MAX_RESULT_ROWS)
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in raw_rows]
            truncated = len(rows) == MAX_RESULT_ROWS
            message = f"Query returned {len(rows)} row(s)." + (
                f" (showing first {MAX_RESULT_ROWS})" if truncated else ""
            )
            return True, rows, message
    except SQLAlchemyError as e:
        return False, [], f"Database error: {e}"
    except Exception as e:
        return False, [], f"Unexpected error during execution: {e}"


def test_connection() -> tuple[bool, str]:
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Database connection successful."
    except Exception as e:
        return False, f"Database connection failed: {e}"
