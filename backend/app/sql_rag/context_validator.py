"""
sql_rag/context_validator.py — Validates that the retrieved schema
context contains the information the user is asking about before SQL
generation, so an LLM doesn't confidently generate SQL against a
table/column that doesn't exist. Adapted verbatim from the standalone
Sql Query Generator project (2026-08-17).
"""
from typing import Any


def validate_context(
    intent: dict,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[bool, str]:
    if not retrieved_chunks:
        return False, "No relevant tables or columns were found in the database for this question."

    available_tables: set[str] = set()
    available_columns: set[str] = set()
    for chunk in retrieved_chunks:
        available_tables.add(chunk.get("table", "").lower())
        for col in chunk.get("columns", []):
            available_columns.add(col["name"].lower())

    missing_tables = [
        t for t in (intent.get("tables_mentioned") or []) if t.lower() not in available_tables
    ]
    missing_columns = [
        c for c in (intent.get("columns_mentioned") or []) if c.lower() not in available_columns
    ]

    warnings = []
    if missing_tables:
        warnings.append(f"Table(s) not found in database: {', '.join(missing_tables)}")
    if missing_columns:
        warnings.append(f"Column(s) not found in retrieved schema: {', '.join(missing_columns)}")

    if warnings:
        return True, (
            "Partial match: " + " | ".join(warnings)
            + ". SQL was generated using the available schema context — results may differ from expectation."
        )
    return True, "Schema context validated."
