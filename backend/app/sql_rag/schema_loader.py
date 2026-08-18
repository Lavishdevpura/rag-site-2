"""
sql_rag/schema_loader.py — Introspects the configured database schema and
prepares text chunks suitable for embedding and BM25 indexing.

Adapted from the standalone Sql Query Generator project (2026-08-17). Each
"chunk" represents one table with its columns, types, and a few sample
rows — giving the LLM real context to generate accurate SQL, and giving
the retrieval step something to match a question against without loading
the entire schema into every prompt.

Note the sample rows below are REAL data pulled straight from the
database and get sent to whichever LLM backend generates the SQL — worth
a deliberate decision (mask/anonymize, or exclude sensitive tables from
the index) once the real schema is known, not something to leave on
autopilot for genuinely sensitive columns.
"""
from typing import Any

from sqlalchemy import create_engine, inspect, text

from sql_rag.config import SQL_DATABASE_URL


def load_schema_chunks() -> list[dict[str, Any]]:
    """
    Introspect the database and return a list of schema chunks.

    Each chunk dict has:
        - "id":      unique identifier  e.g. "table__orders"
        - "table":   table name
        - "text":    human-readable description for embedding / BM25
        - "columns": list of {name, type} dicts
    """
    if not SQL_DATABASE_URL:
        return []

    engine = create_engine(SQL_DATABASE_URL)
    inspector = inspect(engine)
    chunks: list[dict[str, Any]] = []

    table_names = inspector.get_table_names()

    with engine.connect() as conn:
        for table_name in table_names:
            columns = inspector.get_columns(table_name)
            pk_constraint = inspector.get_pk_constraint(table_name)
            fk_list = inspector.get_foreign_keys(table_name)

            col_descriptions = []
            for col in columns:
                col_str = f"  - {col['name']} ({col['type']})"
                if col["name"] in (pk_constraint.get("constrained_columns") or []):
                    col_str += " [PRIMARY KEY]"
                col_descriptions.append(col_str)

            fk_descriptions = []
            for fk in fk_list:
                fk_descriptions.append(
                    f"  - {fk['constrained_columns']} -> "
                    f"{fk['referred_table']}.{fk['referred_columns']}"
                )

            try:
                sample_rows_result = conn.execute(
                    text(f"SELECT * FROM {table_name} LIMIT 5")
                )
                preview_rows = [dict(row._mapping) for row in sample_rows_result]
                sample_str = "\n".join(
                    f"    {row}" for row in preview_rows[:3]
                ) if preview_rows else "    (no data)"
            except Exception:
                preview_rows = []
                sample_str = "    (could not fetch sample rows)"

            text_chunk = (
                f"Table: {table_name}\n"
                f"Columns:\n{chr(10).join(col_descriptions)}\n"
                + (f"Foreign Keys:\n{chr(10).join(fk_descriptions)}\n" if fk_descriptions else "")
                + f"Sample rows:\n{sample_str}"
            )

            try:
                row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0
            except Exception:
                row_count = 0

            chunks.append({
                "id": f"table__{table_name}",
                "table": table_name,
                "text": text_chunk,
                "columns": [{"name": c["name"], "type": str(c["type"])} for c in columns],
                "sample_rows": preview_rows,
                "row_count": row_count,
            })

    return chunks


def get_full_schema_summary() -> str:
    """Concise text summary of the entire schema (small-schema fallback
    when retrieval can't narrow to specific tables)."""
    chunks = load_schema_chunks()
    return "\n\n---\n\n".join(c["text"] for c in chunks)
