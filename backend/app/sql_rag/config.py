"""
sql_rag/config.py — Configuration for the natural-language-to-SQL
capability, adapted from the standalone "Sql Query Generator" project
(2026-08-17) into a dormant, flag-gated addition alongside the existing
document RAG.

Off by default (ENABLE_SQL_RAG=false, SQL_DATABASE_URL unset) until the
real database's engine, schema, and access scope are known — this module
should be safe to import and no-op everywhere until then.
"""
import os

ENABLE_SQL_RAG = os.getenv("ENABLE_SQL_RAG", "false").strip().lower() == "true"

# SQLAlchemy connection string, e.g. "mysql+pymysql://user:pass@host/db" or
# "postgresql+psycopg2://user:pass@host/db". Deliberately no default here —
# unlike the document RAG's turbovec store, there is no safe local fallback
# for someone's real business data; ENABLE_SQL_RAG must be explicitly true
# AND this must be explicitly set, or the pipeline refuses to run at all
# (see pipeline.py's startup check).
SQL_DATABASE_URL = os.getenv("SQL_DATABASE_URL", "").strip()

# sqlglot dialect name for syntax/complexity parsing — auto-derived from
# the SQLAlchemy URL's scheme when not explicitly overridden, since most
# common engines share a name between the two (sqlite/mysql/postgres).
_DIALECT_FROM_SCHEME = {
    "sqlite": "sqlite",
    "mysql": "mysql",
    "mysql+pymysql": "mysql",
    "mysql+mysqlconnector": "mysql",
    "postgresql": "postgres",
    "postgresql+psycopg2": "postgres",
    "postgres": "postgres",
    "mssql": "tsql",
    "mssql+pyodbc": "tsql",
}


def _resolve_dialect() -> str:
    override = os.getenv("SQL_DIALECT", "").strip().lower()
    if override:
        return override
    scheme = SQL_DATABASE_URL.split("://", 1)[0].lower() if SQL_DATABASE_URL else ""
    return _DIALECT_FROM_SCHEME.get(scheme, "sqlite")


SQL_DIALECT = _resolve_dialect()

# Hard row cap on any query result, regardless of what the user asked for —
# a chat answer summarizing thousands of rows isn't useful and risks a slow
# response; this is a safety backstop, not a feature.
MAX_RESULT_ROWS = int(os.getenv("SQL_MAX_RESULT_ROWS", "200"))

# Retrieval tuning for the schema-RAG step (mirrors the standalone
# project's settings.py defaults).
SCHEMA_RETRIEVAL_TOP_K = int(os.getenv("SQL_SCHEMA_TOP_K", "5"))
SCHEMA_HYBRID_ALPHA = float(os.getenv("SQL_SCHEMA_HYBRID_ALPHA", "0.6"))  # semantic weight, 1-alpha = BM25
