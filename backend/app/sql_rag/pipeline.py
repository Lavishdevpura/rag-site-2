"""
sql_rag/pipeline.py — Orchestrates the natural-language-to-SQL flow end
to end: intent -> schema retrieval -> context validation -> SQL
generation -> syntax/read-only/complexity validation -> execution ->
natural-language answer. Built 2026-08-17, ported from the standalone Sql
Query Generator project and adapted into this codebase's conventions.

Entry points for api.py:
  - is_database_question(question) -> bool   fast pre-routing check
  - answer_database_question(question) -> dict   runs the full pipeline
"""
import logging
import re

from sql_rag.config import ENABLE_SQL_RAG, SQL_DATABASE_URL
from sql_rag.context_validator import validate_context
from sql_rag.executor import execute_sql
from sql_rag.intent_identifier import identify_intent
from sql_rag.schema_index import build_context_string, hybrid_retrieve
from sql_rag.sql_generator import generate_sql
from sql_rag.validators import check_complexity, validate_read_only, validate_syntax
from sql_rag.answer_formatter import format_answer

logger = logging.getLogger(__name__)

# Fast regex pre-check for "this reads like a live-data lookup" — deliberately
# conservative (biased toward the existing, proven document-RAG path): only
# routes to the SQL pipeline on a reasonably confident data-lookup signal
# (aggregation words, "status of", "how many", superlatives), not on every
# question that happens to share vocabulary with both. Same fast-regex-then-
# fall-through-to-default philosophy as this codebase's other intent
# classifiers (see project_hybrid_modifier_intent_classifier in memory) —
# there is no LLM fallback here yet since there's no real schema/usage data
# to tune one against; revisit once sir's actual questions are known.
_DB_QUESTION_RE = re.compile(
    r"\b(status of|total (revenue|sales|amount|count)|how many|how much|"
    r"count of|sum of|average|top \d+|highest|lowest|latest|current status|"
    r"list all|show (me )?all)\b",
    re.IGNORECASE,
)


def is_database_question(question: str) -> bool:
    """Conservative pre-routing check — see module docstring. Always
    False when the feature is off or no database is configured, so this
    is a true no-op until both are explicitly set."""
    if not ENABLE_SQL_RAG or not SQL_DATABASE_URL:
        return False
    return bool(_DB_QUESTION_RE.search(question))


async def answer_database_question(question: str) -> dict:
    """
    Run the full pipeline for a question already routed here by
    is_database_question(). Never raises — every failure mode returns a
    result dict with ok=False and a human-readable reason, so the caller
    can fall back to the document-RAG path or show a plain refusal
    instead of a stack trace reaching the user.
    """
    if not ENABLE_SQL_RAG or not SQL_DATABASE_URL:
        return {"ok": False, "reason": "SQL RAG is not enabled/configured.", "answer": None, "table": None}

    try:
        intent = await identify_intent(question)

        retrieved_chunks = hybrid_retrieve(question)
        context_valid, context_message = validate_context(intent, retrieved_chunks)
        schema_context = build_context_string(retrieved_chunks)

        if not retrieved_chunks:
            logger.info("[sql_rag] no schema context retrieved for %r", question[:80])
            return {
                "ok": False,
                "reason": "No matching tables found for this question.",
                "answer": None, "table": None,
            }

        generated_sql = await generate_sql(question, intent, schema_context)

        syntax_ok, syntax_msg = validate_syntax(generated_sql)
        if not syntax_ok:
            logger.warning("[sql_rag] syntax validation failed: %s | sql=%r", syntax_msg, generated_sql)
            return {"ok": False, "reason": "Could not generate a valid query for this question.", "answer": None, "table": None}

        read_only_ok, read_only_msg = validate_read_only(generated_sql)
        if not read_only_ok:
            logger.warning("[sql_rag] blocked non-read-only SQL: %s | sql=%r", read_only_msg, generated_sql)
            return {"ok": False, "reason": "This question can't be answered read-only.", "answer": None, "table": None}

        complexity_level, complexity_warnings = check_complexity(generated_sql)
        if complexity_warnings:
            logger.info("[sql_rag] complexity=%s warnings=%s sql=%r", complexity_level, complexity_warnings, generated_sql)

        exec_ok, rows, exec_msg = execute_sql(generated_sql)
        if not exec_ok:
            logger.warning("[sql_rag] execution failed: %s | sql=%r", exec_msg, generated_sql)
            return {"ok": False, "reason": "That query couldn't be run against the database.", "answer": None, "table": None}

        answer_text, table = await format_answer(question, rows, len(rows))

        logger.info(
            "[sql_rag] TIMING-free ok=True rows=%d complexity=%s context_valid=%s query=%r sql=%r",
            len(rows), complexity_level, context_valid, question[:80], generated_sql,
        )
        return {
            "ok": True,
            "reason": None,
            "answer": answer_text,
            "table": table,
            "sql": generated_sql,
            "row_count": len(rows),
        }

    except Exception:
        logger.exception("[sql_rag] pipeline failed for question=%r", question[:80])
        return {"ok": False, "reason": "Something went wrong answering that from the database.", "answer": None, "table": None}
