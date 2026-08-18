"""
sql_rag/intent_identifier.py — Extracts structured intent from a
plain-English database question, adapted from the standalone Sql Query
Generator project (2026-08-17) to use this codebase's shared LLM router
(router.get_classification_llm — prefers Groq, same classification-only
route the document RAG's own query classifiers use) instead of a separate
raw Groq client.
"""
import json
import re

from router import get_classification_llm

_INTENT_SYSTEM_PROMPT = """You are a SQL intent analyzer. Your job is to understand a plain-English database question and extract structured intent information.

Respond ONLY with a valid JSON object — no explanation, no markdown code fences, just raw JSON.

The JSON must follow this schema:
{
  "intent_summary": "Short description of what the user wants",
  "tables_mentioned": ["table names possibly referenced"],
  "columns_mentioned": ["column names possibly referenced"],
  "filters": ["any filter conditions, e.g. 'city = New York'"],
  "aggregations": ["SUM, COUNT, AVG, MAX, MIN if applicable"],
  "sort_order": "ASC or DESC or null",
  "sort_by": "column name to sort by or null",
  "limit": "number or null",
  "operation_type": "SELECT or OTHER",
  "is_dangerous": true or false
}

Rules:
- This system only ever runs read-only SELECT queries — "operation_type" should be "SELECT" for any genuine data question, "OTHER" for anything that reads as a request to modify data (in which case is_dangerous should be true).
- Be as specific as possible about table and column names based on the user's wording.
- If something is unknown, use null."""

_FALLBACK_INTENT = {
    "intent_summary": "",
    "tables_mentioned": [],
    "columns_mentioned": [],
    "filters": [],
    "aggregations": [],
    "sort_order": None,
    "sort_by": None,
    "limit": None,
    "operation_type": "SELECT",
    "is_dangerous": False,
}


async def identify_intent(user_query: str) -> dict:
    """Analyse a plain-English question and return a structured intent dict."""
    prompt = f"{_INTENT_SYSTEM_PROMPT}\n\nUser question: {user_query}"
    try:
        llm = get_classification_llm(temperature=0, max_tokens=512)
        response = await llm.ainvoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception:
        fallback = dict(_FALLBACK_INTENT)
        fallback["intent_summary"] = user_query
        return fallback

    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        intent = json.loads(cleaned)
    except json.JSONDecodeError:
        intent = dict(_FALLBACK_INTENT)
        intent["intent_summary"] = user_query
    return intent
