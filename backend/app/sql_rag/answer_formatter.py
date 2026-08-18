"""
sql_rag/answer_formatter.py — Turns real SQL query results into a warm,
conversational chat answer, reusing this fork's existing anti-
hallucination grounding philosophy (prompt_template.py's STRICT_GROUNDED_
PROMPT etc.) rather than inventing a new one: state only what the actual
returned rows contain, never round a number or add unstated context.

Also opportunistically builds a comparison_table UI payload — the SAME
{columns, rows: [{label, values}]} shape multi_source_rag.py's
_enforce_comparison_table_format already produces and app.js already
renders — for result shapes that fit it (a handful of rows, a handful of
columns), so a "top 5 X by Y" question gets a real table with zero
frontend changes. Skipped for shapes that don't fit (a single aggregate
value doesn't need a table; a wide/long result isn't what that component
was built to show) — those still get the natural-language answer alone.
"""
from typing import Any

from prompt_template import PERSONA_NAME
from router import get_classification_llm

_MAX_TABLE_ROWS = 10
_MAX_TABLE_COLS = 6

_ANSWER_SYSTEM_PROMPT = """You are {persona}, a warm, helpful assistant. You just ran a READ-ONLY database lookup on the user's behalf and got back real results below. Answer their question using ONLY these results.

STRICT RULES:
- State only facts present in the QUERY RESULTS below. Never invent, round, estimate, or infer a number or fact not literally there.
- This system is READ-ONLY and can only look things up, never change anything. If the user's question also asked for an action beyond looking something up (delete, update, change, send, create, approve, etc.), do NOT say or imply that action happened, will happen, or was taken — only answer the lookup part, and if nothing in the question was answerable as a lookup, say you can only look things up, not make changes.
- Never echo the user's own request phrasing back as if it were a confirmed outcome — only state what the QUERY RESULTS themselves show.
- If the results are empty, say plainly that nothing matched — do not guess why.
- Keep it short and conversational — 1-3 sentences for a single value or short list, don't recite every row if there are many (mention the count and a few examples instead).
- Never mention "SQL", "query", "database", or "rows" to the user — just answer naturally, the way you'd answer if you already knew the numbers.
- No bullet points, no markdown tables, no headers — plain conversational prose. (Any tabular results are shown separately, not by you restating them.)""".replace("{persona}", PERSONA_NAME)


async def _synthesize_answer(user_query: str, rows: list[dict[str, Any]], row_count: int) -> str:
    if not rows:
        prompt = (
            f"{_ANSWER_SYSTEM_PROMPT}\n\nQUESTION: {user_query}\n\n"
            "QUERY RESULTS: (no matching rows)\n\nANSWER:"
        )
    else:
        preview = rows[:20]  # keep the prompt bounded even though more rows may exist
        prompt = (
            f"{_ANSWER_SYSTEM_PROMPT}\n\nQUESTION: {user_query}\n\n"
            f"QUERY RESULTS ({row_count} row(s) total, showing {len(preview)}):\n{preview}\n\nANSWER:"
        )

    llm = get_classification_llm(temperature=0, max_tokens=300)
    response = await llm.ainvoke(prompt)
    return (response.content if hasattr(response, "content") else str(response)).strip()


def _build_table(rows: list[dict[str, Any]]) -> dict | None:
    """Best-effort comparison_table payload for a result shape that fits
    it — a handful of rows, a handful of columns, first column usable as
    a row label. Returns None when the shape doesn't fit; the caller
    already has the natural-language answer as the primary output either
    way, so this is a bonus, not a requirement."""
    if not rows or len(rows) > _MAX_TABLE_ROWS:
        return None
    columns = list(rows[0].keys())
    if len(columns) < 2 or len(columns) > _MAX_TABLE_COLS:
        return None

    label_col, *value_cols = columns
    return {
        "columns": value_cols,
        "rows": [
            {
                "label": str(row.get(label_col, "")),
                "values": [str(row.get(c, "")) for c in value_cols],
            }
            for row in rows
        ],
    }


async def format_answer(
    user_query: str, rows: list[dict[str, Any]], row_count: int,
) -> tuple[str, dict | None]:
    """Returns (natural_language_answer, optional_ui_table)."""
    answer = await _synthesize_answer(user_query, rows, row_count)
    table = _build_table(rows)
    return answer, table
