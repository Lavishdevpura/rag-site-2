"""
sql_rag/sql_generator.py — Generates a SQL SELECT statement from schema
context + user intent, adapted from the standalone Sql Query Generator
project (2026-08-17) to use this codebase's shared LLM router instead of
a separate raw Groq client, and to hard-restrict generation to read-only
SELECT (this integration answers questions, it never has a legitimate
reason to write) — the standalone project's UPDATE/INSERT/DELETE
generation + confirm-to-proceed flow is deliberately not carried over
here.
"""
import re

from router import get_classification_llm
from sql_rag.config import SQL_DIALECT

_SQL_GENERATION_SYSTEM_PROMPT = """You are an expert SQL query generator. Generate a single, correct, read-only SQL SELECT query based on:
1. The user's plain-English question
2. The extracted intent
3. The relevant database schema context provided

STRICT RULES:
- Respond with ONLY the raw SQL query — no explanation, no markdown, no code fences.
- ONLY generate SELECT (or WITH ... SELECT) statements. Never INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, or any statement that modifies data or schema, under any circumstances, even if the question seems to ask for it.
- Use only the tables and columns that exist in the provided schema context.
- If the user asks for something that doesn't exist in the schema, generate: SELECT 'Data not available' AS message;
- Always add appropriate WHERE clauses to avoid full table scans when possible.
- For aggregations, always include GROUP BY.
- Use table aliases for readability when joining multiple tables.
- LIMIT: only add a LIMIT clause if the user explicitly asked for a specific number of rows (e.g. "top 10", "first 5"). Otherwise do NOT add a LIMIT clause."""


async def generate_sql(user_query: str, intent: dict, schema_context: str) -> str:
    limit_val = intent.get("limit")
    limit_instruction = (
        f"The user explicitly requested a limit of {limit_val} rows — include LIMIT {limit_val}."
        if limit_val else "The user did NOT request a row limit — do NOT add a LIMIT clause."
    )

    prompt = f"""{_SQL_GENERATION_SYSTEM_PROMPT}

SQL DIALECT: {SQL_DIALECT}

USER QUESTION:
{user_query}

EXTRACTED INTENT:
{intent}

LIMIT INSTRUCTION:
{limit_instruction}

RELEVANT DATABASE SCHEMA AND CONTEXT:
{schema_context}

Generate the SQL query now."""

    llm = get_classification_llm(temperature=0, max_tokens=1024)
    response = await llm.ainvoke(prompt)
    raw_sql = (response.content if hasattr(response, "content") else str(response)).strip()

    sql = re.sub(r"```(?:sql)?|```", "", raw_sql).strip()

    lines = sql.splitlines()
    sql_lines: list[str] = []
    inside_sql = False
    sql_starters = {"SELECT", "WITH"}
    for line in lines:
        if not inside_sql and any(line.strip().upper().startswith(k) for k in sql_starters):
            inside_sql = True
        if inside_sql:
            sql_lines.append(line)

    return "\n".join(sql_lines).strip() if sql_lines else sql
