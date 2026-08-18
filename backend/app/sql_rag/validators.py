"""
sql_rag/validators.py — Syntax, security, and complexity checks for a
generated SQL statement, adapted from the standalone Sql Query Generator
project (2026-08-17).

Security is hardened relative to the source project for this specific
use case: the standalone project generates a wide range of statements and
asks the user to confirm before running a destructive one (fits a
portfolio demo). This integration exists to ANSWER QUESTIONS — it has no
legitimate reason to ever write, so anything other than a read-only
SELECT (or WITH ... SELECT) is rejected outright here, no confirmation
path. This is a second, independent layer on top of sql_generator.py's
own prompt instructions — an LLM can still be talked into or hallucinate
past a prompt rule, so this AST-level check doesn't trust generation to
have gotten it right. A third, DB-level layer (connecting with genuinely
read-only credentials) is recommended in .env.example and is out of this
code's control — worth confirming once real credentials are set up.
"""
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from sql_rag.config import SQL_DIALECT

_INJECTION_PATTERNS = [
    r"--\s*$",
    r"/\*.*?\*/",
    r";\s*\w+",  # multiple statements
    r"xp_cmdshell",
    r"LOAD\s+DATA",
    r"INTO\s+OUTFILE",
]

MAX_JOINS = 3
MAX_SUBQUERY_DEPTH = 2


def validate_syntax(sql: str) -> tuple[bool, str]:
    if not sql or not sql.strip():
        return False, "SQL query is empty."
    try:
        parsed = sqlglot.parse(sql, dialect=SQL_DIALECT, error_level=sqlglot.ErrorLevel.RAISE)
        if not parsed or any(s is None for s in parsed):
            return False, "SQL could not be parsed."
        return True, "Syntax is valid."
    except ParseError as e:
        msg = "; ".join(str(err) for err in e.errors) if e.errors else str(e)
        return False, f"Syntax error: {msg}"
    except Exception as e:
        return False, f"Unexpected parse error: {e}"


def validate_read_only(sql: str) -> tuple[bool, str]:
    """Hard reject anything that isn't a read-only SELECT (or WITH ...
    SELECT). Unlike the standalone project's DANGEROUS_KEYWORDS gate
    (which allows a confirm-to-proceed path), there is no override here —
    this integration answers questions, it never writes."""
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE | re.DOTALL):
            return False, "Blocked: query contains a suspicious pattern (possible injection)."

    try:
        parsed = sqlglot.parse(sql, dialect=SQL_DIALECT)
    except Exception:
        return False, "Blocked: could not verify this query is read-only."

    for statement in parsed:
        if statement is None:
            continue
        if isinstance(statement, exp.Select):
            continue
        # A CTE ("WITH ... SELECT ...") parses as exp.Select with a `with`
        # arg in sqlglot, already covered by the isinstance check above —
        # anything else here is genuinely not a read query.
        stmt_name = type(statement).__name__
        return False, f"Blocked: only read-only SELECT queries are permitted, got {stmt_name}."

    return True, "Read-only SELECT confirmed."


def check_complexity(sql: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if not sql or not sql.strip():
        return "LOW", []

    try:
        parsed = sqlglot.parse(sql, dialect=SQL_DIALECT)
        for statement in parsed:
            if statement is None or not isinstance(statement, exp.Select):
                continue

            join_count = len(list(statement.find_all(exp.Join)))
            has_where = statement.find(exp.Where) is not None
            has_limit = statement.find(exp.Limit) is not None
            select_star = statement.find(exp.Star) is not None

            def _subquery_depth(node, depth=0):
                d = depth
                for sq in node.find_all(exp.Subquery):
                    d = max(d, _subquery_depth(sq, depth + 1))
                return d

            subq_depth = _subquery_depth(statement)

            if join_count > MAX_JOINS:
                warnings.append(f"High JOIN count ({join_count}).")
            if not has_where and select_star:
                warnings.append("SELECT * without a WHERE clause — may return all rows.")
            elif not has_where and not has_limit:
                warnings.append("No WHERE clause and no LIMIT — this may scan the entire table.")
            if not has_limit:
                warnings.append("No LIMIT clause detected. Result set could be large.")
            if subq_depth > MAX_SUBQUERY_DEPTH:
                warnings.append(f"Deeply nested subqueries (depth {subq_depth}).")
    except Exception:
        pass

    if len(warnings) >= 3:
        level = "HIGH"
    elif len(warnings) >= 1:
        level = "MEDIUM"
    else:
        level = "LOW"
    return level, warnings
