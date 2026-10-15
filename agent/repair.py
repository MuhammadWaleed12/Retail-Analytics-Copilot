"""Routing helpers for the agent's repair loop."""

from typing import Literal


RepairTarget = Literal["executor", "nl_to_sql", "synthesizer"]


def choose_repair_target(
    *, has_sql_error: bool, has_sql_query: bool, repair_count: int
) -> RepairTarget:
    """Choose the node that must run after a repair attempt.

    The first SQL repair edits the existing query, so it must be executed again.
    Later attempts, or failures without a query, need a fresh SQL generation.
    Answer-format failures only need another synthesis pass.
    """
    if not has_sql_error:
        return "synthesizer"
    if repair_count == 0 and has_sql_query:
        return "executor"
    return "nl_to_sql"
