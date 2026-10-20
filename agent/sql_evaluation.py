"""Deterministic helpers for evaluating generated SQL against expected results."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.tools.sqlite_tool import SQLiteTool


@dataclass(frozen=True)
class SQLEvaluationCase:
    """A held-out question with an independently verified scalar answer."""

    question: str
    expected_value: Any
    constraints: str = "{}"
    tolerance: float = 1e-6


@dataclass(frozen=True)
class SQLEvaluationResult:
    """The generated query and outcome for one evaluation case."""

    question: str
    sql_query: str
    expected_value: Any
    actual_value: Any = None
    correct: bool = False
    error: str | None = None


@dataclass(frozen=True)
class SQLEvaluationReport:
    """Aggregate held-out evaluation results."""

    results: tuple[SQLEvaluationResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(result.correct for result in self.results)

    @property
    def score(self) -> float:
        return self.correct / self.total if self.total else 0.0


def extract_sql_query(prediction: Any) -> str:
    """Extract and normalize SQL from DSPy predictions or plain test doubles."""

    if isinstance(prediction, str):
        query = prediction
    elif isinstance(prediction, Mapping):
        query = prediction.get("sql_query", "")
    else:
        query = getattr(prediction, "sql_query", "")

    query = str(query or "").strip()
    query = re.sub(r"^```(?:sql)?\s*", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\s*```$", "", query)
    return query.strip()


def _values_match(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def evaluate_sql_module(
    module: Any,
    cases: Sequence[SQLEvaluationCase],
    db_path: str | Path,
) -> SQLEvaluationReport:
    """Evaluate a module on held-out scalar-answer cases.

    A query only passes when it executes and its first returned value matches the
    independently verified answer. Merely producing valid SQL is insufficient.
    """

    sql_tool = SQLiteTool(str(db_path))
    schema = sql_tool.get_schema()
    results: list[SQLEvaluationResult] = []

    for case in cases:
        sql_query = ""
        try:
            prediction = module(
                question=case.question,
                schema=schema,
                constraints=case.constraints,
            )
            sql_query = extract_sql_query(prediction)
            if not sql_query:
                raise ValueError("model returned an empty SQL query")

            rows, columns, error = sql_tool.execute_query(sql_query)
            if error:
                raise ValueError(error)
            if not rows or not columns:
                raise ValueError("query returned no value")

            actual_value = rows[0][columns[0]]
            results.append(
                SQLEvaluationResult(
                    question=case.question,
                    sql_query=sql_query,
                    expected_value=case.expected_value,
                    actual_value=actual_value,
                    correct=_values_match(
                        actual_value, case.expected_value, case.tolerance
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - model and SQLite failures are results
            results.append(
                SQLEvaluationResult(
                    question=case.question,
                    sql_query=sql_query,
                    expected_value=case.expected_value,
                    error=str(exc),
                )
            )

    return SQLEvaluationReport(tuple(results))


def make_execution_match_metric(db_path: str | Path):
    """Build a DSPy metric that compares candidate and labeled query results."""

    sql_tool = SQLiteTool(str(db_path))

    def execution_match(example: Any, prediction: Any, trace: Any = None) -> bool:
        del trace
        expected_sql = extract_sql_query(example)
        actual_sql = extract_sql_query(prediction)
        if not expected_sql or not actual_sql:
            return False

        expected_rows, expected_columns, expected_error = sql_tool.execute_query(
            expected_sql
        )
        actual_rows, actual_columns, actual_error = sql_tool.execute_query(actual_sql)
        if expected_error is not None or actual_error is not None:
            return False

        expected_values = [tuple(row[column] for column in expected_columns) for row in expected_rows]
        actual_values = [tuple(row[column] for column in actual_columns) for row in actual_rows]
        return expected_values == actual_values

    return execution_match
