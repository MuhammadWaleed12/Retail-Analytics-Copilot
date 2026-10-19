"""Compile and evaluate the NL-to-SQL DSPy module."""

import argparse
from pathlib import Path

import dspy

from agent.dspy_signatures import NLToSQL
from agent.sql_evaluation import (
    SQLEvaluationCase,
    evaluate_sql_module,
    make_execution_match_metric,
)
from agent.tools.sqlite_tool import SQLiteTool

DEFAULT_MODEL = "phi3.5:3.8b-mini-instruct-q4_K_M"


def create_training_set(schema: str) -> list[dspy.Example]:
    """Return labeled optimization examples, separate from held-out cases."""

    records = [
        {
            "question": "What is the total revenue from Beverages category?",
            "schema": schema,
            "constraints": '{"categories": ["Beverages"]}',
            "sql_query": 'SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS revenue FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID WHERE c.CategoryName = "Beverages"',
        },
        {
            "question": "Top 3 products by revenue",
            "schema": schema,
            "constraints": "{}",
            "sql_query": 'SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS revenue FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID GROUP BY p.ProductID, p.ProductName ORDER BY revenue DESC LIMIT 3',
        },
        {
            "question": "Average order value in December 1997",
            "schema": schema,
            "constraints": '{"date_ranges": [["1997-12-01", "1997-12-31"]]}',
            "sql_query": 'SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID) AS aov FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID WHERE o.OrderDate >= "1997-12-01" AND o.OrderDate <= "1997-12-31"',
        },
    ]
    return [
        dspy.Example(**record).with_inputs("question", "schema", "constraints")
        for record in records
    ]


def create_held_out_cases() -> list[SQLEvaluationCase]:
    """Return independently checked questions that are never used for training."""

    return [
        SQLEvaluationCase("How many orders are in the database?", 16_282),
        SQLEvaluationCase("How many customers are based in Germany?", 11),
        SQLEvaluationCase("How many products are discontinued?", 8),
    ]


def configure_ollama(model_name: str) -> None:
    """Configure DSPy to use the requested local Ollama model."""

    lm = dspy.LM(
        model=f"ollama_chat/{model_name}",
        api_base="http://localhost:11434",
        api_key="",
    )
    dspy.configure(lm=lm)


def _print_report(label: str, report) -> None:
    print(f"{label}: {report.correct}/{report.total} ({report.score:.2%})")
    for result in report.results:
        status = "PASS" if result.correct else "FAIL"
        detail = result.error or (
            f"expected={result.expected_value!r}, actual={result.actual_value!r}"
        )
        print(f"  [{status}] {result.question}: {detail}")


def optimize_nl_to_sql(
    db_path: str,
    output_path: str,
    model_name: str = DEFAULT_MODEL,
) -> NLToSQL:
    """Compile NL-to-SQL, evaluate held-out answers, and save its DSPy state."""

    configure_ollama(model_name)
    schema = SQLiteTool(db_path).get_schema()
    training_set = create_training_set(schema)
    held_out_cases = create_held_out_cases()

    base_module = NLToSQL()
    before = evaluate_sql_module(base_module, held_out_cases, db_path)
    _print_report("Before optimization", before)

    optimizer = dspy.BootstrapFewShot(
        metric=make_execution_match_metric(db_path),
        max_bootstrapped_demos=4,
        max_labeled_demos=3,
    )
    optimized_module = optimizer.compile(student=base_module, trainset=training_set)

    after = evaluate_sql_module(optimized_module, held_out_cases, db_path)
    _print_report("After optimization", after)
    print(f"Held-out change: {after.score - before.score:+.2%}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    optimized_module.save(str(destination), save_program=False)
    print(f"Saved optimized module state to {destination}")
    return optimized_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/northwind.sqlite")
    parser.add_argument("--output", default="artifacts/nl_to_sql.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    optimize_nl_to_sql(args.db, args.output, args.model)
