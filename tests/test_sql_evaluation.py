"""Tests for deterministic NL-to-SQL evaluation helpers."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.sql_evaluation import (
    SQLEvaluationCase,
    evaluate_sql_module,
    extract_sql_query,
    make_execution_match_metric,
)

DATABASE = Path(__file__).parents[1] / "data" / "northwind.sqlite"


class StaticModule:
    def __init__(self, query: str):
        self.query = query

    def __call__(self, **kwargs):
        return SimpleNamespace(sql_query=self.query)


class SQLExtractionTests(unittest.TestCase):
    def test_extracts_prediction_field_and_removes_fence(self):
        prediction = SimpleNamespace(sql_query="```sql\nSELECT 1;\n```")
        self.assertEqual(extract_sql_query(prediction), "SELECT 1;")

    def test_extracts_mapping(self):
        self.assertEqual(extract_sql_query({"sql_query": " SELECT 2 "}), "SELECT 2")


class SQLEvaluationTests(unittest.TestCase):
    def test_requires_the_expected_answer_not_just_executable_sql(self):
        case = SQLEvaluationCase("How many orders?", 16_282)

        correct = evaluate_sql_module(
            StaticModule("SELECT COUNT(*) FROM Orders"), [case], DATABASE
        )
        wrong = evaluate_sql_module(StaticModule("SELECT 1"), [case], DATABASE)

        self.assertEqual(correct.score, 1.0)
        self.assertEqual(wrong.score, 0.0)
        self.assertEqual(wrong.results[0].actual_value, 1)
        self.assertIsNone(wrong.results[0].error)

    def test_records_query_errors(self):
        report = evaluate_sql_module(
            StaticModule("SELECT missing FROM nowhere"),
            [SQLEvaluationCase("Broken query", 0)],
            DATABASE,
        )

        self.assertEqual(report.score, 0.0)
        self.assertIn("no such table", report.results[0].error)

    def test_execution_metric_compares_values_not_column_aliases(self):
        metric = make_execution_match_metric(DATABASE)
        labeled = SimpleNamespace(sql_query="SELECT COUNT(*) AS expected FROM Orders")
        prediction = SimpleNamespace(sql_query="SELECT COUNT(*) AS generated FROM Orders")

        self.assertTrue(metric(labeled, prediction))


if __name__ == "__main__":
    unittest.main()
