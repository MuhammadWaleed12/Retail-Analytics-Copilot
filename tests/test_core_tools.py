import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.rag.retrieval import TFIDFRetriever
from agent.tools.sqlite_tool import SQLiteTool


def create_database(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE Orders (id INTEGER PRIMARY KEY, total REAL NOT NULL);
        CREATE TABLE audit_log (message TEXT NOT NULL);
        INSERT INTO Orders (total) VALUES (12.5), (30.0);
        """
    )
    connection.commit()
    connection.close()


class SQLiteToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_rejects_missing_database(self):
        with self.assertRaisesRegex(FileNotFoundError, "Database not found"):
            SQLiteTool(str(self.directory / "missing.sqlite"))

    def test_lists_schema_and_returns_dict_rows(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database))

        self.assertEqual(tool.get_table_names(), ["Orders", "audit_log"])
        self.assertIn("Table: Orders", tool.get_schema())
        self.assertIn("total (REAL)", tool.get_schema())

        rows, columns, error = tool.execute_query(
            "SELECT id, total FROM Orders ORDER BY id"
        )
        self.assertIsNone(error)
        self.assertEqual(columns, ["id", "total"])
        self.assertEqual(
            rows,
            [{"id": 1, "total": 12.5}, {"id": 2, "total": 30.0}],
        )

    def test_returns_results_independent_of_query_prefix(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database))

        for query in (
            (
                "WITH totals AS (SELECT total FROM Orders) "
                "SELECT SUM(total) AS revenue FROM totals"
            ),
            "-- Revenue report\nSELECT SUM(total) AS revenue FROM Orders",
            "/* Revenue report */ SELECT SUM(total) AS revenue FROM Orders",
        ):
            with self.subTest(query=query):
                self.assertEqual(
                    tool.execute_query(query),
                    ([{"revenue": 42.5}], ["revenue"], None),
                )

    def test_empty_cte_preserves_column_names(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database))

        self.assertEqual(
            tool.execute_query(
                "WITH empty AS (SELECT total FROM Orders WHERE 0) "
                "SELECT total FROM empty"
            ),
            ([], ["total"], None),
        )

    @unittest.skipIf(sqlite3.sqlite_version_info < (3, 35, 0), "RETURNING needs SQLite 3.35+")
    def test_returning_rows_are_fetched_and_write_is_committed(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database))

        self.assertEqual(
            tool.execute_query("INSERT INTO Orders (total) VALUES (45.0) RETURNING total"),
            ([{"total": 45.0}], ["total"], None),
        )
        self.assertEqual(
            tool.execute_query("SELECT COUNT(*) AS count FROM Orders")[0],
            [{"count": 3}],
        )

    def test_commits_writes_and_reports_invalid_sql(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database))

        result = tool.execute_query("INSERT INTO Orders (total) VALUES (45.0)")
        self.assertEqual(result, ([], [], None))
        self.assertEqual(
            tool.execute_query("SELECT COUNT(*) AS count FROM Orders")[0],
            [{"count": 3}],
        )

        rows, columns, error = tool.execute_query("SELECT * FROM does_not_exist")
        self.assertEqual(rows, [])
        self.assertEqual(columns, [])
        self.assertIn("no such table", error)


class TFIDFRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_paragraphs_and_ranks_relevant_document(self):
        (self.directory / "sales.md").write_text("Revenue and sales margin guidance.")
        (self.directory / "policy.md").write_text("Returns, refunds, and warranty policy.")
        (self.directory / "shipping.md").write_text("Delivery, freight, and carrier details.")
        retriever = TFIDFRetriever(str(self.directory))

        results = retriever.retrieve("What is the refund policy?", top_k=2)

        self.assertEqual(len(retriever.chunks), 3)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].source, "policy")
        self.assertGreater(results[0].score, results[1].score)

    def test_handles_tokenization_unknown_terms_and_empty_directory(self):
        docs = self.directory / "docs"
        docs.mkdir()
        (docs / "one.md").write_text("Hello, RETAIL-world!")
        retriever = TFIDFRetriever(str(docs))

        self.assertEqual(
            retriever._tokenize("Hello, RETAIL-world!"),
            ["hello", "retail", "world"],
        )
        self.assertEqual(retriever.retrieve("unknown", top_k=1)[0].score, 0.0)

        empty_docs = self.directory / "empty"
        empty_docs.mkdir()
        self.assertEqual(TFIDFRetriever(str(empty_docs)).retrieve("anything"), [])


if __name__ == "__main__":
    unittest.main()
