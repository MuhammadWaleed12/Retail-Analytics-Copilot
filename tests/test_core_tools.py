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

    def test_rejects_database_mutations(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database))

        for query in (
            "INSERT INTO Orders (total) VALUES (45.0)",
            "UPDATE Orders SET total = 0",
            "DELETE FROM Orders",
            "DROP TABLE Orders",
        ):
            with self.subTest(query=query):
                rows, columns, error = tool.execute_query(query)
                self.assertEqual((rows, columns), ([], []))
                self.assertIn("readonly", error.lower())

        self.assertEqual(
            tool.execute_query("SELECT COUNT(*) AS count FROM Orders")[0],
            [{"count": 2}],
        )

    @unittest.skipIf(sqlite3.sqlite_version_info < (3, 35, 0), "RETURNING needs SQLite 3.35+")
    def test_rejects_writes_with_returning(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database))

        rows, columns, error = tool.execute_query(
            "INSERT INTO Orders (total) VALUES (45.0) RETURNING total"
        )

        self.assertEqual((rows, columns), ([], []))
        self.assertIn("readonly", error.lower())
        self.assertEqual(
            tool.execute_query("SELECT COUNT(*) AS count FROM Orders")[0],
            [{"count": 2}],
        )

    def test_reports_invalid_sql(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database))

        rows, columns, error = tool.execute_query("SELECT * FROM does_not_exist")

        self.assertEqual(rows, [])
        self.assertEqual(columns, [])
        self.assertIn("no such table", error)

    def test_rejects_oversized_result_sets(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(str(database), max_rows=1)

        rows, columns, error = tool.execute_query(
            "SELECT id, total FROM Orders ORDER BY id"
        )

        self.assertEqual(rows, [])
        self.assertEqual(columns, ["id", "total"])
        self.assertEqual(
            error,
            "Query returned more than 1 rows; add filters or LIMIT",
        )
        self.assertEqual(
            tool.execute_query("SELECT id, total FROM Orders ORDER BY id LIMIT 1"),
            ([{"id": 1, "total": 12.5}], ["id", "total"], None),
        )

    def test_interrupts_queries_that_exceed_instruction_budget(self):
        database = self.directory / "test.sqlite"
        create_database(database)
        tool = SQLiteTool(
            str(database),
            max_vm_steps=1_000,
            progress_interval=100,
        )

        rows, columns, error = tool.execute_query(
            "WITH RECURSIVE counter(value) AS ("
            "SELECT 1 UNION ALL SELECT value + 1 FROM counter"
            ") SELECT SUM(value) FROM counter"
        )

        self.assertEqual((rows, columns), ([], []))
        self.assertEqual(
            error,
            "Query execution interrupted after exceeding the "
            "1000-instruction budget",
        )

    def test_validates_query_limits(self):
        database = self.directory / "test.sqlite"
        create_database(database)

        for arguments, message in (
            ({"max_rows": 0}, "max_rows must be at least 1"),
            ({"max_vm_steps": 0}, "max_vm_steps must be at least 1"),
            ({"progress_interval": 0}, "progress_interval must be at least 1"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    SQLiteTool(str(database), **arguments)


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
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "policy")
        self.assertGreater(results[0].score, 0.0)

    def test_handles_tokenization_unknown_terms_and_empty_directory(self):
        docs = self.directory / "docs"
        docs.mkdir()
        (docs / "one.md").write_text("Hello, RETAIL-world!")
        retriever = TFIDFRetriever(str(docs))

        self.assertEqual(
            retriever._tokenize("Hello, RETAIL-world!"),
            ["hello", "retail", "world"],
        )
        self.assertEqual(retriever.retrieve("unknown", top_k=1), [])

        empty_docs = self.directory / "empty"
        empty_docs.mkdir()
        self.assertEqual(TFIDFRetriever(str(empty_docs)).retrieve("anything"), [])


if __name__ == "__main__":
    unittest.main()
