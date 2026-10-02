import tempfile
import unittest
from pathlib import Path

from agent.rag.retrieval import TFIDFRetriever


class RetrievalNoMatchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        Path(self.directory.name, "guide.md").write_text(
            "refund policy\n\nshipping carrier\n\nrefund warranty\n\n!!!"
        )
        self.retriever = TFIDFRetriever(self.directory.name)

    def test_no_match_queries_return_no_evidence(self):
        for query in ("", "   ", "?!", "astronomy"):
            with self.subTest(query=query):
                self.assertEqual(self.retriever.retrieve(query), [])

    def test_results_are_not_padded_with_unmatched_chunks(self):
        results = self.retriever.retrieve("refund", top_k=10)

        self.assertEqual(
            {result.content for result in results},
            {"refund policy", "refund warranty"},
        )
        self.assertTrue(all(result.score > 0 for result in results))

    def test_unknown_query_terms_preserve_known_matches(self):
        results = self.retriever.retrieve("astronomy shipping")

        self.assertEqual([result.content for result in results], ["shipping carrier"])

    def test_positive_matches_remain_ranked_and_limited(self):
        results = self.retriever.retrieve("refund policy", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].content, "refund policy")
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(len(self.retriever.retrieve("refund", top_k=1)), 1)

    def test_no_match_after_successful_query_returns_no_evidence(self):
        self.assertTrue(self.retriever.retrieve("refund"))
        self.assertEqual(self.retriever.retrieve("astronomy"), [])
