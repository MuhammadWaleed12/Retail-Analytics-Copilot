import tempfile
import unittest
from pathlib import Path

from agent.rag.retrieval import TFIDFRetriever


class RetrievalIsolationTests(unittest.TestCase):
    def test_later_query_preserves_earlier_result_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "guide.md").write_text("refund policy\n\nshipping carrier")
            retriever = TFIDFRetriever(directory)
            first = retriever.retrieve("refund", top_k=2)
            snapshot = [(chunk.chunk_id, chunk.score) for chunk in first]

            second = retriever.retrieve("shipping", top_k=2)

            self.assertEqual(first[0].content, "refund policy")
            self.assertEqual(second[0].content, "shipping carrier")
            self.assertEqual(
                [(chunk.chunk_id, chunk.score) for chunk in first], snapshot
            )

    def test_caller_mutation_does_not_change_future_results(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "guide.md").write_text("refund policy")
            retriever = TFIDFRetriever(directory)
            result = retriever.retrieve("refund")[0]
            result.content = "caller annotation"
            result.source = "caller source"
            result.chunk_id = "caller id"
            result.score = -1.0

            fresh = retriever.retrieve("refund")[0]

            self.assertEqual(fresh.content, "refund policy")
            self.assertEqual(fresh.source, "guide")
            self.assertEqual(fresh.chunk_id, "guide::chunk0")
            self.assertGreater(fresh.score, 0.0)
