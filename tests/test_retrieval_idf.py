import tempfile
import unittest
from pathlib import Path

from agent.rag.retrieval import TFIDFRetriever


class SmallCorpusRetrievalTests(unittest.TestCase):
    def test_unique_term_ranks_matching_chunk_in_two_chunk_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "guide.md").write_text("shipping carrier\n\nrefund policy")
            retriever = TFIDFRetriever(directory)

            result = retriever.retrieve("refund", top_k=1)[0]

            self.assertEqual(result.content, "refund policy")
            self.assertGreater(result.score, 0.0)

    def test_term_in_all_but_one_chunk_keeps_positive_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "guide.md").write_text(
                "shipping carrier\n\nrefund policy\n\nrefund warranty"
            )
            retriever = TFIDFRetriever(directory)

            results = retriever.retrieve("refund", top_k=2)

            self.assertTrue(all("refund" in result.content for result in results))
            self.assertTrue(all(result.score > 0.0 for result in results))

    def test_single_chunk_and_common_terms_remain_searchable(self):
        for content in ("retail refund", "retail refund\n\nretail shipping"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                Path(directory, "guide.md").write_text(content)
                retriever = TFIDFRetriever(directory)

                self.assertGreater(retriever.retrieve("retail")[0].score, 0.0)
                self.assertTrue(all(weight > 0.0 for weight in retriever.idf.values()))
