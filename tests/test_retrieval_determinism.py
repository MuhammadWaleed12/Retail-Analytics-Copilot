import tempfile
import unittest
from pathlib import Path

from agent.rag.retrieval import TFIDFRetriever


class RetrievalDeterminismTests(unittest.TestCase):
    def test_document_and_tied_result_order_is_filename_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            # Create these in reverse lexical order. Filesystem iteration order is
            # not part of the retrieval contract and can differ across machines.
            Path(directory, "z-last.md").write_text("shared term", encoding="utf-8")
            Path(directory, "a-first.md").write_text("shared term", encoding="utf-8")

            retriever = TFIDFRetriever(directory)
            results = retriever.retrieve("shared", top_k=2)

            self.assertEqual(
                [chunk.chunk_id for chunk in retriever.chunks],
                ["a-first::chunk0", "z-last::chunk0"],
            )
            self.assertEqual(
                [chunk.chunk_id for chunk in results],
                ["a-first::chunk0", "z-last::chunk0"],
            )


if __name__ == "__main__":
    unittest.main()
