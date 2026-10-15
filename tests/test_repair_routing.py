import unittest

from agent.repair import choose_repair_target


class RepairRoutingTests(unittest.TestCase):
    def test_first_sql_repair_reexecutes_fixed_query(self):
        self.assertEqual(
            choose_repair_target(
                has_sql_error=True,
                has_sql_query=True,
                repair_count=0,
            ),
            "executor",
        )

    def test_sql_failure_without_reusable_query_regenerates_sql(self):
        for has_sql_query, repair_count in ((False, 0), (True, 1)):
            with self.subTest(
                has_sql_query=has_sql_query, repair_count=repair_count
            ):
                self.assertEqual(
                    choose_repair_target(
                        has_sql_error=True,
                        has_sql_query=has_sql_query,
                        repair_count=repair_count,
                    ),
                    "nl_to_sql",
                )

    def test_answer_repair_returns_to_synthesizer(self):
        self.assertEqual(
            choose_repair_target(
                has_sql_error=False,
                has_sql_query=True,
                repair_count=0,
            ),
            "synthesizer",
        )


if __name__ == "__main__":
    unittest.main()
