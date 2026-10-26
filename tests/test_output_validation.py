import unittest

from agent.output_validation import matches_format_hint


class OutputValidationTests(unittest.TestCase):
    def test_accepts_signed_integers_consistently(self):
        for value in (-4, "-4", "+4", "0"):
            with self.subTest(value=value):
                self.assertTrue(matches_format_hint(value, "int"))

        for value in (True, "4.0", "There are 4 orders"):
            with self.subTest(value=value):
                self.assertFalse(matches_format_hint(value, "int"))

    def test_requires_finite_floats(self):
        for value in (3.5, "-2.75", "1e3"):
            with self.subTest(value=value):
                self.assertTrue(matches_format_hint(value, "float"))

        for value in (True, "not-a-number", "nan", "inf"):
            with self.subTest(value=value):
                self.assertFalse(matches_format_hint(value, "float"))

    def test_requires_complete_dictionary_syntax(self):
        hint = '{"category": str, "revenue": float}'

        self.assertTrue(matches_format_hint({"category": "Tea"}, hint))
        self.assertTrue(matches_format_hint('{"category": "Tea"}', hint))
        self.assertTrue(matches_format_hint("{'category': 'Tea'}", hint))
        self.assertFalse(matches_format_hint("result: {category: Tea}", hint))
        self.assertFalse(matches_format_hint("{} trailing prose", hint))

    def test_requires_complete_list_syntax(self):
        hint = "list[{product: str, revenue: float}]"

        self.assertTrue(matches_format_hint([], hint))
        self.assertTrue(matches_format_hint('[{"product": "Tea"}]', hint))
        self.assertFalse(matches_format_hint("[product: Tea]", hint))
        self.assertFalse(matches_format_hint("[] trailing prose", hint))

    def test_leaves_free_form_hints_unchanged(self):
        self.assertTrue(matches_format_hint("any answer", "short explanation"))


if __name__ == "__main__":
    unittest.main()
