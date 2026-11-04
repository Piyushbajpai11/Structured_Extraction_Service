"""Checks that evaluation reports classify safe fallbacks honestly."""

import unittest

import evaluate


class EvaluationSummaryTests(unittest.TestCase):
    def test_safe_fallback_is_not_counted_as_a_success(self) -> None:
        result = {
            "expected_pre_call_rejection": False,
            "outcome": "safe_fallback",
            "field_matches": None,
            "reason_presence_match": None,
            "record_match_ignoring_reason_wording": None,
        }

        summary = evaluate.summarize([result])

        self.assertEqual(summary["successful_extractions"], 0)
        self.assertEqual(summary["safe_fallbacks"], 1)
        self.assertEqual(summary["inference_errors"], 0)
        self.assertEqual(summary["full_exact_matches"], 0)


if __name__ == "__main__":
    unittest.main()
