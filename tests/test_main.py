"""Fast checks for the extraction service; these tests never call Ollama."""

import json
import unittest
from unittest.mock import patch

import main


def make_candidate(**overrides: object) -> main.ModelExtraction:
    """Create one valid model response, with optional field overrides."""

    fields: dict[str, object] = {
        "name": "Asha Mehta",
        "contact": "asha.mehta@cedar.example",
        "intent": "pricing_inquiry",
        "urgency": "medium",
        "budget_signal": "not_mentioned",
        "intent_is_unclear": False,
        "additional_actionable_intents": [],
    }
    fields.update(overrides)
    return main.ModelExtraction.model_validate(fields)


class ValidateMessageTests(unittest.TestCase):
    def test_strips_outer_whitespace(self) -> None:
        self.assertEqual(main.validate_message("  Hello there.  "), "Hello there.")

    def test_rejects_empty_message(self) -> None:
        with self.assertRaisesRegex(main.InputValidationError, "cannot be empty"):
            main.validate_message("  \n  ")

    def test_rejects_message_shorter_than_three_characters(self) -> None:
        with self.assertRaisesRegex(main.InputValidationError, "at least 3"):
            main.validate_message("ab")

    def test_rejects_message_over_configured_limit(self) -> None:
        with patch.object(main, "MAX_MESSAGE_CHARS", 5):
            with self.assertRaisesRegex(main.InputValidationError, "limit is 5"):
                main.validate_message("123456")

    def test_rejects_non_text_input(self) -> None:
        with self.assertRaisesRegex(main.InputValidationError, "must be text"):
            main.validate_message(None)  # type: ignore[arg-type]


class SchemaTests(unittest.TestCase):
    def test_rejects_intent_outside_the_allowed_choices(self) -> None:
        with self.assertRaises(main.ValidationError):
            make_candidate(intent="unrecognized_intent")


class ConfigurationTests(unittest.TestCase):
    def test_message_limit_defaults_to_four_thousand(self) -> None:
        self.assertEqual(main._parse_max_message_chars(None), 4_000)

    def test_message_limit_can_be_overridden(self) -> None:
        self.assertEqual(main._parse_max_message_chars("2500"), 2_500)

    def test_message_limit_must_be_a_whole_number_at_least_three(self) -> None:
        for invalid_value in ("lots", "2"):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(RuntimeError):
                    main._parse_max_message_chars(invalid_value)


class ReviewRuleTests(unittest.TestCase):
    def test_clear_message_with_contact_needs_no_review(self) -> None:
        self.assertIsNone(main._review_reason(make_candidate()))

    def test_unclear_intent_requires_review(self) -> None:
        reason = main._review_reason(make_candidate(intent_is_unclear=True))
        self.assertIn("intent is unclear", reason or "")

    def test_multiple_actionable_intents_require_review(self) -> None:
        reason = main._review_reason(
            make_candidate(additional_actionable_intents=["support_request"])
        )
        self.assertIn("multiple actionable intents", reason or "")


class ExtractionFlowTests(unittest.TestCase):
    def test_extraction_uses_model_result_and_returns_public_schema(self) -> None:
        model_result = main.ModelCallResult(
            candidate=make_candidate(),
            prompt_tokens=100,
            completion_tokens=20,
            ollama_total_duration_ms=500,
            retry_count=0,
        )

        with (
            patch.object(main, "_request_model", return_value=model_result) as request,
            patch.object(main, "_log_usage") as log_usage,
        ):
            result = main.extract_message("Please share the pricing details.")

        request.assert_called_once_with("Please share the pricing details.")
        self.assertEqual(
            set(result.model_dump()),
            {
                "name",
                "contact",
                "intent",
                "urgency",
                "budget_signal",
                "requires_human_review",
                "reason_for_review",
            },
        )
        self.assertFalse(result.requires_human_review)
        log_usage.assert_called_once()
        self.assertEqual(log_usage.call_args.kwargs["status"], "success")

    def test_invalid_input_is_rejected_before_model_call(self) -> None:
        with (
            patch.object(main, "_request_model") as request,
            patch.object(main, "_log_usage") as log_usage,
        ):
            with self.assertRaises(main.InputValidationError):
                main.extract_message("")

        request.assert_not_called()
        self.assertEqual(log_usage.call_args.kwargs["status"], "rejected")
        self.assertFalse(log_usage.call_args.kwargs["inference_attempted"])

    def test_model_failure_returns_review_fallback_without_fake_fields(self) -> None:
        invalid_response = json.dumps(
            {"message": {"content": '{"intent":"not-a-real-intent"}'}}
        ).encode("utf-8")

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return invalid_response

        with (
            patch(
                "main.urllib.request.urlopen",
                side_effect=[FakeResponse(), FakeResponse()],
            ) as urlopen,
            patch("main.time.sleep"),
            patch.object(main, "_log_usage") as log_usage,
        ):
            result = main.extract_message("Please share the pricing details.")

        self.assertEqual(urlopen.call_count, 2)
        self.assertIsInstance(result, main.ExtractionFallback)
        result_data = result.model_dump(mode="json")
        self.assertEqual(result_data["status"], "fallback")
        self.assertTrue(result_data["requires_human_review"])
        self.assertIn("2 attempt(s)", result_data["reason_for_review"])
        self.assertNotIn("name", result_data)
        self.assertNotIn("intent", result_data)
        self.assertEqual(log_usage.call_args.kwargs["status"], "fallback")
        self.assertTrue(log_usage.call_args.kwargs["inference_attempted"])
        self.assertEqual(log_usage.call_args.kwargs["retry_count"], 1)

    def test_model_request_retries_once_after_a_timeout(self) -> None:
        candidate_json = json.dumps(
            make_candidate().model_dump(mode="json")
        )
        response_body = json.dumps(
            {
                "message": {"content": candidate_json},
                "prompt_eval_count": 100,
                "eval_count": 20,
                "total_duration": 500_000_000,
            }
        ).encode("utf-8")

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return response_body

        with (
            patch(
                "main.urllib.request.urlopen",
                side_effect=[TimeoutError("request timed out"), FakeResponse()],
            ) as urlopen,
            patch("main.time.sleep") as wait,
        ):
            result = main._request_model("A synthetic pricing message.")

        self.assertEqual(urlopen.call_count, 2)
        wait.assert_called_once_with(main.RETRY_WAIT_SECONDS)
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.candidate.name, "Asha Mehta")
        self.assertEqual(result.ollama_total_duration_ms, 500)

    def test_model_request_retries_once_after_invalid_schema_output(self) -> None:
        valid_content = json.dumps(make_candidate().model_dump(mode="json"))
        invalid_response = json.dumps(
            {"message": {"content": '{"intent":"not-a-real-intent"}'}}
        ).encode("utf-8")
        valid_response = json.dumps(
            {"message": {"content": valid_content}}
        ).encode("utf-8")

        class FakeResponse:
            def __init__(self, response_body: bytes) -> None:
                self.response_body = response_body

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return self.response_body

        with (
            patch(
                "main.urllib.request.urlopen",
                side_effect=[FakeResponse(invalid_response), FakeResponse(valid_response)],
            ) as urlopen,
            patch("main.time.sleep"),
        ):
            result = main._request_model("A synthetic pricing message.")

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.candidate.intent, "pricing_inquiry")


if __name__ == "__main__":
    unittest.main()
