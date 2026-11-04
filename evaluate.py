"""Compare local extraction results with the synthetic gold dataset."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from main import (
    MAX_MESSAGE_CHARS,
    MIN_MESSAGE_CHARS,
    OLLAMA_MODEL,
    InputValidationError,
    extract_message,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR.parent / "milestone-0"
MESSAGES_PATH = DATASET_DIR / "synthetic_messages.jsonl"
EXPECTED_PATH = DATASET_DIR / "expected_outputs.jsonl"
REPORT_PATH = PROJECT_DIR / "evaluation" / "latest_report.json"

OUTPUT_FIELDS = (
    "name",
    "contact",
    "intent",
    "urgency",
    "budget_signal",
    "requires_human_review",
    "reason_for_review",
)


def read_jsonl(path: Path) -> list[dict]:
    """Read JSON Lines data and report which file has a malformed row."""

    records = []
    with path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name} on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.name} on line {line_number}."
                )
            records.append(record)
    return records


def summarize(results: list[dict]) -> dict:
    """Calculate field accuracy and full-record exact-match rates."""

    eligible = [
        result
        for result in results
        if not result["expected_pre_call_rejection"]
    ]
    field_accuracy = {}
    for field in OUTPUT_FIELDS:
        correct = sum(
            result["field_matches"] is not None
            and result["field_matches"].get(field, False)
            for result in eligible
        )
        field_accuracy[field] = {
            "correct": correct,
            "total": len(eligible),
            "percent": round(correct * 100 / len(eligible), 1)
            if eligible
            else None,
        }

    exact_matches = sum(
        result["field_matches"] is not None
        and all(result["field_matches"].values())
        for result in eligible
    )
    expected_rejections = [
        result
        for result in results
        if result["expected_pre_call_rejection"]
    ]
    reason_presence_matches = sum(
        result["reason_presence_match"] is True for result in eligible
    )
    wording_independent_matches = sum(
        result["record_match_ignoring_reason_wording"] is True
        for result in eligible
    )
    return {
        "completed_examples": len(results),
        "expected_pre_call_rejections": len(expected_rejections),
        "correct_pre_call_rejections": sum(
            result["outcome"] == "expected_rejection"
            for result in expected_rejections
        ),
        "model_eligible_examples": len(eligible),
        "successful_extractions": sum(
            result["outcome"] == "success" for result in eligible
        ),
        "inference_errors": sum(
            result["outcome"] == "error" for result in eligible
        ),
        "safe_fallbacks": sum(
            result["outcome"] == "safe_fallback" for result in eligible
        ),
        "unexpected_rejections": sum(
            result["outcome"] == "unexpected_rejection" for result in eligible
        ),
        "unexpected_outputs_for_rejected_inputs": sum(
            result["outcome"] == "unexpected_output"
            for result in expected_rejections
        ),
        "field_accuracy": field_accuracy,
        "reason_presence_accuracy": {
            "correct": reason_presence_matches,
            "total": len(eligible),
            "percent": round(reason_presence_matches * 100 / len(eligible), 1)
            if eligible
            else None,
        },
        "full_exact_matches": exact_matches,
        "full_exact_match_total": len(eligible),
        "full_exact_match_percent": round(exact_matches * 100 / len(eligible), 1)
        if eligible
        else None,
        "record_matches_ignoring_reason_wording": wording_independent_matches,
        "record_match_ignoring_reason_wording_total": len(eligible),
        "record_match_ignoring_reason_wording_percent": round(
            wording_independent_matches * 100 / len(eligible), 1
        )
        if eligible
        else None,
    }


def parse_args() -> argparse.Namespace:
    """Read optional IDs for a faster, targeted prompt iteration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        help="Comma-separated dataset IDs to evaluate, for example P06,P17,P23.",
    )
    return parser.parse_args()


def save_report(
    started_at: str,
    results: list[dict],
    dataset_size: int,
    selected_ids: list[str] | None,
) -> None:
    """Save a checkpoint after each example so progress is not lost."""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at_utc": started_at,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": "ollama_local",
        "model": OLLAMA_MODEL,
        "annotation_version": "0.4",
        "dataset_size": dataset_size,
        "selected_ids": selected_ids,
        "completed_examples": len(results),
        "summary": summarize(results),
        "results": results,
    }
    temporary_path = REPORT_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(REPORT_PATH)


def main() -> None:
    args = parse_args()
    messages = read_jsonl(MESSAGES_PATH)
    expected_rows = read_jsonl(EXPECTED_PATH)
    message_ids = [row.get("id") for row in messages]
    expected_ids = [row.get("id") for row in expected_rows]

    if len(set(message_ids)) != len(message_ids):
        raise SystemExit("The synthetic message file contains duplicate IDs.")
    if message_ids != expected_ids:
        raise SystemExit(
            "The message and expected-output files must have the same IDs "
            "in the same order."
        )

    examples = list(zip(messages, expected_rows))
    selected_ids = None
    if args.ids:
        requested_ids = [item.strip().upper() for item in args.ids.split(",")]
        if not all(requested_ids):
            raise SystemExit("Provide a comma-separated list of non-empty IDs.")
        if len(set(requested_ids)) != len(requested_ids):
            raise SystemExit("The --ids list contains duplicates.")
        unknown_ids = sorted(set(requested_ids) - set(message_ids))
        if unknown_ids:
            raise SystemExit(f"Unknown dataset IDs: {', '.join(unknown_ids)}")
        selected_id_set = set(requested_ids)
        examples = [
            pair for pair in examples if pair[0].get("id") in selected_id_set
        ]
        selected_ids = [pair[0]["id"] for pair in examples]

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results = []
    print(
        f"Evaluating {len(examples)} of {len(messages)} examples with local "
        f"model {OLLAMA_MODEL}. Any selected pre-call rejection should not "
        "reach inference.",
        flush=True,
    )

    for index, (message_row, expected_row) in enumerate(examples, start=1):
        example_id = message_row["id"]
        message = message_row.get("message")
        expected = expected_row.get("expected")
        if not isinstance(message, str):
            raise SystemExit(f"Message {example_id} does not contain text.")

        message_length = len(message.strip())
        expected_pre_call_rejection = expected is None or (
            message_length < MIN_MESSAGE_CHARS
            or message_length > MAX_MESSAGE_CHARS
        )
        result_record = {
            "id": example_id,
            "expected_pre_call_rejection": expected_pre_call_rejection,
            "outcome": None,
            "expected": expected,
            "actual": None,
            "field_matches": None,
            "reason_presence_match": None,
            "record_match_ignoring_reason_wording": None,
            "error_type": None,
            "error_message": None,
        }

        print(f"[{index}/{len(examples)}] {example_id}: ", end="", flush=True)
        case_started = time.perf_counter()
        try:
            extraction = extract_message(message)
            actual = extraction.model_dump(mode="json")
        except InputValidationError as exc:
            result_record["outcome"] = (
                "expected_rejection"
                if expected_pre_call_rejection
                else "unexpected_rejection"
            )
            result_record["error_message"] = str(exc)
        except Exception as exc:  # Keep evaluating later examples after a model failure.
            result_record["outcome"] = "error"
            result_record["error_type"] = type(exc).__name__
            result_record["error_message"] = str(exc)
        else:
            result_record["actual"] = actual
            if expected_pre_call_rejection:
                result_record["outcome"] = "unexpected_output"
            elif actual.get("status") == "fallback":
                result_record["outcome"] = "safe_fallback"
                result_record["error_type"] = "ModelCallError"
                result_record["error_message"] = actual.get("reason_for_review")
            else:
                result_record["outcome"] = "success"
                result_record["field_matches"] = {
                    field: actual.get(field) == expected.get(field)
                    for field in OUTPUT_FIELDS
                }
                result_record["reason_presence_match"] = (
                    (actual.get("reason_for_review") is None)
                    == (expected.get("reason_for_review") is None)
                )
                result_record["record_match_ignoring_reason_wording"] = (
                    all(
                        result_record["field_matches"][field]
                        for field in OUTPUT_FIELDS
                        if field != "reason_for_review"
                    )
                    and result_record["reason_presence_match"]
                )

        results.append(result_record)
        save_report(started_at, results, len(messages), selected_ids)

        duration_seconds = round(time.perf_counter() - case_started, 1)
        if result_record["field_matches"] is not None:
            correct_fields = sum(result_record["field_matches"].values())
            print(
                f"{correct_fields}/{len(OUTPUT_FIELDS)} fields exact, "
                "record match excluding reason wording="
                f"{result_record['record_match_ignoring_reason_wording']}, "
                f"{duration_seconds}s",
                flush=True,
            )
            for field, matches in result_record["field_matches"].items():
                if not matches:
                    print(
                        f"  {field}: expected={expected[field]!r}; "
                        f"actual={actual.get(field)!r}",
                        flush=True,
                    )
        else:
            print(f"{result_record['outcome']}, {duration_seconds}s", flush=True)

    summary = summarize(results)
    print("\nEvaluation summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetailed report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
