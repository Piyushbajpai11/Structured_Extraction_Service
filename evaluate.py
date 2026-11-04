"""Compare local extraction results with the synthetic gold dataset."""

import argparse
import csv
import json
import re
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
DATASET_DIR = PROJECT_DIR / "data"
MESSAGES_PATH = DATASET_DIR / "synthetic_messages.jsonl"
EXPECTED_PATH = DATASET_DIR / "expected_outputs.jsonl"
REPORT_PATH = PROJECT_DIR / "evaluation" / "latest_report.json"
MARKDOWN_REPORT_PATH = PROJECT_DIR / "evaluation" / "latest_report.md"
EXAMPLES_CSV_PATH = PROJECT_DIR / "evaluation" / "latest_examples.csv"
FAILURES_CSV_PATH = PROJECT_DIR / "evaluation" / "latest_failures.csv"

OUTPUT_FIELDS = (
    "name",
    "contact",
    "intent",
    "urgency",
    "budget_signal",
    "requires_human_review",
    "reason_for_review",
)
CLASSIFICATION_FIELDS = (
    "intent",
    "urgency",
    "budget_signal",
    "requires_human_review",
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
    """Calculate exact, normalized-contact, and classification metrics."""

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
    normalized_contact_matches = sum(
        result.get("normalized_contact_match") is True for result in eligible
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
        "classification_metrics": _classification_metrics(eligible),
        "normalized_contact_accuracy": {
            "correct": normalized_contact_matches,
            "total": len(eligible),
            "percent": round(normalized_contact_matches * 100 / len(eligible), 1)
            if eligible
            else None,
        },
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


def _classification_metrics(results: list[dict]) -> dict:
    """Calculate per-label and macro/micro precision, recall, and F1."""

    metrics = {}
    for field in CLASSIFICATION_FIELDS:
        pairs = []
        for result in results:
            expected = result.get("expected")
            actual = result.get("actual")
            if not isinstance(expected, dict):
                continue
            expected_label = expected.get(field)
            predicted_label = (
                actual.get(field, "__no_output__")
                if isinstance(actual, dict)
                else "__no_output__"
            )
            pairs.append((expected_label, predicted_label))

        labels = {expected for expected, _ in pairs}
        labels.update(
            predicted
            for _, predicted in pairs
            if predicted != "__no_output__"
        )
        labels = sorted(labels, key=lambda value: str(value))
        per_label = {}
        total_tp = total_fp = total_fn = 0
        f1_values = []
        for label in labels:
            true_positive = sum(
                expected == label and predicted == label
                for expected, predicted in pairs
            )
            false_positive = sum(
                expected != label and predicted == label
                for expected, predicted in pairs
            )
            false_negative = sum(
                expected == label and predicted != label
                for expected, predicted in pairs
            )
            precision, recall, f1 = _precision_recall_f1(
                true_positive, false_positive, false_negative
            )
            support = sum(expected == label for expected, _ in pairs)
            per_label[str(label)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
            total_tp += true_positive
            total_fp += false_positive
            total_fn += false_negative
            f1_values.append(f1)

        micro_precision, micro_recall, micro_f1 = _precision_recall_f1(
            total_tp, total_fp, total_fn
        )
        metrics[field] = {
            "labels": per_label,
            "macro_average": {
                "precision": round(
                    sum(item["precision"] for item in per_label.values()) / len(per_label),
                    4,
                )
                if per_label
                else None,
                "recall": round(
                    sum(item["recall"] for item in per_label.values()) / len(per_label),
                    4,
                )
                if per_label
                else None,
                "f1": round(sum(f1_values) / len(f1_values), 4)
                if f1_values
                else None,
            },
            "micro_average": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            },
            "examples": len(pairs),
        }
    return metrics


def _precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return round(precision, 4), round(recall, 4), round(f1, 4)


def normalize_contact(value: object) -> str | None:
    """Compare synthetic email/phone contacts while ignoring common formatting."""

    if value is None:
        return None
    if not isinstance(value, str):
        return str(value).strip().casefold()
    contact = value.strip().strip(".,;:<>[]{}()'\"")
    if not contact:
        return None
    if "@" in contact:
        return contact.casefold()
    digits = re.sub(r"\D", "", contact)
    if digits:
        return digits
    return " ".join(contact.casefold().split())


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="")
    temporary_path.replace(path)


def _csv_row(result: dict) -> dict:
    field_matches = result.get("field_matches") or {}
    return {
        "id": result["id"],
        "outcome": result["outcome"],
        "expected_pre_call_rejection": result["expected_pre_call_rejection"],
        "full_exact_match": bool(field_matches)
        and all(field_matches.values()),
        "normalized_contact_match": result.get("normalized_contact_match"),
        "field_matches": json.dumps(field_matches, ensure_ascii=False),
        "expected": json.dumps(result.get("expected"), ensure_ascii=False),
        "actual": json.dumps(result.get("actual"), ensure_ascii=False),
        "error_type": result.get("error_type"),
        "error_message": result.get("error_message"),
    }


def _is_failure_case(result: dict) -> bool:
    """Exclude correctly handled pre-call rejections from model failure lists."""

    if result["outcome"] == "expected_rejection":
        return False
    field_matches = result.get("field_matches")
    return (
        result["outcome"] != "success"
        or not field_matches
        or not all(field_matches.values())
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    headers = (
        "id",
        "outcome",
        "expected_pre_call_rejection",
        "full_exact_match",
        "normalized_contact_match",
        "field_matches",
        "expected",
        "actual",
        "error_type",
        "error_message",
    )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def render_markdown_report(report: dict) -> str:
    summary = report["summary"]
    rows = [
        "# Evaluation report",
        "",
        f"- Started (UTC): `{report['started_at_utc']}`",
        f"- Model: `{report['model']}`",
        f"- Annotation version: `{report['annotation_version']}`",
        f"- Examples completed: {summary['completed_examples']} of {report['dataset_size']}",
        f"- Successful extractions: {summary['successful_extractions']}",
        f"- Correct pre-inference rejections: {summary['correct_pre_call_rejections']} of {summary['expected_pre_call_rejections']}",
        "",
        "## Field accuracy",
        "",
        "| Field | Exact matches | Accuracy | Macro F1 | Micro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, accuracy in summary["field_accuracy"].items():
        class_metrics = summary["classification_metrics"].get(field)
        macro_f1 = (
            class_metrics["macro_average"]["f1"] if class_metrics else None
        )
        micro_f1 = (
            class_metrics["micro_average"]["f1"] if class_metrics else None
        )
        rows.append(
            f"| {field} | {accuracy['correct']}/{accuracy['total']} | "
            f"{accuracy['percent']}% | {macro_f1 if macro_f1 is not None else '—'} | "
            f"{micro_f1 if micro_f1 is not None else '—'} |"
        )

    rows.extend(
        [
            "",
            "## Record-level results",
            "",
            f"- Full exact match: {summary['full_exact_matches']}/{summary['full_exact_match_total']} ({summary['full_exact_match_percent']}%).",
            f"- Match ignoring review-reason wording: {summary['record_matches_ignoring_reason_wording']}/{summary['record_match_ignoring_reason_wording_total']} ({summary['record_match_ignoring_reason_wording_percent']}%).",
            f"- Contact match after normalization: {summary['normalized_contact_accuracy']['correct']}/{summary['normalized_contact_accuracy']['total']} ({summary['normalized_contact_accuracy']['percent']}%).",
            "- Contact normalization lowercases emails and ignores phone punctuation; the expected labels remain unchanged.",
            "- Micro F1 pools decisions across labels. Macro F1 gives each label equal weight, including labels the model missed.",
            "",
            "## Failure cases",
            "",
        ]
    )
    failures = [result for result in report["results"] if _is_failure_case(result)]
    if failures:
        rows.append("| ID | Outcome | Mismatched fields | Error |")
        rows.append("|---|---|---|---|")
        for result in failures:
            mismatches = ", ".join(
                field
                for field, matched in (result.get("field_matches") or {}).items()
                if not matched
            ) or "—"
            error = (result.get("error_message") or "").replace("|", "\\|")
            rows.append(
                f"| {result['id']} | {result['outcome']} | {mismatches} | {error} |"
            )
    else:
        rows.append("No failures in the completed examples.")
    rows.extend(
        [
            "",
            "Detailed per-example records are in `latest_examples.csv`; the filtered failure list is in `latest_failures.csv`.",
            "",
        ]
    )
    return "\n".join(rows)


def write_report_artifacts(report: dict) -> None:
    """Write JSON, Markdown, all-example CSV, and failure-only CSV reports."""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        REPORT_PATH,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(MARKDOWN_REPORT_PATH, render_markdown_report(report))
    csv_rows = [_csv_row(result) for result in report["results"]]
    _write_csv(EXAMPLES_CSV_PATH, csv_rows)
    failure_rows = [
        _csv_row(result)
        for result in report["results"]
        if _is_failure_case(result)
    ]
    _write_csv(FAILURES_CSV_PATH, failure_rows)


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
    write_report_artifacts(report)


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
            "normalized_contact_match": None,
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
                result_record["normalized_contact_match"] = (
                    normalize_contact(actual.get("contact"))
                    == normalize_contact(expected.get("contact"))
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
    print(f"\nJSON report saved to: {REPORT_PATH}")
    print(f"Markdown summary saved to: {MARKDOWN_REPORT_PATH}")
    print(f"All-example CSV saved to: {EXAMPLES_CSV_PATH}")
    print(f"Failure-case CSV saved to: {FAILURES_CSV_PATH}")


if __name__ == "__main__":
    main()
