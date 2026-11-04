"""Summarize token usage, latency, retries, and estimated provider costs."""

import argparse
import json
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
USAGE_LOG_PATH = PROJECT_DIR / "logs" / "usage.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "evaluation" / "latest_cost_summary.json"


def read_usage(path: Path) -> list[dict]:
    """Read usage events and identify malformed log rows."""

    if not path.exists():
        raise SystemExit(
            f"No usage log found at {path}. Run the app or CLI once first."
        )

    events = []
    with path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON in {path.name} on line {line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise SystemExit(
                    f"Expected a JSON object in {path.name} on line {line_number}."
                )
            events.append(event)
    return events


def _non_negative_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def summarize(events: list[dict]) -> dict:
    """Aggregate local or remote usage events without reading message text."""

    inference_events = [event for event in events if event.get("inference_attempted")]
    provider_counts = {}
    model_counts = {}
    for event in events:
        provider = event.get("provider") or "unknown"
        model = event.get("model") or "unknown"
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        model_counts[model] = model_counts.get(model, 0) + 1
    token_events = [
        event
        for event in inference_events
        if _integer(event.get("prompt_tokens")) is not None
        and _integer(event.get("completion_tokens")) is not None
    ]
    latency_values = [
        latency
        for event in events
        if (latency := _non_negative_number(event.get("client_latency_ms")))
        is not None
    ]
    inference_latency_values = [
        latency
        for event in inference_events
        if (latency := _non_negative_number(event.get("client_latency_ms")))
        is not None
    ]
    provider_cost_values = [
        cost
        for event in inference_events
        if (cost := _non_negative_number(event.get("estimated_provider_cost_usd")))
        is not None
    ]
    cost_complete = len(provider_cost_values) == len(inference_events)
    total_provider_cost = (
        round(sum(provider_cost_values), 10) if cost_complete else None
    )

    total_tokens = sum(
        _integer(event.get("prompt_tokens"))
        + _integer(event.get("completion_tokens"))
        for event in token_events
    )
    retry_count = sum(
        _integer(event.get("retry_count")) or 0 for event in inference_events
    )
    inference_attempt_count = sum(
        (_integer(event.get("retry_count")) or 0) + 1
        for event in inference_events
    )
    average_latency = (
        round(sum(latency_values) / len(latency_values), 1)
        if latency_values
        else None
    )
    mean_cost_per_inference = (
        total_provider_cost / len(inference_events)
        if cost_complete and inference_events
        else None
    )

    return {
        "usage_events": len(events),
        "providers": provider_counts,
        "models": model_counts,
        "inference_requests": len(inference_events),
        "successful_requests": sum(event.get("status") == "success" for event in events),
        "safe_fallbacks": sum(event.get("status") == "fallback" for event in events),
        "inference_errors": sum(event.get("status") == "error" for event in events),
        "rejected_before_inference": sum(
            event.get("status") == "rejected" for event in events
        ),
        "inference_attempts_including_retries": inference_attempt_count,
        "retries": retry_count,
        "prompt_tokens": sum(
            _integer(event.get("prompt_tokens")) or 0 for event in token_events
        ),
        "completion_tokens": sum(
            _integer(event.get("completion_tokens")) or 0 for event in token_events
        ),
        "total_tokens": total_tokens,
        "requests_with_token_counts": len(token_events),
        "average_client_latency_ms": average_latency,
        "average_inference_latency_ms": (
            round(sum(inference_latency_values) / len(inference_latency_values), 1)
            if inference_latency_values
            else None
        ),
        "p50_inference_latency_ms": _percentile(inference_latency_values, 0.50),
        "p95_inference_latency_ms": _percentile(inference_latency_values, 0.95),
        "estimated_provider_cost_usd": total_provider_cost,
        "provider_cost_coverage_complete": cost_complete,
        "estimated_provider_cost_per_1000_inference_requests_usd": (
            round(mean_cost_per_inference * 1000, 8)
            if mean_cost_per_inference is not None
            else None
        ),
        "estimated_local_energy_cost_usd": None,
        "note": (
            "Provider cost is an estimate based on observed token counts and "
            "configured rates. Local Ollama calls have $0 provider cost; "
            "electricity and hardware costs are not measured."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=USAGE_LOG_PATH,
        help=f"Usage JSONL file (default: {USAGE_LOG_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save the JSON summary (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(read_usage(args.log))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Usage and cost summary")
    print(f"Usage events: {summary['usage_events']}")
    print(f"Inference requests: {summary['inference_requests']}")
    print(f"Inference attempts including retries: {summary['inference_attempts_including_retries']}")
    print(f"Retries: {summary['retries']}")
    print(
        "Tokens with complete counts: "
        f"{summary['total_tokens']} across "
        f"{summary['requests_with_token_counts']} requests"
    )
    print(
        "Average inference latency: "
        f"{summary['average_inference_latency_ms']} ms"
    )
    print(
        "95th percentile inference latency: "
        f"{summary['p95_inference_latency_ms']} ms"
    )
    cost = summary["estimated_provider_cost_usd"]
    if cost is None:
        print("Estimated provider cost: incomplete or unavailable")
    else:
        print(f"Estimated provider cost: ${cost:.8f}")
    cost_per_thousand = summary[
        "estimated_provider_cost_per_1000_inference_requests_usd"
    ]
    if cost_per_thousand is not None:
        print(f"Estimated cost per 1,000 inference requests: ${cost_per_thousand:.6f}")
    print("Estimated local energy cost: not measured")
    print(f"JSON summary saved to: {args.output}")


if __name__ == "__main__":
    main()
