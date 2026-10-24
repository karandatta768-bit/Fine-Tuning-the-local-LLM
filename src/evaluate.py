"""Compare base and adapter behavior on the held-out processed test split."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path
from inference import answer_customer, load_model

ROOT = Path(__file__).resolve().parents[1]


def run(cases: list[dict], use_adapter: bool, label: str) -> dict:
    load_model(use_adapter)
    records, latencies = [], []
    progress_step = max(1, len(cases) // 4)
    correct = valid = complete = 0
    for case in cases:
        started = time.perf_counter()
        try:
            prediction = answer_customer(case["message"], use_adapter=use_adapter)
        except Exception as exc:
            prediction = {"error": type(exc).__name__, "detail": str(exc)}
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)
        ok = isinstance(prediction, dict) and "error" not in prediction
        fields = ok and {"category", "intent", "response"}.issubset(prediction)
        is_correct = bool(fields and prediction["intent"] == case.get("expected_intent"))
        valid += int(ok)
        complete += int(fields)
        correct += int(is_correct)
        records.append({**case, "prediction": prediction, "intent_correct": is_correct,
                        "latency_seconds": round(elapsed, 3)})
        if len(records) % progress_step == 0 or len(records) == len(cases):
            print(f"{label}: {len(records)}/{len(cases)}", flush=True)
    n = len(cases)
    metrics = {"model": label, "test_cases": n,
               "intent_accuracy": correct / n if n else None,
               "valid_json_rate": valid / n if n else None,
               "required_field_rate": complete / n if n else None,
               "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
               "median_latency_seconds": statistics.median(latencies) if latencies else None}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{label}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Evaluate a deterministic sample instead of the full test split.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    test_path = ROOT / "data/processed/test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError("Held-out test split missing. Run src/prepare_data.py first.")
    cases = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        expected = json.loads(messages[1]["content"])
        cases.append({"message": messages[0]["content"], "expected_intent": expected["intent"]})
    if args.limit is not None and len(cases) > args.limit:
        cases = random.Random(42).sample(cases, args.limit)
        print(f"Using deterministic sample of {len(cases)} held-out cases.")
    baseline = run(cases, False, "baseline")
    print("Baseline:", json.dumps(baseline, indent=2))
    try:
        finetuned = run(cases, True, "finetuned")
        metrics = {"baseline": baseline, "finetuned": finetuned}
    except FileNotFoundError as exc:
        print(f"Fine-tuned evaluation skipped: {exc}")
        metrics = {"baseline": baseline, "finetuned": None,
                   "note": "Train the adapter, then rerun evaluation for the comparison."}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
