"""Run every configured model on every image in a manifest.

Usage:
  python scripts/run_benchmark.py --dataset nutrition5k --limit 3   # quick check
  python scripts/run_benchmark.py --dataset all                     # full run

Writes one JSONL file per dataset and model: results/runs/<dataset>/<model>.jsonl
The run is resumable: images that already succeeded are skipped, failed ones are retried.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.parsing import parse_json_response  # noqa: E402
from src.prompts import NUTRITION_PROMPT  # noqa: E402
from src.providers import call_model, load_models  # noqa: E402

MANIFEST_DIR = Path("data/manifests")
RUNS_DIR = Path("results/runs")
DATASETS = ["mmfood", "nutrition5k"]
MAX_ATTEMPTS = 3


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def done_ids(out_path):
    """Images that already have a successful (error-free) result."""
    return {r["image_id"] for r in read_jsonl(out_path) if r.get("error") is None}


def run_one(model, record):
    """Call the model with retries. Always returns a result row, with `error` set on failure."""
    image_bytes = Path(record["image_path"]).read_bytes()
    row = {"image_id": record["image_id"], "model": model.name, "model_id": model.model_id}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = call_model(model, image_bytes, NUTRITION_PROMPT, "image/jpeg")
        except Exception as e:  # rate limits, timeouts, server errors
            row["error"] = f"{type(e).__name__}: {e}"[:300]
            if attempt < MAX_ATTEMPTS:
                time.sleep(5 * attempt)
            continue
        prediction = parse_json_response(r.text)
        row.update({
            "error": None,
            "parse_ok": prediction is not None,
            "prediction": prediction,
            "raw_text": r.text,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cost_usd": round(r.cost_usd(model), 6),
            "latency_s": round(r.latency_s, 3),
        })
        break
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--models", nargs="*", help="model names from configs/models.yaml (default: all)")
    parser.add_argument("--limit", type=int, help="only the first N images (for a quick check)")
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between calls")
    parser.add_argument("--max-cost", type=float, default=5.0, help="stop if this run spends more (USD)")
    args = parser.parse_args()

    load_dotenv()
    models = [m for m in load_models() if not args.models or m.name in args.models]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    spent = 0.0

    for dataset in datasets:
        records = read_jsonl(MANIFEST_DIR / f"{dataset}.jsonl")[: args.limit]
        for model in models:
            out_path = RUNS_DIR / dataset / f"{model.name}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            todo = [r for r in records if r["image_id"] not in done_ids(out_path)]
            print(f"\n{dataset} / {model.name}: {len(todo)} to run ({len(records) - len(todo)} already done)")

            with out_path.open("a") as out:
                for i, record in enumerate(todo, 1):
                    row = run_one(model, record)
                    out.write(json.dumps(row) + "\n")
                    out.flush()  # keep progress even if the run is interrupted
                    spent += row.get("cost_usd") or 0
                    status = "ERR " if row["error"] else ("OK  " if row["parse_ok"] else "JSON")
                    print(f"  [{i:3}/{len(todo)}] {status} {record['image_id']:22} total ${spent:.3f}")
                    if spent > args.max_cost:
                        sys.exit(f"Stopped: spent ${spent:.2f} > --max-cost {args.max_cost}")
                    time.sleep(args.sleep)

    print(f"\nDone. Spent ${spent:.3f} in this run.")


if __name__ == "__main__":
    main()