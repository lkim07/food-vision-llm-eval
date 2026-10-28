"""Normalize raw Colab prediction logs into one consistent JSONL format.

Usage: python scripts/clean_results.py
Reads results/raw/*_predictions_*.jsonl, writes results/clean/<model>__<split>.jsonl
"""
import json
import math
import re
from pathlib import Path

RAW_DIR = Path("results/raw")
OUT_DIR = Path("results/clean")

# Raw file suffix -> canonical model name
MODEL_NAMES = {
    "gemini_2.0_flash_light": "gemini-2.0-flash-lite",
    "gpt_4o_mini": "gpt-4o-mini",
    "claude": "claude-3-haiku",
}


def parse_maybe_json(value):
    """Turn JSON strings (optionally wrapped in ``` fences) into objects; NaN -> None."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip())
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def clean_record(raw):
    truth = {k: parse_maybe_json(v) for k, v in raw["truth"].items()}
    prediction = parse_maybe_json(raw["json"])
    return {
        "image_id": Path(raw["image_path"]).stem,
        "split": truth["food_type"],
        "parse_ok": isinstance(prediction, dict),
        "prediction": prediction,
        "truth": {
            "dish_name": truth["label"],
            "cooking_method": truth["cooking_method"],
            "portion_size": truth["portion_size"],
            # The original column was misnamed "calories_kcal" but holds all four values
            "nutrition": truth["calories_kcal"],
        },
        "latency_s": raw.get("latency"),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(RAW_DIR.glob("*_predictions_*.jsonl")):
        split, _, model_key = path.stem.partition("_predictions_")
        model = MODEL_NAMES[model_key]
        records = [clean_record(json.loads(line)) for line in path.open() if line.strip()]
        out_path = OUT_DIR / f"{model}__{split}.jsonl"
        with out_path.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        failed = sum(not r["parse_ok"] for r in records)
        print(f"{out_path.name:40} {len(records):4} records, {failed} unparsed")


if __name__ == "__main__":
    main()