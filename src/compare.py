"""Score the re-run (results/runs) against both label sets and compare model rankings.

Usage: python -m src.compare
Writes:
  results/rerun_summary.csv      one row per dataset x model
  results/rerun_summary.md       Markdown tables for the README / FINDINGS
  docs/assets/rank_shift.png     how model accuracy changes between label sets
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.providers import load_models
from src.scoring import NUTRIENTS, name_match, score_nutrition, score_portion

MANIFEST_DIR = Path("data/manifests")
RUNS_DIR = Path("results/runs")
ASSETS_DIR = Path("docs/assets")
DATASETS = {"mmfood": "MM-Food-100K (GPT-4o labels)", "nutrition5k": "Nutrition5k (measured labels)"}

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # assigned in configs/models.yaml order
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
NUTRIENT_LABELS = {"calories_kcal": "calories", "protein_g": "protein", "fat_g": "fat", "carbohydrate_g": "carbs"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_success(rows):
    """Keep the last error-free result per image (retries append new rows)."""
    best = {}
    for row in rows:
        if row.get("error") is None:
            best[row["image_id"]] = row
    return best


def score_run(manifest, results):
    """One row per manifest image. Missing or unparseable predictions count as wrong."""
    rows = []
    for record in manifest:
        result = results.get(record["image_id"])
        pred = (result or {}).get("prediction") or {}
        truth = record["truth"]
        row = {
            "answered": result is not None,
            "parse_ok": bool(result and result.get("parse_ok")),
            "cost_usd": (result or {}).get("cost_usd"),
            "latency_s": (result or {}).get("latency_s"),
            "name_correct": name_match(pred.get("dish_name"), truth["dish_name"]) if truth["dish_name"] else None,
        }
        portion = score_portion(pred.get("portion_size"), truth["portion_size"])
        # No gram values in the prediction counts as a miss, since the truth has a weight
        row["portion_within_tol"] = portion["within_tol"] if portion else False
        nutrition = score_nutrition(pred.get("nutritional_profile"), truth["nutrition"]) or {}
        for key in NUTRIENTS:
            s = nutrition.get(key)
            row[f"{key}_within_tol"] = s["within_tol"] if s else None
            row[f"{key}_abs_err"] = s["abs_pct_error"] if s else None
            row[f"{key}_err"] = s["pct_error"] if s else None
        rows.append(row)
    return pd.DataFrame(rows)


def pct(series):
    s = series.dropna().astype(float)
    return round(100 * s.mean(), 1) if len(s) else None


def summarize(df):
    out = {
        "images": len(df),
        "valid_json_%": pct(df["parse_ok"]),
        "portion_within_20_%": pct(df["portion_within_tol"]),
    }
    if df["name_correct"].notna().any():
        out["dish_name_%"] = pct(df["name_correct"])
    for key in NUTRIENTS:
        out[f"{NUTRIENT_LABELS[key]}_within_20_%"] = pct(df[f"{key}_within_tol"])
    out["macro_avg_within_20_%"] = round(
        sum(out[f"{NUTRIENT_LABELS[k]}_within_20_%"] for k in NUTRIENTS) / len(NUTRIENTS), 1)
    for key in NUTRIENTS:
        out[f"{NUTRIENT_LABELS[key]}_median_abs_err_%"] = round(100 * df[f"{key}_abs_err"].median(), 1)
    for key in NUTRIENTS:
        out[f"{NUTRIENT_LABELS[key]}_median_bias_%"] = round(100 * df[f"{key}_err"].median(), 1)
    out["cost_per_image_usd"] = round(df["cost_usd"].mean(), 5)
    out["median_latency_s"] = round(df["latency_s"].median(), 2)
    return out


def rank_shift_chart(summary, models, out_path):
    """Slopegraph: macro-average accuracy on each label set, one line per model."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = [0, 1]
    for color, model in zip(COLORS, models):
        ys = [summary.query("dataset == @d and model == @model")["macro_avg_within_20_%"].iloc[0]
              for d in DATASETS]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8)
        ax.annotate(f"{model}  {ys[1]:.0f}%", (1, ys[1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10, color=TEXT)
        ax.annotate(f"{ys[0]:.0f}%", (0, ys[0]), xytext=(-10, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=10, color=TEXT)
    ax.set_xticks(xs, [label.replace(" (", "\n(") for label in DATASETS.values()], color=TEXT)
    ax.set_xlim(-0.35, 1.9)
    ax.set_ylabel("Macronutrients within ±20%, average of 4 (%)", color=MUTED)
    ax.set_title("Macronutrient accuracy by label source", loc="left", fontsize=13, color=TEXT)
    ax.tick_params(colors=MUTED, length=0)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    models = [m.name for m in load_models()]
    rows = []
    for dataset in DATASETS:
        manifest = read_jsonl(MANIFEST_DIR / f"{dataset}.jsonl")
        for model in models:
            results = latest_success(read_jsonl(RUNS_DIR / dataset / f"{model}.jsonl"))
            rows.append({"dataset": dataset, "model": model, **summarize(score_run(manifest, results))})
    summary = pd.DataFrame(rows)

    summary["rank"] = summary.groupby("dataset")["macro_avg_within_20_%"].rank(ascending=False, method="min")
    summary.to_csv("results/rerun_summary.csv", index=False)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rank_shift_chart(summary, models, ASSETS_DIR / "rank_shift.png")

    with open("results/rerun_summary.md", "w") as f:
        for dataset, label in DATASETS.items():
            table = summary[summary["dataset"] == dataset].drop(columns="dataset").dropna(axis=1, how="all")
            f.write(f"## {label}\n\n{table.to_markdown(index=False)}\n\n")
    for dataset in DATASETS:
        print(f"\n=== {DATASETS[dataset]}")
        print(summary[summary["dataset"] == dataset].drop(columns="dataset").set_index("model").T.to_string())


if __name__ == "__main__":
    main()