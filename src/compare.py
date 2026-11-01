"""Score the re-run (results/runs) against both label sets and compare models.

Usage: python -m src.compare
Writes:
  results/rerun_summary.csv / .md     per dataset x model metrics
  results/rerun_significance.md       paired bootstrap 95% CIs for model differences
  docs/assets/rank_shift.png          macro accuracy on GPT-4o labels vs measured labels
  docs/assets/tolerance_curve.png     accuracy at 10-50% tolerance

Metrics
  PMAE   mean absolute error as a % of the mean label value (Nutrition5k paper's metric)
  MAE    mean absolute error in grams / kcal
  within_20  share of images within +/-20% (FDA nutrition label tolerance)
Missing or unparseable predictions count as misses for within_X and are excluded from MAE/PMAE.
"""
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.providers import load_models
from src.scoring import NUTRIENTS, name_match, score_nutrition, total_grams

MANIFEST_DIR = Path("data/manifests")
RUNS_DIR = Path("results/runs")
ASSETS_DIR = Path("docs/assets")
DATASETS = {"mmfood": "MM-Food-100K (GPT-4o labels)", "nutrition5k": "Nutrition5k (measured labels)"}
TOLERANCES = [0.10, 0.20, 0.30, 0.50]
N_BOOTSTRAP = 2000

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # assigned in configs/models.yaml order
TEXT, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#ebeae6"
SHORT = {"calories_kcal": "calories", "protein_g": "protein", "fat_g": "fat", "carbohydrate_g": "carbs"}
UNITS = {"calories_kcal": "kcal", "protein_g": "g", "fat_g": "g", "carbohydrate_g": "g"}


# ---------------------------------------------------------------------------
# Loading and per-image scoring
# ---------------------------------------------------------------------------
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
    """One row per manifest image with raw predictions, labels and errors."""
    rows = []
    for record in manifest:
        result = results.get(record["image_id"])
        pred = (result or {}).get("prediction") or {}
        truth = record["truth"]
        row = {
            "image_id": record["image_id"],
            "parse_ok": bool(result and result.get("parse_ok")),
            "cost_usd": (result or {}).get("cost_usd"),
            "latency_s": (result or {}).get("latency_s"),
            "name_correct": name_match(pred.get("dish_name"), truth["dish_name"]) if truth["dish_name"] else None,
            "mass_pred": total_grams(pred.get("portion_size")),
            "mass_true": total_grams(truth["portion_size"]),
        }
        nutrition = score_nutrition(pred.get("nutritional_profile"), truth["nutrition"]) or {}
        pred_profile = pred.get("nutritional_profile") if isinstance(pred.get("nutritional_profile"), dict) else {}
        for key in NUTRIENTS:
            s = nutrition.get(key)
            row[f"{key}_true"] = float(truth["nutrition"][key]) if s is not None else np.nan
            row[f"{key}_pred"] = _num(pred_profile.get(key)) if s is not None else np.nan
            row[f"{key}_rel_err"] = s["pct_error"] if s and s["pct_error"] is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def within(df, key, tol):
    """1.0 if within tolerance, 0.0 if outside or missing, NaN if the label is unscorable."""
    err = df[f"{key}_rel_err"].abs()
    return np.where(df[f"{key}_true"].isna(), np.nan, (err <= tol).astype(float))


def pmae(df, key):
    """Mean absolute error as a % of the mean label, over images with a prediction."""
    ok = df[f"{key}_pred"].notna() & df[f"{key}_true"].notna()
    if not ok.any():
        return np.nan
    return 100 * (df.loc[ok, f"{key}_pred"] - df.loc[ok, f"{key}_true"]).abs().mean() / df.loc[ok, f"{key}_true"].mean()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summarize(df):
    out = {"images": len(df), "valid_json_%": round(100 * df["parse_ok"].mean(), 1)}
    if df["name_correct"].notna().any():
        out["dish_name_%"] = round(100 * df["name_correct"].dropna().astype(float).mean(), 1)

    mass_ok = df["mass_pred"].notna() & df["mass_true"].notna()
    if mass_ok.any():
        m = df[mass_ok]
        out["mass_PMAE_%"] = round(100 * (m["mass_pred"] - m["mass_true"]).abs().mean() / m["mass_true"].mean(), 1)

    for key in NUTRIENTS:
        out[f"{SHORT[key]}_PMAE_%"] = round(pmae(df, key), 1)
    out["macro_PMAE_%"] = round(np.mean([out[f"{SHORT[k]}_PMAE_%"] for k in NUTRIENTS]), 1)
    for key in NUTRIENTS:
        ok = df[f"{key}_pred"].notna() & df[f"{key}_true"].notna()
        mae = (df.loc[ok, f"{key}_pred"] - df.loc[ok, f"{key}_true"]).abs().mean()
        out[f"{SHORT[key]}_MAE_{UNITS[key]}"] = round(mae, 1)
    for key in NUTRIENTS:
        out[f"{SHORT[key]}_within_20_%"] = round(100 * np.nanmean(within(df, key, 0.20)), 1)
    out["macro_within_20_%"] = round(np.mean([out[f"{SHORT[k]}_within_20_%"] for k in NUTRIENTS]), 1)
    for key in NUTRIENTS:
        out[f"{SHORT[key]}_median_bias_%"] = round(100 * df[f"{key}_rel_err"].median(), 1)
    out["cost_per_image_usd"] = round(df["cost_usd"].mean(), 5)
    out["median_latency_s"] = round(df["latency_s"].median(), 2)
    return out


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------
def _macro_within(df, idx):
    """Macro-average within-20% for each bootstrap resample (rows of idx)."""
    per_nutrient = [np.nanmean(within(df, k, 0.20)[idx], axis=1) for k in NUTRIENTS]
    return 100 * np.mean(per_nutrient, axis=0)


def _macro_pmae(df, idx):
    values = []
    for key in NUTRIENTS:
        pred, true = df[f"{key}_pred"].to_numpy()[idx], df[f"{key}_true"].to_numpy()[idx]
        ok = ~np.isnan(pred) & ~np.isnan(true)
        abs_err = np.where(ok, np.abs(pred - true), 0).sum(axis=1)
        values.append(100 * abs_err / np.where(ok, true, 0).sum(axis=1))
    return np.mean(values, axis=0)


def bootstrap(scored, seed=0):
    """Resample the same images for every model, so differences are paired."""
    n = len(next(iter(scored.values())))
    idx = np.random.default_rng(seed).integers(0, n, size=(N_BOOTSTRAP, n))
    samples = {m: {"macro_within_20": _macro_within(df, idx), "macro_PMAE": _macro_pmae(df, idx)}
               for m, df in scored.items()}
    rows = []
    for metric in ["macro_within_20", "macro_PMAE"]:
        for a, b in itertools.combinations(scored, 2):
            diff = samples[a][metric] - samples[b][metric]
            low, high = np.percentile(diff, [2.5, 97.5])
            rows.append({"metric": metric, "comparison": f"{a} - {b}",
                         "mean_diff": round(diff.mean(), 1), "ci95_low": round(low, 1),
                         "ci95_high": round(high, 1), "significant": not (low <= 0 <= high)})
    cis = {m: {k: np.percentile(v, [2.5, 97.5]).round(1).tolist() for k, v in s.items()}
           for m, s in samples.items()}
    return pd.DataFrame(rows), cis


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def rank_shift_chart(summary, models, out_path):
    """Slopegraph: macro within-20% on each label set, one line per model."""
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=SURFACE)
    _style(ax)
    for color, model in zip(COLORS, models):
        ys = [summary.query("dataset == @d and model == @model")["macro_within_20_%"].iloc[0] for d in DATASETS]
        ax.plot([0, 1], ys, color=color, linewidth=2, marker="o", markersize=8)
        ax.annotate(f"{model}  {ys[1]:.0f}%", (1, ys[1]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=10, color=TEXT)
        ax.annotate(f"{ys[0]:.0f}%", (0, ys[0]), xytext=(-10, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=10, color=TEXT)
    ax.set_xticks([0, 1], [label.replace(" (", "\n(") for label in DATASETS.values()], color=TEXT)
    ax.set_xlim(-0.35, 1.9)
    ax.set_ylabel("Macronutrients within ±20%, average of 4 (%)", color=MUTED)
    ax.set_title("GPT-4o mini's lead on GPT-4o labels disappears on measured labels",
                 loc="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def tolerance_curve_chart(scored_by_dataset, models, out_path):
    """Macro accuracy at each tolerance, one panel per dataset, shared y-axis."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, facecolor=SURFACE)
    xs = [int(t * 100) for t in TOLERANCES]
    for ax, (dataset, label) in zip(axes, DATASETS.items()):
        _style(ax)
        for color, model in zip(COLORS, models):
            df = scored_by_dataset[dataset][model]
            ys = [100 * np.mean([np.nanmean(within(df, k, t)) for k in NUTRIENTS]) for t in TOLERANCES]
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, label=model)
        ax.set_title(label, loc="left", fontsize=11, color=TEXT)
        ax.set_xticks(xs, [f"±{x}%" for x in xs], color=TEXT)
        ax.axvline(20, color=MUTED, linewidth=1, linestyle=":")
    axes[0].set_ylabel("Macronutrients within tolerance, average of 4 (%)", color=MUTED)
    axes[0].set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="lower center", ncol=len(models))
    fig.suptitle("Accuracy at different error tolerances (dotted line: ±20% used in tables)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    models = [m.name for m in load_models()]
    scored = {}
    rows = []
    for dataset in DATASETS:
        manifest = read_jsonl(MANIFEST_DIR / f"{dataset}.jsonl")
        scored[dataset] = {}
        for model in models:
            df = score_run(manifest, latest_success(read_jsonl(RUNS_DIR / dataset / f"{model}.jsonl")))
            scored[dataset][model] = df
            rows.append({"dataset": dataset, "model": model, **summarize(df)})
    summary = pd.DataFrame(rows)
    summary["rank_by_within_20"] = summary.groupby("dataset")["macro_within_20_%"].rank(ascending=False, method="min")
    summary["rank_by_PMAE"] = summary.groupby("dataset")["macro_PMAE_%"].rank(ascending=True, method="min")
    summary.to_csv("results/rerun_summary.csv", index=False)

    with open("results/rerun_summary.md", "w") as f:
        for dataset, label in DATASETS.items():
            table = summary[summary["dataset"] == dataset].drop(columns="dataset").dropna(axis=1, how="all")
            f.write(f"## {label}\n\n{table.to_markdown(index=False)}\n\n")

    with open("results/rerun_significance.md", "w") as f:
        f.write(f"Paired bootstrap over images, {N_BOOTSTRAP} resamples, 95% CI. "
                "A difference is significant if its CI excludes 0.\n"
                "macro_within_20: higher is better. macro_PMAE: lower is better.\n\n")
        for dataset, label in DATASETS.items():
            diffs, cis = bootstrap(scored[dataset])
            ci_table = pd.DataFrame([{"model": m, "macro_within_20_CI95": c["macro_within_20"],
                                      "macro_PMAE_CI95": c["macro_PMAE"]} for m, c in cis.items()])
            f.write(f"## {label}\n\n{ci_table.to_markdown(index=False)}\n\n{diffs.to_markdown(index=False)}\n\n")
            print(f"\n=== {label}: model differences (95% CI)")
            print(diffs.to_string(index=False))

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rank_shift_chart(summary, models, ASSETS_DIR / "rank_shift.png")
    tolerance_curve_chart(scored, models, ASSETS_DIR / "tolerance_curve.png")

    for dataset, label in DATASETS.items():
        print(f"\n=== {label}")
        print(summary[summary["dataset"] == dataset].drop(columns="dataset").set_index("model").T.to_string())


if __name__ == "__main__":
    main()