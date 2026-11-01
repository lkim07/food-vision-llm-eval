"""Render one Nutrition5k dish with each model's estimate next to the measured values.

Usage:
  python scripts/make_demo.py                      # picks a typical dish automatically
  python scripts/make_demo.py --image-id dish_123  # a specific dish
Writes docs/assets/demo_example.png. Uses saved runs; makes no API calls.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.providers import load_models  # noqa: E402
from src.scoring import NUTRIENTS, total_grams  # noqa: E402

MANIFEST = Path("data/manifests/nutrition5k.jsonl")
RUNS_DIR = Path("results/runs/nutrition5k")
OUT_PATH = Path("docs/assets/demo_example.png")
ROWS = [("Calories", "calories_kcal", "kcal"), ("Protein", "protein_g", "g"),
        ("Fat", "fat_g", "g"), ("Carbs", "carbohydrate_g", "g")]
TEXT, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
DISPLAY = {"gpt-4o-mini": "GPT-4o\nmini", "gemini-2.5-flash-lite": "Gemini 2.5\nFlash-Lite",
           "claude-haiku-4.5": "Claude\nHaiku 4.5"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_predictions(models):
    """{model: {image_id: prediction}} using the last successful result per image."""
    preds = {}
    for model in models:
        preds[model] = {}
        for row in read_jsonl(RUNS_DIR / f"{model}.jsonl"):
            if row.get("error") is None and row.get("prediction"):
                preds[model][row["image_id"]] = row["prediction"]
    return preds


def calorie_error(record, preds, models):
    """Mean absolute % calorie error across models, or None if any model is missing."""
    truth = record["truth"]["nutrition"]["calories_kcal"]
    errors = []
    for m in models:
        p = preds[m].get(record["image_id"], {}).get("nutritional_profile", {})
        try:
            errors.append(abs(float(p["calories_kcal"]) - truth) / truth)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
    return float(np.mean(errors))


def pick_typical(manifest, preds, models):
    """The dish whose average calorie error is closest to the median: not cherry-picked."""
    scored = [(calorie_error(r, preds, models), r) for r in manifest]
    scored = [(e, r) for e, r in scored if e is not None]
    median = np.median([e for e, _ in scored])
    return min(scored, key=lambda er: abs(er[0] - median))[1]


def fmt(value, unit):
    try:
        return f"{float(value):.0f} {unit}"
    except (TypeError, ValueError):
        return "–"


def render(record, preds, models):
    truth = record["truth"]
    header = ["", "Measured"] + [DISPLAY.get(m, m.replace("-", "\n", 1)) for m in models]
    cells = []
    for label, key, unit in ROWS:
        row = [label, fmt(truth["nutrition"][key], unit)]
        row += [fmt(preds[m].get(record["image_id"], {}).get("nutritional_profile", {}).get(key), unit)
                for m in models]
        cells.append(row)
    mass_row = ["Total weight", fmt(total_grams(truth["portion_size"]), "g")]
    mass_row += [fmt(total_grams(preds[m].get(record["image_id"], {}).get("portion_size")), "g") for m in models]
    cells.append(mass_row)

    fig, (ax_img, ax_tab) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                                         gridspec_kw={"width_ratios": [1, 1.6]})
    ax_img.imshow(Image.open(record["image_path"]))
    ax_img.axis("off")
    ax_tab.axis("off")
    table = ax_tab.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.1)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(cell.get_height() * 1.6)  # room for two-line model names
        cell.set_edgecolor("#d6d5d0")
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(TEXT)
        if row == 0 or col == 0:
            cell.get_text().set_weight("bold")
        if col == 1 and row > 0:
            cell.set_facecolor("#ebeae6")  # highlight the measured column
    fig.suptitle(f"Nutrition5k {record['image_id']}: model estimates vs. weighed ingredients",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.text(0.01, 0.02, "Dish chosen automatically as the one with median calorie error across models.",
             fontsize=8, color=MUTED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH} for {record['image_id']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", help="Nutrition5k dish id; default picks a typical dish")
    args = parser.parse_args()

    models = [m.name for m in load_models()]
    manifest = read_jsonl(MANIFEST)
    preds = load_predictions(models)
    if args.image_id:
        record = next(r for r in manifest if r["image_id"] == args.image_id)
    else:
        record = pick_typical(manifest, preds, models)
    render(record, preds, models)


if __name__ == "__main__":
    main()