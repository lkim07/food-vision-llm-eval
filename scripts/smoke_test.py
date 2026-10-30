"""Send one image to every configured model to check keys, model IDs and parsing.

Usage: python scripts/smoke_test.py path/to/food.jpg
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompts import NUTRITION_PROMPT  # noqa: E402
from src.providers import call_model, load_models  # noqa: E402

MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def main(image_path):
    load_dotenv()
    path = Path(image_path)
    if not path.is_file():
        sys.exit(f"Image not found: {path.resolve()}")
    mime = MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        sys.exit(f"Unsupported image type '{path.suffix}'. Use one of: {', '.join(MIME_TYPES)}")
    image_bytes = path.read_bytes()

    for model in load_models():
        try:
            r = call_model(model, image_bytes, NUTRITION_PROMPT, mime)
        except Exception as e:  # show every failure, keep testing other models
            print(f"[FAIL] {model.name}: {type(e).__name__}: {e}")
            continue
        try:
            profile = json.loads(r.text)["nutritional_profile"]
            status = "OK  "
        except (json.JSONDecodeError, KeyError, TypeError):
            profile, status = r.text[:120], "BAD JSON"
        print(f"[{status}] {model.name:22} {r.latency_s:5.1f}s  "
              f"{r.input_tokens}+{r.output_tokens} tok  ${r.cost_usd(model):.5f}  {profile}")


if __name__ == "__main__":
    main(sys.argv[1])
