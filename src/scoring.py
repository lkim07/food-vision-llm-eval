"""Scoring functions for food-vision LLM predictions.

Every function scores ONE prediction against ONE ground-truth record.
Functions return None when a record cannot be scored (e.g. missing ground truth),
so unscorable records are excluded instead of silently counted as right or wrong.
"""
import re
import unicodedata

NUTRIENTS = ["calories_kcal", "protein_g", "fat_g", "carbohydrate_g"]
TOLERANCE = 0.20  # +/-20% counts as "within tolerance"

# Exact values used in the few-shot prompt examples (calories, protein, fat, carbs)
FEW_SHOT_PROFILES = {
    (95, 0.5, 0.3, 25),     # apple
    (31, 2.5, 0.3, 6),      # broccoli
    (450, 28, 25, 22),      # fried chicken
    (680, 32, 20, 90),      # spaghetti bolognese
}

# ---------------------------------------------------------------------------
# Dish name
# ---------------------------------------------------------------------------
DROP_WORDS = {
    "fresh", "organic", "raw", "ripe", "whole", "sliced", "mixed", "assorted",
    "red", "green", "yellow", "white", "the", "a", "of", "on", "with", "and", "in",
}

# Dataset label quirks -> canonical name (applied after normalization)
NAME_SYNONYMS = {
    "sweetcorn": "corn",
    "sweet corn": "corn",
    "corn cob": "corn",
    "capsicum": "bell pepper",
    "paprika": "bell pepper",   # the fruit/veg dataset's "paprika" images are bell peppers
    "raddish": "radish",        # typo in dataset labels
    "jalepeno": "jalapeno",     # typo in dataset labels
    "jalapeno pepper": "jalapeno",
    "chilli pepper": "chili pepper",
    "sweetpotato": "sweet potato",
    "beetroot": "beet",
}


def _singular(token):
    if token.endswith("oes"):
        return token[:-2]          # tomatoes -> tomato
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]          # peppers -> pepper
    return token


def normalize_name(name):
    """Lowercase, strip accents/punctuation, drop descriptor words, singularize."""
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFD", name.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [_singular(t) for t in text.split() if t not in DROP_WORDS]
    joined = " ".join(tokens)
    return NAME_SYNONYMS.get(joined, joined)


def name_match(pred, truth):
    """True if one normalized name's tokens are a subset of the other's.

    "beef pho" vs "pho" -> True; "fried chicken" vs "chicken salad" -> False.
    An empty prediction never matches.
    """
    p, t = set(normalize_name(pred).split()), set(normalize_name(truth).split())
    if not p or not t:
        return False
    return p <= t or t <= p


# ---------------------------------------------------------------------------
# Nutrition
# ---------------------------------------------------------------------------
def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_nutrition(pred_profile, truth_profile):
    """Per-nutrient signed % error, absolute % error and within-tolerance flag.

    Returns None if there is no usable ground truth. A nutrient whose truth is
    missing or zero is skipped (None); a nutrient the model failed to return
    is counted as outside tolerance.
    """
    if not isinstance(truth_profile, dict):
        return None
    pred_profile = pred_profile if isinstance(pred_profile, dict) else {}
    scores = {}
    for key in NUTRIENTS:
        truth = _to_float(truth_profile.get(key))
        if truth is None or truth == 0:
            scores[key] = None
            continue
        pred = _to_float(pred_profile.get(key))
        if pred is None:
            scores[key] = {"pct_error": None, "abs_pct_error": None, "within_tol": False}
            continue
        err = (pred - truth) / truth
        scores[key] = {"pct_error": err, "abs_pct_error": abs(err), "within_tol": abs(err) <= TOLERANCE}
    return scores


def is_few_shot_copy(pred_profile):
    """True if the predicted macros exactly equal one of the prompt's few-shot examples."""
    if not isinstance(pred_profile, dict):
        return False
    values = tuple(_to_float(pred_profile.get(k)) for k in NUTRIENTS)
    return values in FEW_SHOT_PROFILES


# ---------------------------------------------------------------------------
# Portion size
# ---------------------------------------------------------------------------
def total_grams(portions):
    """Sum grams from entries like "beef:100g". Unparseable entries are ignored.

    Returns None if no entry has a gram value.
    """
    if not isinstance(portions, list):
        return None
    grams = []
    for item in portions:
        match = re.search(r":\s*([\d.]+)\s*g\b", str(item))
        if match:
            grams.append(float(match.group(1)))
    return sum(grams) if grams else None


def score_portion(pred_portions, truth_portions):
    """Compare TOTAL plate weight. Returns None if either side has no gram values."""
    pred, truth = total_grams(pred_portions), total_grams(truth_portions)
    if pred is None or not truth:
        return None
    err = (pred - truth) / truth
    return {"pct_error": err, "abs_pct_error": abs(err), "within_tol": abs(err) <= TOLERANCE}


# ---------------------------------------------------------------------------
# Cooking method
# ---------------------------------------------------------------------------
COOKING_ROOTS = {
    "fry": ["fried", "frying", "fry", "deep", "pan", "stir", "sauteed", "saute"],
    "boil": ["boiled", "boiling", "boil", "blanched", "poached", "simmered", "braised", "stewed"],
    "steam": ["steamed", "steaming", "steam"],
    "bake": ["baked", "baking", "bake", "roasted", "roast", "oven"],
    "grill": ["grilled", "grilling", "grill", "barbecue", "barbeque", "bbq", "flame"],
    "raw": ["raw", "uncooked", "fresh"],
}
_ROOT_OF = {word: root for root, words in COOKING_ROOTS.items() for word in words}


def cooking_roots(method):
    if isinstance(method, list):
        method = " ".join(map(str, method))
    if not isinstance(method, str):
        return set()
    words = re.sub(r"[^a-z ]", " ", method.lower()).split()
    return {_ROOT_OF[w] for w in words if w in _ROOT_OF}


def cooking_match(pred, truth):
    """True if prediction and truth share at least one cooking-method root.

    Returns None if the truth has no recognizable method.
    """
    t = cooking_roots(truth)
    if not t:
        return None
    return bool(cooking_roots(pred) & t)