from src.scoring import (
    cooking_match, is_few_shot_copy, name_match, score_nutrition, score_portion, total_grams,
)


def test_name_match_basic_and_synonyms():
    assert name_match("Fried Chicken", "fried chicken")
    assert name_match("bell peppers", "capsicum")
    assert name_match("corn on the cob", "sweetcorn")
    assert name_match("beef pho", "Pho")


def test_name_match_rejects_old_false_positives():
    assert not name_match("", "orange")                       # empty prediction
    assert not name_match("eggplant", "egg")                  # substring, not same food
    assert not name_match("fried chicken", "chicken salad")   # one shared word


def test_nutrition_skips_missing_truth_and_flags_missing_pred():
    truth = {"calories_kcal": 400, "protein_g": 30, "fat_g": 0, "carbohydrate_g": 15}
    pred = {"calories_kcal": 450, "protein_g": 28}
    scores = score_nutrition(pred, truth)
    assert scores["calories_kcal"]["within_tol"]              # +12.5%
    assert scores["fat_g"] is None                            # truth is 0 -> unscorable
    assert scores["carbohydrate_g"]["within_tol"] is False    # model omitted it
    assert score_nutrition(pred, None) is None


def test_portion_uses_total_grams_and_ignores_unparseable():
    assert total_grams(["noodles:200g", "beef:100g"]) == 300
    assert total_grams(["potato:unknown"]) is None
    assert score_portion(["potato:unknown"], ["potato:200g"]) is None
    assert score_portion(["chicken:200g"], ["chicken:300g"])["within_tol"] is False


def test_few_shot_copy_detection():
    assert is_few_shot_copy({"calories_kcal": 450, "protein_g": 28, "fat_g": 25, "carbohydrate_g": 22})
    assert not is_few_shot_copy({"calories_kcal": 451, "protein_g": 28, "fat_g": 25, "carbohydrate_g": 22})


def test_cooking_match():
    assert cooking_match("pan fried", "Frying")
    assert not cooking_match("baked", "boiled")
    assert cooking_match("fried", "unknown") is None