"""Prompts sent to every model. Kept in one place so all models get identical input."""

# Zero-shot: describes the schema in words instead of showing example values,
# because example values leaked into the capstone outputs (see docs/FINDINGS.md).
NUTRITION_PROMPT = """You are a food analysis model. Analyze the food in the image and estimate
its nutrition for the ENTIRE portion shown.

Return ONLY a JSON object with exactly these keys:
- "dish_name": string, the common name of the dish
- "ingredients": list of strings
- "portion_size": list of strings formatted "ingredient:<grams>g", e.g. "rice:150g"
- "cooking_method": string, e.g. "grilled", "fried", "raw"
- "nutritional_profile": object with numeric keys
  "calories_kcal", "protein_g", "fat_g", "carbohydrate_g"

Use numbers, not ranges or strings, for every nutritional value."""
