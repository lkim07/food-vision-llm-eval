"""One interface for every model provider.

Usage:
    from src.providers import load_models, call_model
    for model in load_models():
        response = call_model(model, image_bytes, prompt)
"""
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path("configs/models.yaml")


@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str
    price_per_mtok: dict


@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float

    def cost_usd(self, model: ModelConfig) -> float:
        price = model.price_per_mtok
        return (self.input_tokens * price["input"] + self.output_tokens * price["output"]) / 1_000_000


def load_models(path=CONFIG_PATH):
    with open(path) as f:
        return [ModelConfig(**m) for m in yaml.safe_load(f)["models"]]


def call_model(model: ModelConfig, image_bytes: bytes, prompt: str, mime_type="image/jpeg") -> ModelResponse:
    """Send one image + prompt to a model and time the call."""
    # Imported lazily so you only need the SDK of the providers you actually run
    if model.provider == "openai":
        from src.providers.openai_provider import call
    elif model.provider == "anthropic":
        from src.providers.anthropic_provider import call
    elif model.provider == "gemini":
        from src.providers.gemini_provider import call
    else:
        raise ValueError(f"Unknown provider: {model.provider}")

    start = time.perf_counter()
    text, input_tokens, output_tokens = call(model.model_id, image_bytes, prompt, mime_type)
    return ModelResponse(text, input_tokens, output_tokens, time.perf_counter() - start)
