import os
import base64
import warnings

import pytest

from litellm import completion


def raise_value_error_if_env_missing(prefix: str, example_model_str: str):
    """Raises ValueError if either MODELS or API_KEY env vars are missing for a given provider."""
    if os.getenv(f"{prefix}_MODELS") and not os.getenv(f"{prefix}_API_KEY"):
        raise ValueError(f"If {prefix}_MODELS is set, {prefix}_API_KEY must also be set."
                        f" Example: {example_model_str}")
    if os.getenv(f"{prefix}_API_KEY") and not os.getenv(f"{prefix}_MODELS"):
        raise ValueError(f"If {prefix}_API_KEY is set, {prefix}_MODELS must also be set."
                        f" Example: {example_model_str}")



PROVIDERS = []

# ------------------ Ollama ------------------
OLLAMA_EXAMPLE = "OLLAMA_URL=http://localhost:11434; OLLAMA_MODELS=ollama/llava:7b,ollama/llava:13b"
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS")
if OLLAMA_MODELS and not OLLAMA_URL:
    raise ValueError("If OLLAMA_MODELS is set, OLLAMA_URL must also be set."
                     f" Example: {OLLAMA_EXAMPLE}")
if OLLAMA_URL:
    if not OLLAMA_MODELS:
        raise ValueError("If OLLAMA_URL is set, OLLAMA_MODELS must also be set."
                         f" Example: {OLLAMA_EXAMPLE}")
    for model in OLLAMA_MODELS.split(","):
        PROVIDERS.append(
            {
                "model": model,
                "api_base": OLLAMA_URL,
                "extra": {}
            }
        )

# ------------------ Mistral ------------------
MISTRAL_EXAMPLE = "MISTRAL_API_KEY=<your_key>; MISTRAL_MODELS=mistral/pixtral-12b-2409"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODELS = os.getenv("MISTRAL_MODELS")
if MISTRAL_MODELS and not MISTRAL_API_KEY:
    raise ValueError("If MISTRAL_MODELS is set, MISTRAL_API_KEY must also be set."
                     f" Example: {MISTRAL_EXAMPLE}")
if MISTRAL_API_KEY:
    if not MISTRAL_MODELS:
        raise ValueError(
            "If MISTRAL_API_KEY is set, MISTRAL_MODELS must also be set."
            f" Example: {MISTRAL_EXAMPLE}"
        )
    for model in MISTRAL_MODELS.split(","):
        PROVIDERS.append({
            "model": model,
            "api_base": "https://api.mistral.ai",
            "extra": {"api_key": MISTRAL_API_KEY}
        })

# ------------------ Anthropic (Claude) ------------------
ANTHROPIC_EXAMPLE = "ANTHROPIC_API_KEY=<your_key>; ANTHROPIC_MODELS=claude-sonnet-4-5-20250929"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODELS = os.getenv("ANTHROPIC_MODELS")
if ANTHROPIC_MODELS and not ANTHROPIC_API_KEY:
    raise ValueError("If ANTHROPIC_MODELS is set, ANTHROPIC_API_KEY must also be set."
                     f" Example: {ANTHROPIC_EXAMPLE}")
if ANTHROPIC_API_KEY:
    if not ANTHROPIC_MODELS:
        raise ValueError(
            "If ANTHROPIC_API_KEY is set, ANTHROPIC_MODELS must also be set."
            f" Example: {ANTHROPIC_EXAMPLE}"
        )
    for model in ANTHROPIC_MODELS.split(","):
        PROVIDERS.append({
            "model": model,
            "api_base": "https://api.anthropic.com",
            "extra": {"api_key": ANTHROPIC_API_KEY}
        })

# ------------------ Hugging Face ------------------
raise_value_error_if_env_missing("HF", "HF_API_KEY=<your_key>; HF_MODELS=microsoft/Phi-3-vision-128k-instruct")

HF_API_KEY = os.getenv("HF_API_KEY")
if HF_API_KEY:
    for model in os.getenv("HF_MODELS").split(","):
        PROVIDERS.append({
            "model": model.strip(),
            "extra": {"api_key": HF_API_KEY}
        })


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda x: x["model"] + (' on ' + x["api_base"] if x.get("api_base") else ''))
@pytest.mark.repeated(times=100, threshold=1)
def test_nutrition_extraction(provider):
    test_case = 'IMG_B768CE83-9FEC-461A-BE63-CDDF64EBEB58'
    image_format = 'jpeg'
    with open(f'images/{test_case}.{image_format}', "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    prompt = """The image is the label of a packaged food product.
        Give me the number of calories per serving as integer.
        Do not print anything else, just the integer value.
        Do not include units like "calories" or "kcal".
        No explanation, do not write a complete sentence,
        please just write the integer value."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f'data:image/{image_format};base64,' + encoded_image
                    }
                }
            ]
        }
    ]
    kwargs = dict(
        model=provider["model"],
        messages=messages,
        api_key=provider.get("extra", {}).get("api_key"),
        max_tokens=64,
    )
    if provider.get("api_base"):
        kwargs["api_base"] = provider["api_base"]
    with warnings.catch_warnings():
        response = completion(**kwargs)

    actual = response.get('choices')[0].to_dict().get('message', {}).get('content').strip()
    print("Response:", actual)
    assert actual.strip() == "180"
    assert int(actual) == 180