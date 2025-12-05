import os
import base64
import warnings
from textwrap import dedent
from typing import Any, Dict, List
import re
import yaml
from pathlib import Path

import pytest

from litellm import completion
from pillow_heif import register_heif_opener
from PIL import Image


# Get the LLM providers from environment variables

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


# Retrieve the evaluations cases from a YAML file

PROMPT_FOLDER = Path(__file__).resolve().parents[1] / "prompts"

# -------- YAML loader / parser (multi-step format) ----------
def load_evaluation_cases() -> List[Dict[str, Any]]:
    folder_path = Path('/evaluation')
    if not folder_path.exists() or not folder_path.is_dir():
        raise RuntimeError(f"Folder not found: {folder}")

    for path in folder_path.glob("*.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

    all_tests = []
    top_level_prompt = data.get('prompt')
    top_level_prompt_path = data.get('prompt_path')
    if top_level_prompt and top_level_prompt_path:
        raise ValueError(f"Cannot have both 'prompt' and 'prompt_path'."
                         "Either write the prompt into the YAML file or provide a prompt file path.")
    top_level_repeat = data.get("repeat", data.get("repeat"))
    top_level_threshold = data.get("threshold", data.get("threshold"))
    for test in data.get("cases", []):
        test_id = test.get("id")
        repeat = test.get("repeat", top_level_repeat)
        threshold = test.get("threshold", top_level_threshold)
        steps = test.get("steps", [])
        if not test_id:
            raise ValueError("Every test must have an 'id'.")
        if not steps:
            raise ValueError(f"Test '{test_id}' must contain at least one step.")
        parsed_steps = []
        for idx, step in enumerate(steps):
            step_content = []
            inp = step.get("input", {})
            prompt = inp.get("prompt") or top_level_prompt
            prompt_path = inp.get("prompt_path") or top_level_prompt_path
            img_path = inp.get("image_path")
            max_tokens = int(inp.get("max_tokens")) if inp.get("max_tokens") is not None else None
            if not (prompt or prompt_path) and not img_path:
                raise ValueError(f"Test '{test_id}', step {idx}: need text or image.")
            if (prompt and not top_level_prompt) and (prompt_path and not top_level_prompt_path):
                raise ValueError(f"Test '{test_id}', step {idx}: cannot have both 'prompt' and 'prompt_path'."
                                 "Either write the prompt into the YAML file or provide a prompt file path.")
            if prompt_path and not prompt_path.endswith('.md'):
                raise ValueError(f"Test '{test_id}', step {idx}: prompt_path must point to a .md file.")
            image_url = None
            if img_path:
                p = folder_path / Path(img_path)
                if not p.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")
                img_bytes = p.read_bytes()
                fmt = img_path.split(".")[-1].lower()
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                image_url = f"data:image/{fmt};base64,{b64}"
            if prompt:
                step_content.append({
                    "type": "text",
                    "text": prompt
                })
            if prompt_path:
                p = Path(PROMPT_FOLDER) / prompt_path
                if not p.exists():
                    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
                prompt = p.read_text()
                step_content.append({
                    "type": "text",
                    "text": prompt
                })
            if image_url:
                step_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                })
            expectations = step.get("expectations", [])
            parsed_steps.append({
                "content": step_content,
                "max_tokens": max_tokens,
                "expectations": expectations,
            })
        all_tests.append({"id": test_id,
                          "repeat": repeat,
                          "threshold": threshold,
                          "steps": parsed_steps})
    return all_tests


def expect_approx_pct(output: str, spec: Dict[str, Any]) -> None:
    value = float(spec["value"])
    tol_pct = float(spec["tolerance_pct"])

    v = float(output)
    lower = value * (1 - tol_pct / 100.0)
    upper = value * (1 + tol_pct / 100.0)

    if not (lower <= v <= upper):
        raise AssertionError(
            f"approx_pct failed: output={v}, expected≈{value} ±{tol_pct}%, range=({lower}, {upper})"
        )


def expect_in_range(actual: str, spec: Dict[str, Any]) -> None:
    v = float(actual)
    lo = float(spec["min"])
    hi = float(spec["max"])

    if not (lo <= v <= hi):
        raise AssertionError(
            f"in_range failed: output={v}, expected between {lo} and {hi}"
        )


def expect_approx_pct(actual: str, spec: Dict[str, Any]) -> None:
    value = float(spec["value"])
    tol_pct = float(spec["tolerance_pct"])

    v = float(actual)
    lower = value * (1 - tol_pct / 100.0)
    upper = value * (1 + tol_pct / 100.0)

    if not (lower <= v <= upper):
        raise AssertionError(
            f"approx_pct failed: output={v}, expected≈{value} ±{tol_pct}%, range=({lower}, {upper})"
        )


EVALUATION_CASES = load_evaluation_cases()


# @pytest.mark.parametrize(
#     "id, steps",
#     [(c["id"], c["steps"]) for c in EVALUATION_CASES],
#     ids=[c["id"] for c in EVALUATION_CASES],
# )
# @pytest.mark.parametrize("provider", PROVIDERS, ids=lambda x: x["model"] + (' on ' + x["api_base"] if x.get("api_base") else ''))
# @pytest.mark.repeated(times=5, threshold=0)
@pytest.mark.parametrize(
    "id, steps",
    [
        pytest.param(
            c["id"], c["steps"],
            marks=pytest.mark.repeated(times=c.get("repeat", 100), threshold=c.get("threshold", 0))
        )
        for c in EVALUATION_CASES
    ],
    ids=[c["id"] for c in EVALUATION_CASES]
)
@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda x: x["model"] + (' on ' + x["api_base"] if x.get("api_base") else ''))
def test_extract_calories(id, steps, provider):
    register_heif_opener()

    for step in steps:
        # TODO: Add support for multi-turn evaluation cases for models. (might do this, or implement it in agents)
        message_content = step["content"]
        max_tokens = step.get("max_tokens")
        expectations = step.get("expectations", [])

        messages = [
            {
                "role": "user",
                "content": message_content
            }
        ]
        kwargs = dict(
            model=provider["model"],
            messages=messages,
            api_key=provider.get("extra", {}).get("api_key"),
        )
        if provider.get("api_base"):
            kwargs["api_base"] = provider["api_base"]
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        with warnings.catch_warnings():
            response = completion(**kwargs)

        actual = response.get('choices')[0].to_dict().get('message', {}).get('content').strip()
        print(f"Test ID: {id}, Response: {actual}")

        for expectation in expectations:
            if expectation["type"] == "contains":
                assert expectation["value"] in actual, f"Expectation failed: response does not contain '{expectation['value']}'"
            elif expectation["type"] == "equals":
                assert actual == expectation["value"], f"Expectation failed: response '{actual}' != expected '{expectation['value']}'"
            elif expectation["type"] in {"regex", "regexp", "regular_expression", "match"}:
                if not re.search(expectation["value"], actual):
                    raise AssertionError(f"Expectation failed: response '{actual}' does not match regex '{expectation['value']}'")
            elif expectation["type"] in {"in_range", "range", "within_range"}:
                expect_in_range(actual, expectation)
            elif expectation["type"] in {"approx_pct", "approximate_percentage", "percent_error", "within_percentage"}:
                expect_approx_pct(actual, expectation)
            else:
                raise ValueError(f"Unknown expectation type: {expectation['type']}")
