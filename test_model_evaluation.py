import os
import warnings
from typing import Any, Dict, List
import re
import yaml
import json
from pathlib import Path

import pytest

from litellm import completion
from helpers.evaluation_helpers import (
    convert_image_to_jpeg_base64,
    expect_equality,
    expect_in_range,
    expect_approx_pct
)


# Get the LLM providers from environment variables


# Load providers from YAML files in the providers directory
PROVIDERS = []

# Get PROJECT_PATH from environment (required)
PROJECT_PATH_STR = os.getenv('PROJECT_PATH')
if not PROJECT_PATH_STR:
    raise RuntimeError("PROJECT_PATH environment variable is required")
PROJECT_PATH = Path(PROJECT_PATH_STR)
if not PROJECT_PATH.exists():
    raise RuntimeError(f"PROJECT_PATH does not exist: {PROJECT_PATH}")

PROVIDERS_DIR = PROJECT_PATH / "providers"
if not PROVIDERS_DIR.exists():
    raise RuntimeError(f"Providers directory not found: {PROVIDERS_DIR}")

PROMPT_FOLDER = PROJECT_PATH / "prompts"
RESULTS_DIR = PROJECT_PATH / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${VAR_NAME} with environment variable values."""
    if isinstance(obj, str):
        # Replace ${VAR_NAME} with environment variable
        import re
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace_var, obj)
    elif isinstance(obj, dict):
        return {k: substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]
    return obj

# Load provider YAML files
for provider_file in sorted(PROVIDERS_DIR.glob("*.yaml")):
    with open(provider_file, "r") as f:
        provider_config = yaml.safe_load(f)

    # Substitute environment variables
    provider_config = substitute_env_vars(provider_config)

    # Skip disabled providers
    if not provider_config.get("_enabled", True):
        continue

    # Ensure the provider has required fields
    if "model" not in provider_config:
        raise ValueError(f"Provider {provider_file.name} missing 'model' field")

    # Initialize extra dict if not present
    if "extra" not in provider_config:
        provider_config["extra"] = {}

    # Remove _enabled before adding to PROVIDERS (it's not a LiteLLM kwarg)
    provider_config.pop("_enabled", None)

    PROVIDERS.append(provider_config)

# -------- YAML loader / parser (multi-step format) ----------
def load_evaluation_cases() -> List[Dict[str, Any]]:
    # Get evaluation path from environment variable or use default
    evaluation_path_str = os.getenv('EVALUATION_PATH', 'evaluation')

    # If EVALUATION_PATH is relative, make it relative to PROJECT_PATH
    evaluation_path = Path(evaluation_path_str)
    if not evaluation_path.is_absolute():
        evaluation_path = PROJECT_PATH / evaluation_path

    if not evaluation_path.exists():
        raise RuntimeError(f"Evaluation path not found: {evaluation_path}")

    # If it's a file, load just that file
    if evaluation_path.is_file():
        if not evaluation_path.suffix == '.yaml':
            raise ValueError(f"Evaluation file must be a YAML file: {evaluation_path}")
        yaml_files = [evaluation_path]
    else:
        # If it's a directory, find all YAML files recursively
        yaml_files = sorted(evaluation_path.glob("**/*.yaml"))

    if not yaml_files:
        raise RuntimeError(f"No YAML files found in: {evaluation_path}")

    all_tests = []
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        # Base directory for resolving relative paths in this YAML file
        yaml_dir = yaml_file.parent

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
                    # Resolve image path relative to the YAML file's directory
                    p = yaml_dir / Path(img_path)
                    if not p.exists():
                        raise FileNotFoundError(f"Image not found: {img_path} (resolved to {p})")

                    image_url = convert_image_to_jpeg_base64(p, quality=60)

                if prompt:
                    step_content.append({
                        "type": "text",
                        "text": prompt
                    })
                if prompt_path:
                    p = PROMPT_FOLDER / prompt_path
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


EVALUATION_CASES = load_evaluation_cases()


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
        kwargs = {}
        allowed_keys = ['model', 'api_base', 'api_key', 'max_tokens']
        for key in allowed_keys:
            if provider.get(key):
                kwargs[key] = provider[key]
        kwargs["messages"] = messages

        with warnings.catch_warnings():
            start_time = time.time()
            response = completion(**kwargs)
            end_time = time.time()

            # Store timing data for this iteration
            call_duration = end_time - start_time
            if not hasattr(request.node, "_timing_data"):
                request.node._timing_data = []
            request.node._timing_data.append(call_duration)

        actual = response.get('choices')[0].to_dict().get('message', {}).get('content').strip()

        # Try to parse as JSON if the expectation value is a dict
        actual_parsed = actual
        for expectation in expectations:
            if isinstance(expectation.get("value"), dict):
                try:
                    actual_parsed = json.loads(actual)
                except (json.JSONDecodeError, ValueError):
                    # If parsing fails, keep as string
                    pass
                break

        for expectation in expectations:
            if expectation["type"] == "contains":
                assert expectation["value"] in actual, f"Expectation failed: response does not contain '{expectation['value']}'"
            elif expectation["type"] == "equals":
                expect_equality(actual_parsed, expectation)
            elif expectation["type"] in {"regex", "regexp", "regular_expression", "match"}:
                if not re.search(expectation["value"], actual):
                    raise AssertionError(f"Expectation failed: response '{actual}' does not match regex '{expectation['value']}'")
            elif expectation["type"] in {"in_range", "range", "within_range"}:
                expect_in_range(actual, expectation)
            elif expectation["type"] in {"approx_pct", "approximate_percentage", "percent_error", "within_percentage"}:
                expect_approx_pct(actual, expectation)
            else:
                raise ValueError(f"Unknown expectation type: {expectation['type']}")
