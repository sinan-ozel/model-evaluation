import os
import base64

import pytest

from litellm import completion
import litellm

SUPPLIERS = []

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
        SUPPLIERS.append(
            {
                "model": model,
                "api_base": OLLAMA_URL,
                "extra": {}
            }
        )

@pytest.mark.parametrize("supplier", SUPPLIERS, ids=lambda s: s["model"] + ' on ' + s["api_base"])
def test_nutrition_extraction(supplier):
    test_case = 'IMG_B768CE83-9FEC-461A-BE63-CDDF64EBEB58'
    with open(f'images/{test_case}.jpeg', "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    model = supplier["model"]
    # TODO: Remove the prompt - that's not the endpoint, this is not a chat service.
    prompt = """The image is the label of a packaged food product.
        Give me the number of calories per serving as integer.
        Do not print anything else, just the integer value.
    """
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
                        "url": encoded_image
                    }
                }
            ]
        }
    ]
    response = completion(
        model=model,
        messages=messages,
        api_base=supplier["api_base"],
        max_tokens=64,
    )
    actual = response.to_dict().get('choices')[0].get('message', {}).get('content').strip()
    print("Response:", actual)
    assert int(actual) == 180