"""Helper functions for model evaluation."""

import base64
import io
from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image
from pillow_heif import register_heif_opener

# Register HEIF opener for PIL
register_heif_opener()


def convert_image_to_jpeg_base64(image_path: Path, quality: int = 60) -> str:
    """
    Open an image, convert to RGB if needed, encode as JPEG, and return base64 data URL.

    Args:
        image_path: Path to the image file
        quality: JPEG quality (1-100)

    Returns:
        Base64-encoded data URL string (data:image/jpeg;base64,...)
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Encode to JPEG in memory
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format='JPEG', quality=quality)
    img_bytes_io.seek(0)
    img_bytes = img_bytes_io.read()

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def expect_equality(actual: Union[str, dict, int, float, None], spec: Dict[str, Any]) -> None:
    """
    Check if output equals expected value.
    Supports dict, str, int, float, and None types.
    For dict comparison, compares only non-None values in expected dict,
    but actual must not have extra keys beyond what's in expected.

    Special handling for LLM responses:
    - If expected is dict and actual is string containing JSON in markdown code blocks,
      extracts and parses the JSON for comparison.
    - For string comparisons, strips trailing whitespace and newlines from actual.

    Raises AssertionError if not equal.
    """
    import json
    import re

    expected = spec["value"]

    # If expected is dict but actual is string, try to extract JSON from the text.
    # Support direct JSON, JSON inside markdown code fences, or pretty-printed JSON.
    def _try_parse_json(text: str):
        # Try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to extract from markdown code fence
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except Exception:
                pass

        # Fallback: find first '{' and last '}' and try to parse that slice
        first = text.find('{')
        last = text.rfind('}')
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(text[first:last+1])
            except Exception:
                pass

        return None

    if isinstance(expected, dict) and isinstance(actual, str):
        parsed = _try_parse_json(actual)
        if parsed is not None:
            actual = parsed

    # Strip trailing whitespace from strings
    if isinstance(actual, str) and isinstance(expected, str):
        actual = actual.rstrip()

    # Handle dict comparison - only check non-None values in expected
    if isinstance(expected, dict) and isinstance(actual, dict):
        # Check for extra keys in actual
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        extra_keys = actual_keys - expected_keys
        if extra_keys:
            raise AssertionError(
                f"equality failed: actual has extra keys {extra_keys}"
            )

        # Check each expected key
        for key, expected_value in expected.items():
            if expected_value is not None:
                if key not in actual:
                    raise AssertionError(
                        f"equality failed for key '{key}': key missing in actual"
                    )
                actual_value = actual[key]
                if actual_value != expected_value:
                    raise AssertionError(
                        f"equality failed for key '{key}': output='{actual_value}', expected='{expected_value}'"
                    )
    elif actual != expected:
        raise AssertionError(
            f"equality failed: output='{actual}', expected='{expected}'"
        )


def expect_in_range(actual: Union[str, int, float], spec: Dict[str, Any]) -> None:
    """
    Check if output is within the range specified by spec['min'] and spec['max'].
    Note that output, spec['min'], and spec['max'] can be strings, but are converted to float for comparison.
    Raises AssertionError if not within range.
    """
    v = float(actual)
    lo = float(spec["min"])
    hi = float(spec["max"])

    if not (lo <= v <= hi):
        raise AssertionError(
            f"in_range failed: output={v}, expected between {lo} and {hi}"
        )


def expect_approx_pct(output: Union[str, int, float], spec: Dict[str, Any]) -> None:
    """
    Check if output is approximately equal to spec['value'] within spec['tolerance_pct'] percent.
    Note that output, spec['value'], and spec['tolerance_pct'] can be strings, but are converted to float for comparison.
    Raises AssertionError if not within range.
    """
    value = float(spec["value"])
    tol_pct = float(spec["tolerance_pct"])

    v = float(output)
    lower = value * (1 - tol_pct / 100.0)
    upper = value * (1 + tol_pct / 100.0)

    if not (lower <= v <= upper):
        raise AssertionError(
            f"approx_pct failed: output={v}, expected≈{value} ±{tol_pct}%, range=({lower}, {upper})"
        )
