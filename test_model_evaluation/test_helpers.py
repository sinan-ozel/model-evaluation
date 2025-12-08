import pytest
import sys
from pathlib import Path
from PIL import Image
import io
import base64

# Import from helpers/evaluation_helpers.py
sys.path.insert(0, '/')

from helpers.evaluation_helpers import (
    expect_equality,
    expect_in_range,
    expect_approx_pct,
    convert_image_to_jpeg_base64
)


class TestExpectEquality:
    """Test the expect_equality function."""

    def test_equality_passes_with_matching_strings(self):
        """Should pass when actual equals expected string value."""
        expect_equality("180", {"value": "180"})

    def test_equality_passes_with_matching_dicts(self):
        """Should pass when actual equals expected dict value."""
        expect_equality({"a": 1}, {"value": {"a": 1}})

    def test_equality_fails_with_different_strings(self):
        """Should raise AssertionError when strings don't match."""
        with pytest.raises(AssertionError, match="equality failed"):
            expect_equality("180", {"value": "200"})

    def test_equality_fails_with_different_types(self):
        """Should raise AssertionError when types don't match."""
        with pytest.raises(AssertionError, match="equality failed"):
            expect_equality("180", {"value": 180})

    def test_equality_fails_with_dict(self):
        """Should raise AssertionError when types don't match."""
        with pytest.raises(AssertionError, match="equality failed"):
            expect_equality({"a": 1}, {"value": {"a": 2}})

    def test_equality_passes_with_matching_dicts_with_different_sortinf(self):
        """Should pass when actual equals expected dict value."""
        expect_equality({"a": 1, "b": 2}, {"value": {"b": 2, "a": 1}})

    def test_equality_fails_with_dict_if_key_missing(self):
        """Should raise AssertionError when keys are missing."""
        with pytest.raises(AssertionError, match="equality failed"):
            expect_equality({"a": 1}, {"value": {"a": 1, "b": 2}})

    def test_equality_fails_with_dict_if_extra_key(self):
        """Should raise AssertionError when keys are missing."""
        with pytest.raises(AssertionError, match="equality failed"):
            expect_equality({"a": 1, "b": 2}, {"value": {"a": 1}})

    def test_equality_fails_with_dict_first_is_string(self):
        """Should raise AssertionError when keys are missing."""
        with pytest.raises(AssertionError, match="equality failed"):
            expect_equality('{"a": 1}', {"value": {"a": 1}})



class TestExpectInRange:
    """Test the expect_in_range function."""

    def test_in_range_passes_with_value_within_range(self):
        """Should pass when value is within min and max."""
        expect_in_range("50", {"min": "0", "max": "100"})

    def test_in_range_passes_at_lower_boundary(self):
        """Should pass when value equals min."""
        expect_in_range("0", {"min": "0", "max": "100"})

    def test_in_range_passes_at_upper_boundary(self):
        """Should pass when value equals max."""
        expect_in_range("100", {"min": "0", "max": "100"})

    def test_in_range_accepts_numeric_types(self):
        """Should accept int and float inputs and convert to float."""
        expect_in_range(50, {"min": 0, "max": 100})
        expect_in_range(50.5, {"min": 0.0, "max": 100.0})

    def test_in_range_fails_below_minimum(self):
        """Should raise AssertionError when value is below min."""
        with pytest.raises(AssertionError, match="in_range failed"):
            expect_in_range("-10", {"min": "0", "max": "100"})

    def test_in_range_fails_above_maximum(self):
        """Should raise AssertionError when value is above max."""
        with pytest.raises(AssertionError, match="in_range failed"):
            expect_in_range("150", {"min": "0", "max": "100"})


class TestExpectApproxPct:
    """Test the expect_approx_pct function."""

    def test_approx_pct_passes_with_exact_value(self):
        """Should pass when output exactly matches value."""
        expect_approx_pct("100", {"value": "100", "tolerance_pct": "10"})

    def test_approx_pct_passes_within_positive_tolerance(self):
        """Should pass when output is within tolerance above value."""
        expect_approx_pct("105", {"value": "100", "tolerance_pct": "10"})

    def test_approx_pct_passes_within_negative_tolerance(self):
        """Should pass when output is within tolerance below value."""
        expect_approx_pct("95", {"value": "100", "tolerance_pct": "10"})

    def test_approx_pct_passes_at_upper_boundary(self):
        """Should pass when output equals upper boundary (value + tolerance%)."""
        expect_approx_pct("110", {"value": "100", "tolerance_pct": "10"})

    def test_approx_pct_passes_at_lower_boundary(self):
        """Should pass when output equals lower boundary (value - tolerance%)."""
        expect_approx_pct("90", {"value": "100", "tolerance_pct": "10"})

    def test_approx_pct_accepts_numeric_types(self):
        """Should accept int and float inputs and convert to float."""
        expect_approx_pct(105, {"value": 100, "tolerance_pct": 10})
        expect_approx_pct(105.0, {"value": 100.0, "tolerance_pct": 10.0})

    def test_approx_pct_fails_above_tolerance(self):
        """Should raise AssertionError when output exceeds upper tolerance."""
        with pytest.raises(AssertionError, match="approx_pct failed"):
            expect_approx_pct("111", {"value": "100", "tolerance_pct": "10"})

    def test_approx_pct_fails_below_tolerance(self):
        """Should raise AssertionError when output is below lower tolerance."""
        with pytest.raises(AssertionError, match="approx_pct failed"):
            expect_approx_pct("89", {"value": "100", "tolerance_pct": "10"})

    def test_approx_pct_with_small_tolerance(self):
        """Should work correctly with small tolerance percentages."""
        expect_approx_pct("100.5", {"value": "100", "tolerance_pct": "1"})
        with pytest.raises(AssertionError):
            expect_approx_pct("102", {"value": "100", "tolerance_pct": "1"})


class TestConvertImageToJpegBase64:
    """Test the convert_image_to_jpeg_base64 function."""

    @pytest.fixture
    def temp_rgb_image(self, tmp_path):
        """Create a temporary RGB image for testing."""
        img_path = tmp_path / "test_rgb.png"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)
        return img_path

    @pytest.fixture
    def temp_rgba_image(self, tmp_path):
        """Create a temporary RGBA image for testing."""
        img_path = tmp_path / "test_rgba.png"
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(img_path)
        return img_path

    def test_converts_rgb_image_to_base64(self, temp_rgb_image):
        """Should convert RGB image to base64 data URL."""
        result = convert_image_to_jpeg_base64(temp_rgb_image)
        assert result.startswith("data:image/jpeg;base64,")

        # Verify it's valid base64
        base64_data = result.split(",", 1)[1]
        decoded = base64.b64decode(base64_data)
        assert len(decoded) > 0

    def test_converts_rgba_to_rgb(self, temp_rgba_image):
        """Should convert RGBA image to RGB before encoding."""
        result = convert_image_to_jpeg_base64(temp_rgba_image)
        assert result.startswith("data:image/jpeg;base64,")

        # Verify the image can be decoded and is RGB
        base64_data = result.split(",", 1)[1]
        decoded = base64.b64decode(base64_data)
        img = Image.open(io.BytesIO(decoded))
        assert img.mode == 'RGB'

    def test_quality_parameter_affects_size(self, temp_rgb_image):
        """Should produce smaller file with lower quality."""
        result_high = convert_image_to_jpeg_base64(temp_rgb_image, quality=95)
        result_low = convert_image_to_jpeg_base64(temp_rgb_image, quality=10)

        # Lower quality should produce smaller base64 string
        assert len(result_low) < len(result_high)

    def test_default_quality_is_60(self, temp_rgb_image):
        """Should use quality=60 by default."""
        result_default = convert_image_to_jpeg_base64(temp_rgb_image)
        result_60 = convert_image_to_jpeg_base64(temp_rgb_image, quality=60)

        # Results should be identical
        assert result_default == result_60

    def test_returns_valid_jpeg_data(self, temp_rgb_image):
        """Should return data that can be decoded back to a JPEG image."""
        result = convert_image_to_jpeg_base64(temp_rgb_image)
        base64_data = result.split(",", 1)[1]
        decoded = base64.b64decode(base64_data)

        # Should be able to open as image
        img = Image.open(io.BytesIO(decoded))
        assert img.format == 'JPEG'
        assert img.size == (100, 100)
