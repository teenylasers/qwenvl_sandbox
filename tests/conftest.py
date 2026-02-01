"""Shared fixtures for VLM training tests."""

import io

import pytest
from datasets import Dataset
from PIL import Image


@pytest.fixture
def dummy_pil_image():
    """A small 4x4 RGB PIL image for tests."""
    return Image.new("RGB", (4, 4), color=(128, 64, 32))


@pytest.fixture
def dummy_image_bytes(dummy_pil_image):
    """PNG bytes of the dummy image."""
    buf = io.BytesIO()
    dummy_pil_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def make_hf_dataset():
    """Factory to build a HuggingFace Dataset from a list of dicts."""

    def _make(rows: list[dict], features=None) -> Dataset:
        if not rows:
            return Dataset.from_dict({})
        keys = rows[0].keys()
        data = {k: [r[k] for r in rows] for k in keys}
        if features:
            return Dataset.from_dict(data, features=features)
        return Dataset.from_dict(data)

    return _make
