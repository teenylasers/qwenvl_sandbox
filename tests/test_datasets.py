"""Tests for dataset loading functions."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import datasets as ds_lib
import pytest
import requests as requests_lib
from datasets import Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import (
    _download_image,
    load_rlhfv_sft,
    load_spatial_vlm,
    load_llava_instruct,
    load_pixmo_points,
    load_sharegpt4v,
    load_pixmo_docs,
    load_rlhfv_preference,
    load_sft_dataset,
    load_preference_dataset,
    load_vsr_benchmark,
    load_cvbench,
)


# =============================================================================
# Unit Tests
# =============================================================================


class TestDownloadImage:
    """Tests for _download_image utility."""

    @patch("src.data.datasets.requests.get")
    def test_successful_download(self, mock_get, dummy_pil_image):
        buf = io.BytesIO()
        dummy_pil_image.save(buf, format="PNG")
        mock_resp = MagicMock()
        mock_resp.content = buf.getvalue()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _download_image("http://example.com/img.jpg")
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        mock_get.assert_called_once_with("http://example.com/img.jpg", timeout=10)

    @patch("src.data.datasets.requests.get")
    def test_custom_timeout(self, mock_get, dummy_pil_image):
        buf = io.BytesIO()
        dummy_pil_image.save(buf, format="PNG")
        mock_resp = MagicMock()
        mock_resp.content = buf.getvalue()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _download_image("http://example.com/img.jpg", timeout=30)
        mock_get.assert_called_once_with("http://example.com/img.jpg", timeout=30)

    @patch("src.data.datasets.requests.get", side_effect=requests_lib.Timeout)
    def test_timeout_returns_none(self, mock_get):
        assert _download_image("http://example.com/img.jpg") is None

    @patch("src.data.datasets.requests.get")
    def test_http_error_returns_none(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests_lib.HTTPError
        mock_get.return_value = mock_resp
        assert _download_image("http://example.com/img.jpg") is None

    @patch("src.data.datasets.requests.get")
    def test_invalid_bytes_returns_none(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.content = b"not-an-image"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        assert _download_image("http://example.com/img.jpg") is None


class TestRlhfvSft:
    """Tests for load_rlhfv_sft."""

    def _mock_ds(self, make_hf_dataset, dummy_pil_image, n=1):
        rows = [
            {
                "text": json.dumps({"question": f"Q{i}", "chosen": f"A{i}"}),
                "image": dummy_pil_image,
            }
            for i in range(n)
        ]
        return make_hf_dataset(
            rows,
            features=ds_lib.Features({
                "text": ds_lib.Value("string"),
                "image": ds_lib.Image(),
            }),
        )

    @patch("src.data.datasets.load_dataset")
    def test_basic_formatting(self, mock_load, make_hf_dataset, dummy_pil_image):
        mock_load.return_value = self._mock_ds(make_hf_dataset, dummy_pil_image)
        result = load_rlhfv_sft()
        assert len(result) == 1
        assert result[0]["question"] == "Q0"
        assert result[0]["answer"] == "A0"
        assert "images" in result.column_names

    @patch("src.data.datasets.load_dataset")
    def test_messages_structure(self, mock_load, make_hf_dataset, dummy_pil_image):
        mock_load.return_value = self._mock_ds(make_hf_dataset, dummy_pil_image)
        result = load_rlhfv_sft()
        msgs = result[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Q0"}
        assert msgs[1] == {"role": "assistant", "content": "A0"}

    @patch("src.data.datasets.load_dataset")
    def test_max_samples(self, mock_load, make_hf_dataset, dummy_pil_image):
        mock_load.return_value = self._mock_ds(make_hf_dataset, dummy_pil_image, n=5)
        result = load_rlhfv_sft(max_samples=2)
        assert len(result) == 2


class TestSpatialVlm:
    """Tests for load_spatial_vlm."""

    @patch("src.data.datasets.load_dataset")
    def test_basic_formatting(self, mock_load, make_hf_dataset, dummy_pil_image):
        mock_ds = make_hf_dataset(
            [{"question": "Where is the cat?", "answer": "On the left", "image": dummy_pil_image}],
            features=ds_lib.Features({
                "question": ds_lib.Value("string"),
                "answer": ds_lib.Value("string"),
                "image": ds_lib.Image(),
            }),
        )
        mock_load.return_value = mock_ds
        result = load_spatial_vlm()
        assert len(result) == 1
        assert result[0]["question"] == "Where is the cat?"
        assert result[0]["answer"] == "On the left"

    @patch("src.data.datasets.load_dataset")
    def test_fallback_field_names(self, mock_load, make_hf_dataset, dummy_pil_image):
        mock_ds = make_hf_dataset(
            [{"prompt": "Where?", "response": "Left", "image": dummy_pil_image}],
            features=ds_lib.Features({
                "prompt": ds_lib.Value("string"),
                "response": ds_lib.Value("string"),
                "image": ds_lib.Image(),
            }),
        )
        mock_load.return_value = mock_ds
        result = load_spatial_vlm()
        assert result[0]["question"] == "Where?"
        assert result[0]["answer"] == "Left"

    @patch("src.data.datasets.load_dataset", side_effect=Exception("not available"))
    def test_graceful_failure(self, mock_load):
        result = load_spatial_vlm()
        assert len(result) == 0


class TestLlavaInstruct:
    """Tests for load_llava_instruct."""

    @patch("src.data.datasets._download_image")
    @patch("src.data.datasets.load_dataset")
    def test_conversation_extraction(self, mock_load, mock_download, make_hf_dataset, dummy_pil_image):
        mock_download.return_value = dummy_pil_image
        mock_ds = make_hf_dataset([{
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image."},
                {"from": "gpt", "value": "A beautiful landscape."},
            ],
            "image": "000000001.jpg",
        }])
        mock_load.return_value = mock_ds
        result = load_llava_instruct()
        assert len(result) == 1
        assert result[0]["question"] == "Describe this image."
        assert result[0]["answer"] == "A beautiful landscape."

    @patch("src.data.datasets._download_image")
    @patch("src.data.datasets.load_dataset")
    def test_image_tag_stripping(self, mock_load, mock_download, make_hf_dataset, dummy_pil_image):
        mock_download.return_value = dummy_pil_image
        mock_ds = make_hf_dataset([{
            "conversations": [
                {"from": "human", "value": "<image>What is this?"},
                {"from": "gpt", "value": "A dog."},
            ],
            "image": "000000002.jpg",
        }])
        mock_load.return_value = mock_ds
        result = load_llava_instruct()
        assert "<image>" not in result[0]["question"]
        assert result[0]["question"] == "What is this?"

    @patch("src.data.datasets._download_image", return_value=None)
    @patch("src.data.datasets.load_dataset")
    def test_failed_download_filtered(self, mock_load, mock_download, make_hf_dataset):
        mock_ds = make_hf_dataset([{
            "conversations": [
                {"from": "human", "value": "Describe"},
                {"from": "gpt", "value": "Answer"},
            ],
            "image": "missing.jpg",
        }])
        mock_load.return_value = mock_ds
        result = load_llava_instruct()
        assert len(result) == 0


class TestPixmoPoints:
    """Tests for load_pixmo_points."""

    @patch("src.data.datasets._download_image")
    @patch("src.data.datasets.load_dataset")
    def test_counting_format(self, mock_load, mock_download, make_hf_dataset, dummy_pil_image):
        mock_download.return_value = dummy_pil_image
        mock_ds = make_hf_dataset([{
            "label": "cats",
            "count": 3,
            "collection_method": "counting",
            "points": [],
            "image_url": "http://example.com/img.jpg",
        }])
        mock_load.return_value = mock_ds
        result = load_pixmo_points()
        assert len(result) == 1
        assert "How many cats" in result[0]["question"]
        assert result[0]["answer"] == "3"

    @patch("src.data.datasets._download_image")
    @patch("src.data.datasets.load_dataset")
    def test_pointing_format(self, mock_load, mock_download, make_hf_dataset, dummy_pil_image):
        mock_download.return_value = dummy_pil_image
        mock_ds = make_hf_dataset([{
            "label": "dogs",
            "count": 2,
            "collection_method": "pointing",
            "points": [{"x": 0.5, "y": 0.3}, {"x": 0.8, "y": 0.7}],
            "image_url": "http://example.com/img.jpg",
        }])
        mock_load.return_value = mock_ds
        result = load_pixmo_points()
        assert len(result) == 1
        assert "Point to all dogs" in result[0]["question"]
        assert "(0.5, 0.3)" in result[0]["answer"]
        assert "(0.8, 0.7)" in result[0]["answer"]

    @patch("src.data.datasets._download_image", return_value=None)
    @patch("src.data.datasets.load_dataset")
    def test_failed_download_filtered(self, mock_load, mock_download, make_hf_dataset):
        mock_ds = make_hf_dataset([{
            "label": "cats",
            "count": 1,
            "collection_method": "counting",
            "points": [],
            "image_url": "http://example.com/missing.jpg",
        }])
        mock_load.return_value = mock_ds
        result = load_pixmo_points()
        assert len(result) == 0


class TestSharegpt4v:
    """Tests for load_sharegpt4v."""

    @patch("src.data.datasets._download_image")
    @patch("src.data.datasets.load_dataset")
    def test_coco_url_construction(self, mock_load, mock_download, make_hf_dataset, dummy_pil_image):
        mock_download.return_value = dummy_pil_image
        mock_ds = make_hf_dataset([{
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe"},
                {"from": "gpt", "value": "A park scene"},
            ],
            "image": "coco/train2017/000000000009.jpg",
        }])
        mock_load.return_value = mock_ds
        result = load_sharegpt4v()
        assert len(result) == 1
        # Verify _download_image was called with the correct COCO URL
        mock_download.assert_called_once_with(
            "http://images.cocodataset.org/train2017/000000000009.jpg"
        )

    @patch("src.data.datasets._download_image")
    @patch("src.data.datasets.load_dataset")
    def test_non_coco_path_filtered(self, mock_load, mock_download, make_hf_dataset):
        mock_ds = make_hf_dataset([{
            "conversations": [
                {"from": "human", "value": "Describe"},
                {"from": "gpt", "value": "Answer"},
            ],
            "image": "some/other/path.jpg",
        }])
        mock_load.return_value = mock_ds
        result = load_sharegpt4v()
        assert len(result) == 0
        mock_download.assert_not_called()


class TestPixmoDocs:
    """Tests for load_pixmo_docs.

    Uses a plain iterable mock instead of Dataset because load_pixmo_docs
    iterates directly (``for example in ds``), and the real HuggingFace
    dataset returns ``questions`` as a list-of-dicts per row.
    """

    @staticmethod
    def _make_subset(rows):
        """Return a list (iterable) mimicking a HF Dataset for direct iteration."""
        return rows

    @patch("src.data.datasets.load_dataset")
    def test_multi_qa_flattening(self, mock_load, dummy_pil_image):
        charts = self._make_subset([{
            "image": dummy_pil_image,
            "questions": [
                {"question": "Q1", "answer": "A1"},
                {"question": "Q2", "answer": "A2"},
            ],
        }])
        mock_load.side_effect = [charts, Exception("skip"), Exception("skip"), Exception("skip")]
        result = load_pixmo_docs()
        assert len(result) == 2
        assert result["question"] == ["Q1", "Q2"]
        assert result["answer"] == ["A1", "A2"]

    @patch("src.data.datasets.load_dataset")
    def test_skips_missing_images(self, mock_load):
        charts = self._make_subset([{
            "image": None,
            "questions": [{"question": "Q", "answer": "A"}],
        }])
        mock_load.side_effect = [charts, Exception("skip"), Exception("skip"), Exception("skip")]
        result = load_pixmo_docs()
        assert len(result) == 0

    @patch("src.data.datasets.load_dataset")
    def test_subset_failure_continues(self, mock_load, dummy_pil_image):
        tables = self._make_subset([{
            "image": dummy_pil_image,
            "questions": [{"question": "Q1", "answer": "A1"}],
        }])
        # "charts" fails, "tables" succeeds, rest fail
        mock_load.side_effect = [Exception("fail"), tables, Exception("skip"), Exception("skip")]
        result = load_pixmo_docs()
        assert len(result) == 1


class TestRlhfvPreference:
    """Tests for load_rlhfv_preference."""

    def _mock_ds(self, make_hf_dataset, dummy_pil_image, n=1):
        rows = [
            {
                "text": json.dumps({
                    "question": f"Q{i}",
                    "chosen": f"Good{i}",
                    "rejected": f"Bad{i}",
                }),
                "image": dummy_pil_image,
            }
            for i in range(n)
        ]
        return make_hf_dataset(
            rows,
            features=ds_lib.Features({
                "text": ds_lib.Value("string"),
                "image": ds_lib.Image(),
            }),
        )

    @patch("src.data.datasets.load_dataset")
    def test_dpo_format(self, mock_load, make_hf_dataset, dummy_pil_image):
        mock_load.return_value = self._mock_ds(make_hf_dataset, dummy_pil_image)
        result = load_rlhfv_preference()
        assert len(result) == 1
        assert result[0]["prompt"] == "Q0"
        assert result[0]["chosen"] == "Good0"
        assert result[0]["rejected"] == "Bad0"

    @patch("src.data.datasets.load_dataset")
    def test_max_samples(self, mock_load, make_hf_dataset, dummy_pil_image):
        mock_load.return_value = self._mock_ds(make_hf_dataset, dummy_pil_image, n=5)
        result = load_rlhfv_preference(max_samples=2)
        assert len(result) == 2


class TestLoadSftDataset:
    """Tests for load_sft_dataset orchestrator."""

    @staticmethod
    def _simple_ds(n):
        return Dataset.from_dict({
            "images": [[]] * n,
            "question": [f"Q{i}" for i in range(n)],
            "answer": [f"A{i}" for i in range(n)],
            "messages": [[]] * n,
        })

    @patch("src.data.datasets.load_spatial_vlm")
    @patch("src.data.datasets.load_rlhfv_sft")
    def test_combines_multiple(self, mock_rlhfv, mock_spatial):
        mock_rlhfv.return_value = self._simple_ds(3)
        mock_spatial.return_value = self._simple_ds(2)
        result = load_sft_dataset(["rlhfv", "spatial"], shuffle=False)
        assert len(result) == 5

    @patch("src.data.datasets.load_rlhfv_sft")
    def test_all_empty_raises(self, mock_rlhfv):
        mock_rlhfv.return_value = self._simple_ds(0)
        with pytest.raises(ValueError, match="No datasets"):
            load_sft_dataset(["rlhfv"], shuffle=False)

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="No datasets"):
            load_sft_dataset(["nonexistent"], shuffle=False)

    @patch("src.data.datasets.load_rlhfv_sft")
    def test_shuffle_deterministic(self, mock_rlhfv):
        mock_rlhfv.return_value = self._simple_ds(10)
        r1 = load_sft_dataset(["rlhfv"], shuffle=True, seed=42)
        mock_rlhfv.return_value = self._simple_ds(10)
        r2 = load_sft_dataset(["rlhfv"], shuffle=True, seed=42)
        assert list(r1["question"]) == list(r2["question"])


class TestLoadPreferenceDataset:
    """Tests for load_preference_dataset wrapper."""

    @patch("src.data.datasets.load_rlhfv_preference")
    def test_delegates_to_rlhfv(self, mock_pref):
        mock_pref.return_value = Dataset.from_dict({
            "image": [None],
            "prompt": ["Q"],
            "chosen": ["Good"],
            "rejected": ["Bad"],
        })
        result = load_preference_dataset(max_samples=1)
        assert len(result) == 1
        mock_pref.assert_called_once_with(max_samples=1)


class TestEvalDatasets:
    """Tests for evaluation dataset loaders."""

    @patch("src.data.datasets.load_dataset", side_effect=Exception("unavailable"))
    def test_vsr_fallback(self, mock_load):
        result = load_vsr_benchmark()
        assert len(result) == 0
        assert "caption" in result.column_names
        assert "label" in result.column_names

    @patch("src.data.datasets.load_dataset", side_effect=Exception("unavailable"))
    def test_cvbench_fallback(self, mock_load):
        result = load_cvbench()
        assert len(result) == 0
        assert "question" in result.column_names
        assert "answer" in result.column_names


# =============================================================================
# Integration Tests (require network access)
# =============================================================================


@pytest.mark.slow
class TestIntegrationSft:
    """Integration tests that download real data from HuggingFace."""

    def test_rlhfv_sft_real(self):
        result = load_rlhfv_sft(max_samples=2)
        assert len(result) == 2
        assert "images" in result.column_names
        assert "question" in result.column_names
        assert "answer" in result.column_names
        assert "messages" in result.column_names
        assert len(result[0]["question"]) > 0

    def test_rlhfv_preference_real(self):
        result = load_rlhfv_preference(max_samples=2)
        assert len(result) == 2
        assert "prompt" in result.column_names
        assert "chosen" in result.column_names
        assert "rejected" in result.column_names
        assert len(result[0]["chosen"]) > 0


@pytest.mark.slow
class TestIntegrationEval:
    """Integration tests for evaluation datasets."""

    def test_vsr_benchmark_real(self):
        result = load_vsr_benchmark(max_samples=2)
        if len(result) > 0:
            assert "caption" in result.column_names or "image" in result.column_names
