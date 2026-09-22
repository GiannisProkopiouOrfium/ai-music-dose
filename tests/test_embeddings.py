"""Tests for embedding extractors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from intrinsic_ai_music_detection.config import MuQConfig
from intrinsic_ai_music_detection.features.embeddings import get_extractor


class TestGetExtractor:
    """Tests for the extractor factory function."""

    def test_encodec_name(self) -> None:
        ext = get_extractor("encodec")
        assert ext.embedding_dim == 128
        assert ext.sample_rate == 24000

    def test_clap_name(self) -> None:
        ext = get_extractor("clap")
        assert ext.embedding_dim == 512
        assert ext.sample_rate == 48000

    def test_mert_name(self) -> None:
        ext = get_extractor("mert")
        assert ext.embedding_dim == 1024
        assert ext.sample_rate == 24000

    def test_muq_name(self) -> None:
        ext = get_extractor("muq")
        assert ext.embedding_dim == 1024
        assert ext.sample_rate == 24000
        assert ext.distance_metric == "cosine"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown extractor"):
            get_extractor("nonexistent")


class TestMuQExtractor:
    """Tests for MuQExtractor with mocked model."""

    def test_extract_shape(self) -> None:
        from intrinsic_ai_music_detection.features.embeddings import MuQExtractor

        ext = MuQExtractor(device="cpu")
        # Mock the model
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 375, 1024)
        mock_model = MagicMock(return_value=mock_output)
        ext._model = mock_model

        audio = np.random.randn(24000 * 5).astype(np.float32)  # 5 seconds
        result = ext.extract(audio, sr=24000)

        assert result.shape == (375, 1024)
        assert result.dtype == np.float32

    def test_wrong_sample_rate(self) -> None:
        from intrinsic_ai_music_detection.features.embeddings import MuQExtractor

        ext = MuQExtractor(device="cpu")
        audio = np.random.randn(48000).astype(np.float32)

        with pytest.raises(ValueError, match="24000"):
            ext.extract(audio, sr=48000)

    def test_config_defaults(self) -> None:
        cfg = MuQConfig()
        assert cfg.model_name == "OpenMuQ/MuQ-large-msd-iter"
        assert cfg.sample_rate == 24000
        assert cfg.embedding_dim == 1024
        assert cfg.distance_metric == "cosine"
