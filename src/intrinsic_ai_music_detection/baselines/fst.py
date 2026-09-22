"""FST (Fusion Segment Transformer) baseline wrapper.

Wraps the pre-trained FST model from Mippia/FST-AI-Music-Detection
(ICASSP 2026) for comparison against our intrinsic dimension approach.

The FST is a two-stage model:
  Stage 1 (MERT-AudioCAT): Extracts content embeddings from beat-segmented audio
  Stage 2 (FST): Bi-directional attention fusion of content + structural streams

Reference: https://arxiv.org/abs/2601.13647
License: GPL-3.0

Setup
-----
    # Clone FST repo
    git clone https://github.com/Mippia/FST-AI-Music-Detection.git vendors/fst

    # Download checkpoints (Google Drive):
    # Stage-1: https://drive.google.com/file/d/1frT4Mn0l6rso407Sy3eWCKbZmgwuVceN
    # Stage-2: https://drive.google.com/file/d/1E_xPsosYWI4UjKT8XQCbZW4ILvsWnmda
    # Place in vendors/fst/checkpoints/
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

FST_REPO_DIR = Path("vendors/fst")


@dataclass
class FSTConfig:
    """Configuration for the FST baseline."""

    repo_dir: Path = field(default_factory=lambda: FST_REPO_DIR)
    stage1_ckpt: str = "checkpoints/stage1_mert_audiocat.ckpt"
    stage2_ckpt: str = "checkpoints/stage2_fst.ckpt"
    sample_rate: int = 24_000
    fixed_samples: int = 240_000  # 10s at 24kHz
    max_segments: int = 48
    device: str = "cpu"


class FSTBaseline:
    """Wrapper for the pre-trained FST model.

    Performs two-stage inference:
    1. MERT-AudioCAT extracts segment-level embeddings
    2. FusionSegmentTransformer classifies based on content + structure
    """

    def __init__(self, cfg: FSTConfig | None = None) -> None:
        self.cfg = cfg or FSTConfig()
        self._stage1 = None
        self._stage2 = None
        self._fst_imported = False

    def _ensure_fst_on_path(self) -> None:
        """Add FST repo to sys.path so we can import its modules."""
        if self._fst_imported:
            return
        repo_dir = str(self.cfg.repo_dir.resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        self._fst_imported = True

    def _load_models(self) -> None:
        """Lazily load both stages of the FST pipeline."""
        self._ensure_fst_on_path()

        try:
            from model import MERT_AudioCAT, MusicAudioClassifier
        except ImportError as e:
            raise ImportError(
                f"Could not import FST model. Clone the repo first:\n"
                f"  git clone https://github.com/Mippia/FST-AI-Music-Detection.git {self.cfg.repo_dir}\n"
                f"Original error: {e}"
            ) from e

        device = self.cfg.device

        # Stage 1: MERT-AudioCAT
        stage1_path = self.cfg.repo_dir / self.cfg.stage1_ckpt
        if not stage1_path.exists():
            raise FileNotFoundError(
                f"Stage-1 checkpoint not found at {stage1_path}. "
                "Download from: https://drive.google.com/file/d/1frT4Mn0l6rso407Sy3eWCKbZmgwuVceN"
            )

        self._stage1 = MERT_AudioCAT.load_from_checkpoint(str(stage1_path))
        self._stage1.to(device)
        self._stage1.eval()
        logger.info("FST Stage-1 (MERT-AudioCAT) loaded on %s", device)

        # Stage 2: Fusion Segment Transformer
        stage2_path = self.cfg.repo_dir / self.cfg.stage2_ckpt
        if not stage2_path.exists():
            raise FileNotFoundError(
                f"Stage-2 checkpoint not found at {stage2_path}. "
                "Download from: https://drive.google.com/file/d/1E_xPsosYWI4UjKT8XQCbZW4ILvsWnmda"
            )

        embed_dim = 768  # MERT-v1-95M hidden dim
        self._stage2 = MusicAudioClassifier.load_from_checkpoint(
            str(stage2_path),
            input_dim=embed_dim,
            backbone="fusion_segment_transformer",
            is_emb=True,
        )
        self._stage2.to(device)
        self._stage2.eval()
        logger.info("FST Stage-2 (FusionSegmentTransformer) loaded on %s", device)

    @property
    def stage1(self) -> Any:
        if self._stage1 is None:
            self._load_models()
        return self._stage1

    @property
    def stage2(self) -> Any:
        if self._stage2 is None:
            self._load_models()
        return self._stage2

    def _segment_audio(self, audio_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment audio using beat tracking (FST's preprocessing).

        Returns
        -------
        segments : tensor of shape ``[max_segments, 1, fixed_samples]``
        padding_mask : bool tensor of shape ``[max_segments]``
        """
        self._ensure_fst_on_path()

        import torchaudio

        try:
            from preprocess import find_optimal_segment_length, get_segments_from_wav
        except ImportError:
            # Fallback: uniform 10s segments without beat tracking
            logger.warning("beat_this not available, using uniform 10s segments")
            return self._segment_audio_uniform(audio_path)

        audio_path = str(audio_path)

        try:
            beats, downbeats = get_segments_from_wav(audio_path, device=self.cfg.device)
        except Exception:
            logger.warning("Beat tracking failed for %s, using uniform segments", audio_path)
            return self._segment_audio_uniform(audio_path)

        optimal_length, cleaned_downbeats = find_optimal_segment_length(downbeats)

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(torch.float32)

        if sample_rate != self.cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.cfg.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        sr = self.cfg.sample_rate
        fixed_samples = self.cfg.fixed_samples

        if waveform.shape[1] <= fixed_samples:
            padding = torch.zeros(1, fixed_samples, dtype=torch.float32)
            waveform = torch.cat([waveform, padding], dim=1)

        segments = []
        for start_time in cleaned_downbeats:
            start_sample = int(start_time * sr)
            end_sample = start_sample + fixed_samples
            if end_sample > waveform.size(1):
                continue
            segment = waveform[:, start_sample:end_sample]
            segments.append(segment)
            if len(segments) >= self.cfg.max_segments:
                break

        if not segments:
            return torch.zeros((self.cfg.max_segments, 1, fixed_samples)), torch.ones(
                self.cfg.max_segments, dtype=torch.bool
            )

        stacked = torch.stack(segments)
        num_segments = stacked.shape[0]
        padding_mask = torch.zeros(self.cfg.max_segments, dtype=torch.bool)

        if num_segments < self.cfg.max_segments:
            pad = torch.zeros((self.cfg.max_segments - num_segments, 1, fixed_samples))
            stacked = torch.cat([stacked, pad], dim=0)
            padding_mask[num_segments:] = True

        return stacked, padding_mask

    def _segment_audio_uniform(self, audio_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Fallback: split audio into uniform 10s segments."""
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(audio_path))
        waveform = waveform.to(torch.float32)

        if sample_rate != self.cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.cfg.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        fixed_samples = self.cfg.fixed_samples
        total_samples = waveform.shape[1]

        segments = []
        for start in range(0, total_samples - fixed_samples + 1, fixed_samples):
            segments.append(waveform[:, start : start + fixed_samples])
            if len(segments) >= self.cfg.max_segments:
                break

        if not segments:
            return torch.zeros((self.cfg.max_segments, 1, fixed_samples)), torch.ones(
                self.cfg.max_segments, dtype=torch.bool
            )

        stacked = torch.stack(segments)
        num_segments = stacked.shape[0]
        padding_mask = torch.zeros(self.cfg.max_segments, dtype=torch.bool)

        if num_segments < self.cfg.max_segments:
            pad = torch.zeros((self.cfg.max_segments - num_segments, 1, fixed_samples))
            stacked = torch.cat([stacked, pad], dim=0)
            padding_mask[num_segments:] = True

        return stacked, padding_mask

    def predict(self, audio_path: str | Path) -> dict:
        """Run FST inference on a single audio file.

        Returns
        -------
        dict with keys:
            prediction: "fake" or "real"
            fake_prob: float in [0, 1]
            real_prob: float in [0, 1]
            raw_logit: float
        """
        device = self.cfg.device
        segments, padding_mask = self._segment_audio(audio_path)

        segments = segments.to(device).to(torch.float32)
        padding_mask = padding_mask.to(device)

        with torch.no_grad():
            # Stage 1: extract MERT-AudioCAT embeddings
            seg_input = segments.squeeze(1)  # [48, fixed_samples]
            logits, embeddings = self.stage1(seg_input)
            # embeddings: [48, 768]

            # Stage 2: classify with FST
            emb_input = embeddings.unsqueeze(0)  # [1, 48, 768]
            mask_input = padding_mask.unsqueeze(0)  # [1, 48]

            output = self.stage2(emb_input, mask_input)
            logit = output.squeeze().float()

            # Scaled sigmoid (matching FST inference.py)
            scale_factor = 1.0
            raw_prob = torch.sigmoid(logit * scale_factor).item()
            fake_prob = float(np.clip(raw_prob, 0.01, 0.99))

        return {
            "prediction": "fake" if fake_prob > 0.5 else "real",
            "fake_prob": fake_prob,
            "real_prob": 1.0 - fake_prob,
            "raw_logit": logit.item(),
        }

    def predict_batch(self, audio_paths: list[str | Path]) -> list[dict]:
        """Run FST inference on multiple files."""
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                result["audio_path"] = str(path)
                results.append(result)
            except Exception as e:
                logger.warning("FST prediction failed for %s: %s", path, e)
                results.append(
                    {
                        "audio_path": str(path),
                        "prediction": "error",
                        "fake_prob": float("nan"),
                        "real_prob": float("nan"),
                        "raw_logit": float("nan"),
                        "error": str(e),
                    }
                )
        return results
