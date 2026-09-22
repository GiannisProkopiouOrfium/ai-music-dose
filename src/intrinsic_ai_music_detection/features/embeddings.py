"""Embedding extractors: CLAP, MERT, EnCodec.

Each extractor takes raw audio (numpy array + sample rate) and returns
an embedding matrix of shape ``[N, D]`` where *N* is the number of
temporal frames and *D* is the embedding dimension.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import torch

from intrinsic_ai_music_detection.config import (
    CLAPConfig,
    CombPrintConfig,
    EnCodecConfig,
    MERT95MConfig,
    MERTConfig,
    MuQConfig,
    MusicDETSpecConfig,
    SpectrogramConfig,
    Wav2Vec2Config,
)
from intrinsic_ai_music_detection.data.audio_utils import load_audio, normalize_audio, sliding_window_chunked

logger = logging.getLogger(__name__)


def _bypass_torch_load_check() -> None:
    """Bypass CVE-2025-32434 check in newer transformers for torch < 2.6.

    Required when CUDA driver only supports torch 2.5.x.  We trust the
    official HuggingFace checkpoints we load.  Must patch both the source
    module AND modeling_utils (which may already have imported the binding).
    """
    _noop = lambda: None  # noqa: E731
    try:
        import transformers.utils.import_utils as _tiu

        _tiu.check_torch_load_is_safe = _noop
    except (ImportError, AttributeError):
        pass
    try:
        import transformers.modeling_utils as _tmu

        _tmu.check_torch_load_is_safe = _noop
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Protocol for all extractors
# ---------------------------------------------------------------------------


class EmbeddingExtractor(Protocol):
    """Common interface for embedding extractors."""

    embedding_dim: int
    sample_rate: int
    distance_metric: str

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Return embeddings of shape ``[N, D]``."""
        ...

    def extract_from_file(self, path: str | Path) -> npt.NDArray[np.float32]:
        """Load audio from *path* and extract embeddings."""
        ...


# ---------------------------------------------------------------------------
# CLAP
# ---------------------------------------------------------------------------


class CLAPExtractor:
    """Extract CLAP embeddings using LAION-CLAP.

    Slices audio into overlapping chunks (default 10 s / 5 s hop) and produces
    one 512-d embedding per chunk.
    """

    def __init__(self, cfg: CLAPConfig | None = None, device: str = "cpu") -> None:
        self.cfg = cfg or CLAPConfig()
        self.embedding_dim = self.cfg.embedding_dim
        self.sample_rate = self.cfg.sample_rate
        self.distance_metric = self.cfg.distance_metric
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        import laion_clap

        # PyTorch 2.6+ changed torch.load default to weights_only=True.
        # laion-clap checkpoints contain numpy globals that aren't allowlisted,
        # so we temporarily patch torch.load to use weights_only=False during
        # CLAP checkpoint loading.  This is safe because we trust the
        # official CLAP checkpoint.
        _original_torch_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        # Newer transformers removed position_ids from RoBERTa state dicts,
        # causing strict load_state_dict to fail on CLAP checkpoints.
        # Temporarily patch to use strict=False.
        _original_load_state_dict = torch.nn.Module.load_state_dict

        def _patched_load_state_dict(self_module, state_dict, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["strict"] = False
            return _original_load_state_dict(self_module, state_dict, *args, **kwargs)

        torch.load = _patched_load  # type: ignore[assignment]
        torch.nn.Module.load_state_dict = _patched_load_state_dict  # type: ignore[assignment]
        try:
            self._model = laion_clap.CLAP_Module(enable_fusion=self.cfg.enable_fusion, device=self.device)
            self._model.load_ckpt(model_id=1)  # music_audioset_epoch_15
        finally:
            torch.load = _original_torch_load  # type: ignore[assignment]
            torch.nn.Module.load_state_dict = _original_load_state_dict  # type: ignore[assignment]

        logger.info("CLAP model loaded on %s", self.device)

    @property
    def model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            self._load_model()
        return self._model

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Extract CLAP embeddings from raw audio.

        Parameters
        ----------
        audio : 1-D float32 array, already at 48 kHz
        sr : sample rate (must be 48000)

        Returns
        -------
        embeddings : ndarray of shape ``[N, 512]``
        """
        if sr != self.sample_rate:
            raise ValueError(f"CLAP requires {self.sample_rate} Hz audio, got {sr}")

        chunks = sliding_window_chunked(
            audio,
            sr,
            chunk_duration=self.cfg.chunk_duration,
            hop_duration=self.cfg.hop_duration,
        )

        if not chunks:
            raise ValueError("Audio too short to produce any CLAP chunks")

        # CLAP expects int16-quantized audio as float
        batch = np.stack(chunks, axis=0)  # [N, chunk_samples]

        # Vectorized peak normalization to [-1, 1]
        peaks = np.abs(batch).max(axis=1, keepdims=True)
        batch = batch / np.maximum(peaks, 1e-7)

        with torch.no_grad():
            embeddings = self.model.get_audio_embedding_from_data(batch)
        return np.asarray(embeddings, dtype=np.float32)

    def extract_from_file(self, path: str | Path) -> npt.NDArray[np.float32]:
        audio, sr = load_audio(path, target_sr=self.sample_rate)
        audio = normalize_audio(audio)
        return self.extract(audio, sr)


# ---------------------------------------------------------------------------
# MERT
# ---------------------------------------------------------------------------


class MERTExtractor:
    """Extract MERT-v1-330M embeddings.

    Produces frame-level embeddings at ~75 Hz from the last hidden state
    (1024-d per frame).
    """

    def __init__(self, cfg: MERTConfig | None = None, device: str = "cpu") -> None:
        self.cfg = cfg or MERTConfig()
        self.embedding_dim = self.cfg.embedding_dim
        self.sample_rate = self.cfg.sample_rate
        self.distance_metric = self.cfg.distance_metric
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        _bypass_torch_load_check()

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.cfg.model_name, trust_remote_code=True).to(self.device)
        self._model.eval()
        logger.info("MERT model loaded on %s", self.device)

    @property
    def model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):  # type: ignore[no-untyped-def]
        if self._processor is None:
            self._load_model()
        return self._processor

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Extract MERT embeddings.

        Parameters
        ----------
        audio : 1-D float32 array at 24 kHz
        sr : sample rate (must be 24000)

        Returns
        -------
        embeddings : ndarray of shape ``[T, 1024]``
        """
        if sr != self.sample_rate:
            raise ValueError(f"MERT requires {self.sample_rate} Hz audio, got {sr}")

        # Process in chunks to avoid OOM on long audio
        chunk_samples = self.sample_rate * 30  # 30s chunks
        if len(audio) > chunk_samples:
            parts = []
            for start in range(0, len(audio), chunk_samples):
                chunk = audio[start : start + chunk_samples]
                parts.append(self._extract_chunk(chunk, sr))
            return np.concatenate(parts, axis=0)

        return self._extract_chunk(audio, sr)

    def _extract_chunk(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Extract embeddings from a single audio chunk."""
        inputs = self.processor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)

        use_amp = self.device != "cpu" and torch.cuda.is_available()
        with torch.no_grad(), torch.autocast(self.device, enabled=use_amp):
            outputs = self.model(input_values, output_hidden_states=True)

        # Use the last hidden state
        last_hidden = outputs.hidden_states[-1]  # [1, T, 1024]
        embeddings = last_hidden.squeeze(0).cpu().float().numpy()
        del input_values, outputs, last_hidden
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return embeddings.astype(np.float32)

    def extract_from_file(self, path: str | Path) -> npt.NDArray[np.float32]:
        audio, sr = load_audio(path, target_sr=self.sample_rate)
        audio = normalize_audio(audio)
        return self.extract(audio, sr)


# ---------------------------------------------------------------------------
# EnCodec
# ---------------------------------------------------------------------------


class SpectrogramExtractor:
    """Classical log-mel spectrogram front end (the MusicDET-style spectral domain).

    Returns ``[T, n_mels]`` log-power mel frames at 75 fps (24 kHz / hop 320),
    i.e. the SAME frame rate and SAME dimensionality as our EnCodec latents, so
    an ``--embeddings spec`` run is a controlled swap of the representation only
    (see ``SpectrogramConfig`` for the full rationale).

    No learned parameters and no GPU: this is a fixed signal-processing
    transform, which is also why it is the natural front end for the efficiency
    comparison — it is the same class of transform MusicDET pays almost nothing
    for, versus EnCodec's CNN forward pass.
    """

    def __init__(self, cfg: "SpectrogramConfig | None" = None, device: str = "cpu") -> None:
        from ..config import SpectrogramConfig

        self.cfg = cfg or SpectrogramConfig()
        self.embedding_dim = (
            self._linear_band()[2] if getattr(self.cfg, "spec_type", "mel") == "linear" else self.cfg.n_mels
        )
        self.sample_rate = self.cfg.sample_rate
        self.distance_metric = self.cfg.distance_metric
        self.device = device  # accepted for API parity; unused (CPU transform)

    def _linear_band(self) -> tuple[int, int, int]:
        """Return ``(k_lo, k_hi, width)`` of the modelled linear-STFT bin range.

        Single source of truth for the crop, so the advertised ``embedding_dim``
        can never disagree with what ``extract`` actually returns.
        """
        n_bins = self.cfg.n_fft // 2 + 1
        k_lo = int(round((self.cfg.fmin or 0.0) * self.cfg.n_fft / self.cfg.sample_rate))
        k_hi = (
            int(round(self.cfg.fmax * self.cfg.n_fft / self.cfg.sample_rate)) if self.cfg.fmax is not None else n_bins
        )
        k_lo, k_hi = max(0, k_lo), min(n_bins, k_hi)
        if k_hi - k_lo < 1:
            raise ValueError(f"empty STFT band: fmin={self.cfg.fmin} fmax={self.cfg.fmax}")
        return k_lo, k_hi, k_hi - k_lo

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Return log-mel frames of shape ``[T, n_mels]``."""
        import librosa

        a = np.asarray(audio, dtype=np.float32).ravel()
        if sr != self.cfg.sample_rate:
            a = librosa.resample(a, orig_sr=sr, target_sr=self.cfg.sample_rate, res_type="soxr_hq")
        if a.size < self.cfg.n_fft:
            raise ValueError(f"audio too short for STFT: {a.size} samples < n_fft {self.cfg.n_fft}")

        if getattr(self.cfg, "spec_type", "mel") == "linear":
            # One-sided LINEAR power STFT, F = n_fft//2 + 1 (MusicDET's front end).
            mel = (
                np.abs(
                    librosa.stft(
                        y=a,
                        n_fft=self.cfg.n_fft,
                        hop_length=self.cfg.hop_length,
                        win_length=self.cfg.n_fft,
                    )
                )
                ** 2.0
            )
            # Crop to [fmin, fmax) by BIN INDEX (MusicDET's band-edge formula
            # k = round(hz * n_fft / sr)). At MusicDETSpecConfig's settings
            # fmax == Nyquist, so this drops only the Nyquist bin: 257 -> 256,
            # which is what makes the dimension divisible by n_bands.
            k_lo, k_hi, _ = self._linear_band()
            mel = mel[k_lo:k_hi]
        else:
            mel = librosa.feature.melspectrogram(
                y=a,
                sr=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
                fmin=self.cfg.fmin,
                fmax=self.cfg.fmax,
                power=2.0,
            )
        # log power. Plain log (not librosa's per-clip top_db reference) keeps
        # every clip on one absolute scale — a per-clip reference would inject a
        # loudness-dependent offset and partly undo the -23 LUFS canonicalization.
        #
        # With log_floor_db set, the power is first clamped to a finite dynamic
        # range below the clip's own peak. Without it, an EMPTY band maps to the
        # single exact constant log(log_offset), which turns "this file is
        # band-limited" into a near-deterministic feature — a delivery-chain
        # measurement rather than a synthesis artifact. See SpectrogramConfig.
        floor_db = getattr(self.cfg, "log_floor_db", None)
        peak = float(np.max(mel)) if mel.size else 0.0
        if floor_db is not None and peak > 0.0:
            # mel is a POWER spectrum, so dB = 10*log10(power).
            mel = np.maximum(mel, peak * (10.0 ** (-abs(floor_db) / 10.0)))
            # Every entry is now strictly positive, so log_offset is unnecessary
            # here — and adding it would re-introduce the very interaction we are
            # removing whenever the clamped floor lands near log_offset itself.
            log_mel = np.log(mel).astype(np.float32)
        else:
            log_mel = np.log(mel + self.cfg.log_offset).astype(np.float32)
        feats = log_mel.T  # [T, n_mels]

        if self.cfg.per_frame_normalize:
            feats = feats - feats.mean(axis=1, keepdims=True)
        return np.ascontiguousarray(feats, dtype=np.float32)

    def extract_from_file(self, path: str | Path) -> npt.NDArray[np.float32]:
        from ..data.audio_utils import load_audio

        audio, sr = load_audio(str(path), target_sr=self.cfg.sample_rate)
        return self.extract(audio, sr)


class CombPrintExtractor:
    """Per-span deconvolution-comb profiles, as a frame sequence for the flow.

    Returns ``[T, n_lags]``: one row per time span, each row the frequency-axis
    autocorrelation of that span's spectral peak residual. Downstream this looks
    exactly like any other frame matrix, so the existing windowing, pooling,
    caching and flow machinery applies unchanged.

    The point of the arm: train the one-class flow on REAL music in a space where
    the artifact provably lives, rather than in a generic embedding where the
    likelihood is dominated by corpus identity. See CombPrintConfig.
    """

    def __init__(self, cfg=None, device: str = "cpu") -> None:
        from ..config import CombPrintConfig

        self.cfg = cfg or CombPrintConfig()
        kind = self.cfg.profile_kind
        if kind not in ("hull", "autocorr", "both"):
            raise ValueError(f"unknown profile_kind {kind!r}")
        self.embedding_dim = (
            self.cfg.n_bins
            if kind == "hull"
            else self.cfg.n_lags
            if kind == "autocorr"
            else self.cfg.n_bins + self.cfg.n_lags
        )
        self.sample_rate = self.cfg.sample_rate
        self.distance_metric = self.cfg.distance_metric
        self.device = device  # accepted for API parity; this is a CPU transform

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        import librosa

        from .comb_artifacts import hull_residual, peak_residual

        cfg = self.cfg
        a = np.asarray(audio, dtype=np.float32).ravel()
        if sr != cfg.sample_rate:
            a = librosa.resample(a, orig_sr=sr, target_sr=cfg.sample_rate, res_type="soxr_hq")
        sr = cfg.sample_rate

        span = int(cfg.span_duration * sr)
        hop = max(int(cfg.span_hop * sr), 1)
        # The long STFT is what resolves a comb; a span shorter than one frame
        # cannot produce a profile at all.
        if a.size < max(span, cfg.n_fft):
            raise ValueError(
                f"audio too short for comb profiles: {a.size} samples < " f"max(span {span}, n_fft {cfg.n_fft})"
            )

        freqs = librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)
        hi = cfg.f_max if cfg.f_max is not None else 0.98 * (sr / 2.0)
        band = (freqs >= cfg.f_min) & (freqs <= hi)
        if band.sum() < 64:
            raise ValueError(f"empty analysis band: f_min={cfg.f_min} f_max={hi}")
        f_band = freqs[band]
        bin_hz = float(f_band[1] - f_band[0])

        lo_lag = max(int(round(cfg.min_spacing_hz / bin_hz)), 1)
        hi_lag = min(int(round(cfg.max_spacing_hz / bin_hz)), band.sum() - 1)
        if hi_lag <= lo_lag:
            raise ValueError("lag range collapsed; check min/max spacing against n_fft")
        # Log-spaced lags: comb spacings of interest span two decades, and a
        # linear grid would spend almost every dimension on the widest ones.
        lag_grid = np.unique(np.geomspace(lo_lag, hi_lag, num=cfg.n_lags).round().astype(int))

        rows: list[np.ndarray] = []
        for start in range(0, max(len(a) - span, 0) + 1, hop):
            chunk = a[start : start + span]
            if len(chunk) < cfg.n_fft:
                break
            spec = np.abs(librosa.stft(chunk, n_fft=cfg.n_fft)) ** 2
            mean_power = spec.mean(axis=1)[band]
            db = 10.0 * np.log10(np.maximum(mean_power, 1e-20))
            parts: list[np.ndarray] = []

            if cfg.profile_kind in ("hull", "both"):
                # Afchar's descriptor: the PATTERN of peaks above the noise
                # floor, which is the part the decoder's strides set. Collapsing
                # this to a single autocorrelation peak — our first attempt — is
                # why we sat at 0.899 where they report >99%.
                parts.append(hull_residual(db, area=cfg.hull_area, max_db=cfg.hull_max_db, n_bins=cfg.n_bins))

            if cfg.profile_kind in ("autocorr", "both"):
                resid = peak_residual(db, smooth_bins=cfg.smooth_bins)
                resid = resid - resid.mean()
                if not np.isfinite(resid).all() or resid.std() < 1e-12:
                    continue
                n = len(resid)
                padded = int(2 ** np.ceil(np.log2(2 * n)))
                fft = np.fft.rfft(resid, n=padded)
                acf = np.fft.irfft(fft * np.conj(fft), n=padded)[:n]
                if acf[0] <= 0:
                    continue
                acf = acf / acf[0]
                row = np.interp(lag_grid, np.arange(n), acf)
                if len(row) < cfg.n_lags:
                    row = np.pad(row, (0, cfg.n_lags - len(row)), mode="edge")
                parts.append(row[: cfg.n_lags])

            if not parts:
                continue
            row = np.concatenate(parts)
            if not np.isfinite(row).all():
                continue
            rows.append(row.astype(np.float32))

        if not rows:
            raise ValueError("no comb profiles produced")
        return np.ascontiguousarray(np.stack(rows), dtype=np.float32)

    def extract_from_file(self, path: str | Path) -> npt.NDArray[np.float32]:
        from ..data.audio_utils import load_audio

        audio, sr = load_audio(str(path), target_sr=self.cfg.sample_rate)
        return self.extract(audio, sr)


class EnCodecExtractor:
    """Extract continuous pre-quantization latents from EnCodec.

    Uses ``model.encoder(wav)`` to get the continuous representation
    *before* the Residual Vector Quantizer, yielding shape ``[T, 128]``
    at 75 Hz (24 kHz / 320x downsampling). Note: because only the encoder is
    used, ``set_target_bandwidth`` has no effect on these latents — do not
    describe them as "X kbps" anywhere.
    """

    def __init__(self, cfg: EnCodecConfig | None = None, device: str = "cpu") -> None:
        self.cfg = cfg or EnCodecConfig()
        self.embedding_dim = self.cfg.embedding_dim
        self.sample_rate = self.cfg.sample_rate
        self.distance_metric = self.cfg.distance_metric
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        from encodec import EncodecModel

        if self.cfg.model_name == "encodec_48khz":
            self._model = EncodecModel.encodec_model_48khz()
        else:
            self._model = EncodecModel.encodec_model_24khz()

        self._model.set_target_bandwidth(self.cfg.target_bandwidth)
        self._model.to(self.device)
        self._model.eval()
        logger.info("EnCodec model loaded on %s", self.device)

    @property
    def model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            self._load_model()
        return self._model

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Extract continuous EnCodec latents.

        Parameters
        ----------
        audio : 1-D float32 array at 24 kHz
        sr : sample rate

        Returns
        -------
        embeddings : ndarray of shape ``[T, 128]``
        """
        if sr != self.sample_rate:
            raise ValueError(f"EnCodec requires {self.sample_rate} Hz audio, got {sr}")

        # Process in chunks to bound memory on long audio
        chunk_samples = self.sample_rate * 60  # 60s chunks
        if len(audio) > chunk_samples:
            parts = []
            for start in range(0, len(audio), chunk_samples):
                chunk = audio[start : start + chunk_samples]
                parts.append(self._extract_chunk(chunk))
            return np.concatenate(parts, axis=0)

        return self._extract_chunk(audio)

    def _extract_chunk(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Extract embeddings from a single audio chunk."""
        # EnCodec expects [B, C, samples]
        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encoder(wav)  # [1, 128, T]

        # Transpose to [T, 128]
        embeddings = emb.squeeze(0).permute(1, 0).cpu().numpy()
        del wav, emb
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return embeddings.astype(np.float32)

    def extract_from_file(self, path: str | Path) -> npt.NDArray[np.float32]:
        audio, sr = load_audio(path, target_sr=self.sample_rate)
        audio = normalize_audio(audio)
        return self.extract(audio, sr)


# ---------------------------------------------------------------------------
# MuQ
# ---------------------------------------------------------------------------


class MuQExtractor:
    """Extract MuQ frame-level embeddings.

    MuQ is a self-supervised music representation model (HuBERT-style)
    pre-trained with Mel-RVQ targets. Produces frame-level embeddings
    at ~75 Hz from the last hidden state (1024-d per frame).

    Requires: ``pip install muq``
    """

    def __init__(self, cfg: MuQConfig | None = None, device: str = "cpu") -> None:
        self.cfg = cfg or MuQConfig()
        self.embedding_dim = self.cfg.embedding_dim
        self.sample_rate = self.cfg.sample_rate
        self.distance_metric = self.cfg.distance_metric
        self.device = device
        self._model = None

    def _load_model(self) -> None:
        from muq import MuQ

        _bypass_torch_load_check()

        self._model = MuQ.from_pretrained(self.cfg.model_name)
        self._model = self._model.to(self.device).eval()
        logger.info("MuQ model loaded on %s", self.device)

    @property
    def model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            self._load_model()
        return self._model

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Extract MuQ embeddings.

        Parameters
        ----------
        audio : 1-D float32 array at 24 kHz
        sr : sample rate (must be 24000)

        Returns
        -------
        embeddings : ndarray of shape ``[T, 1024]``
        """
        if sr != self.sample_rate:
            raise ValueError(f"MuQ requires {self.sample_rate} Hz audio, got {sr}")

        # Process in chunks to avoid OOM on long audio
        chunk_samples = self.sample_rate * 30  # 30s chunks
        if len(audio) > chunk_samples:
            parts = []
            for start in range(0, len(audio), chunk_samples):
                chunk = audio[start : start + chunk_samples]
                parts.append(self._extract_chunk(chunk))
            return np.concatenate(parts, axis=0)

        return self._extract_chunk(audio)

    def _extract_chunk(self, audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Extract embeddings from a single audio chunk."""
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)

        # Disable autocast for MuQ — fp16 produces NaN on torch 2.5 / T4
        with torch.no_grad():
            output = self.model(wav, output_hidden_states=True)

        last_hidden = output.last_hidden_state  # [1, T, 1024]
        embeddings = last_hidden.squeeze(0).cpu().float().numpy()
        del wav, output, last_hidden
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return embeddings.astype(np.float32)

    def extract_from_file(self, path: str | Path) -> npt.NDArray[np.float32]:
        audio, sr = load_audio(path, target_sr=self.sample_rate)
        audio = normalize_audio(audio)
        return self.extract(audio, sr)


# ---------------------------------------------------------------------------
# Wav2Vec2 / XLS-R (speech SSL, anti-spoofing front-end)
# ---------------------------------------------------------------------------


class Wav2Vec2Extractor:
    """Extract Wav2Vec2-XLS-R frame-level embeddings (anti-spoofing transfer).

    XLS-R is a speech SSL model widely used as the front-end for synthetic
    audio / voice-spoofing detection. As a *frozen* low-level representation
    (sensitive to vocoder/synthesis artifacts rather than musical semantics)
    it is complementary to MERT/MuQ for the cross-generator generalisation
    problem. Produces last-hidden-state frame embeddings (~50 Hz, 1024-d).
    """

    def __init__(self, cfg: Wav2Vec2Config | None = None, device: str = "cpu") -> None:
        self.cfg = cfg or Wav2Vec2Config()
        self.embedding_dim = self.cfg.embedding_dim
        self.sample_rate = self.cfg.sample_rate
        self.distance_metric = self.cfg.distance_metric
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        _bypass_torch_load_check()

        self._processor = AutoFeatureExtractor.from_pretrained(self.cfg.model_name)
        self._model = AutoModel.from_pretrained(self.cfg.model_name).to(self.device)
        self._model.eval()
        logger.info("Wav2Vec2-XLS-R model loaded on %s", self.device)

    @property
    def model(self):  # type: ignore[no-untyped-def]
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):  # type: ignore[no-untyped-def]
        if self._processor is None:
            self._load_model()
        return self._processor

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Extract XLS-R embeddings.

        Parameters
        ----------
        audio : 1-D float32 array at 16 kHz
        sr : sample rate (must be 16000)

        Returns
        -------
        embeddings : ndarray of shape ``[T, 1024]``
        """
        if sr != self.sample_rate:
            raise ValueError(f"Wav2Vec2-XLS-R requires {self.sample_rate} Hz audio, got {sr}")

        # Process in chunks to avoid OOM on long audio
        chunk_samples = self.sample_rate * 30  # 30s chunks
        if len(audio) > chunk_samples:
            parts = []
            for start in range(0, len(audio), chunk_samples):
                chunk = audio[start : start + chunk_samples]
                parts.append(self._extract_chunk(chunk, sr))
            return np.concatenate(parts, axis=0)

        return self._extract_chunk(audio, sr)

    def _extract_chunk(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        """Extract embeddings from a single audio chunk."""
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        use_amp = self.device != "cpu" and torch.cuda.is_available()
        with torch.no_grad(), torch.autocast(self.device, enabled=use_amp):
            outputs = self.model(input_values, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]  # [1, T, 1024]
        embeddings = last_hidden.squeeze(0).cpu().float().numpy()
        del input_values, outputs, last_hidden
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return embeddings.astype(np.float32)

    def extract_from_file(self, path: str | Path) -> npt.NDArray[np.float32]:
        audio, sr = load_audio(path, target_sr=self.sample_rate)
        audio = normalize_audio(audio)
        return self.extract(audio, sr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_extractor(
    name: str,
    device: str = "cpu",
    **kwargs,  # type: ignore[no-untyped-def]
) -> CLAPExtractor | MERTExtractor | EnCodecExtractor | MuQExtractor | Wav2Vec2Extractor:
    """Instantiate an embedding extractor by name.

    Supported names
    ---------------
    clap        LAION-CLAP (48 kHz, 512-d)
    mert        MERT-v1-330M (24 kHz, 1024-d)  — best quality
    mert-95m    MERT-v1-95M  (24 kHz, 768-d)   — ~3.5× faster, speed/quality trade-off
    encodec     EnCodec-24kHz (24 kHz, 128-d)
    spec/logmel classical log-mel spectrogram (24 kHz, 128-d @ 75 fps) —
                dimension- and rate-matched control for the EnCodec arm
    muq         MuQ-large     (24 kHz, 1024-d)
    xls-r       Wav2Vec2-XLS-R-300M (16 kHz, 1024-d) — speech SSL anti-spoofing front-end
    """
    extractors = {
        "clap": (CLAPExtractor, CLAPConfig),
        "mert": (MERTExtractor, MERTConfig),
        "mert-95m": (MERTExtractor, MERT95MConfig),
        "encodec": (EnCodecExtractor, EnCodecConfig),
        "spec": (SpectrogramExtractor, SpectrogramConfig),
        "logmel": (SpectrogramExtractor, SpectrogramConfig),
        "spec-musicdet": (SpectrogramExtractor, MusicDETSpecConfig),
        # The flow, inside the deconvolution-artifact space.
        "combprint": (CombPrintExtractor, CombPrintConfig),
        "muq": (MuQExtractor, MuQConfig),
        "xls-r": (Wav2Vec2Extractor, Wav2Vec2Config),
        "wav2vec2": (Wav2Vec2Extractor, Wav2Vec2Config),
    }
    if name not in extractors:
        raise ValueError(f"Unknown extractor: {name!r}. Choose from {list(extractors)}")

    cls, cfg_cls = extractors[name]
    cfg = cfg_cls(**kwargs) if kwargs else cfg_cls()
    return cls(cfg=cfg, device=device)
