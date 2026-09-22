"""Experiment configuration using dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

# ---------------------------------------------------------------------------
# Embedding configs
# ---------------------------------------------------------------------------


@dataclass
class CLAPConfig:
    model_name: str = "630k-audioset-best"
    sample_rate: int = 48_000
    embedding_dim: int = 512
    chunk_duration: float = 10.0  # seconds per chunk fed to CLAP
    hop_duration: float = 5.0  # hop between chunks
    enable_fusion: bool = False
    distance_metric: str = "cosine"


@dataclass
class MERTConfig:
    model_name: str = "m-a-p/MERT-v1-330M"
    sample_rate: int = 24_000
    embedding_dim: int = 1024
    distance_metric: str = "cosine"


@dataclass
class MERT95MConfig:
    """Smaller MERT model for speed/accuracy trade-off comparison.

    MERT-v1-95M is ~3.5x faster than MERT-v1-330M with a 768-d embedding.
    Use for subsample pilot comparison before committing to the full 330M run.
    """

    model_name: str = "m-a-p/MERT-v1-95M"
    sample_rate: int = 24_000
    embedding_dim: int = 768
    distance_metric: str = "cosine"


@dataclass
class EnCodecConfig:
    model_name: str = "encodec_24khz"
    sample_rate: int = 24_000
    embedding_dim: int = 128
    target_bandwidth: float = 6.0
    distance_metric: str = "euclidean"


@dataclass
class SpectrogramConfig:
    """Classical log-mel front end — the MusicDET-style spectral-domain control.

    Deliberately configured to be dimension- and rate-matched to
    ``EnCodecConfig`` so that swapping ``--embeddings encodec`` for
    ``--embeddings spec`` changes ONLY the representation:

      * ``sample_rate`` 24 kHz — same canonical audio.
      * ``hop_length`` 320 @ 24 kHz = **75 frames/s**, identical to EnCodec's
        latent frame rate (24000/320), so 4 s/2 s windowing pools the same
        number of frames in both arms.
      * ``n_mels`` 128 = EnCodec's latent width, so the flow sees the same
        input dimensionality and parameter count.

    Rationale: MusicDET (arXiv:2605.18072) runs its one-class flow on a raw
    (banded) STFT energy spectrogram, whereas we run ours on a learned neural
    codec latent. Those are two confounded differences (representation AND
    granularity). This front end isolates the representation axis inside our
    own pipeline, protocol, and canonicalization.
    """

    sample_rate: int = 24_000
    n_fft: int = 1024
    hop_length: int = 320  # -> exactly 75 fps at 24 kHz, matching EnCodec
    n_mels: int = 128  # -> matches EnCodec's 128-d latent
    fmin: float = 20.0
    fmax: float | None = None  # None -> Nyquist
    log_offset: float = 1e-10
    # Dynamic-range floor in dB below each track's own peak power, applied BEFORE
    # the log. ``None`` reproduces the historical `log(power + log_offset)`.
    #
    # Why this knob exists. With log_offset alone, a frequency band containing NO
    # energy maps to exactly log(1e-10) = -23.026 — one specific constant, in
    # every frame, of every track that is band-limited. Both benchmarks are
    # band-asymmetric (FakeMusicCaps resamples every generated clip to 16 kHz
    # while its reals are a YouTube pull; SONICS fakes arrive band-limited
    # against high-quality real m4a), and MP3-64k @ 24 kHz cuts near 10.5 kHz,
    # i.e. ABOVE an 8 kHz ceiling — so canonicalisation does not close the gap.
    # A one-class flow can then reach near-perfect separation by noticing "these
    # dimensions are pinned at the floor", which is a delivery-chain measurement,
    # not a synthesis artifact.
    #
    # Clamping to a finite range relative to the per-track peak makes an empty
    # band merely quiet rather than a unique constant, and removes the absolute
    # level dependence at the same time. 80 dB is a conventional display range.
    # Changing it changes the stored embedding, so it is part of the cache key
    # (see features/cache_keys.spec_floor_tag).
    log_floor_db: float | None = None
    per_frame_normalize: bool = False
    embedding_dim: int = 128
    distance_metric: str = "euclidean"
    # "mel"    -> n_mels log-mel bins (our default, EnCodec-matched)
    # "linear" -> n_fft//2+1 LINEAR power-STFT bins, as MusicDET actually uses.
    #             Mel compresses high frequencies logarithmically, averaging away
    #             the fine band-edge / spectral-hole structure where codec and
    #             vocoder artifacts live; a linear STFT keeps every bin.
    spec_type: str = "mel"


@dataclass
class MusicDETSpecConfig(SpectrogramConfig):
    """Front end matched to MusicDET's spec-nf, read from their source.

    model.py:611-654 — sr 16 kHz, n_fft 512, win 512, hop 160 (=100 fps),
    power=2, one-sided linear STFT with F = n_fft//2 + 1 = 257 bins, band edges
    given in Hz over (0, 8000). Their preprocessing (preprocess.py) resamples
    48k->16k mono and applies NO loudness normalisation and NO MP3 round-trip.

    Using this makes our representation axis directly comparable to theirs; it
    deliberately does NOT adopt their preprocessing, so our canonicalisation
    (and the compression robustness it buys) is held constant.
    """

    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 160  # -> 100 fps at 16 kHz, matching MusicDET
    fmin: float = 0.0
    fmax: float | None = 8000.0
    spec_type: str = "linear"
    # NOTE 257 = n_fft//2+1 is PRIME, so their own `assert F % n_bands == 0`
    # could never pass for n_bands=2. Resolving that: they crop to band_hz via
    # k = int(round(hz * n_fft / sr)), and round(8000*512/16000) = 256, so the
    # modelled range is bins [0:256] -- 256 bins, divisible by 2/4/8. We match
    # that crop, which is also what makes our banded flow runnable on this arm.
    embedding_dim: int = 256


@dataclass
class CombPrintConfig:
    """Front end that puts the flow INSIDE the deconvolution-artifact space.

    Why this exists
    ---------------
    Afchar et al. (arXiv:2506.19108) report >99% accuracy on Suno/Udio from the
    deconvolution comb, using the full 445-d peak-residual *profile* with a
    classifier. Our `comb_strength` collapses that profile to a single
    autocorrelation peak and reaches 0.899 / 0.730 — good, but it is one number
    where they use hundreds, and the discarded structure is exactly the part that
    distinguishes one decoder's stride pattern from another's.

    This extractor keeps the profile and hands it to the one-class flow, which
    unifies the two halves of the project:

    * the FEATURE is mechanistic (a decoder-stride signature), not a generic
      embedding that inherits the corpus;
    * the MODEL is trained on real music only, with no generator output and no
      fake label anywhere — which is the claim the project set out to make and
      could not support on EnCodec latents or spectrograms.

    Representation
    --------------
    Per sub-span: long STFT -> time-averaged log spectrum -> subtract a narrow
    median envelope -> autocorrelate across frequency -> keep ``n_lags``
    log-spaced lags spanning ``[min_spacing_hz, max_spacing_hz]``. A row is
    therefore "which spectral periodicities does this span contain", and a comb
    at 300 Hz shows up as a peak at that lag regardless of the musical content.

    ``smooth_bins`` defaults to 5 because the sweep is monotone down to it
    (0.899 at 5 against 0.637 at 65 on FakeMusicCaps). It is the one free
    parameter and must be selected on a DIFFERENT corpus from the one reported.
    """

    sample_rate: int = 16_000
    n_fft: int = 16_384
    span_duration: float = 2.0  # seconds averaged into one profile
    span_hop: float = 1.0  # seconds between profiles
    f_min: float = 500.0
    f_max: float | None = None  # None -> 0.98 * Nyquist, so the band adapts
    # "hull"     -> Afchar's actual descriptor: peak height above a LOWER
    #               envelope, clipped and max-normalised, resampled to n_bins.
    #               This is what their >99% result uses.
    # "autocorr" -> frequency-axis autocorrelation of a median-filter residual
    #               (our first attempt; keeps periodicity, discards the pattern).
    # "both"     -> concatenation, so the flow sees pattern AND periodicity.
    profile_kind: str = "hull"
    hull_area: int = 10  # running-minimum width, their `area`
    hull_max_db: float = 5.0  # their `max_normalise` clip
    n_bins: int = 445  # OURS, not theirs; published dimension is 4458 (retraction 8)
    smooth_bins: int = 5  # median width, "autocorr" mode only
    n_lags: int = 128  # autocorrelation dimension
    min_spacing_hz: float = 40.0
    max_spacing_hz: float = 4_000.0
    distance_metric: str = "euclidean"


@dataclass
class MuQConfig:
    model_name: str = "OpenMuQ/MuQ-large-msd-iter"
    sample_rate: int = 24_000
    embedding_dim: int = 1024
    distance_metric: str = "cosine"


@dataclass
class Wav2Vec2Config:
    """Wav2Vec2-XLS-R speech SSL front-end (anti-spoofing transfer).

    XLS-R is the canonical front-end for synthetic-audio / voice-spoofing
    detection (e.g. wav2vec2+AASIST in ASVspoof, SONICS). Unlike MERT/MuQ
    (trained on real *music* for understanding), XLS-R is trained on speech
    and is sensitive to vocoder/synthesis artifacts — a low-level
    representation complementary to music SSL for the cross-generator wall.

    Operates at 16 kHz (Nyquist 8 kHz), matching the --lowpass-hz 8000
    bandwidth-equalisation control. Uses frame-level last-hidden-state
    (~50 Hz, 1024-d).
    """

    model_name: str = "facebook/wav2vec2-xls-r-300m"
    sample_rate: int = 16_000
    embedding_dim: int = 1024
    distance_metric: str = "cosine"


# ---------------------------------------------------------------------------
# PHD / ID estimation configs
# ---------------------------------------------------------------------------


@dataclass
class PHDConfig:
    alpha: float = 1.0
    n_reruns: int = 3
    n_points: int = 7
    n_points_min: int = 3
    min_subsample: int = 40
    intermediate_points: int = 7


@dataclass
class IDEstimationConfig:
    phd: PHDConfig = field(default_factory=PHDConfig)
    estimators: list[str] = field(default_factory=lambda: ["phd", "twonn", "mle"])


# ---------------------------------------------------------------------------
# Fakeprint baseline config
# ---------------------------------------------------------------------------


@dataclass
class FakeprintConfig:
    n_fft: int = 16_384
    sample_rate: int = 44_100
    f_min: int = 5_000
    f_max: int = 16_000
    max_duration: int = 180  # seconds


# ---------------------------------------------------------------------------
# Data configs
# ---------------------------------------------------------------------------


@dataclass
class S3Config:
    bucket: str = "ai-plagiarism-data"
    prefix: str = "ai-data-augmentation"
    registry_key: str = "ai-data-augmentation/registry/plagiarism_dataset.csv"
    region: str = "eu-west-1"


@dataclass
class DataConfig:
    s3: S3Config = field(default_factory=S3Config)
    local_cache_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    embeddings_dir: Path = field(default_factory=lambda: Path("data/processed/embeddings"))
    id_scores_dir: Path = field(default_factory=lambda: Path("data/processed/id_scores"))
    max_duration: int = 360  # seconds — skip tracks longer than this
    min_frames: int = 80  # minimum embedding frames for stable PHD (matches GPTID MINIMAL_CLOUD)


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig:
    window_size: float = 1.0  # seconds
    hop_size: float = 0.5  # seconds


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    clap: CLAPConfig = field(default_factory=CLAPConfig)
    mert: MERTConfig = field(default_factory=MERTConfig)
    encodec: EnCodecConfig = field(default_factory=EnCodecConfig)
    muq: MuQConfig = field(default_factory=MuQConfig)
    id_estimation: IDEstimationConfig = field(default_factory=IDEstimationConfig)
    fakeprint: FakeprintConfig = field(default_factory=FakeprintConfig)
    data: DataConfig = field(default_factory=DataConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    device: str = "cuda"
    seed: int = 42


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load config from YAML file, falling back to defaults."""
    if path is None:
        return ExperimentConfig()

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = ExperimentConfig()
    if raw is None:
        return cfg

    # Simple shallow merge — extend as needed
    for section_name, section_cls in [
        ("clap", CLAPConfig),
        ("mert", MERTConfig),
        ("encodec", EnCodecConfig),
        ("muq", MuQConfig),
        ("fakeprint", FakeprintConfig),
        ("audio", AudioConfig),
    ]:
        if section_name in raw:
            setattr(cfg, section_name, section_cls(**raw[section_name]))

    if "device" in raw:
        cfg.device = raw["device"]
    if "seed" in raw:
        cfg.seed = raw["seed"]

    return cfg
