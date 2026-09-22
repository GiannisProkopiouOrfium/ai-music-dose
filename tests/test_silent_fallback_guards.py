"""Regression tests for the silent-corruption family (report §8.4).

Four bugs of the same shape have already invalidated results in this project:
a fallback or guard that produces a plausible wrong answer instead of failing.
The standing rule is that every fallback must be opt-in and every new flag needs
a test asserting rows survive. These are those tests.

Each test names the specific failure it prevents.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    """Import a top-level script from scripts/ as a module.

    They are not a package, so they cannot be imported normally; this keeps the
    tests honest by exercising the real file rather than a copy.
    """
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# §8.2 — score_external_corpus.py silently re-trained instead of loading
# ---------------------------------------------------------------------------


def test_missing_flow_checkpoint_is_a_hard_error_by_default(tmp_path):
    """The bug: a missing --flow-path fell through to training a fresh default
    flow. Two runs scoring two DIFFERENT detectors both took that path and
    emitted byte-identical score files (mean -194.96444750715756 in both).
    """
    sec = _load_script("score_external_corpus")
    with pytest.raises(SystemExit) as excinfo:
        sec._load_or_train_flow(
            flow_path=tmp_path / "does_not_exist.pt",
            sonics_cache=tmp_path / "cache",
            sonics_manifest=tmp_path / "manifest.csv",
            wf_window_duration=4.0,
            wf_hop_duration=2.0,
            max_duration=55.0,
            flow_epochs=1,
            device="cpu",
            seed=42,
            # allow_retrain omitted => must default to refusing
        )
    message = str(excinfo.value)
    assert "does_not_exist.pt" in message, "the error must name the path that is missing"
    assert "--allow-retrain" in message, "the error must name the opt-in escape hatch"


def test_retrain_requires_an_existing_cache_for_the_requested_embedding(tmp_path):
    """The second half of the same bug: the retrain path was hard-coded to the
    'encodec' cache subdirectory. Requesting a retrain for a spectrogram arm
    therefore trained on codec latents and then scored spectrogram features with
    it — both are 128-d, so every shape check passed.
    """
    sec = _load_script("score_external_corpus")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("track_id,label,status\nt1,real,ok\n")
    cache_root = tmp_path / "cache"
    (cache_root / "encodec").mkdir(parents=True)  # exists for encodec, NOT for spec

    with pytest.raises(SystemExit) as excinfo:
        sec._load_or_train_flow(
            flow_path=tmp_path / "missing.pt",
            sonics_cache=cache_root,
            sonics_manifest=manifest,
            wf_window_duration=4.0,
            wf_hop_duration=2.0,
            max_duration=55.0,
            flow_epochs=1,
            device="cpu",
            seed=42,
            allow_retrain=True,
            embedding="spec",
        )
    assert "spec" in str(excinfo.value)


def test_existing_checkpoint_is_loaded_not_retrained(tmp_path, monkeypatch):
    """The load path must still work — a guard that blocks everything is the
    other half of this failure family (cf. --flow-only discarding every track).
    """
    sec = _load_script("score_external_corpus")
    flow_path = tmp_path / "flow.pt"
    flow_path.write_bytes(b"not really a checkpoint")

    sentinel = object()
    loaded: dict = {}

    class _FakeLoading:
        @staticmethod
        def load_detector(path, device="cpu"):
            loaded["path"] = Path(path)
            return sentinel

    monkeypatch.setitem(sys.modules, "intrinsic_ai_music_detection.models.loading", _FakeLoading)

    result = sec._load_or_train_flow(
        flow_path=flow_path,
        sonics_cache=tmp_path,
        sonics_manifest=tmp_path / "m.csv",
        wf_window_duration=4.0,
        wf_hop_duration=2.0,
        max_duration=55.0,
        flow_epochs=1,
        device="cpu",
        seed=42,
    )
    assert result is sentinel
    assert loaded["path"] == flow_path


# ---------------------------------------------------------------------------
# The log-floor: an empty band must not map to one exact constant
# ---------------------------------------------------------------------------


def test_spectrogram_log_floor_removes_the_band_limited_constant():
    """Without a floor, a band with no energy maps to log(1e-10) in EVERY frame
    of EVERY band-limited track — one exact constant that a one-class flow can
    key on. Since FakeMusicCaps resamples every fake to 16 kHz while its reals
    are full-band, that constant is close to a class label.
    """
    librosa = pytest.importorskip("librosa")
    from intrinsic_ai_music_detection.config import SpectrogramConfig
    from intrinsic_ai_music_detection.features.embeddings import SpectrogramExtractor

    sr = 24_000
    t = np.arange(sr * 2) / sr
    # A band-limited signal: energy only near 440 Hz, nothing in the high mels.
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    no_floor = SpectrogramExtractor(SpectrogramConfig()).extract(audio, sr)
    floored = SpectrogramExtractor(SpectrogramConfig(log_floor_db=80.0)).extract(audio, sr)

    assert no_floor.shape == floored.shape

    constant = float(np.log(1e-10))
    n_at_constant = int(np.sum(np.isclose(no_floor, constant, atol=1e-3)))
    assert n_at_constant > 0, "expected empty bands to be pinned at log(1e-10) without a floor"
    assert (
        int(np.sum(np.isclose(floored, constant, atol=1e-3))) == 0
    ), "with a floor set, no bin may sit at the log(log_offset) constant"
    # The floor must compress the dynamic range, not merely shift it.
    assert np.ptp(floored) < np.ptp(no_floor)


def test_spectrogram_default_is_unchanged():
    """Every number reported before this change was produced with log_offset
    alone. The default must therefore still be the old behaviour, or historical
    results silently stop being reproducible.
    """
    from intrinsic_ai_music_detection.config import SpectrogramConfig

    assert SpectrogramConfig().log_floor_db is None


# ---------------------------------------------------------------------------
# Precision is prevalence-dependent and the assumption must be visible
# ---------------------------------------------------------------------------


def test_precision_at_fpr_states_its_prevalence():
    fuse = _load_script("fuse_labelfree_scores")
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 1.0, 5_000)
    fake = rng.normal(3.0, 1.0, 5_000)

    declared = fuse._precision_metrics(real, fake, prevalence=0.01)
    assert declared["precision_prevalence"] == 0.01
    assert declared["precision_prevalence_source"] == "declared"

    implied = fuse._precision_metrics(real, fake, prevalence=None)
    assert implied["precision_prevalence_source"] == "eval_set_ratio"
    assert implied["precision_prevalence"] == pytest.approx(0.5, abs=1e-6)

    # TPR is prevalence-free; precision is not. If these moved together the
    # metric would be silently ignoring the declared rate.
    assert declared["tpr_at_fpr01"] == implied["tpr_at_fpr01"]
    assert declared["precision_at_fpr01"] < implied["precision_at_fpr01"]


def test_complexity_join_does_not_clobber_the_label_column(tmp_path):
    """The bug: eval_complexity_compensated.py --per-track-csv emits
    track_id,label,algorithm,nll,flac_bits_per_sec. Merging all of that into the
    window-flow table collided on 'label' and 'algorithm', so pandas renamed them
    label_x/label_y — and the fusion died with KeyError: 'label' ~20 lines later,
    nowhere near the cause. Only the two needed columns may be joined.
    """
    import pandas as pd

    window_flow = pd.DataFrame(
        {
            "track_id": ["a", "b", "c"],
            "label": ["real", "fake", "real"],
            "algorithm": ["", "udio-30s", ""],
            "encodec_wf_mean": [-1.0, -5.0, -1.2],
        }
    )
    complexity = pd.DataFrame(
        {
            "track_id": ["a", "b", "c"],
            "label": ["real", "fake", "real"],
            "algorithm": ["", "udio-30s", ""],
            "nll": [1.0, 5.0, 1.2],
            "flac_bits_per_sec": [251_723.0, 209_093.0, 250_000.0],
        }
    )

    trimmed = complexity[["track_id", "flac_bits_per_sec"]].drop_duplicates("track_id")
    merged = window_flow.merge(trimmed, on="track_id", how="left")

    assert "label" in merged.columns, "the join must not rename the label column"
    assert "label_x" not in merged.columns and "label_y" not in merged.columns
    assert len(merged) == len(window_flow), "the join must not fan out rows"
    assert merged["flac_bits_per_sec"].notna().all()


def test_precision_falls_as_ai_music_gets_rarer():
    """Sanity on the direction: at a fixed operating point, rarer positives mean
    a larger share of the alarms are false. A precision number quoted without its
    prevalence is therefore uninterpretable.
    """
    fuse = _load_script("fuse_labelfree_scores")
    rng = np.random.default_rng(1)
    real = rng.normal(0.0, 1.0, 5_000)
    fake = rng.normal(3.0, 1.0, 5_000)

    precisions = [
        fuse._precision_metrics(real, fake, prevalence=p)["precision_at_fpr01"] for p in (0.5, 0.1, 0.01, 0.001)
    ]
    assert precisions == sorted(precisions, reverse=True)
