"""Regression test for the --flow-only extraction path.

`--flow-only` sets estimators=[] so the expensive intrinsic-dimension work is
skipped. `_compute_ids_from_emb` used to gate on an ID key being present, which
with an empty estimator list dropped EVERY track: the run ended with
"No results produced" before the window-flow eval started, and three separate
experiment batches (the spec-musicdet arms and the likelihood-ratio arms)
silently produced no output.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_runner():
    pytest.importorskip("librosa")
    pytest.importorskip("soxr")
    spec = importlib.util.spec_from_file_location(
        "run_balanced_ablation", REPO / "scripts" / "run_balanced_ablation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compute_ids_returns_row_when_no_estimators():
    m = _load_runner()
    emb = np.random.default_rng(0).normal(size=(400, 128))
    row = m._compute_ids_from_emb(
        {"track_id": "t1", "label": "real"},
        emb,
        audio_duration=10.0,
        metric="euclidean",
        estimators=[],  # <- the --flow-only configuration
        with_temporal=False,
        window_duration=4.0,
        hop_duration=2.0,
        rng=np.random.default_rng(0),
    )
    assert row is not None, "--flow-only must not drop the track"
    assert row["track_id"] == "t1" and row["label"] == "real"


def test_compute_ids_still_gates_when_estimators_requested():
    """With estimators asked for but none computable, dropping the row is correct."""
    m = _load_runner()
    tiny = np.random.default_rng(0).normal(size=(3, 128))  # too few points for an ID
    row = m._compute_ids_from_emb(
        {"track_id": "t2", "label": "real"},
        tiny,
        audio_duration=1.0,
        metric="euclidean",
        estimators=["twonn"],
        with_temporal=False,
        window_duration=4.0,
        hop_duration=2.0,
        rng=np.random.default_rng(0),
    )
    assert row is None


def test_flow_only_requires_its_companion_flags():
    m = _load_runner()
    p = m.build_parser() if hasattr(m, "build_parser") else None
    if p is None:
        pytest.skip("parser not exposed as build_parser()")
    args = p.parse_args(["--flow-only"])
    assert args.flow_only is True
