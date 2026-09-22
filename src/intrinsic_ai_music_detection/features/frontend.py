"""Single source of truth for "which front end, at what frame rate, in what window".

Why this module exists
----------------------
Ten of the twelve silent-corruption bugs in this project are one shape:

    code written for EnCodec assumes its 75 fps or its 128-d output, and
    silently produces zero rows — or the wrong vectors — for any other
    front end.

``build_reconstruction_control.py``, ``run_robustness_battery.py`` and
``measure_efficiency.py`` each acquired that bug independently, and each was
fixed independently with a slightly different local patch. That is the same
situation ``features/cache_keys.py`` was created to end for cache keys: four
call sites re-deriving one fact, three of them wrongly.

The rules this module enforces
------------------------------
1. **The front end follows the CHECKPOINT, never a default.** EnCodec latents
   and 128-bin log-mel are both 128-d, so a wrong front end passes every shape
   check and yields meaningless numbers. A disagreement between the checkpoint's
   ``embedding_name`` and an explicit request is a hard error, not a warning.
2. **The frame rate is DERIVED from the array**, as ``n_rows / analysis_seconds``
   — never from a constant. EnCodec gives 75 rows/s, ``combprint`` gives ~1
   profile/s; a constant that is right for one is a data-destroying filter for
   the other.
3. **An empty result is an error.** ``pool_windows`` returning zero rows used to
   propagate as ``None`` per track, then as an empty CSV, then as
   ``KeyError: 'manipulation'`` five lines later, hiding the cause after
   32,960 tracks of work. ``windows_or_raise`` refuses to return nothing.

Anything that needs to know a front end's rate, dimension or window size should
call into here rather than compute it again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# A window must contain at least this many rows to be worth pooling. Below it,
# the pooled vector is dominated by one or two frames and its statistics are
# meaningless — but note this is a FLOOR, never a filter: falling below it is
# reported, not silently dropped.
MIN_WINDOW_ROWS = 2


@dataclass(frozen=True)
class FrontEnd:
    """A resolved front end plus everything derived from it.

    ``frame_rate_hz`` is None until an actual extracted array has been seen —
    it is a property of the array, not of the config, which is exactly the
    assumption whose violation caused bugs 8, 10 and 11.
    """

    name: str
    extractor: Any
    embedding_dim: int
    sample_rate: int

    def extract(self, audio: npt.NDArray[np.float32], sr: int) -> npt.NDArray[np.float32]:
        emb = self.extractor.extract(audio, sr)
        if emb is None or len(emb) == 0:
            raise EmptyFrontEndOutput(
                f"front end {self.name!r} produced no rows for a {len(audio) / max(sr, 1):.2f}s "
                "buffer. This is a hard error: an empty frame matrix silently becomes an empty "
                "CSV downstream."
            )
        return np.asarray(emb, dtype=np.float32)

    def frame_rate_from(self, emb: npt.NDArray[np.float32], seconds: float) -> float:
        """Rows per second, measured. The ONLY sanctioned way to get a frame rate."""
        if seconds <= 0:
            raise ValueError(f"non-positive analysis duration {seconds!r}")
        return float(len(emb)) / float(seconds)


class EmptyFrontEndOutput(RuntimeError):
    """Raised instead of returning an empty array. See rule 3 in the module docstring."""


class FrontEndMismatch(SystemExit):
    """Raised when the requested front end disagrees with the checkpoint's.

    Deliberately derived from ``SystemExit`` (i.e. ``BaseException``, not
    ``Exception``) and not from ``Exception``, matching the existing
    ``raise SystemExit(...)`` convention in ``measure_efficiency.py`` and
    ``run_robustness_battery.py``. A front-end mismatch must not be swallowed by
    the ``except Exception: return None`` per-track handlers those scripts use —
    that is precisely how a wrong front end becomes an empty CSV instead of an
    error message.
    """


def resolve_frontend(
    embedding: str | None = None,
    flow: Any | None = None,
    device: str = "cpu",
) -> FrontEnd:
    """Resolve the front end to use, honouring the checkpoint above all else.

    Parameters
    ----------
    embedding : explicit request, or None to take the checkpoint's tag
    flow : a loaded checkpoint exposing ``embedding_name`` (optional)
    device : passed through to the extractor

    Raises
    ------
    FrontEndMismatch
        when ``embedding`` and the checkpoint's ``embedding_name`` disagree. This
        is deliberately fatal: several front ends share 128 dimensions, so the
        mismatch is invisible to every shape check downstream.
    """
    # NOTE: the mismatch guard runs BEFORE importing .embeddings. That import
    # pulls in librosa/torch, so importing first would turn a clear
    # "you asked for the wrong front end" into a ModuleNotFoundError on any box
    # without the audio stack — including the local dev machine, where this
    # guard most needs to be checkable.
    trained_on = getattr(flow, "embedding_name", None) if flow is not None else None

    if trained_on and embedding and trained_on != embedding:
        raise FrontEndMismatch(
            f"FRONT-END MISMATCH: checkpoint was trained on {trained_on!r} but --embedding is "
            f"{embedding!r}. Scoring a {trained_on!r}-trained model with {embedding!r} features "
            "produces meaningless numbers that pass every shape check (several front ends are "
            f"128-d). Re-run with --embedding {trained_on}."
        )

    name = embedding or trained_on
    if name is None:
        raise FrontEndMismatch(
            "No front end specified and the checkpoint carries no embedding_name tag. "
            "Pass --embedding explicitly rather than defaulting to EnCodec — defaulting is "
            "how bug 11 (measure_efficiency.py) happened."
        )
    if flow is not None and not trained_on:
        logger.warning(
            "Checkpoint has no embedding tag (saved before tagging existed) — using %r as "
            "requested. Verify this matches how the flow was trained.",
            name,
        )

    from .embeddings import get_extractor

    extractor = get_extractor(name, device=device)
    return FrontEnd(
        name=name,
        extractor=extractor,
        embedding_dim=int(getattr(extractor, "embedding_dim", 0)),
        sample_rate=int(getattr(extractor, "sample_rate", 0)),
    )


def frames_for_window(
    n_rows: int,
    total_seconds: float,
    window_duration: float,
    hop_duration: float,
) -> tuple[int, int]:
    """Window and hop in ROWS, derived from the array's own measured frame rate.

    This replaces every ``int(window_duration * 75)`` in the repo. At EnCodec's
    75 rows/s a 4 s window is 300 rows; at ``combprint``'s ~1 profile/s the same
    4 s is 4 rows, and asking a 7-row matrix for 300 rows is what produced
    "0 rows" after a full pass over 32,960 tracks.

    When the clip is genuinely too short for the requested window, the window is
    shrunk to fit rather than the track being dropped — and the shrink is
    reported by the caller via :func:`windows_or_raise`, never silently.
    """
    if n_rows <= 0:
        raise EmptyFrontEndOutput("cannot derive a window from a zero-row frame matrix")
    if total_seconds <= 0:
        raise ValueError(f"non-positive analysis duration {total_seconds!r}")

    fps = float(n_rows) / float(total_seconds)
    win = int(round(window_duration * fps))
    hop = int(round(hop_duration * fps))

    win = max(win, MIN_WINDOW_ROWS)
    hop = max(hop, 1)

    if win > n_rows:
        # Fit at least three windows into what we have, so the pooled statistics
        # still mean something, rather than collapsing to a single whole-track vector.
        win = max(min(win, max(n_rows // 3, MIN_WINDOW_ROWS)), MIN_WINDOW_ROWS)
        hop = max(win // 2, 1)
    return win, hop


def windows_or_raise(
    emb: npt.NDArray[np.float32],
    total_seconds: float,
    window_duration: float,
    hop_duration: float,
    context: str = "",
) -> npt.NDArray[np.float32]:
    """Pool ``emb`` into windows, raising rather than returning an empty array.

    The standing rule from twelve silent-corruption bugs: *a guard written for
    one configuration becomes a data-destroying filter in another.* So the only
    acceptable failure mode here is a loud one.
    """
    from .pooling import pool_windows

    win, hop = frames_for_window(len(emb), total_seconds, window_duration, hop_duration)
    pooled = pool_windows(emb, win, hop)
    if pooled is None or len(pooled) == 0:
        raise EmptyFrontEndOutput(
            f"pooling produced zero windows{f' ({context})' if context else ''}: "
            f"{len(emb)} rows over {total_seconds:.2f}s → window={win} hop={hop}. "
            "Refusing to return an empty result."
        )
    return np.asarray(pooled, dtype=np.float32)
