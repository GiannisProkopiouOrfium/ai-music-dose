"""Single source of truth for embedding-cache keys.

Why this module exists
----------------------
The key format was duplicated as an inline f-string at four call sites. That is
how §9.7.6 happened: a key that does not name every variable which changes the
stored array turns the cache into a silent corruption channel — two different
configurations read each other's vectors, every shape check passes, and the
resulting numbers look plausible enough to be written up (twice, in that case).

Two rules follow, and this module enforces both:

1. **Anything that changes the stored array must appear in the key.** Adding a
   preprocessing knob without adding it here is a data-corruption bug, not a
   missing feature.
2. **Existing keys must never move.** There are hundreds of GB of populated
   caches on EC2; a reformatting that changes the digest silently invalidates
   them (best case: an expensive re-extraction; worst case: a half-populated
   cache mixing two conventions). ``embedding_cache_key`` therefore reproduces
   the historical string EXACTLY for the historical arguments, and new knobs may
   only ever append when they are set to a non-default value.

The digest is over a plain-text descriptor; ``tests/test_cache_keys.py`` pins
known-good digests so a future edit cannot move them by accident.
"""

from __future__ import annotations

import hashlib


def cache_key_string(
    track_id: str,
    target_sr: int,
    max_duration: float,
    preprocess_mode: str,
    lowpass_hz: float | None = None,
    fixed_duration: float | None = None,
    analysis_region: str = "full",
    extra_tags: tuple[str, ...] = (),
) -> str:
    """Return the plain-text descriptor hashed into a cache key.

    Parameters mirror the extraction pipeline. ``extra_tags`` carries knobs added
    after the original format; each is appended verbatim and MUST be empty when
    the knob sits at its historical default, so old keys stay reachable.
    """
    lp_tag = f"_lp{int(lowpass_hz)}" if lowpass_hz else ""
    fd_tag = f"_fd{fixed_duration:g}" if fixed_duration else ""
    reg_tag = f"_r{analysis_region}" if analysis_region != "full" else ""
    suffix = "".join(extra_tags)
    return f"{track_id}_{target_sr}_{max_duration}_{preprocess_mode}" f"{lp_tag}{fd_tag}{reg_tag}{suffix}"


def embedding_cache_key(
    track_id: str,
    target_sr: int,
    max_duration: float,
    preprocess_mode: str,
    lowpass_hz: float | None = None,
    fixed_duration: float | None = None,
    analysis_region: str = "full",
    extra_tags: tuple[str, ...] = (),
) -> str:
    """MD5 hex digest naming one cached embedding matrix (``<key>.npy``)."""
    return hashlib.md5(
        cache_key_string(
            track_id,
            target_sr,
            max_duration,
            preprocess_mode,
            lowpass_hz=lowpass_hz,
            fixed_duration=fixed_duration,
            analysis_region=analysis_region,
            extra_tags=extra_tags,
        ).encode()
    ).hexdigest()


def band_match_tag(resample_hz: float | None) -> str:
    """Cache tag for the decimate-and-return bandwidth control.

    Distinct from the ``_lp`` tag because the two are NOT interchangeable: a
    low-pass leaves a 0.9287 channel-alone AUC on canonical SONICS where
    decimation leaves 0.5766. Sharing one tag would let an inadequate control
    silently read an adequate one's cache.
    """
    return "" if resample_hz is None else f"_rs{int(resample_hz)}"


def spec_floor_tag(log_floor_db: float | None) -> str:
    """Cache tag for the spectrogram dynamic-range floor.

    Empty when ``None`` (the historical ``log(mel + 1e-10)`` behaviour), so
    existing spectrogram caches keep their keys.
    """
    return "" if log_floor_db is None else f"_fl{log_floor_db:g}"


def level_match_tag(level_match: bool, peak_normalise: bool = False) -> str:
    """Cache tag for DC removal + silence trimming.

    Separate from the bandwidth tag because the two controls are independent and
    address different confound families: under ``resample_hz=15000`` the
    bandwidth family drops to 0.5778 on SONICS while the level family is still
    at 0.7773. Sharing a tag would let a run with only one control read the
    other's vectors.
    """
    if not level_match:
        return ""
    return "_lvlpk" if peak_normalise else "_lvl"


def musicdet_preproc_tag(drop_dc: bool = False, rank_gaussianise: bool = False) -> str:
    """Cache tag for the MusicDET preprocessing choices we had never ported (arm F5).

    Two knobs, read from their ``model.py``:

    * ``drop_dc`` — they crop ``x[:, :, 1:, 1:]``, discarding the first frequency
      bin and the first time frame. We keep bins ``[0:256]`` including DC, and DC
      is still a 0.53-0.64 channel descriptor for us, so this is not cosmetic.
    * ``rank_gaussianise`` — the clean version of the two BatchNorms and the SELU
      that sit between their spectrogram and their flow. A quantile transform
      fitted on the REAL training split maps every feature to a standard normal,
      which removes marginal scale entirely and leaves the density only
      cross-feature dependency to model.

    Both change the stored vectors, so both must be in the key (rule 1). Both
    return "" at their historical defaults, so the hundreds of GB of existing
    caches keep their digests (rule 2).
    """
    tag = ""
    if drop_dc:
        tag += "_nodc"
    if rank_gaussianise:
        tag += "_rg"
    return tag


def window_normalise_tag(window_normalise: bool = False) -> str:
    """Cache tag for MusicDET's per-segment RMS normalisation (arm F5).

    They RMS-normalise each 4.04 s segment; we normalise per track (LUFS). These
    produce different vectors for the same audio, so they cannot share a key.
    """
    return "_wnorm" if window_normalise else ""
