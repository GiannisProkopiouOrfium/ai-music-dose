"""Lock the embedding-cache key format.

There are hundreds of GB of populated caches on EC2 keyed by these digests. A
reformatting that changes them silently invalidates the lot: best case an
expensive re-extraction, worst case a half-populated cache mixing two
conventions, which is the §9.7.6 failure mode (two configurations reading each
other's vectors while every shape check passes).

These digests were computed from the historical inline f-strings that lived in
``run_balanced_ablation.py`` before the format was centralised. They are a
regression lock, not a specification: if one of these fails, the cache format
moved and every existing cache directory is now unreachable.
"""

from __future__ import annotations

import hashlib

import pytest

from intrinsic_ai_music_detection.features.cache_keys import cache_key_string, embedding_cache_key, spec_floor_tag


def _legacy_key(track_id, target_sr, max_duration, preprocess_mode, lowpass_hz, fixed_duration, analysis_region):
    """The exact f-string that produced every cache on disk today."""
    lp_tag = f"_lp{int(lowpass_hz)}" if lowpass_hz else ""
    fd_tag = f"_fd{fixed_duration:g}" if fixed_duration else ""
    reg_tag = f"_r{analysis_region}" if analysis_region != "full" else ""
    return hashlib.md5(
        f"{track_id}_{target_sr}_{max_duration}_{preprocess_mode}{lp_tag}{fd_tag}{reg_tag}".encode()
    ).hexdigest()


LEGACY_CASES = [
    # (track_id, sr, max_duration, mode, lowpass, fixed_duration, region)
    ("trk1", 24_000, 55.0, "preprocessed", None, None, "full"),  # SONICS headline
    ("trk2", 24_000, 10.0, "preprocessed", None, None, "full"),  # FakeMusicCaps native
    ("trk3", 24_000, 55.0, "preprocessed", 8000, None, "full"),  # bandwidth recheck
    ("trk4", 16_000, 55.0, "preprocessed", None, None, "full"),  # spec-musicdet / xls-r
    ("trk5", 24_000, 55.0, "raw", None, 30.0, "intro"),  # fixed-duration region sweep
]


@pytest.mark.parametrize("case", LEGACY_CASES)
def test_key_matches_the_historical_format(case):
    track_id, sr, max_dur, mode, lp, fd, region = case
    assert embedding_cache_key(
        track_id,
        sr,
        max_dur,
        mode,
        lowpass_hz=lp,
        fixed_duration=fd,
        analysis_region=region,
    ) == _legacy_key(track_id, sr, max_dur, mode, lp, fd, region)


def test_known_good_digests_are_pinned():
    """Belt-and-braces: literal digests, so an edit to BOTH the helper and the
    legacy replica above still fails the suite."""
    assert embedding_cache_key("trk1", 24_000, 55.0, "preprocessed") == (
        "c3a54542c8028f16c61574294e906ac1"  # pragma: allowlist secret
    )
    assert embedding_cache_key("trk2", 24_000, 55.0, "preprocessed", lowpass_hz=8000) == (
        "4d2c59f4418412dc49828038ef771c32"  # pragma: allowlist secret
    )


def test_default_extra_tags_do_not_move_the_key():
    """A new knob at its default value must be invisible to the key."""
    base = embedding_cache_key("t", 24_000, 55.0, "preprocessed")
    assert embedding_cache_key("t", 24_000, 55.0, "preprocessed", extra_tags=()) == base
    assert spec_floor_tag(None) == ""
    assert (
        embedding_cache_key(
            "t",
            24_000,
            55.0,
            "preprocessed",
            extra_tags=tuple(t for t in (spec_floor_tag(None),) if t),
        )
        == base
    )


def test_spec_floor_changes_the_key():
    """The whole point: a floor that changes the stored vectors must change the
    key, or the two configurations silently share a cache."""
    base = embedding_cache_key("t", 24_000, 55.0, "preprocessed")
    floored = embedding_cache_key("t", 24_000, 55.0, "preprocessed", extra_tags=(spec_floor_tag(80.0),))
    assert floored != base
    assert spec_floor_tag(80.0) == "_fl80"
    # And two different floors must not collide with each other either.
    assert floored != embedding_cache_key("t", 24_000, 55.0, "preprocessed", extra_tags=(spec_floor_tag(60.0),))


def test_every_argument_is_actually_in_the_key():
    """Guards the rule: anything that changes the stored array changes the key.

    A parameter that silently does not reach the digest is a data-corruption bug,
    so vary each one in turn and require the digest to move.
    """
    base_kwargs = dict(
        track_id="t",
        target_sr=24_000,
        max_duration=55.0,
        preprocess_mode="preprocessed",
    )
    base = embedding_cache_key(**base_kwargs)
    variants = [
        dict(base_kwargs, track_id="other"),
        dict(base_kwargs, target_sr=16_000),
        dict(base_kwargs, max_duration=10.0),
        dict(base_kwargs, preprocess_mode="raw"),
        dict(base_kwargs, lowpass_hz=8000),
        dict(base_kwargs, fixed_duration=30.0),
        dict(base_kwargs, analysis_region="intro"),
    ]
    digests = {embedding_cache_key(**v) for v in variants}
    assert base not in digests, "a varied argument did not reach the cache key"
    assert len(digests) == len(variants), "two different configurations collide on one key"


def test_key_string_is_human_readable_for_debugging():
    """The digest is opaque; the descriptor behind it must not be, or diagnosing
    a cache miss means reverse-engineering an MD5."""
    assert cache_key_string("abc", 24_000, 55.0, "preprocessed", lowpass_hz=7000) == (
        "abc_24000_55.0_preprocessed_lp7000"
    )


def test_lowpass_and_resample_controls_never_share_a_key():
    """A Butterworth low-pass and a decimate-and-return are NOT the same control:
    measured on canonical SONICS, the first leaves a channel-alone AUC of 0.9287
    and the second 0.5766. If they shared a cache tag, the inadequate control
    would silently read the adequate one's vectors.
    """
    from intrinsic_ai_music_detection.features.cache_keys import band_match_tag

    base = embedding_cache_key("t", 24_000, 55.0, "preprocessed")
    lowpassed = embedding_cache_key("t", 24_000, 55.0, "preprocessed", lowpass_hz=7000)
    decimated = embedding_cache_key("t", 24_000, 55.0, "preprocessed", extra_tags=(band_match_tag(14_000),))
    both = embedding_cache_key("t", 24_000, 55.0, "preprocessed", lowpass_hz=6000, extra_tags=(band_match_tag(14_000),))
    assert len({base, lowpassed, decimated, both}) == 4
    assert band_match_tag(None) == ""
    assert band_match_tag(14_000) == "_rs14000"


def test_band_match_and_spec_floor_tags_compose_in_a_fixed_order():
    """Tag order must be deterministic, or the same configuration hashes two ways
    depending on which knob was set first."""
    from intrinsic_ai_music_detection.features.cache_keys import band_match_tag

    tags = tuple(t for t in (band_match_tag(14_000), spec_floor_tag(80.0)) if t)
    assert tags == ("_rs14000", "_fl80")
    assert cache_key_string("t", 24_000, 55.0, "preprocessed", extra_tags=tags) == (
        "t_24000_55.0_preprocessed_rs14000_fl80"
    )


# --- arm F5: MusicDET preprocessing tags -----------------------------------
# Rule 2 of the module: existing keys must never move. These assert that the new
# knobs are invisible at their defaults and distinct when set.


def test_musicdet_preproc_tags_are_empty_at_defaults():
    from intrinsic_ai_music_detection.features.cache_keys import musicdet_preproc_tag, window_normalise_tag

    assert musicdet_preproc_tag() == ""
    assert musicdet_preproc_tag(drop_dc=False, rank_gaussianise=False) == ""
    assert window_normalise_tag() == ""
    base = embedding_cache_key("t", 24_000, 55.0, "preprocessed")
    assert (
        embedding_cache_key(
            "t",
            24_000,
            55.0,
            "preprocessed",
            extra_tags=(musicdet_preproc_tag(), window_normalise_tag()),
        )
        == base
    )


def test_each_musicdet_knob_gives_a_distinct_key():
    from intrinsic_ai_music_detection.features.cache_keys import musicdet_preproc_tag, window_normalise_tag

    def key(**kw):
        return embedding_cache_key(
            "t",
            24_000,
            55.0,
            "preprocessed",
            extra_tags=(
                musicdet_preproc_tag(
                    drop_dc=kw.get("drop_dc", False),
                    rank_gaussianise=kw.get("rank_gaussianise", False),
                ),
                window_normalise_tag(kw.get("window_normalise", False)),
            ),
        )

    variants = {
        "base": key(),
        "dc": key(drop_dc=True),
        "rg": key(rank_gaussianise=True),
        "wn": key(window_normalise=True),
        "all": key(drop_dc=True, rank_gaussianise=True, window_normalise=True),
    }
    assert len(set(variants.values())) == len(variants), (
        "two MusicDET preprocessing configurations collide on one cache key — that is "
        "the silent-corruption channel this module exists to close"
    )
