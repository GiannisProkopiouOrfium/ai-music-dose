"""Deconvolution-comb detection: the one artifact with a mechanistic derivation.

Afchar, Meseguer-Brocal, Akesbi & Hennequin, "A Fourier Explanation of AI-music
Artifacts" (arXiv:2506.19108, ISMIR 2025 Best Paper) prove that transposed
convolutions — the upsampling primitive in every neural vocoder and codec
decoder — imprint **periodic spectral peaks**, and that the pattern is a property
of the *architecture* (its stride and kernel sizes), not of the training data or
the learned weights.

Why that matters more here than anywhere else in the bibliography
-----------------------------------------------------------------
Every other signal we have tried is statistical, so it inherits the corpus. FLAC
complexity turned out to be mastering (AUC 0.52 on content-identical pairs).
Our one-class likelihood is dominated by production identity. A two-line energy
ratio scored AUC 1.0000 on SONICS purely because we collected the reals at
48 kHz. A comb whose spacing is fixed by a decoder's stride is none of those
things: it is a claim about the generator's *mechanism*, so it has a reason to
transfer across corpora, which is exactly the property we cannot otherwise buy.

Why the published band does NOT apply to us, and why that is fine
-----------------------------------------------------------------
Their released descriptor takes the peak residual over 3-15 kHz, which is where
the comb is most visible for 44.1 kHz models, and they state SONICS at 16 kHz is
"unsuited". Our audio is worse still: after the bandwidth control that both
corpora need (``--resample-hz 15000``), nothing above 7.5 kHz survives.

But the band is a visibility choice, not the mechanism. A transposed-conv stack
with output rate ``f`` and per-layer strides ``s_1..s_L`` produces peaks spaced
at ``f / s_L``, ``f / (s_L s_{L-1})``, and so on — a *harmonic series whose
fundamental is set by the strides*. For the 16 kHz decoders in FakeMusicCaps
(AudioLDM2, MusicLDM, Mustango all invert mel with HiFi-GAN-family vocoders) the
lower members of that series fall inside 0-8 kHz, i.e. inside the band that
survives our control.

So instead of fixing a band and looking for peaks, this module fixes nothing and
looks for **periodicity** in whatever band survives: it measures the peak
residual over the available spectrum and then asks whether that residual is
itself periodic in frequency. That is band-adaptive by construction.

Log-frequency variant, from a second paper
------------------------------------------
Dugelay et al. (arXiv:2607.27454) make a supervised detector invariant to
frequency scaling by working on a log-frequency axis with cross-correlation.
A pitch shift or a speed change multiplies every frequency by a constant, which
is a *translation* on a log axis — so a comb's autocorrelation-in-log-frequency
is unchanged in shape and merely shifts. ``comb_features(..., log_axis=True)``
takes that invariance; the linear-axis version localises the actual stride.

No training, no labels, no corpus statistics: this is a physical criterion.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Fine enough to resolve a comb: at 24 kHz, 2^14 bins give ~1.5 Hz resolution,
# and the spacings of interest are hundreds of Hz.
DEFAULT_N_FFT = 16_384


def peak_residual(log_spectrum: npt.NDArray[np.float64], smooth_bins: int = 65) -> npt.NDArray[np.float64]:
    """Isolate narrow spectral peaks by subtracting a smooth spectral envelope.

    The published method subtracts a lower convex hull. A wide median filter is
    used here instead because it is O(n log n) rather than O(n * area), is not
    sensitive to a window-size parameter tuned on 44.1 kHz audio, and leaves the
    same quantity: everything that varies faster across frequency than the
    musical envelope does.

    ``smooth_bins`` must be much wider than a comb tooth and much narrower than
    the envelope's own structure.
    """
    from scipy.ndimage import median_filter

    if smooth_bins % 2 == 0:
        smooth_bins += 1
    envelope = median_filter(log_spectrum, size=smooth_bins, mode="nearest")
    return log_spectrum - envelope


def lower_envelope(log_spectrum: npt.NDArray[np.float64], area: int = 10, smooth: int = 3) -> npt.NDArray[np.float64]:
    """Lower envelope of a spectrum: a running minimum, then a light smoothing.

    This is the operator Afchar et al. actually use (arXiv:2506.19108) — their
    ``lower_hull`` slides a window, keeps the local minima and interpolates
    between them — and it is NOT the same as the median filter used by
    ``peak_residual``. A median tracks the middle of the distribution; a lower
    envelope tracks the **noise floor between the teeth**, which is what makes
    the residual "how far above the floor does this peak stand" rather than
    "how far from typical is this bin". For isolating narrow peaks above a
    floor, the lower envelope is the right operator and the median is not.

    ``features/fakeprints.py`` has their exact loop implementation, but its
    ``if abs_idx not in idx`` membership test on a Python list makes it O(n^2) —
    ~56M operations for the ~7,500 bins we get at n_fft=16384, per span. A
    running minimum is O(n) and gives the same envelope.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    if area % 2 == 0:
        area += 1
    floor = minimum_filter1d(log_spectrum, size=area, mode="nearest")
    if smooth > 1:
        floor = uniform_filter1d(floor, size=smooth, mode="nearest")
    return floor


def lower_hull_indices(x: npt.NDArray[np.float64], area: int = 10) -> npt.NDArray[np.int64]:
    """Indices Afchar's ``lower_hull`` keeps, computed in O(n) instead of O(n·area).

    Their loop slides a window of length ``area`` and records the position of
    each window's minimum, de-duplicated and in first-seen order::

        for i in range(len(x)-area+1):
            patch = x[i:i+area]
            abs_idx = np.argmin(patch) + i
            if abs_idx not in idx: idx.append(abs_idx)

    ``np.argmin`` returns the FIRST index of a tied minimum, so the monotonic
    deque below uses a strict ``>`` when popping, which preserves the earlier
    index on ties and reproduces their output exactly. The endpoints are then
    forced in, as they do.

    This is deliberately NOT the same operator as ``lower_envelope``'s running
    minimum: a running minimum sits at the floor everywhere, whereas
    interpolating between hull points rises between the minima. The residual
    above them therefore differs systematically, which is exactly the kind of
    "parallel, weaker reimplementation" that ``tests/test_afchar_parity.py``
    exists to prevent.
    """
    from collections import deque

    n = len(x)
    if n < area:
        return np.array([0, n - 1] if n > 1 else [0], dtype=np.int64)

    idx: list[int] = []
    seen: set[int] = set()
    dq: deque[int] = deque()

    for i in range(n):
        while dq and x[dq[-1]] > x[i]:
            dq.pop()
        dq.append(i)
        if i >= area - 1:
            while dq[0] < i - area + 1:
                dq.popleft()
            j = dq[0]
            if j not in seen:
                seen.add(j)
                idx.append(j)

    if idx[0] != 0:
        idx.insert(0, 0)
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return np.asarray(idx, dtype=np.int64)


def afchar_hull_curve(
    x_freq: npt.NDArray[np.float64],
    log_spectrum: npt.NDArray[np.float64],
    area: int = 10,
    min_db: float = -45.0,
) -> npt.NDArray[np.float64]:
    """The lower-hull curve of ``compute_fakeprints.curve_profile``.

    Hull points from :func:`lower_hull_indices`, interpolated **quadratically**
    across the frequency axis and clipped from below at ``min_db``.
    """
    from scipy import interpolate

    hull_idx = lower_hull_indices(log_spectrum, area=area)
    if len(hull_idx) < 3:
        # interp1d(kind="quadratic") needs three points; below that the hull is
        # degenerate and a straight line is the only defensible curve.
        curve = np.interp(x_freq, x_freq[hull_idx], log_spectrum[hull_idx])
    else:
        curve = interpolate.interp1d(x_freq[hull_idx], log_spectrum[hull_idx], kind="quadratic")(x_freq)
    return np.clip(curve, min_db, None)


def mean_log_spectrum(
    audio: npt.NDArray[np.float32],
    sr: int,
    n_fft: int = DEFAULT_N_FFT,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Time-averaged spectrum **in dB**, as Afchar et al. compute it.

    Returns ``(freqs, mean_db)``.

    The operator ordering matters and we had it backwards. Their code is::

        stft = 10 * np.log10(np.clip(|STFT|**2, 1e-10, 1e6))   # to dB FIRST
        fp   = np.mean(stft, axis=(0, 2))                       # THEN average

    i.e. the mean of the log, which is a **geometric** mean of power. Ours took
    the arithmetic mean of power and logged afterwards. The difference is not
    cosmetic for this feature: an arithmetic mean of power is dominated by the
    loudest frames — the music — while a geometric mean suppresses transients and
    exposes the persistent noise floor between them, which is precisely where a
    stationary decoder comb lives. On a signal that is 99% music and 1% comb,
    that is most of the available SNR.
    """
    import librosa

    a = np.asarray(audio, dtype=np.float32).ravel()
    power = np.abs(librosa.stft(a, n_fft=n_fft)) ** 2
    db = 10.0 * np.log10(np.clip(power, 1e-10, 1e6))
    return librosa.fft_frequencies(sr=sr, n_fft=n_fft), db.mean(axis=1)


def afchar_fakeprint(
    audio: npt.NDArray[np.float32],
    sr: int,
    n_fft: int = DEFAULT_N_FFT,
    f_min: float = 1_000.0,
    f_max: float = 8_000.0,
    area: int = 10,
    min_db: float = -45.0,
    max_db: float = 5.0,
    n_bins: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Their fakeprint end to end. Returns ``(freqs_in_band, residual)``.

    Reproduces ``compute_fakeprints.py``'s chain exactly: dB-domain time average
    → band cut → lower hull (area 10, quadratic interp, clipped at −45 dB) →
    residual clipped to [0, 5] dB → divided by its maximum.

    Band defaults follow the paper's 16 kHz setting, ``[1 kHz, 8 kHz]``. Their
    44.1 kHz setting is ``[5 kHz, 16 kHz]``; pass it explicitly for that regime.

    ``n_bins`` resamples onto a fixed grid and defaults to **None**. The published
    dimension is **4458** (arXiv:2607.25530 sec 4.2), not 445 --- see RETRACTION 8
    in ``scripts/eval_nmf_peak_detector.py``. Their setting transfers to 16 kHz
    essentially intact; 445 is a ~16x decimation nobody proposed, and it flattens
    the one-to-three-bin teeth the detector exists to measure.
    """
    freqs, mean_db = mean_log_spectrum(audio, sr, n_fft=n_fft)
    band = (freqs > f_min) & (freqs < f_max)
    if band.sum() < 64:
        raise ValueError(
            f"empty analysis band [{f_min}, {f_max}] at sr={sr}, n_fft={n_fft}: "
            f"{int(band.sum())} bins survive (need ≥64)"
        )
    f_band = freqs[band]
    c_band = mean_db[band]

    hull = afchar_hull_curve(f_band, c_band, area=area, min_db=min_db)
    residual = np.clip(c_band - hull, 0.0, None)
    residual = np.clip(residual, 0.0, max_db)

    if n_bins and n_bins != len(residual):
        grid = np.linspace(0, len(residual) - 1, n_bins)
        residual = np.interp(grid, np.arange(len(residual)), residual)
        f_band = np.interp(grid, np.arange(len(f_band)), f_band)

    return f_band, residual / (1e-6 + float(np.max(residual)))


def hull_residual(
    log_spectrum: npt.NDArray[np.float64],
    area: int = 10,
    max_db: float = 5.0,
    n_bins: int | None = 445,
) -> npt.NDArray[np.float64]:
    """Afchar's fakeprint: peak height above the lower envelope, normalised.

    Clipping to ``max_db`` and dividing by the maximum is their
    ``max_normalise``; it makes the descriptor invariant to how loud the peaks
    are and sensitive only to their *pattern*, which is the part set by the
    decoder's strides.

    ``n_bins`` resamples onto a fixed grid so the descriptor has a constant
    dimension whatever the sample rate and band. Theirs is **4458**-d from 3-15 kHz
    at 44.1 kHz (2.69 Hz/bin); the 16 kHz audio both benchmarks ship gives a
    different bin count for the same physical band, so the dimension must be
    re-derived per sample rate rather than carried across. The default 445 below
    is OURS and is kept only for continuity with earlier runs.
    """
    residual = np.clip(log_spectrum - lower_envelope(log_spectrum, area=area), 0.0, max_db)
    if n_bins and n_bins != len(residual):
        residual = np.interp(
            np.linspace(0, len(residual) - 1, n_bins),
            np.arange(len(residual)),
            residual,
        )
    peak = float(np.max(residual))
    return residual / (1e-6 + peak)


def comb_strength(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    min_spacing_hz: float = 40.0,
    max_spacing_hz: float = 4_000.0,
) -> tuple[float, float, float]:
    """Score how comb-like a peak residual is.

    Returns ``(strength, spacing_hz, sharpness)``:

    ``strength``
        Height of the largest non-trivial peak of the residual's
        autocorrelation-in-frequency, normalised by the zero lag. A genuine comb
        repeats, so its autocorrelation has a strong peak at the tooth spacing;
        broadband noise does not.
    ``spacing_hz``
        The lag of that peak — an estimate of the decoder's imprint spacing, and
        the interpretable part: it should cluster by generator architecture
        rather than by corpus.
    ``sharpness``
        Peak height relative to the local autocorrelation floor, so a slowly
        varying residual cannot masquerade as a comb.
    """
    x = np.asarray(residual, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 16 or not np.isfinite(x).all() or x.std() < 1e-12:
        return float("nan"), float("nan"), float("nan")

    # Autocorrelation via FFT, keeping non-negative lags.
    padded = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, n=padded)
    acf = np.fft.irfft(spec * np.conj(spec), n=padded)[:n]
    if acf[0] <= 0:
        return float("nan"), float("nan"), float("nan")
    acf = acf / acf[0]

    lo = max(int(round(min_spacing_hz / bin_hz)), 1)
    hi = min(int(round(max_spacing_hz / bin_hz)), n - 1)
    if hi <= lo:
        return float("nan"), float("nan"), float("nan")

    band = acf[lo:hi]
    k = int(np.argmax(band))
    strength = float(band[k])
    spacing = float((lo + k) * bin_hz)
    floor = float(np.median(np.abs(band)))
    sharpness = float(strength / (floor + 1e-12))
    return strength, spacing, sharpness


# Plausible decoder fundamentals f_s / prod(strides), in Hz, for published neural
# audio decoders. Used ONLY by ``comb_prior_strength``, which is an ABLATION and
# never a headline: with harmonics admitted the candidate set is no longer small,
# and a reviewer is entitled to call any such prior cherry-picked.
#
# Frequencies are properties of the GENERATOR's native rate, not of our analysis
# rate: resampling to 16 kHz truncates the band above Nyquist but does not move a
# surviving tooth. So the native rate of each architecture is the one used here.
DECODER_FUNDAMENTALS_HZ: tuple[float, ...] = (
    21.53,  # Stable Audio 44.1 kHz autoencoder, prod(strides) = 2048
    50.00,  # EnCodec 32 kHz (MusicGen), prod(strides) = 640
    75.00,  # EnCodec 24 kHz / SoundStream 24 kHz, prod(strides) = 320
    86.13,  # HiFi-GAN / BigVGAN 22.05 kHz, 256; DAC 44.1 kHz, 512
    93.75,  # 24 kHz mel hop 256 (Vocos-family)
    100.00,  # 16 kHz mel hop 160 (the AudioLDM2 / MusicLDM / Mustango vocoders)
)


#: Lowered lag floor for the R27.62 probe. 20 Hz admits lag 22 = Stable Audio Open's
#: own 21.53 Hz fundamental, which the published 40 Hz floor rejects at every M.
LO20_HZ: float = 20.0

#: Widened null ceiling for the R27.64 probe. ``comb_prior_harmonic_null`` caps the null
#: at ``n_harm * hi_hz``, so 1000 gives 4 kHz at M=4 against the published 600 Hz.
WIDE_HI_HZ: float = 1000.0


def _normalised_acf(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    """Autocorrelation-in-frequency, non-negative lags, normalised by the zero lag.

    Deliberately duplicated from :func:`comb_strength` rather than factored out of
    it. ``comb_strength`` is byte-pinned by ``tests/test_afchar_parity.py`` and is
    the source of every published number in the paper; the cost of a fifteen-line
    duplicate is far below the cost of a refactor that silently moves it.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 16 or not np.isfinite(x).all() or x.std() < 1e-12:
        return None
    padded = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(x, n=padded)
    acf = np.fft.irfft(spec * np.conj(spec), n=padded)[:n]
    if acf[0] <= 0:
        return None
    return acf / acf[0]


def _harmonic_curve(
    acf: npt.NDArray[np.float64],
    lo: int,
    hi: int,
    n_harm: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Mean of ``acf`` over the first ``n_harm`` multiples of each candidate lag.

    ``H(k) = (1/M) * sum_{m=1..M} acf[m*k]`` for ``k`` in ``[lo, hi]``.

    The number of terms is **constant across k** — that is why ``hi`` is capped at
    ``(n-1)//n_harm`` by the caller rather than letting large k fall back on fewer
    harmonics. A variable term count would suppress the noise floor unevenly and
    bias the argmax toward whichever k had the fewest terms, which is precisely the
    kind of readout artefact this statistic exists to remove.
    """
    ks = np.arange(lo, hi + 1, dtype=np.int64)
    acc = np.zeros(len(ks), dtype=np.float64)
    for m in range(1, n_harm + 1):
        acc += acf[ks * m]
    return ks, acc / float(n_harm)


def _nan_harmonic() -> dict[str, float]:
    nan = float("nan")
    return {
        "comb_harm_strength": nan,
        "comb_harm_spacing_hz": nan,
        "comb_harm_sharpness": nan,
        "comb_harm_top2_spacing_hz": nan,
        "comb_harm_top2_strength": nan,
        "comb_harm_top3_spacing_hz": nan,
        "comb_harm_top3_strength": nan,
    }


def comb_harmonic(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    n_harm: int = 4,
    min_spacing_hz: float = 40.0,
    max_fundamental_hz: float = 1_500.0,
    tol: float = 0.05,
) -> dict[str, float]:
    """Read the decoder comb as the HARMONIC SERIES the physics says it is.

    Afchar et al. (arXiv:2506.19108) and Pons et al. (ICASSP 2021) derive the
    upsampling artifact as a *series* of spectral peaks at multiples of
    ``f_s / prod(strides)``. :func:`comb_strength` reads that series with a single
    ``argmax`` over ~4,000 autocorrelation lags. This reads all M of them.

    What this buys, MEASURED on a symmetric fixture
    -----------------------------------------------
    The maximum of a noisy autocorrelation over K lags grows like
    ``sigma * sqrt(2 ln K)``, so a real-music score is an extreme value. Averaging
    M harmonics divides that floor by ``~sqrt(M)`` while leaving a genuine comb's
    amplitude alone — a matched filter for the series. Comb-vs-noise AUC, 300 draws
    per class, comb buried in noise of the same amplitude
    (``tests/test_comb_harmonic.py::TestTheSNRArgument``):

    | tooth height | M=1 (``comb_strength``) | M=2 | M=4 | M=8 |
    |---|---|---|---|---|
    | 0.30 strong | 1.000 | 1.000 | 1.000 | 1.000 |
    | **0.10 weak** | **0.526** | 0.590 | 0.687 | **0.824** |
    | 0.05 absent | 0.488 | 0.472 | 0.468 | 0.511 |

    So the gain is confined to the **weak-comb** regime, which is where the hard
    families sit; strong combs saturate either way, and a comb that is not there
    stays undetected at every M. That last row matters: this does not manufacture
    a detection where there is nothing to detect.

    What this does NOT buy — a hypothesis of ours that the fixture falsified
    -----------------------------------------------------------------------
    We proposed that harmonic summing would resolve the sub-multiple ambiguity —
    that because every measured spacing reads as a harmonic of a smaller
    fundamental (107.42 Hz for Stable Audio Open ~ 5 x 21.5; 200.195 Hz for the
    HiFi-GAN vocoders ~ 2 x 100; 250.0 Hz for MusicGen = 5 x 50), summing
    harmonics would recover the fundamental and stabilise the spacing estimate.
    **It does not, and the reason is arithmetic:** if teeth exist at every multiple
    of 50 with the multiples of 150 tallest, then 150 is *itself* a valid period,
    so ``acf[150]``, ``acf[300]``, ``acf[450]`` are all large and 150 wins the
    harmonic sum too. Measured on 200 noise draws of one comb, concentration within
    +-5% of the modal estimate is **0.055 (argmax) vs 0.060 (harmonic)** in the weak
    regime — no material improvement. Recorded here rather than removed, per the
    honesty discipline; ``comb_harm_spacing_hz`` is therefore NOT a more reliable
    identity than ``comb_spacing_hz`` and must not be sold as one.

    Returns ``comb_harm_{strength,spacing_hz,sharpness}`` plus the second and third
    peaks of the harmonic curve. The first peak is ``comb_harm_spacing_hz`` and is
    not repeated.

    Direction is fixed a priori: **more harmonic comb => more decoder.** Nothing
    here is fitted, so nothing can mis-fit the sign.
    """
    acf = _normalised_acf(residual)
    if acf is None:
        return _nan_harmonic()

    n = len(acf)
    lo = max(int(round(min_spacing_hz / bin_hz)), 1)
    hi = min(int(round(max_fundamental_hz / bin_hz)), (n - 1) // max(n_harm, 1))
    if hi <= lo:
        return _nan_harmonic()

    ks, curve = _harmonic_curve(acf, lo, hi, n_harm)
    floor = float(np.median(np.abs(curve)))

    # Non-maximum suppression at the same +-tol the C8 gate uses, so "a different
    # peak" means the same thing here as it does in the gate.
    out: dict[str, float] = {}
    work = curve.copy()
    for rank in (1, 2, 3):
        if not np.isfinite(work).any() or np.all(work == -np.inf):
            break
        j = int(np.argmax(work))
        k = int(ks[j])
        spacing = float(k * bin_hz)
        strength = float(curve[j])
        if rank == 1:
            out["comb_harm_strength"] = strength
            out["comb_harm_spacing_hz"] = spacing
            out["comb_harm_sharpness"] = float(strength / (floor + 1e-12))
        else:
            out[f"comb_harm_top{rank}_spacing_hz"] = spacing
            out[f"comb_harm_top{rank}_strength"] = strength
        work[np.abs(ks - k) <= tol * k] = -np.inf

    return {**_nan_harmonic(), **out}


def decoy_fundamentals(
    n_sets: int,
    size: int = len(DECODER_FUNDAMENTALS_HZ),
    lo_hz: float = 20.0,
    hi_hz: float = 150.0,
    seed: int = 20260907,
    reject_within: float = 0.05,
) -> list[tuple[float, ...]]:
    """Decoy prior sets of the SAME SIZE as the real one, at wrong frequencies.

    Why this exists — read before quoting any ``prior_strength`` number
    -------------------------------------------------------------------
    ``comb_prior_strength`` restricts the lag search to a handful of candidates,
    and that alone lowers real music's score: the maximum of a noisy
    autocorrelation over K candidates grows like ``sigma * sqrt(2 ln K)``, so going
    from ~4,000 lags to 6 shrinks the baseline by ``sqrt(ln 6 / ln 4000) ~ 0.47``
    **whatever the candidates are**. That is a legitimate, content-free effect.

    It is not the only possible explanation, and the other one is a leak.
    ``DECODER_FUNDAMENTALS_HZ`` was assembled partly by working *backwards* from
    spacings measured on FakeMusicCaps — 100 Hz because the measured comb is at
    200.195 = 2 x 100, 50 Hz because MusicGen's is at 250 = 5 x 50, 21.53 Hz
    because Stable Audio Open's is at 107.42 ~ 5 x 21.5. A prior chosen with the
    answers in view is not a prior.

    These two explanations are separated by a null, not by argument: score the same
    tracks against ``n_sets`` prior sets of the same cardinality drawn from the same
    range but deliberately *not* at the measured fundamentals. If the decoys reach
    the same macro AUC, the gain is candidate-set SIZE and the specific frequencies
    are irrelevant — which is the stronger and cleaner result. If the decoys
    collapse, the gain is the set's CONTENT and the FakeMusicCaps number is
    contaminated by the way the list was built.

    Sets are drawn log-uniformly (spacing priors are naturally multiplicative) with
    a fixed seed, and any draw within ``reject_within`` of a real prior is rejected
    so the decoys are genuinely wrong. A decoy can still land near a true comb by
    chance; that is what makes this a null DISTRIBUTION rather than a single
    contrast, and it is the same construction as ``comb_match_null`` in
    ``scripts/make_paper_figures.py``.
    """
    rng = np.random.default_rng(seed)
    real = np.asarray(DECODER_FUNDAMENTALS_HZ, dtype=float)
    out: list[tuple[float, ...]] = []
    while len(out) < n_sets:
        draw = np.exp(rng.uniform(np.log(lo_hz), np.log(hi_hz), size=size))
        if np.any(np.abs(draw[:, None] - real[None, :]) <= reject_within * real[None, :]):
            continue
        out.append(tuple(round(float(v), 3) for v in np.sort(draw)))
    return out


def comb_prior_strength(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    n_harm: int = 4,
    fundamentals_hz: tuple[float, ...] = DECODER_FUNDAMENTALS_HZ,
) -> dict[str, float]:
    """Harmonic sum restricted to an enumerable set of decoder fundamentals.

    Returns ``{"strength": ..., "hz": ...}`` — the best score and WHICH candidate
    achieved it. The second is diagnostic and load-bearing: if real music and
    generated audio both peak at the smallest candidate, the score is reading
    low-lag autocorrelation (residual smoothness) rather than a decoder comb, and
    that is a nuisance rather than a mechanism.

    **Its number means nothing without :func:`decoy_fundamentals`.** See that
    function for why: the candidate list was built partly from measured
    FakeMusicCaps spacings, so a decoy null is required to say whether the gain is
    the set's size or its content.
    """
    acf = _normalised_acf(residual)
    if acf is None:
        return {"strength": float("nan"), "hz": float("nan")}
    n = len(acf)
    ks = sorted({int(round(f / bin_hz)) for f in fundamentals_hz})
    ks = [k for k in ks if k >= 1 and k * n_harm <= n - 1]
    if not ks:
        return {"strength": float("nan"), "hz": float("nan")}
    best, best_k = -np.inf, ks[0]
    for k in ks:
        v = float(np.mean([acf[m * k] for m in range(1, n_harm + 1)]))
        if v > best:
            best, best_k = v, k
    return {"strength": float(best), "hz": float(best_k * bin_hz)}


def comb_prior_max(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    n_harm: int = 8,
    fundamentals_hz: tuple[float, ...] = DECODER_FUNDAMENTALS_HZ,
    min_spacing_hz: float = 40.0,
    tol_bins: int = 0,
) -> dict[str, float]:
    """Peak autocorrelation over the LAG SET a published decoder could produce.

    Where :func:`comb_prior_strength` averages ``acf`` over the first M multiples
    of each candidate fundamental, this takes the **maximum over the union of
    those multiples**: is the residual periodic at *any* physically realisable
    ``m * f_s / prod(strides)``?

    Why the max is the right operator and the mean is not
    ----------------------------------------------------
    A decoder puts teeth at every multiple of its fundamental, but only some of
    them survive into the analysis band with usable amplitude — which member is
    tallest is a property of the filter, not of the stride. Averaging over M
    therefore dilutes: each harmonic that misses adds a noise term. Measured on
    FakeMusicCaps, the mean form degrades monotonically with M
    (0.9426 at M=2, 0.9257 at M=4, 0.8513 at M=8) for exactly this reason.

    Taking the max instead lets a larger M *help*, because it only ever adds
    candidates. It is also the form that can reach a comb whose visible tooth is a
    high harmonic of a low fundamental: MusicGen's comb sits at 250 Hz, which none
    of the priors' first two multiples reach, yet 250 = 5 x 50 and EnCodec 32 kHz
    has ``prod(strides) = 640``, i.e. a 50 Hz fundamental.

    The candidate set stays small — at most ``6 * n_harm`` lags against ~4,000 —
    so the extreme-value baseline for real music is still suppressed by
    ``sqrt(ln 48 / ln 4000) ~ 0.68`` at M = 8.

    ``min_spacing_hz`` is the same 40 Hz floor :func:`comb_strength` uses, so the
    very short lags where an autocorrelation reflects residual smoothness rather
    than a comb are excluded here too. Without it the smallest prior (21.53 Hz)
    would let this statistic read smoothness, which is a nuisance and not a
    mechanism.

    ``tol_bins`` widens each candidate to ``k +- tol_bins``, and defaults to 0, which
    is the published behaviour bin for bin. It exists because the analysis lattice
    cannot represent most of these frequencies exactly: 100 Hz is 102.4 bins at the
    FakeMusicCaps resolution, so a comb at exactly 100 Hz has its autocorrelation peak
    straddling lags 102 and 103 while this reads only 102. The worst fractional part
    over the candidate set is 0.4, and 93.75 Hz is the only one that lands exactly.
    ``tol_bins = 1`` is therefore the smallest tolerance the rounding forces rather
    than a fitted width -- but it also widens the search, so it must be given to the
    NULL as well as to the prior or the comparison is rigged. See
    :func:`comb_prior_lattice_null`, which takes the same argument.

    **Read :func:`decoy_fundamentals` before quoting any number from this.**
    """
    acf = _normalised_acf(residual)
    if acf is None:
        return {"strength": float("nan"), "hz": float("nan")}
    n = len(acf)
    lo = max(int(round(min_spacing_hz / bin_hz)), 1)
    lags = sorted({int(round(m * f / bin_hz)) for f in fundamentals_hz for m in range(1, n_harm + 1)})
    if tol_bins:
        lags = sorted({k + d for k in lags for d in range(-tol_bins, tol_bins + 1)})
    lags = [k for k in lags if lo <= k <= n - 1]
    if not lags:
        return {"strength": float("nan"), "hz": float("nan")}
    vals = acf[np.asarray(lags, dtype=np.int64)]
    j = int(np.argmax(vals))
    return {"strength": float(vals[j]), "hz": float(lags[j] * bin_hz)}


@lru_cache(maxsize=64)
def decoy_lattice_lags(
    n: int,
    bin_hz: float,
    n_harm: int = 2,
    lo_hz: float = 20.0,
    hi_hz: float = 150.0,
    reject_within: float = 0.05,
    min_spacing_hz: float = 40.0,
) -> tuple[int, ...]:
    """EVERY lag a WRONG but physically plausible fundamental could produce.

    The professor's question, answered in the right space
    -----------------------------------------------------
    The proposal was: search all lags exhaustively, zero out the neighbourhoods of the
    expected decoder fundamentals, and take the max of the rest. Done on the LAG axis
    that fails, and instructively. A decoder emits peaks at *every* multiple of its
    spacing, ``P`` at M = 2 holds only the first two, so everything from 3*Delta up
    survives into the complement carrying the very evidence being measured: the score
    subtracts its own signal from itself. Simulated at the FakeMusicCaps geometry over
    200 seeds, a MusicGen-like comb (Delta = 50 Hz, fifth harmonic tallest) scores AUC
    **0.0000** under that null -- perfectly inverted -- with the complement's maximum
    sitting at 249.0 Hz, which is 5 x 49.8. See ``tests/test_null_alternatives.py``.

    Exclude on the FUNDAMENTAL axis instead and the leak closes exactly. A candidate
    fundamental ``g`` contributes lags ``{m*g : m = 1..M}`` and the prior contributes
    ``{m'*Delta : m' = 1..M}``, so the two collide precisely when
    ``g ~ (m'/m) * Delta`` for some ``m, m' <= M``. Rejecting a RELATIVE neighbourhood
    of every such ratio removes every collision at that depth, with no drift term --
    the tolerance is a fraction of the frequency, so it does not walk away from the
    comb as the harmonic order rises the way a fixed-bin mask does.

    ⚠ CORRECTED 2026-09-17. The first implementation excluded only
    ``{Delta/2, Delta, 2*Delta}``, which is the ``m, m' <= 2`` case. At M = 4 that
    leaves eight ratios unexcluded (1/4, 1/3, 2/3, 3/4, 4/3, 3/2, 3, 4) and at M = 8
    it leaves forty, so the "deterministic" null overlapped the prior's own lag set by
    13 of 18 lags at M = 4 and 33 of 36 at M = 8 on FakeMusicCaps -- WORSE than the
    random decoys it was built to replace. Caught by measuring the overlap rather than
    trusting the argument. Any `_DETNULL` run made before this fix has contaminated
    M = 4 and M = 8 lattice columns; M = 2 was correct.

    Why the overlap matters more than it looks
    ------------------------------------------
    The margin is ``real - max(null)``. If the null contains the prior's own winning
    lag, the margin is <= 0, and is EXACTLY 0 whenever that lag is the null's maximum
    too. Measured on the shipped random decoys, the fraction of the prior's lags that
    the decoy union also contains rises 4/10 -> 10/18 -> 21/36 (FakeMusicCaps, M = 2,
    4, 8), and the number of decoy sets touching the prior rises 4 -> 12 -> 19 of 24.
    That is why ``comb_priormax8_margin`` collapses -- an atom of probability at
    exactly zero, which destroys both the recall and any quantile threshold built on
    it. A null that is disjoint from the prior BY CONSTRUCTION is what lets a deep
    prior work at all, and a deep prior is what MusicGen (its tooth enters P at M = 8)
    and Suno (at M = 4) need.

    Why this has no step parameter
    ------------------------------
    A log grid over ``[lo_hz, hi_hz]`` would need a spacing, and an unregistered knob
    is exactly what disqualified the drift-aware complement. It is also unnecessary:
    the analysis lattice is discrete, so beyond a fine enough spacing extra grid
    points stop producing new lags. This enumerates the SATURATED set directly -- for
    every integer lag in range it asks whether some admissible fundamental reaches it
    -- so the construction is exhaustive by definition and has no spacing at all.

    Deterministic: no seed, no draws. That also dissolves the wrinkle that 8 of the 24
    random decoy sets contain a lag within one bin of a real candidate, because
    ``decoy_fundamentals`` rejects draws near a real FUNDAMENTAL but cannot stop their
    HARMONICS colliding.

    Sized to match, not to flatter. On FakeMusicCaps this yields lag counts within a
    few percent of the 24 random sets' 118 (and 94 on SONICS), so the extreme-value
    penalty ``sigma*sqrt(2 ln K)`` is matched and the two nulls are comparable.

    A decoy lag can still coincide with a real tooth the prior itself cannot reach --
    MusicGen's 250 Hz comb is 2 x 125, and 125 Hz is admissible here. That inflates
    the null and shrinks the margin, which is the conservative direction, and the 24
    random sets have the same property.
    """
    lo = max(int(round(min_spacing_hz / bin_hz)), 1)
    # Every ratio m'/m with m, m' <= n_harm: the complete collision set at this depth.
    ratios = {mp / m for m in range(1, n_harm + 1) for mp in range(1, n_harm + 1)}
    banned = sorted(
        ((1.0 - reject_within) * t, (1.0 + reject_within) * t)
        for d in DECODER_FUNDAMENTALS_HZ
        for t in (r * d for r in ratios)
    )

    def _reachable(g_lo: float, g_hi: float) -> bool:
        """Is ANY fundamental in ``[g_lo, g_hi]`` admissible?

        Rounding to the lag grid maps an INTERVAL of fundamentals onto each lag, not a
        point, so testing only the interval's centre wrongly rejects a lag whose
        centre happens to fall inside a banned neighbourhood while part of the
        interval lies outside it. Sweep the banned intervals in order and see whether
        any gap survives.
        """
        g_lo, g_hi = max(g_lo, lo_hz), min(g_hi, hi_hz)
        if g_lo > g_hi:
            return False
        cursor = g_lo
        for b_lo, b_hi in banned:
            if b_hi <= cursor:
                continue
            if b_lo > cursor:
                return True  # a gap before this ban starts
            cursor = max(cursor, b_hi)
            if cursor > g_hi:
                return False
        return cursor <= g_hi

    keep: set[int] = set()
    for m in range(1, n_harm + 1):
        k_lo = max(lo, int(np.floor(m * lo_hz / bin_hz)))
        k_hi = min(n - 1, int(np.ceil(m * hi_hz / bin_hz)))
        for k in range(k_lo, k_hi + 1):
            # The fundamentals that round to lag k at harmonic m.
            if _reachable((k - 0.5) * bin_hz / m, (k + 0.5) * bin_hz / m):
                keep.add(k)
    return tuple(sorted(keep))


def comb_prior_lattice_null(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    n_harm: int = 2,
    fundamentals_hz: tuple[float, ...] = DECODER_FUNDAMENTALS_HZ,
    min_spacing_hz: float = 40.0,
    tol_bins: int = 0,
) -> dict[str, float]:
    """The real prior against the exhaustive wrong-fundamental null of the same shape.

    Returns the null's maximum and an exact empirical p-value: the fraction of
    admissible wrong lags whose autocorrelation reaches the plausible prior's best.
    That p-value is the deterministic analogue of "0 of 24 decoy sets reach it", and
    it is stronger, because the null is exhaustive rather than sampled -- there is no
    seed to have been lucky with.

    ``pval`` is deliberately a fraction rather than a count: the number of admissible
    lags depends on the corpus's bin width, so a count is not comparable across the
    two benchmarks and a fraction is.
    """
    nan = {"strength": float("nan"), "pval": float("nan"), "n_lags": float("nan")}
    acf = _normalised_acf(residual)
    if acf is None:
        return nan
    n = len(acf)
    real = comb_prior_max(
        residual, bin_hz=bin_hz, n_harm=n_harm, fundamentals_hz=fundamentals_hz, min_spacing_hz=min_spacing_hz
    )["strength"]
    lags = np.asarray(
        decoy_lattice_lags(n, bin_hz, n_harm=n_harm, min_spacing_hz=min_spacing_hz), dtype=np.int64
    )
    if tol_bins:
        lags = np.unique(np.clip(np.concatenate([lags + d for d in range(-tol_bins, tol_bins + 1)]), 0, n - 1))
    if not len(lags) or not np.isfinite(real):
        return nan
    vals = acf[lags]
    return {
        "strength": float(np.max(vals)),
        "pval": float(np.mean(vals >= real)),
        "n_lags": float(len(lags)),
    }


def comb_prior_harmonic_null(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    n_harm: int = 4,
    fundamentals_hz: tuple[float, ...] = DECODER_FUNDAMENTALS_HZ,
    min_spacing_hz: float = 40.0,
    hi_hz: float = 150.0,
    tol_bins: int = 1,
) -> dict[str, float]:
    """The professor's proposal, in the form the measurements actually support.

    *Zero out the neighbourhoods of the expected decoder fundamentals and take the max
    of the rest* -- applied to the whole HARMONIC SERIES of each expected fundamental,
    and confined to the lag range a plausible fundamental can reach.

    Why the two earlier nulls each get half of it right
    ----------------------------------------------------
    :func:`decoy_lattice_lags` excludes candidate FUNDAMENTALS near a rational multiple
    ``(m'/m)*Delta`` with ``m, m' <= M``. That makes the null disjoint from the PRIOR's
    lag set -- which removed the zero atom entirely, 19.5% -> 0.0% -- but it does not
    protect the comb's own HIGHER harmonics. At M = 4 the ratio set stops at 4, so a
    candidate near ``(5/4)*Delta`` is admissible and its 4th multiple lands on ``5*Delta``,
    which is signal. Measured on FakeMusicCaps: **11 true-comb harmonics sit inside the
    M=2 lattice null and 3 inside the M=4 one**, and one of them is Stable Audio Open's
    own visible tooth at ``21.53 x 5 = 107.65 Hz``. The detector subtracts part of its
    own evidence -- the same failure as the lag-axis complement, one level out. It shows
    up as MusicGen reading 0.6466 at M=2 (its visible tooth is the 5th harmonic) against
    0.8668 at M=4, where the prior protects more of the series.

    The ratio exclusion is also far too broad in the other direction: rejecting +-5% of
    eleven ratios leaves only **74** admissible lags at M=4 on FakeMusicCaps and **52**
    on SONICS, against the drawn decoys' 275 and 220. A null a quarter the size cancels
    a quarter as much of the residual smoothness that inverts Udio, which is the whole
    of the measured SONICS deficit (pooled AUC 0.7646 against the incumbent's 0.7901,
    DeLong p = 1.2e-59).

    What this does instead
    ----------------------
    Take **every** lag in ``[min_spacing_hz, n_harm * hi_hz]`` and remove only what must
    be removed: ``+-tol_bins`` around ``k * Delta`` for every real fundamental and every
    integer ``k``, not just ``k <= M``. That is the exact and minimal exclusion -- it is
    precisely the set of lags where a real decoder could have put energy.

    The result has both properties for the first time: **441 lags at M=4 on
    FakeMusicCaps and 252 on SONICS** (CORRECTED from 253) (3-6x the ratio lattice, comparable to the decoys)
    with **zero true-comb harmonics inside**, and it is still fully deterministic -- no
    seed, no draws, no grid spacing.

    ``tol_bins = 1`` is forced by rounding rather than fitted: the analysis lattice
    cannot place ``k*Delta`` on an exact integer bin, so one bin either side is the
    smallest exclusion that covers where the peak actually lands. The residual
    prior-frequency error is negligible over this range -- Suno's 49.99 Hz against a
    50.00 Hz prior drifts 0.08 bins by its 8th harmonic.
    """
    nan = {"strength": float("nan"), "pval": float("nan"), "n_lags": float("nan")}
    acf = _normalised_acf(residual)
    if acf is None:
        return nan
    n = len(acf)
    lo = max(int(round(min_spacing_hz / bin_hz)), 1)
    hi = min(n - 1, int(round(n_harm * hi_hz / bin_hz)))
    if hi <= lo:
        return nan
    real = comb_prior_max(
        residual, bin_hz=bin_hz, n_harm=n_harm, fundamentals_hz=fundamentals_hz, min_spacing_hz=min_spacing_hz
    )["strength"]
    if not np.isfinite(real):
        return nan

    keep = np.ones(hi - lo + 1, dtype=bool)
    for f in fundamentals_hz:
        k = 1
        while True:
            lag = int(round(k * f / bin_hz))
            if lag - tol_bins > hi:
                break
            a, b = max(lag - tol_bins, lo), min(lag + tol_bins, hi)
            if b >= a:
                keep[a - lo : b - lo + 1] = False
            k += 1
    lags = np.flatnonzero(keep) + lo
    if not len(lags):
        return nan
    vals = acf[lags]
    return {"strength": float(np.max(vals)), "pval": float(np.mean(vals >= real)), "n_lags": float(len(lags))}


def comb_prior_union_null(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    n_harm: int = 2,
    n_decoy_sets: int = 24,
    fundamentals_hz: tuple[float, ...] = DECODER_FUNDAMENTALS_HZ,
    min_spacing_hz: float = 40.0,
    exclude_bins: int = 1,
) -> dict[str, float]:
    """The null both the decoy and the lattice results point at: their union, minus the prior.

    Why neither of the two alone is right
    --------------------------------------
    Measured on the corpora, the two hypothesis-varying nulls fail in opposite ways.

    * The **24 random decoy sets** have good coverage -- 257 lags at M = 4 on
      FakeMusicCaps -- but they are *contaminated*: 10 of the prior's 18 lags are
      inside the decoy union at that depth, and 12 of the 24 sets touch the prior.
      Because ``margin = real - max(null)``, a shared lag forces the margin to <= 0
      and to exactly 0 when it is the null's maximum, so 26% of FakeMusicCaps tracks
      score exactly zero. That atom caps recall at every quantile and makes any
      threshold degenerate.
    * The **deterministic lattice** is disjoint from the prior by construction, so its
      zero atom is exactly 0.0000 on both corpora -- but it is *small*: 52 lags at
      M = 4 on SONICS against the decoys' 209. A smaller null cancels less of the
      track's own residual smoothness, which is precisely the nuisance that inverts
      Udio, and the measured Udio margin is correspondingly worse (0.530 / 0.527
      against the decoys' 0.597 / 0.584).

    Coverage and disjointness are independent properties, and the union has both:
    every lag either null proposes, minus anything within ``exclude_bins`` of a lag
    the prior itself searches. 242 lags at M = 4 on FakeMusicCaps and 179 on SONICS,
    with zero overlap.

    The exclusion is what makes this legitimate rather than merely bigger. Removing
    the prior's own lags from a null is not a handicap on the null -- those lags were
    never admissible members of it, because a null is supposed to answer "what could a
    WRONG hypothesis achieve on this residual" and the prior's lags are the right
    hypothesis. The random construction included them only because
    ``decoy_fundamentals`` rejects draws near a real FUNDAMENTAL and cannot stop their
    HARMONICS colliding.

    Still seeded, unlike :func:`comb_prior_lattice_null`, because it contains the drawn
    decoys -- so it does not inherit the lattice's determinism. It is reported beside
    the lattice, not instead of it.
    """
    nan = {"strength": float("nan"), "pval": float("nan"), "n_lags": float("nan")}
    acf = _normalised_acf(residual)
    if acf is None:
        return nan
    n = len(acf)
    lo = max(int(round(min_spacing_hz / bin_hz)), 1)
    real = comb_prior_max(
        residual, bin_hz=bin_hz, n_harm=n_harm, fundamentals_hz=fundamentals_hz, min_spacing_hz=min_spacing_hz
    )["strength"]
    if not np.isfinite(real):
        return nan

    prior = {int(round(m * f / bin_hz)) for f in fundamentals_hz for m in range(1, n_harm + 1)}
    candidates = set(decoy_lattice_lags(n, bin_hz, n_harm=n_harm, min_spacing_hz=min_spacing_hz))
    for decoy in decoy_fundamentals(n_decoy_sets):
        candidates |= {int(round(m * f / bin_hz)) for f in decoy for m in range(1, n_harm + 1)}
    lags = np.asarray(
        sorted(
            k
            for k in candidates
            if lo <= k <= n - 1 and all(abs(k - p) > exclude_bins for p in prior)
        ),
        dtype=np.int64,
    )
    if not len(lags):
        return nan
    vals = acf[lags]
    return {"strength": float(np.max(vals)), "pval": float(np.mean(vals >= real)), "n_lags": float(len(lags))}


def comb_acf_floor(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    min_spacing_hz: float = 40.0,
    max_spacing_hz: float = 4_000.0,
) -> float:
    """``median(|acf|)`` over the band ``comb_strength`` searches: the ANALYTIC null.

    ``comb_harmonic`` already divides by this quantity computed on the HARMONIC curve,
    whose noise is about ``sqrt(M)`` below the plain autocorrelation's. Subtracting
    that floor from a plain-autocorrelation peak therefore mixes two scales. This
    emits the matching one, so a prior-restricted analytic margin can be formed
    without the mismatch -- see ``scripts/derive_analytic_null.py``, which documents
    the approximate cross-form it replaces.

    The band is ``comb_strength``'s own, so the floor is the bulk of exactly the
    lag range the peak was found in.
    """
    acf = _normalised_acf(residual)
    if acf is None:
        return float("nan")
    n = len(acf)
    lo = max(int(round(min_spacing_hz / bin_hz)), 1)
    hi = min(int(round(max_spacing_hz / bin_hz)), n - 1)
    if hi <= lo:
        return float("nan")
    return float(np.median(np.abs(acf[lo:hi])))


def comb_matched_filter(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    f_start: float,
    fundamentals_hz: tuple[float, ...] = DECODER_FUNDAMENTALS_HZ,
    n_offsets: int = 1,
    min_teeth: int = 4,
) -> dict[str, float]:
    """Sum the residual AT the tooth positions, instead of autocorrelating it.

    The constraint an autocorrelation throws away
    ---------------------------------------------
    ``comb_strength`` and every harmonic variant above are built on the
    autocorrelation of the residual, which is **offset-blind**: it measures the
    spacing between teeth but not *where* those teeth sit on the frequency axis.
    The physics gives us that too. A transposed-convolution stack imprints copies
    of the spectrum at multiples of ``f_s / prod(strides)`` — teeth at
    ``m * Delta`` for integer ``m``, i.e. **aligned to DC**, not at an arbitrary
    offset. Discarding the alignment discards a real constraint.

    This scores the residual as a matched filter over that comb: for candidate
    fundamental ``Delta`` and offset ``phi``, average the residual at every
    ``m * Delta + phi`` inside the band and express it as a z-score against the
    residual's own mean and spread,

        z(Delta, phi) = (mean_teeth - mean_residual) / (sd_residual / sqrt(K)),

    with ``K`` the number of teeth. The ``sqrt(K)`` makes candidates with
    different tooth counts comparable, and the whole quantity is self-normalising,
    so it needs no corpus statistics.

    Why it should beat the autocorrelation readout, argued before measuring
    ----------------------------------------------------------------------
    The matched filter is **linear** in the residual where an autocorrelation is
    quadratic, so its noise floor falls as ``1/sqrt(K)`` with K the tooth count —
    at ``Delta = 50`` Hz over a 7 kHz band that is 140 teeth, against the handful
    of harmonics ``comb_harmonic`` can average. And requiring DC alignment removes
    the offsets a musical harmonic series would need, which an ACF cannot exclude.

    ``n_offsets`` > 1 scans ``phi`` across ``[0, Delta)`` and takes the best. That
    is the CONTROL, not the detector: it is an upper bound that discards the
    alignment constraint, so if the DC-aligned score does not beat it in AUC then
    alignment carries no information and this whole argument is void. Scanning
    also re-introduces the multiple-comparison cost the restricted candidate set
    exists to avoid.

    ``f_start`` is the absolute frequency of ``residual[0]`` — required, because
    alignment is meaningless without it. Passing the band's lower edge is what
    makes ``m * Delta`` an absolute frequency rather than an offset into an array.

    **Read :func:`decoy_fundamentals` before quoting any number from this.**
    """
    r = np.asarray(residual, dtype=np.float64)
    n = len(r)
    if n < 16 or not np.isfinite(r).all():
        return {"z": float("nan"), "hz": float("nan"), "offset_hz": float("nan"), "n_teeth": float("nan")}
    mu, sd = float(r.mean()), float(r.std())
    if sd < 1e-12:
        return {"z": float("nan"), "hz": float("nan"), "offset_hz": float("nan"), "n_teeth": float("nan")}
    f_end = f_start + (n - 1) * bin_hz

    best_z, best_hz, best_off, best_k = -np.inf, float("nan"), float("nan"), 0
    for f0 in fundamentals_hz:
        if f0 <= 0:
            continue
        offsets = [0.0] if n_offsets <= 1 else list(np.linspace(0.0, f0, n_offsets, endpoint=False))
        for phi in offsets:
            m_lo = int(np.ceil((f_start - phi) / f0))
            m_hi = int(np.floor((f_end - phi) / f0))
            if m_hi - m_lo + 1 < min_teeth:
                continue
            freqs = np.arange(m_lo, m_hi + 1) * f0 + phi
            idx = np.rint((freqs - f_start) / bin_hz).astype(np.int64)
            idx = idx[(idx >= 0) & (idx < n)]
            if len(idx) < min_teeth:
                continue
            z = (float(r[idx].mean()) - mu) / (sd / np.sqrt(len(idx)) + 1e-12)
            if z > best_z:
                best_z, best_hz, best_off, best_k = z, float(f0), float(phi), len(idx)

    if not np.isfinite(best_z):
        return {"z": float("nan"), "hz": float("nan"), "offset_hz": float("nan"), "n_teeth": float("nan")}
    return {"z": float(best_z), "hz": best_hz, "offset_hz": best_off, "n_teeth": float(best_k)}


def comb_matched_filter_refined(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    f_start: float,
    delta_init_hz: float,
    refine_frac: float = 0.02,
    n_refine: int = 161,
    min_teeth: int = 4,
) -> dict[str, float]:
    """Matched filter at a spacing REFINED around a data-derived estimate.

    Why a fixed prior list cannot drive a matched filter
    ----------------------------------------------------
    :func:`comb_matched_filter` assumes teeth at exact multiples of a candidate
    fundamental, and that assumption is brittle in a way the autocorrelation is
    not: an error of ``d`` in the spacing displaces the K-th tooth by ``K*d``, so
    the filter walks off the comb. Measured on a fixture at the FakeMusicCaps
    geometry, weak-tooth regime:

    | comb | prior used | drift by 7 kHz | matched-filter AUC |
    |---|---|---|---|
    | MusicGen 250.0 = 5 x 50.0 | 50.0 exact | 0 bins | **0.760** |
    | chirp 399.902 ~ 8 x 49.99 | 50.0 | ~1.7 bins | 0.586 |
    | HiFi-GAN 200.195 vs 2 x 100.0 | 100.0 | ~7 bins | 0.488 |
    | Stable Audio 107.42 vs 5 x 21.53 | 21.53 | ~15 bins | 0.326 |

    Where the prior is exact the gain over ``comb_strength`` (0.513) is large;
    where it is off by 0.1% the filter fails. A fixed list cannot supply the
    needed precision, because ``f_s / prod(strides)`` depends on the generator's
    native sample rate, which is not observable from the audio.

    The resolution: **estimate the spacing from the data, then filter at it.** An
    autocorrelation is drift-tolerant — it needs one lag right, not K — so it
    gives a good ``delta_init_hz``; this refines locally and takes the linear
    SNR gain. Coarse-to-fine, with the coarse stage doing what it is good at.

    The refinement grid spans ``+-refine_frac`` around the estimate in
    ``n_refine`` steps. It is a real multiple-comparison cost and the decoy null
    prices it: the decoys get the same grid and the same refinement.
    """
    if not np.isfinite(delta_init_hz) or delta_init_hz <= 0:
        return {"z": float("nan"), "hz": float("nan"), "n_teeth": float("nan")}
    grid = delta_init_hz * np.linspace(1.0 - refine_frac, 1.0 + refine_frac, n_refine)
    best = comb_matched_filter(
        residual,
        bin_hz=bin_hz,
        f_start=f_start,
        fundamentals_hz=tuple(float(g) for g in grid if g > 0),
        n_offsets=1,
        min_teeth=min_teeth,
    )
    return {"z": best["z"], "hz": best["hz"], "n_teeth": best["n_teeth"]}


def comb_surrogate_null(
    residual: npt.NDArray[np.float64],
    bin_hz: float,
    n_surrogates: int = 20,
    n_blocks: int = 16,
    seed: int = 0,
    min_spacing_hz: float = 40.0,
    max_spacing_hz: float = 4_000.0,
) -> float:
    """``comb_strength`` calibrated against the track's OWN null.

    The confound this exists to remove
    ----------------------------------
    ``comb_strength`` is compared across tracks, but its null depends on the
    track's own residual — how smooth it is, how heavy its peak-height
    distribution is. That is not a nuisance we can wave away: on SONICS the Udio
    families' residual is *flatter* than real music's in every measure
    (``comb_residual_std`` 0.208 vs 0.273), and their raw ``comb_strength`` (0.055)
    sits **below** real music's (0.072). Some of that gap is a difference in null,
    not a difference in comb.

    The surrogate
    -------------
    Permute the residual **within contiguous frequency blocks**. That destroys any
    periodic alignment while preserving, block by block, the marginal distribution
    of peak heights and the coarse shape of the residual across frequency. The
    score is ``(observed - mean_surrogate) / sd_surrogate``: how many nulls out the
    track's own periodicity is, given its own amplitude statistics.

    Per-track, no corpus, no labels, nothing fitted. The seed is fixed so the value
    is a deterministic function of the residual.
    """
    kw = {"min_spacing_hz": min_spacing_hz, "max_spacing_hz": max_spacing_hz}
    observed, _, _ = comb_strength(residual, bin_hz=bin_hz, **kw)
    if not np.isfinite(observed):
        return float("nan")

    x = np.asarray(residual, dtype=np.float64)
    if len(x) < 16 * max(n_blocks, 1):
        return float("nan")
    blocks = np.array_split(np.arange(len(x)), n_blocks)
    rng = np.random.default_rng(seed)

    draws: list[float] = []
    for _ in range(n_surrogates):
        y = x.copy()
        for b in blocks:
            y[b] = x[rng.permutation(b)]
        s, _, _ = comb_strength(y, bin_hz=bin_hz, **kw)
        if np.isfinite(s):
            draws.append(s)

    if len(draws) < 2:
        return float("nan")
    return float((observed - np.mean(draws)) / (np.std(draws) + 1e-12))


def _residual_from(
    log_spec: npt.NDArray[np.float64],
    f_band: npt.NDArray[np.float64],
    residual_op: str,
    hull_area: int,
    hull_clip_db: float,
    smooth_bins: int,
) -> npt.NDArray[np.float64]:
    """Peak residual under one operator choice.

    ``hull_clip_db`` — MEASURED, and it is the knob that matters
    -----------------------------------------------------------
    Afchar's ``max_normalise`` clips the residual at 5 dB and divides by its
    maximum. That is deliberate and correct **for their readout**: a logistic
    regression over the whole 445-d profile only needs the *pattern* of which
    bins peak, so making the descriptor invariant to peak loudness helps it.

    It is the wrong thing for **our** readout. ``comb_strength`` is an
    autocorrelation peak, which needs the residual's amplitude structure. Once
    teeth exceed 5 dB they all saturate to 1.0, the residual becomes a near-binary
    mask, and noise bins that also exceed 5 dB inflate the autocorrelation floor.
    On synthetic spectra with 3-bin teeth on a noisy floor, separation
    (comb minus clean ``comb_strength``) is:

    | tooth height | hull + clip 5 dB + max-norm | hull, no clip | median-65 |
    |---|---|---|---|
    | 4 dB | 0.0245 | 0.0453 | 0.0594 |
    | 8 dB | 0.0483 | 0.2255 | 0.2471 |
    | 20 dB | 0.0493 | 0.6229 | 0.6639 |
    | 40 dB | **0.0493 — saturated** | 0.6229 | 0.6639 |

    So: use ``hull_clip_db=5`` for the PROFILE paths (NMF, logistic regression,
    the combprint front end), where it is their published operator; use
    ``hull_clip_db=0`` for the SCALAR autocorrelation path. The operator and the
    readout are coupled, and porting one without the other loses signal.

    Note also that on this fixture the median is *not* worse than the unclipped
    hull for the autocorrelation readout — a lower envelope leaves a noise
    pedestal that a median does not. Which wins on real audio is an empirical
    question; ``eval_comb_detector.py --operator-grid`` answers it on one pass of
    identical audio rather than by argument.
    """
    if residual_op == "hull":
        residual = np.clip(log_spec - afchar_hull_curve(f_band, log_spec, area=hull_area), 0.0, None)
        if hull_clip_db and hull_clip_db > 0:
            residual = np.clip(residual, 0.0, hull_clip_db)
            residual = residual / (1e-6 + float(np.max(residual)))
        return residual
    return peak_residual(log_spec, smooth_bins=smooth_bins)


def comb_stationarity(
    profiles: npt.NDArray[np.float64],
    bin_hz: float,
    min_spacing_hz: float = 40.0,
    max_spacing_hz: float = 4_000.0,
) -> dict[str, float]:
    """Separate a DECODER comb from a MUSICAL harmonic comb, by stationarity.

    The confound this exists to remove
    ----------------------------------
    A sustained musical note puts a comb of harmonics at its f0 spacing into the
    spectrum, and ``comb_strength`` cannot tell that apart from a decoder comb on
    a time-averaged spectrum alone. This is not hypothetical for us: our measured
    "real music" spacing clusters sit at **401.9 Hz** (FakeMusicCaps) and
    **295.2 Hz** (SONICS), squarely inside the range of musical f0. No control in
    the protocol addresses it.

    The physical discriminator
    --------------------------
    A decoder comb's spacing is fixed by the architecture's strides, so it is
    **the same in every span of the track**. A harmonic comb moves with the
    melody. So given per-span residual profiles ``[T, F]``:

    * ``comb_stat_strength`` — autocorrelation peak of the **mean** profile,
      divided by the mean of the per-span autocorrelation peaks. A stationary
      comb survives averaging across spans and this ratio approaches 1; a comb
      whose spacing wanders averages away and the ratio falls.
    * ``comb_spacing_dispersion`` — standard deviation of the per-span spacing
      estimate, in Hz. Near zero for a decoder, large for a melody.

    ``profiles`` must be ``[T, F]`` with ``T >= 2``; with one span there is no
    stationarity to measure and the features are NaN rather than a fabricated 1.0.
    """
    nan = float("nan")
    p = np.asarray(profiles, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2:
        return {
            "comb_stat_strength": nan,
            "comb_spacing_dispersion": nan,
            "comb_mean_profile_strength": nan,
        }

    mean_strength, _, _ = comb_strength(
        p.mean(axis=0), bin_hz=bin_hz, min_spacing_hz=min_spacing_hz, max_spacing_hz=max_spacing_hz
    )

    per_span_strength: list[float] = []
    per_span_spacing: list[float] = []
    for row in p:
        s, lag, _ = comb_strength(row, bin_hz=bin_hz, min_spacing_hz=min_spacing_hz, max_spacing_hz=max_spacing_hz)
        if np.isfinite(s) and np.isfinite(lag):
            per_span_strength.append(s)
            per_span_spacing.append(lag)

    if not per_span_strength or not np.isfinite(mean_strength):
        return {
            "comb_stat_strength": nan,
            "comb_spacing_dispersion": nan,
            "comb_mean_profile_strength": float(mean_strength),
        }

    denom = float(np.mean(per_span_strength))
    return {
        # Ratio in [0, ~1]: how much of a typical span's comb survives averaging
        # across spans. Stationary => survives.
        "comb_stat_strength": float(mean_strength / (denom + 1e-12)),
        "comb_spacing_dispersion": float(np.std(per_span_spacing)),
        "comb_mean_profile_strength": float(mean_strength),
    }


def comb_harmonic_stationarity(
    profiles: npt.NDArray[np.float64],
    bin_hz: float,
    n_harm: int = 4,
    min_spacing_hz: float = 40.0,
    max_fundamental_hz: float = 1_500.0,
) -> dict[str, float]:
    """:func:`comb_stationarity`, but on the harmonic-sum readout.

    Same physical question — is the comb at the *same* fundamental in every span,
    as a decoder's must be, or does it move with the melody? — asked of an
    estimator that resolves the fundamental instead of betting on one member of
    the series. ``comb_harm_spacing_dispersion`` is the one to watch: a decoder
    pins it near zero, and it is the within-track analogue of the cross-track
    consensus in ``scripts/eval_consensus_spacing.py``.

    Reported separately from :func:`comb_stationarity` rather than replacing it,
    because ``comb_stat_strength`` carries a published SONICS number (0.6864) and
    must not move.
    """
    nan = float("nan")
    empty = {
        "comb_harm_stat_strength": nan,
        "comb_harm_spacing_dispersion": nan,
        "comb_harm_mean_profile_strength": nan,
    }
    p = np.asarray(profiles, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2:
        return empty

    kw = {"n_harm": n_harm, "min_spacing_hz": min_spacing_hz, "max_fundamental_hz": max_fundamental_hz}
    mean_h = comb_harmonic(p.mean(axis=0), bin_hz=bin_hz, **kw)
    mean_strength = mean_h["comb_harm_strength"]

    per_span_strength: list[float] = []
    per_span_spacing: list[float] = []
    for row in p:
        h = comb_harmonic(row, bin_hz=bin_hz, **kw)
        if np.isfinite(h["comb_harm_strength"]) and np.isfinite(h["comb_harm_spacing_hz"]):
            per_span_strength.append(h["comb_harm_strength"])
            per_span_spacing.append(h["comb_harm_spacing_hz"])

    if not per_span_strength or not np.isfinite(mean_strength):
        return {**empty, "comb_harm_mean_profile_strength": float(mean_strength)}

    return {
        "comb_harm_stat_strength": float(mean_strength / (float(np.mean(per_span_strength)) + 1e-12)),
        "comb_harm_spacing_dispersion": float(np.std(per_span_spacing)),
        "comb_harm_mean_profile_strength": float(mean_strength),
    }


CALIBRATED_NULL_KEY: dict[str, str] = {
    "hmargin": "hmax_strength",
    "latmargin": "latmax_strength",
    "unionmargin": "unionmax_strength",
    # --- the 2x2 floor x ceiling probe (ledger R27.62, R27.64). Both factors are free
    # in the same extraction, so they are emitted together rather than in two passes.
    # `lo20`  : min_spacing_hz 40 -> 20, which admits Stable Audio's own 21.53 Hz
    #           fundamental (lag 22), currently rejected at EVERY M.
    # `wide`  : the null's ceiling M*150 Hz -> M*1000 Hz.
    # The floor is given to the PRIOR and the NULL together; giving it to one only
    # would rig the comparison, which is the mistake `comb_prior_max`'s tol_bins
    # docstring already warns about.
    "lo20margin": "lo20_hmax_strength",
    "widemargin": "wide_hmax_strength",
    "lo20widemargin": "lo20wide_hmax_strength",
}

#: Which prior column each calibrated score subtracts its null from. Only the
#: ``lo20`` arms move the prior; ``wide`` changes the null's ceiling alone.
CALIBRATED_REAL_KEY: dict[str, str] = {
    "hmargin": "strength",
    "latmargin": "strength",
    "unionmargin": "strength",
    "lo20margin": "lo20_strength",
    "widemargin": "strength",
    "lo20widemargin": "lo20_strength",
}


def comb_calibrated_score(feats: dict[str, float], name: str) -> float:
    """One calibrated score from a :func:`comb_features` dict, by column name.

    Why this exists in the library rather than in a script
    -------------------------------------------------------
    The margin is ``real - max(null)``, and until now it was only ever formed at
    CSV level by ``scripts/derive_analytic_null.py``. That left the two deployment
    scripts -- ``run_robustness_battery.py`` and ``measure_efficiency.py`` -- able to
    score only ``comb_strength`` and ``comb_stat_strength``, so **neither could measure
    the detector the paper proposes.** The paper's robustness and cost figures are
    consequently the raw comb arm's, not the proposal's (ledger R27.44).

    Defining the arithmetic once, here, means the per-track CSV path and the per-buffer
    deployment path cannot drift apart; ``tests/test_calibrated_score.py`` asserts they
    agree bin for bin.

    ``comb_priormax{M}_hmargin`` needs **no decoys at all** -- the harmonic-protected
    null is deterministic -- so it can be timed with ``null_priors=0``, which is both
    faster and the configuration a deployer would actually run.
    """
    if name in feats:
        value = feats[name]
        if value is None or not np.isfinite(value):
            raise ValueError(f"{name} is not finite")
        return float(value)

    m = re.fullmatch(
        r"comb_priormax(\d+)_(hmargin|latmargin|unionmargin|lo20margin|widemargin|lo20widemargin)", name
    )
    if m:
        order, kind = m.group(1), m.group(2)
        real = feats.get(f"comb_priormax{order}_{CALIBRATED_REAL_KEY[kind]}")
        null = feats.get(f"comb_priormax{order}_{CALIBRATED_NULL_KEY[kind]}")
        if real is None or null is None:
            raise KeyError(
                f"{name} needs comb_priormax{order}_{CALIBRATED_REAL_KEY[kind]} and "
                f"comb_priormax{order}_{CALIBRATED_NULL_KEY[kind]}; pass n_harm_grid=({order},)"
            )
        if not (np.isfinite(real) and np.isfinite(null)):
            raise ValueError(f"{name} inputs are not finite")
        return float(real) - float(null)

    m = re.fullmatch(r"comb_priormax(\d+)_margin", name)
    if m:
        order = m.group(1)
        real = feats.get(f"comb_priormax{order}_strength")
        decoys = [
            v
            for k, v in feats.items()
            if re.fullmatch(rf"comb_priormax{order}_decoy\d+_strength", k) and v is not None and np.isfinite(v)
        ]
        if real is None or not decoys:
            raise KeyError(f"{name} needs comb_priormax{order}_strength and its decoys; pass null_priors>0")
        return float(real) - float(max(decoys))

    raise KeyError(f"unknown calibrated score {name!r}")


def comb_features(
    audio: npt.NDArray[np.float32],
    sr: int,
    n_fft: int = DEFAULT_N_FFT,
    f_min: float = 1_000.0,
    f_max: float | None = None,
    smooth_bins: int = 65,
    log_axis: bool = False,
    residual_op: str = "hull",
    average: str = "db",
    hull_area: int = 10,
    hull_clip_db: float = 5.0,
    span_duration: float = 4.0,
    n_harm_grid: tuple[int, ...] = (2, 4, 8),
    surrogates: int = 20,
    min_spans: int = 8,
    null_priors: int = 0,
) -> dict[str, float]:
    """Deconvolution-comb descriptors for one audio buffer.

    ``f_max`` defaults to just under Nyquist, so the descriptor automatically
    uses whatever band the bandwidth control left — no band constant to become
    wrong when the control changes.

    ``log_axis`` resamples the residual onto a log-frequency grid first, making
    the result invariant to frequency scaling (pitch shift, speed change), per
    Dugelay et al. arXiv:2607.27454. Report both: the linear version identifies
    the stride, the log version survives the manipulation battery.

    Operator selection — read this before changing a default
    --------------------------------------------------------
    ``residual_op`` and ``average`` select between **Afchar et al.'s operator**
    (``"hull"`` / ``"db"``, the defaults) and the earlier local approximation
    (``"median"`` / ``"power"``). The earlier one is kept only so the corrected
    and uncorrected numbers can be produced in a single pass for the CORRECTED
    table; it is not an alternative worth tuning. See ``mean_log_spectrum`` and
    ``lower_hull_indices`` for why each differs.

    ``f_min`` defaults to 1 kHz, their 16 kHz setting. The previous 500 Hz floor
    admitted the band where musical energy dominates and no decoder comb lives.

    The harmonic channels
    ---------------------
    ``n_harm_grid``, ``surrogates`` and ``min_spans`` drive the additions described
    in :func:`comb_harmonic`, :func:`comb_surrogate_null` and
    :func:`_span_overlap_factor`. They are strictly ADDITIVE: every pre-existing
    key keeps the value it had before they were introduced, which is what lets the
    new columns be compared like-for-like against the published ``0.9177`` and
    ``0.6864`` from a single pass over the same audio. Set ``surrogates=0`` to skip
    the per-track null, which is the only one of the three with a real cost.
    """
    import librosa

    if residual_op not in ("hull", "median"):
        raise ValueError(f"residual_op must be 'hull' or 'median', got {residual_op!r}")
    if average not in ("db", "power"):
        raise ValueError(f"average must be 'db' or 'power', got {average!r}")

    a = np.asarray(audio, dtype=np.float32).ravel()
    if a.size < n_fft:
        return {}

    # Time-average the spectrum: the comb is stationary across the file (it is a
    # property of the decoder), so averaging suppresses everything that is not,
    # which is the whole musical signal. The DOMAIN of that average matters —
    # see mean_log_spectrum.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    if average == "db":
        _, mean_db = mean_log_spectrum(a, sr, n_fft=n_fft)
    else:
        spec = np.abs(librosa.stft(a, n_fft=n_fft)) ** 2
        mean_db = 10.0 * np.log10(np.maximum(spec.mean(axis=1), 1e-20))

    hi = f_max if f_max is not None else 0.98 * (sr / 2.0)
    band = (freqs >= f_min) & (freqs <= hi)
    if band.sum() < 64:
        return {}

    log_spec = mean_db[band]
    f_band = freqs[band]
    bin_hz = float(f_band[1] - f_band[0])

    residual = _residual_from(log_spec, f_band, residual_op, hull_area, hull_clip_db, smooth_bins)

    if log_axis:
        # Uniform grid in log f, so a multiplicative frequency change becomes a
        # pure translation and the autocorrelation is unchanged.
        lo = max(f_band[0], 1.0)
        grid = np.geomspace(lo, f_band[-1], num=len(f_band))
        residual = np.interp(grid, f_band, residual)
        # "Spacing" is now a ratio in log units, not Hz; the strength and
        # sharpness remain comparable, the spacing does not.
        bin_hz = float(np.log(grid[1] / grid[0]))

    strength, spacing, sharpness = comb_strength(
        residual,
        bin_hz=bin_hz,
        min_spacing_hz=40.0 if not log_axis else 4 * bin_hz,
        max_spacing_hz=4_000.0 if not log_axis else 200 * bin_hz,
    )

    prefix = "comb_log_" if log_axis else "comb_"
    out = {
        f"{prefix}strength": strength,
        f"{prefix}spacing_hz": spacing,
        f"{prefix}sharpness": sharpness,
        f"{prefix}residual_std": float(np.std(residual)),
        f"{prefix}residual_kurtosis": float(
            np.mean((residual - residual.mean()) ** 4) / (np.var(residual) ** 2 + 1e-12)
        ),
    }

    # The harmonic readout and the per-track surrogate null are LINEAR-AXIS only.
    # Harmonics are additive in frequency; on a log axis they are not evenly
    # spaced, so a harmonic sum there would be summing the wrong lags.
    #
    # M is emitted as a GRID rather than a single choice, for the same reason the
    # operator grid exists: the fixture says the gain is monotone in M
    # (0.526 -> 0.590 -> 0.687 -> 0.824) but a fixture is not a corpus, and the
    # useful M is capped by the band (hi = (n-1)//M). Emitting every M on identical
    # audio settles it in one pass. DISCIPLINE: M is selected once on
    # FakeMusicCaps and then FIXED for SONICS, as sigma and F already are.
    if not log_axis:
        decoys = decoy_fundamentals(null_priors) if null_priors else []
        for m in n_harm_grid:
            for k, v in comb_harmonic(residual, bin_hz=bin_hz, n_harm=m).items():
                out[k.replace("comb_harm_", f"comb_harm{m}_", 1)] = v
            real_prior = comb_prior_strength(residual, bin_hz=bin_hz, n_harm=m)
            out[f"comb_harm{m}_prior_strength"] = real_prior["strength"]
            # Which candidate won. If real and generated both peak at the smallest,
            # the score is reading residual smoothness, not a decoder comb.
            out[f"comb_harm{m}_prior_hz"] = real_prior["hz"]
            # The max-over-the-lag-set form, which is the better-motivated one and
            # the only one that can reach a comb sitting on a high harmonic.
            pmax = comb_prior_max(residual, bin_hz=bin_hz, n_harm=m)
            out[f"comb_priormax{m}_strength"] = pmax["strength"]
            out[f"comb_priormax{m}_hz"] = pmax["hz"]
            # The DETERMINISTIC null: every lag a wrong-but-plausible fundamental
            # could produce, enumerated exhaustively. No seed, no draws, and it still
            # varies the HYPOTHESIS, so unlike a complement null it can still answer
            # the provenance question the decoys exist for.
            lat = comb_prior_lattice_null(residual, bin_hz=bin_hz, n_harm=m)
            out[f"comb_priormax{m}_latmax_strength"] = lat["strength"]
            out[f"comb_priormax{m}_latpval"] = lat["pval"]
            # Coverage AND disjointness: every lag either null proposes, minus the
            # prior's own. See comb_prior_union_null for why neither alone suffices.
            uni = comb_prior_union_null(residual, bin_hz=bin_hz, n_harm=m, n_decoy_sets=null_priors or 24)
            out[f"comb_priormax{m}_unionmax_strength"] = uni["strength"]
            out[f"comb_priormax{m}_unionpval"] = uni["pval"]
            # The harmonic-protected null: the only one that is BOTH disjoint from the
            # comb's whole series AND comparable in size to the decoys. Deterministic.
            hn = comb_prior_harmonic_null(residual, bin_hz=bin_hz, n_harm=m)
            out[f"comb_priormax{m}_hmax_strength"] = hn["strength"]
            out[f"comb_priormax{m}_hpval"] = hn["pval"]
            out[f"comb_priormax{m}_hnlags"] = hn["n_lags"]
            # --- the 2x2 probe: floor {40, 20} Hz x null ceiling {M*150, M*1000} Hz.
            # R27.62: the 40 Hz floor's stated rationale ("short lags read smoothness")
            # is measurable on the HULL residual and absent on the MEDIAN residual we
            # use, and the floor rejects Stable Audio's 21.53 Hz fundamental at every M.
            # R27.64: widening the ceiling is free and, on the fixture, very slightly
            # better -- but it does not remove M, which still defines the PRIOR.
            # Four extra columns, one extra pass of the same autocorrelation.
            out[f"comb_priormax{m}_lo20_strength"] = comb_prior_max(
                residual, bin_hz=bin_hz, n_harm=m, min_spacing_hz=LO20_HZ
            )["strength"]
            for tag, kw in (
                ("lo20", dict(min_spacing_hz=LO20_HZ)),
                ("wide", dict(hi_hz=WIDE_HI_HZ)),
                ("lo20wide", dict(min_spacing_hz=LO20_HZ, hi_hz=WIDE_HI_HZ)),
            ):
                probe = comb_prior_harmonic_null(residual, bin_hz=bin_hz, n_harm=m, **kw)
                out[f"comb_priormax{m}_{tag}_hmax_strength"] = probe["strength"]
                out[f"comb_priormax{m}_{tag}_hnlags"] = probe["n_lags"]
            # The lattice-tolerance arm, prior and null both widened by one bin.
            out[f"comb_priormax{m}_pm1_strength"] = comb_prior_max(
                residual, bin_hz=bin_hz, n_harm=m, tol_bins=1
            )["strength"]
            lat1 = comb_prior_lattice_null(residual, bin_hz=bin_hz, n_harm=m, tol_bins=1)
            out[f"comb_priormax{m}_pm1_latmax_strength"] = lat1["strength"]
            out[f"comb_priormax{m}_pm1_latpval"] = lat1["pval"]
            # The null: same cardinality, wrong frequencies. See decoy_fundamentals.
            # Both prior forms get it, or the contrast would only cover one of them.
            for j, decoy in enumerate(decoys):
                out[f"comb_harm{m}_decoy{j:02d}_strength"] = comb_prior_strength(
                    residual, bin_hz=bin_hz, n_harm=m, fundamentals_hz=decoy
                )["strength"]
                out[f"comb_priormax{m}_decoy{j:02d}_strength"] = comb_prior_max(
                    residual, bin_hz=bin_hz, n_harm=m, fundamentals_hz=decoy
                )["strength"]
        # The DC-aligned matched filter, and its offset-scanning control. These do
        # not depend on M: they read the tooth POSITIONS, not the lag spectrum.
        # The analytic per-track null on the PLAIN autocorrelation, so a
        # prior-restricted margin can be formed without mixing it with the harmonic
        # curve's floor (whose noise is ~sqrt(M) smaller). M-independent.
        out["comb_acf_floor"] = comb_acf_floor(residual, bin_hz=bin_hz)
        mf = comb_matched_filter(residual, bin_hz=bin_hz, f_start=float(f_band[0]))
        out["comb_mf_z"] = mf["z"]
        out["comb_mf_hz"] = mf["hz"]
        out["comb_mf_n_teeth"] = mf["n_teeth"]
        ctrl = comb_matched_filter(residual, bin_hz=bin_hz, f_start=float(f_band[0]), n_offsets=8)
        out["comb_mfoff_z"] = ctrl["z"]
        out["comb_mfoff_offset_hz"] = ctrl["offset_hz"]
        for j, decoy in enumerate(decoys):
            out[f"comb_mf_decoy{j:02d}_z"] = comb_matched_filter(
                residual, bin_hz=bin_hz, f_start=float(f_band[0]), fundamentals_hz=decoy
            )["z"]
        if surrogates and surrogates > 0:
            out["comb_surrogate_z"] = comb_surrogate_null(residual, bin_hz=bin_hz, n_surrogates=surrogates)

    # Stationarity (arm F4) — computed on the LINEAR axis only, because the
    # question is whether the spacing in Hz is fixed by the decoder's strides.
    if not log_axis and span_duration and span_duration > 0:
        # One pass at the denser hop serves both statistics: `spans[::k]` is
        # bin-for-bin the non-overlapping set, so the published columns below are
        # computed on exactly the audio they were always computed on.
        k = _span_overlap_factor(len(a), int(span_duration * sr), min_spans=min_spans)
        spans = _span_residuals(
            a,
            sr,
            n_fft=n_fft,
            f_min=f_min,
            f_max=hi,
            span_duration=span_duration,
            residual_op=residual_op,
            average=average,
            hull_area=hull_area,
            hull_clip_db=hull_clip_db,
            smooth_bins=smooth_bins,
            overlap_factor=k,
        )
        legacy = spans[::k]
        if len(legacy) >= 2:
            out.update(comb_stationarity(np.stack(legacy, axis=0), bin_hz=bin_hz))
        if len(spans) >= 2:
            stack = np.stack(spans, axis=0)
            for m in n_harm_grid:
                for key, val in comb_harmonic_stationarity(stack, bin_hz=bin_hz, n_harm=m).items():
                    out[key.replace("comb_harm_", f"comb_harm{m}_", 1)] = val
        # Deliberately NOT prefixed "comb_": eval_comb_detector.py selects feature
        # columns by that substring, and the span count is a diagnostic, not a
        # candidate detector. Scoring track length would be gate C3's confound.
        out["n_spans_used"] = float(len(spans))
    return out


def _span_overlap_factor(n_samples: int, span_samples: int, min_spans: int, max_factor: int = 8) -> int:
    """Smallest integer ``k`` (hop = span/k) that yields at least ``min_spans`` spans.

    A MEASUREMENT ARTEFACT, not a preference. ``comb_stat_strength`` is the best
    chirp channel on SONICS (0.972 / 0.927 / 0.979, above ``comb_strength``) and
    the *worst* relative showing on FakeMusicCaps — because FMC tracks are capped
    at 9 s, so at a 4 s non-overlapping span they yield **two** spans against
    SONICS's thirty. A stationarity ratio over two spans is noise. Overlapping the
    spans recovers the statistic on short audio at no cost on long audio.

    ``k`` is restricted to **exact divisors** of ``span_samples`` so that
    ``spans[::k]`` reproduces the non-overlapping set bin-for-bin. That is what
    keeps ``comb_stat_strength`` and ``comb_spacing_dispersion`` — one of which
    carries a published SONICS number — identical to their pre-existing values
    while the new harmonic channels get the denser set. On SONICS ``k`` is 1 and
    nothing changes at all.

    The divisor restriction is not fussiness. With ``span_samples = 64000`` and an
    unrestricted ``k = 7``, ``64000 // 7 * 7 = 63994``: the subsampled spans would
    start four samples off, the "unchanged" published column would move by a
    small unexplained amount, and nothing would report an error. That is the
    silent-corruption family this repository has been bitten by twice
    (ledger L1, L2), so it is a divisor or it is ``k = 1``.
    """
    if span_samples <= 0 or n_samples < span_samples:
        return 1
    for k in range(1, max_factor + 1):
        if span_samples % k:
            continue
        if (n_samples - span_samples) // (span_samples // k) + 1 >= min_spans:
            return k
    return 1


def _span_residuals(
    audio: npt.NDArray[np.float32],
    sr: int,
    n_fft: int,
    f_min: float,
    f_max: float,
    span_duration: float,
    residual_op: str,
    average: str,
    hull_area: int,
    hull_clip_db: float,
    smooth_bins: int,
    overlap_factor: int = 1,
) -> list[npt.NDArray[np.float64]]:
    """Peak residuals computed independently per span.

    ``overlap_factor`` k sets the hop to ``span/k``; k = 1 is the original
    non-overlapping geometry and is the default, so every existing caller and
    every published column is unaffected. Because k divides the span exactly,
    ``result[::k]`` is bin-for-bin the k = 1 result.

    A span shorter than ``n_fft`` cannot produce a spectrum at all, so spans are
    silently skipped rather than zero-padded — padding would inject a synthetic
    stationary component and bias exactly the statistic this feeds.
    """
    import librosa

    span_samples = int(span_duration * sr)
    if span_samples < n_fft:
        return []

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= f_min) & (freqs <= f_max)
    if band.sum() < 64:
        return []
    f_band = freqs[band]

    hop = max(span_samples // max(overlap_factor, 1), 1)
    out: list[npt.NDArray[np.float64]] = []
    for start in range(0, len(audio) - span_samples + 1, hop):
        chunk = audio[start : start + span_samples]
        if average == "db":
            _, mean_db = mean_log_spectrum(chunk, sr, n_fft=n_fft)
        else:
            spec = np.abs(librosa.stft(chunk, n_fft=n_fft)) ** 2
            mean_db = 10.0 * np.log10(np.maximum(spec.mean(axis=1), 1e-20))
        out.append(_residual_from(mean_db[band], f_band, residual_op, hull_area, hull_clip_db, smooth_bins))
    return out
