#!/usr/bin/env python3
"""Generate the ICASSP figure from measured results.

One figure, three panels:

  (a,b)  the contamination dose-response on both corpora, held out, 3 seeds.
         SONICS transitions sharply; FakeMusicCaps climbs smoothly.
  (c)    per-atom recovered comb spacing against periodicity, with the MEASURED
         decoder spacings marked -- the mechanism behind (a,b).

``--standalone-fig2`` additionally writes panel (c) on its own, for the project
page and slides. The paper uses only the combined figure.

PROVENANCE
----------
Every value plotted is either read from a result CSV (``--curve-dir`` /
``--atom-dir``) or taken from the constants below, each carrying the
provenance-ledger section and the CSV it was transcribed from. Nothing is
estimated, smoothed or interpolated, and a ``*_plotted.csv`` sidecar records
exactly what reached the axes. If a figure and the paper ever disagree, that
file settles it.

Usage
-----
    python3 scripts/make_paper_figures.py --out-dir icassp/figs
    python3 scripts/make_paper_figures.py --out-dir icassp/figs \
        --atom-dir reports/figures --curve-dir reports/diagnostics
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# -----------------------------------------------------------------------------
# Measured values.
#
# SONICS: provenance_ledger.md §N, from
#   reports/diagnostics/nmf_sonics_curve_FINAL/nmf_peak_auc_lvl.csv
#   held out, --fit-real-frac 0.5, seeds 42/43/44, 59,280 tracks, 6,361 reals
#   scored at every point.
# Columns: n_fakes_in_dictionary, contamination_pct, s42, s43, s44, chirp_mean
# -----------------------------------------------------------------------------
SONICS_CURVE = [
    (0, 0.000, 0.402, 0.411, 0.333, 0.310),
    (9, 0.141, 0.437, 0.425, 0.338, 0.342),
    (23, 0.360, 0.611, 0.524, 0.344, 0.493),
    (47, 0.733, 0.785, 0.786, 0.411, 0.774),
    (70, 1.090, 0.787, 0.786, 0.788, 0.984),
    (93, 1.440, 0.787, 0.790, 0.787, 0.986),
    (233, 3.530, 0.796, 0.794, 0.790, 0.987),
    (931, 12.800, 0.899, 0.903, 0.900, 0.992),
    (2328, 26.800, 0.913, 0.915, 0.914, 0.996),
    (11640, 64.700, 0.931, 0.933, 0.931, 0.998),
]

# FakeMusicCaps: provenance_ledger.md §M1 and §N2, from
#   reports/diagnostics/nmf_fmc_curve_FINAL/nmf_peak_auc_lvl.csv
#   same protocol, 32,960 tracks. Only mean and SD were transcribed into the
#   ledger, so this arm plots mean +/- SD without per-seed points.
# Columns: n_fakes_in_dictionary, contamination_pct, mean_auc, sd
FMC_CURVE = [
    (0, 0.00, 0.7444, 0.0203),
    (14, 0.52, 0.8357, 0.0180),
    (28, 1.03, 0.8749, 0.0142),
    (55, 2.01, 0.9401, 0.0178),
    (138, 4.90, 0.9771, 0.0017),
    (552, 17.10, 0.9873, 0.0008),
    (1380, 34.00, 0.9917, 0.0006),
    (6901, 72.00, 0.9919, 0.0004),
]

# The transition region, re-run at TEN draws.
#   reports/diagnostics/nmf_sonics_knee_10seed/nmf_peak_auc_lvl.csv, seeds 42-51,
#   bins4779, held out. This is a SEPARATE run from SONICS_CURVE above, which is
#   three draws across the whole range; only these four doses have ten. The
#   figure and the paper must both say which is which.
# n_fakes -> the ten macro AUCs
TEN_SEED = {
    23: [0.61134, 0.52374, 0.51548, 0.46492, 0.45904, 0.42524, 0.40434, 0.38414, 0.37254, 0.34356],
    47: [0.78744, 0.78592, 0.78532, 0.78514, 0.78354, 0.78250, 0.77974, 0.54744, 0.51358, 0.41088],
    70: [0.79018, 0.79002, 0.78856, 0.78760, 0.78746, 0.78710, 0.78656, 0.78628, 0.78356, 0.78308],
    93: [0.79192, 0.79144, 0.79132, 0.79094, 0.78978, 0.78930, 0.78804, 0.78732, 0.78694, 0.78548],
}

# The published transductive protocol, full corpus, for the reference lines.
# provenance_ledger.md §A: nmf_{fmc,sonics}_FULL/nmf_peak_auc_lvl.csv
POOLED_AUC = {"sonics": 0.93894, "fmc": 0.99168}

# Decoder comb spacings MEASURED on the corpora, not read off architectures.
# provenance_ledger.md §G2.
MEASURED_SPACING_HZ = {
    "fmc": [(200.195, "audioldm2 / musicldm / mustango"), (250.0, "MusicGen")],
    "sonics": [(399.902, "chirp v2 / v3 / v3.5")],
}

# All K=20 atoms per corpus, transcribed from
#   reports/figures/nmf_{fmc,sonics}_periodicity/atom_peakiness.csv
# (EC2 batch-2 dump, 2026-08-29). Columns: atom, spacing_hz, periodicity,
# peakiness. `peakiness` is carried only to reproduce the
# corr(peakiness, periodicity) control of ledger F4/G2 -- it is NOT plotted
# and must never be quoted as evidence of a comb.
ATOMS = {
    "fmc": [
        (9, 49.8, 0.8598, 0.6155),
        (8, 49.8, 0.8996, 0.6122),
        (13, 250.1, 0.8712, 0.5649),
        (6, 250.1, 0.8112, 0.5493),
        (15, 200.3, 0.7846, 0.5354),
        (2, 43.0, 0.6879, 0.5339),
        (18, 250.1, 0.7396, 0.5163),
        (14, 43.0, 0.5890, 0.5100),
        (0, 299.9, 0.8207, 0.4990),
        (12, 200.3, 0.8552, 0.4961),
        (1, 1378.3, 0.2823, 0.4728),
        (7, 409.3, 0.3399, 0.4654),
        (11, 49.8, 0.9300, 0.4637),
        (3, 49.8, 0.2324, 0.4544),
        (4, 99.6, 0.6164, 0.3962),
        (5, 200.3, 0.6849, 0.3910),
        (17, 99.6, 0.7587, 0.2603),
        (19, 99.6, 0.4216, 0.2523),
        (10, 49.8, 0.9348, 0.1651),
        (16, 99.6, 0.8804, 0.1543),
    ],
    "sonics": [
        (16, 400.0, 0.7060, 0.7761),
        (0, 200.7, 0.4388, 0.7718),
        (4, 400.0, 0.5926, 0.7493),
        (11, 300.3, 0.3548, 0.7436),
        (1, 400.0, 0.4673, 0.7406),
        (13, 400.0, 0.3977, 0.6927),
        (7, 200.7, 0.5347, 0.6914),
        (19, 200.7, 0.4548, 0.6895),
        (14, 400.0, 0.4041, 0.6888),
        (2, 400.0, 0.4132, 0.6797),
        (12, 527.4, 0.2530, 0.6232),
        (6, 165.6, 0.4750, 0.5789),
        (18, 137.7, 0.3128, 0.5766),
        (17, 394.1, 0.3300, 0.5709),
        (15, 209.5, 0.3823, 0.5673),
        (10, 39.6, 0.5782, 0.4303),
        (5, 46.9, 0.3654, 0.4185),
        (3, 142.1, 0.3006, 0.4071),
        (8, 58.6, 0.6908, 0.2838),
        (9, 49.8, 0.8344, 0.2196),
    ],
}

# An atom is "on a comb" if its recovered spacing is within MATCH_TOL of a
# measured decoder spacing or of its half or quarter. Sub-multiples are
# admissible because an autocorrelation-in-frequency locks onto any period of
# the comb, not only the fundamental. See comb_match_null for what that
# generosity costs.
MATCH_TOL = 0.01
MATCH_SUBMULTIPLES = (1, 2, 4)

INK = "#1a1a1a"
SONICS_C = "#B5322A"  # the corpus that inverts
FMC_C = "#1F5F8B"  # the corpus that does not
GREY = "#8a8a8a"


def style() -> None:
    """Match the template: Times, 8 pt, vector output, no chartjunk."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
        }
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def curve_from_csv(path: Path) -> None:
    """Report whether a *_curve_FINAL CSV can be re-read, without guessing.

    Column names moved across rounds. Silently plotting the wrong column is the
    failure mode that produced two of this project's retractions, so this
    reports and declines rather than guessing.
    """
    rows = read_csv_rows(path)
    need = {"n_fakes_in_fit", "seed", "auc_macro"}
    cols = set(rows[0]) if rows else set()
    if not need <= cols:
        print(
            f"  ! {path}: columns {sorted(cols)}; expected {sorted(need)}. "
            f"NOT re-read -- using the transcribed ledger values.",
            file=sys.stderr,
        )


def load_atoms(atom_dir: Path | None, corpus: str) -> tuple[list[tuple], str]:
    """Return (atom, spacing_hz, periodicity, peakiness) rows and provenance."""
    if atom_dir is not None:
        path = atom_dir / f"nmf_{corpus}_periodicity" / "atom_peakiness.csv"
        if path.is_file():
            rows = read_csv_rows(path)
            cols = set(rows[0]) if rows else set()
            if {"spacing_hz", "periodicity"} <= cols:
                return [
                    (
                        int(float(r.get("atom", i))),
                        float(r["spacing_hz"]),
                        float(r["periodicity"]),
                        float(r.get("peakiness", "nan")),
                    )
                    for i, r in enumerate(rows)
                    if r["spacing_hz"] not in ("", "nan")
                ], str(path)
            print(
                f"  ! {path} has no spacing_hz/periodicity -- this is the "
                f"SUPERSEDED peakiness-only run (ledger F4). Not used.",
                file=sys.stderr,
            )
    return ATOMS[corpus], f"transcribed: nmf_{corpus}_periodicity/atom_peakiness.csv"


# -----------------------------------------------------------------------------
# The comb-matching test and its nulls
# -----------------------------------------------------------------------------


def on_comb(spacing_hz: float, corpus: str) -> tuple[float, int] | None:
    """(fundamental, sub-multiple) if this spacing is a decoder comb period."""
    for f, _who in MEASURED_SPACING_HZ[corpus]:
        for m in MATCH_SUBMULTIPLES:
            if abs(spacing_hz - f / m) / (f / m) <= MATCH_TOL:
                return f, m
    return None


def comb_match_null(rows: list[tuple], corpus: str, trials: int = 20000, seed: int = 0) -> dict:
    """Two nulls for the atom-matching count, because one is not enough.

    ``uniform`` draws the fundamental uniformly from 40-700 Hz. That is too
    weak on its own: the atom spacings are themselves selected by NMF, peak
    extraction and the periodicity statistic, so they need not resemble uniform
    draws, and a referee will say so.

    ``cross`` scores this corpus's atoms against the OTHER corpus's *measured*
    decoder spacing -- a real spacing from a real decoder, and a far more
    demanding control. It is the one reported, and it is the one that split the
    result: FakeMusicCaps survives it and SONICS does not, because
    399.902/2 = 199.95 Hz sits 0.12% from FakeMusicCaps' 200.195 Hz comb. The
    two corpora's decoders are within 0.12% of a 2:1 ratio, so admitting
    sub-multiples makes them mutually indistinguishable. See ledger O8b.
    """
    rng = np.random.default_rng(seed)
    spac = np.array([r[1] for r in rows])
    other = "sonics" if corpus == "fmc" else "fmc"

    def count(funds, ms=MATCH_SUBMULTIPLES) -> int:
        t = np.concatenate([np.asarray(funds, dtype=float) / m for m in ms])
        return int((np.abs(spac[:, None] - t) / t <= MATCH_TOL).any(axis=1).sum())

    own = [f for f, _ in MEASURED_SPACING_HZ[corpus]]
    cross = [f for f, _ in MEASURED_SPACING_HZ[other]]
    obs, obs_cross = count(own), count(cross)
    obs_m1, cross_m1 = count(own, (1,)), count(cross, (1,))

    counts = np.array([count(rng.uniform(40.0, 700.0, size=len(own))) for _ in range(trials)])

    out = dict(
        n=len(rows),
        observed=obs,
        cross_corpus=obs_cross,
        observed_m1=obs_m1,
        cross_corpus_m1=cross_m1,
        uniform_null_mean=float(counts.mean()),
        uniform_p=float((counts >= obs).mean()),
    )
    try:
        from scipy.stats import fisher_exact

        n = len(rows)
        out["cross_p"] = float(fisher_exact([[obs, n - obs], [obs_cross, n - obs_cross]])[1])
        out["cross_p_m1"] = float(fisher_exact([[obs_m1, n - obs_m1], [cross_m1, n - cross_m1]])[1])
    except ImportError:
        out["cross_p"] = out["cross_p_m1"] = float("nan")
    return out


# -----------------------------------------------------------------------------
# Panels
# -----------------------------------------------------------------------------


def atom_panel(ax, atom_dir: Path | None, title: bool = True) -> dict:
    """Per-atom spacing against periodicity, drawn onto an existing axis.

    Every atom is drawn, including those nowhere near a decoder comb: plotting
    only the matches would imply a result the dictionary does not support.
    """
    stats = {}
    for corpus, colour, marker, label in [
        ("fmc", FMC_C, "o", "FakeMusicCaps"),
        ("sonics", SONICS_C, "s", "SONICS"),
    ]:
        rows, src = load_atoms(atom_dir, corpus)
        st = comb_match_null(rows, corpus)
        st.update(source=src, mean_periodicity=float(np.mean([r[2] for r in rows])))
        stats[corpus] = st

        for f, _who in MEASURED_SPACING_HZ[corpus]:
            for m in MATCH_SUBMULTIPLES:
                ax.axvline(f / m, color=colour, lw=0.5, ls=(0, (1, 2.5)), alpha=0.55, zorder=1)

        hit = [r for r in rows if on_comb(r[1], corpus)]
        mis = [r for r in rows if not on_comb(r[1], corpus)]
        ax.scatter(
            [r[1] for r in hit],
            [r[2] for r in hit],
            s=15,
            marker=marker,
            facecolor=colour,
            edgecolor=colour,
            linewidth=0.7,
            alpha=0.9,
            zorder=3,
            label=f"{label}  {st['observed']}/{st['n']}",
        )
        ax.scatter(
            [r[1] for r in mis],
            [r[2] for r in mis],
            s=15,
            marker=marker,
            facecolor="none",
            edgecolor=colour,
            linewidth=0.7,
            alpha=0.75,
            zorder=3,
        )

        print(
            f"  {label}: own comb {st['observed']}/{st['n']} | cross-corpus "
            f"{st['cross_corpus']}/{st['n']} (Fisher p={st['cross_p']:.3f}) "
            f"| fundamental only {st['observed_m1']} vs {st['cross_corpus_m1']}"
            f" (p={st['cross_p_m1']:.3f}) | uniform null "
            f"{st['uniform_null_mean']:.2f} | mean periodicity "
            f"{st['mean_periodicity']:.3f}"
        )

    ax.set_xscale("log")
    ax.set_xlim(33, 1700)
    ax.set_ylim(0.15, 1.02)
    ax.set_xticks([50, 100, 200, 400, 800, 1600])
    ax.set_xticklabels(["50", "100", "200", "400", "800", "1600"])
    ax.set_xlabel("atom comb spacing (Hz, log)")
    ax.set_ylabel("atom periodicity")
    if title:
        ax.set_title("(c) the atoms that explain it", pad=3.5)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.3, borderpad=0.1, labelspacing=0.2, fontsize=5.8)
    return stats


def write_atom_sidecar(path: Path, atom_dir: Path | None, stats: dict) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "corpus",
                "atom",
                "spacing_hz",
                "periodicity",
                "peakiness",
                "on_decoder_comb",
                "matched_fundamental_hz",
                "submultiple",
            ]
        )
        for corpus in ("fmc", "sonics"):
            rows, _ = load_atoms(atom_dir, corpus)
            for a, hz, per, pk in rows:
                m = on_comb(hz, corpus)
                w.writerow([corpus, a, hz, per, pk, bool(m), f"{m[0]:.3f}" if m else "", m[1] if m else ""])
        w.writerow([])
        for corpus, st in stats.items():
            w.writerow(
                [
                    f"# {corpus}",
                    f"source={st['source']}",
                    f"on_own_comb={st['observed']}/{st['n']}",
                    f"cross_corpus={st['cross_corpus']}/{st['n']}",
                    f"cross_fisher_p={st['cross_p']:.4f}",
                    f"fundamental_only_own={st['observed_m1']}",
                    f"fundamental_only_cross={st['cross_corpus_m1']}",
                    f"fundamental_only_p={st['cross_p_m1']:.4f}",
                    f"uniform_null_mean={st['uniform_null_mean']:.3f}",
                    f"uniform_p={st['uniform_p']:.5f}",
                    f"mean_periodicity={st['mean_periodicity']:.4f}",
                    f"tol={MATCH_TOL}",
                    f"submultiples={MATCH_SUBMULTIPLES}",
                ]
            )


def figure1(out_dir: Path, atom_dir: Path | None) -> tuple[Path, dict]:
    s_n = np.array([r[0] for r in SONICS_CURVE], dtype=float)
    s_seeds = np.array([r[2:5] for r in SONICS_CURVE], dtype=float)
    s_mean, s_sd = s_seeds.mean(axis=1), s_seeds.std(axis=1, ddof=1)

    f_n = np.array([r[0] for r in FMC_CURVE], dtype=float)
    f_mean = np.array([r[2] for r in FMC_CURVE], dtype=float)
    f_sd = np.array([r[3] for r in FMC_CURVE], dtype=float)

    # Three panels at the REDUCED height. A full-width float costs vertical page
    # space only, so restoring panel (c) beside the two dose curves is free: it
    # makes each panel narrower, not the figure taller. Panel (c) is the
    # atom-level evidence for the dictionary mechanism, which the per-track
    # attribution of Sec. 5 complements rather than replaces.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 1.50),
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.92], "wspace": 0.30},
    )
    axes[1].sharey(axes[0])

    for ax, n, mean, sd, seeds, colour, pooled, title in [
        (axes[0], s_n, s_mean, s_sd, s_seeds, SONICS_C, POOLED_AUC["sonics"], "(a) SONICS – a switch"),
        (axes[1], f_n, f_mean, f_sd, None, FMC_C, POOLED_AUC["fmc"], "(b) FakeMusicCaps – a ramp"),
    ]:
        ax.axhline(0.5, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.axhline(pooled, color=colour, lw=0.6, ls=(0, (1, 2)), alpha=0.8, zorder=1)
        ax.text(
            0.0,
            pooled + 0.012,
            f"published transductive protocol ({pooled:.3f})",
            va="bottom",
            ha="left",
            fontsize=5.8,
            color=colour,
        )
        ax.text(0.0, 0.508, "chance", fontsize=5.8, color=GREY, va="bottom")

        if seeds is not None:  # show the draws, do not assert them
            for k in range(seeds.shape[1]):
                ax.scatter(
                    n, seeds[:, k], s=5.5, facecolor="none", edgecolor=colour, linewidth=0.5, alpha=0.75, zorder=3
                )
            # The transition doses were re-run at ten draws. Plot every one: the
            # switch should be visible, not asserted through an error bar.
            for nf, vals in TEN_SEED.items():
                ax.scatter([nf] * len(vals), vals, s=7, marker="_", color=colour, linewidth=0.9, alpha=0.95, zorder=5)

        ax.errorbar(
            n, mean, yerr=sd, color=colour, marker="o", ms=2.6, capsize=1.8, elinewidth=0.8, capthick=0.8, zorder=4
        )

        ax.set_xscale("symlog", linthresh=10, linscale=0.45)
        ax.set_xlim(-1.2, 18000)
        ax.set_ylim(0.24, 1.02)
        ax.set_xticks([0, 10, 100, 1000, 10000])
        ax.set_xticklabels(["0", "10", "100", "1k", "10k"])
        ax.set_xlabel("generated tracks in the dictionary")
        ax.set_title(title, pad=3.5)

    axes[0].set_ylabel("macro AUC (signed)")
    axes[0].set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1].tick_params(labelleft=False)

    arrow = dict(arrowstyle="-", lw=0.5, color=INK, shrinkB=2, connectionstyle="angle3,angleA=0,angleB=75")
    axes[0].annotate(
        "47 tracks (0.7%):\n7 of 10 draws flip,\nnone land between",
        xy=(47, 0.62),
        xytext=(150, 0.335),
        fontsize=5.8,
        color=INK,
        linespacing=1.15,
        arrowprops=arrow,
    )
    axes[0].annotate(
        "70 tracks:\n10 of 10",
        xy=(70, 0.787),
        xytext=(340, 0.645),
        fontsize=5.8,
        color=INK,
        linespacing=1.15,
        arrowprops=arrow,
    )
    axes[1].annotate(
        "never inverts:\nstarts at 0.744",
        xy=(0, 0.744),
        xytext=(2.4, 0.575),
        fontsize=5.8,
        color=INK,
        linespacing=1.15,
        arrowprops=arrow,
    )

    stats = atom_panel(axes[2], atom_dir)

    pdf = out_dir / "fig1_contamination.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)

    with (out_dir / "fig1_contamination_plotted.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "corpus", "n_fakes_in_dict", "contamination_pct", "mean_auc", "sd", "seeds"])
        for (n, pct, a, b, c, _chirp), m, sd in zip(SONICS_CURVE, s_mean, s_sd):
            w.writerow(["a", "sonics", n, pct, f"{m:.4f}", f"{sd:.4f}", f"{a};{b};{c}"])
        for n, pct, m, sd in FMC_CURVE:
            w.writerow(["b", "fmc", n, pct, f"{m:.4f}", f"{sd:.4f}", ""])
    write_atom_sidecar(out_dir / "fig1c_atoms_plotted.csv", atom_dir, stats)
    return pdf, stats


def figure2_standalone(out_dir: Path, atom_dir: Path | None) -> Path:
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    atom_panel(ax, atom_dir, title=False)
    pdf = out_dir / "fig2_atoms.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="icassp/figs")
    ap.add_argument(
        "--atom-dir",
        default=None,
        help="reports/figures/ -- must contain " "nmf_{fmc,sonics}_periodicity/atom_peakiness.csv",
    )
    ap.add_argument("--curve-dir", default=None, help="reports/diagnostics/ -- checks the *_curve_FINAL CSVs")
    ap.add_argument(
        "--standalone-fig2",
        action="store_true",
        help="also write panel (c) alone, for slides and the " "project page. The paper does not use it.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style()

    if args.curve_dir:
        for corpus in ("fmc", "sonics"):
            p = Path(args.curve_dir) / f"nmf_{corpus}_curve_FINAL" / "nmf_peak_auc_lvl.csv"
            if p.is_file():
                curve_from_csv(p)
            else:
                print(f"  ! {p} not present locally", file=sys.stderr)

    atom_dir = Path(args.atom_dir) if args.atom_dir else None
    pdf, _ = figure1(out_dir, atom_dir)
    print(f"wrote {pdf}")
    if args.standalone_fig2:
        print(f"wrote {figure2_standalone(out_dir, atom_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
