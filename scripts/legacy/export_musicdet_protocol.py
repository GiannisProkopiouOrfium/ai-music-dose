"""Export our canonical corpus as MusicDET protocol files, to run THEIR model on OUR data.

This is the decisive control for the FakeMusicCaps gap: if MusicDET's released
code, trained one-class on the same real tracks we train on and evaluated on the
same fakes, reproduces its published FMC numbers, then our gap is *method*. If it
lands near ours, the gap is the *corpus/protocol* and their published number is
not comparable to ours in the way the paper implies.

Protocol format, read from their ``dataset.py:89-95``::

    <filename> <real|fake>          # whitespace-separated, one per line

``filename`` is joined onto ``--fakemusiccaps_audio``; ``only_real=True`` keeps
lines whose second field is exactly ``real``. ``test.py`` expects the eval
directory to contain ``TTM01_eval.txt`` .. ``TTM05_eval.txt``.

Split policy
------------
Their pipeline wants a fixed train/dev/eval split, while our headline protocol is
leave-one-generator-out over folds. These are NOT the same protocol, so this
script makes a deterministic seeded split of the REAL tracks only and documents
it. Every fake goes to eval (they are never trained on under ``--only_real``).
Each ``TTM0k_eval.txt`` contains the held-out reals plus exactly one generator's
fakes, so per-generator EER is computed against a common real set — matching how
we report per-generator numbers.

Usage:
    python scripts/export_musicdet_protocol.py \\
        --manifest   data/processed/canonical_fmc_native/combined_manifest.csv \\
        --audio-root data/processed/canonical_fmc_native \\
        --out-dir    data/processed/musicdet_protocol_fmc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def _rel(path: str, root: Path) -> str | None:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True, help="pass this same path as --fakemusiccaps_audio")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--path-col", default="canonical_path")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--algo-order",
        default="",
        help="comma-separated algorithm names fixing the TTM01..TTM05 mapping; "
        "default is alphabetical (the mapping is always printed and written to mapping.csv)",
    )
    args = ap.parse_args()

    root = Path(args.audio_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest, low_memory=False)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    for col in ("label", args.path_col):
        if col not in df.columns:
            raise SystemExit(f"manifest lacks required column {col!r}")

    df = df.copy()
    df["relpath"] = [_rel(p, root) for p in df[args.path_col].astype(str)]
    n_bad = int(df["relpath"].isna().sum())
    if n_bad:
        logger.warning("%d rows are outside --audio-root and are dropped", n_bad)
        df = df[df["relpath"].notna()]
    if df.empty:
        raise SystemExit("no rows left after resolving paths against --audio-root")

    reals = df[df["label"] == "real"]
    fakes = df[df["label"] == "fake"]
    if reals.empty or fakes.empty:
        raise SystemExit(f"need both labels; got {len(reals)} real / {len(fakes)} fake")

    # deterministic real split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(reals))
    n_tr = int(round(args.train_frac * len(reals)))
    n_dv = int(round(args.dev_frac * len(reals)))
    if n_tr < 1 or n_dv < 1 or n_tr + n_dv >= len(reals):
        raise SystemExit("train/dev fractions leave no evaluation reals")
    tr, dv, ev = (reals.iloc[idx[:n_tr]], reals.iloc[idx[n_tr : n_tr + n_dv]], reals.iloc[idx[n_tr + n_dv :]])

    def write(path: Path, frame: pd.DataFrame) -> None:
        # NB: plain zip over columns -- itertuples() mangles any column name that
        # is not a valid identifier, which silently broke an earlier version.
        with open(path, "w", encoding="utf-8") as fh:
            for rel, lab in zip(frame["relpath"], frame["label"]):
                fh.write(f"{rel} {lab}\n")

    write(out / "train.txt", tr)
    # dev is loaded with only_real=False, so give it reals + a fake sample for a
    # meaningful validation signal
    dev_fakes = fakes.sample(n=min(len(dv), len(fakes)), random_state=args.seed)
    write(out / "dev.txt", pd.concat([dv, dev_fakes]))

    algos = sorted(fakes["algorithm"].dropna().astype(str).unique()) if "algorithm" in fakes.columns else []
    if args.algo_order:
        wanted = [a.strip() for a in args.algo_order.split(",") if a.strip()]
        missing = set(wanted) - set(algos)
        if missing:
            raise SystemExit(f"--algo-order names not present in manifest: {sorted(missing)}")
        algos = wanted + [a for a in algos if a not in wanted]
    if not algos:
        raise SystemExit("manifest has no 'algorithm' column values for fakes; cannot build TTM0k files")

    eval_dir = out / "eval"
    eval_dir.mkdir(exist_ok=True)
    mapping = []
    for k, alg in enumerate(algos, start=1):
        sub = fakes[fakes["algorithm"].astype(str) == alg]
        name = f"TTM0{k}_eval.txt"
        write(eval_dir / name, pd.concat([ev, sub]))
        mapping.append({"file": name, "algorithm": alg, "n_fake": len(sub), "n_real": len(ev)})
        logger.info("%s <- %-16s %6d fake + %6d real", name, alg, len(sub), len(ev))
    pd.DataFrame(mapping).to_csv(out / "mapping.csv", index=False)

    if len(algos) != 5:
        logger.warning(
            "their test.py hard-codes TTM01..TTM05; this manifest has %d generators, so it will "
            "look for files that do not exist (or skip yours). Adjust before running test.py.",
            len(algos),
        )

    logger.info(
        "reals: %d train / %d dev / %d eval (seed %d) | fakes: %d, all eval-only",
        len(tr),
        len(dv),
        len(ev),
        args.seed,
        len(fakes),
    )
    logger.info("wrote %s", out)
    logger.info(
        "NOTE: this fixed split is NOT our leave-one-generator-out protocol. Report any number "
        "from it as 'their model, our corpus, their split' — never merged into our LOGO table."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
