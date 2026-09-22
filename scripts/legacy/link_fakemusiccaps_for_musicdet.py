"""Build MusicDET's expected flat audio directory as a SYMLINK FARM over our raw files.

Rung 0 of the reproduction ladder (§9.7.21) needs MusicDET's code running on
MusicDET's protocol with MusicDET's data layout. Their shipped protocol files
(``TTM01_train.txt`` etc.) already parse; only the audio directory
``datasets/fakemusiccaps/all_audio_wav/`` is missing. Their naming is flat:

    <ytid>.wav            real (MusicCaps)
    TTM0k_<ytid>.wav      fake from generator k

Rather than guess which of our generator directories is TTM01..TTM05 — an
assumption that would silently mislabel every per-generator number — this script
works backwards from the filenames THEIR protocol files ask for, and searches our
raw trees for a file with that exact basename. Anything not found is reported,
never substituted. Symlinks mean no disk cost, which matters on a full volume.

Coverage is the output that matters: if it is not ~100%, rung 0 is not a faithful
reproduction and must not be compared against their published table.

Usage:
    python scripts/link_fakemusiccaps_for_musicdet.py \\
        --protocol-dir ~/MusicDET/datasets/fakemusiccaps/labels \\
        --search-dir   data/external/fakemusiccaps \\
        --search-dir   data/raw/fakemusiccaps_real_only \\
        --out-dir      ~/MusicDET/datasets/fakemusiccaps/all_audio_wav
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def wanted_from_protocols(proto_dir: Path) -> dict[str, str]:
    """Map required basename -> label, read from every protocol .txt."""
    want: dict[str, str] = {}
    files = sorted(proto_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt protocol files in {proto_dir}")
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                want[os.path.basename(parts[0])] = parts[1]
    logger.info("%d protocol files -> %d distinct filenames required", len(files), len(want))
    return want


TTM_RE = re.compile(r"^(TTM0\d)_(.+)$")


def parse_generator_map(spec: str, dirs: list[Path]) -> dict[str, Path]:
    """Parse ``TTM01=dirname,TTM02=...`` into TTM tag -> directory."""
    out: dict[str, Path] = {}
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        if "=" not in item:
            raise SystemExit(f"--generator-map entry must be TTM0k=dirname, got {item!r}")
        tag, name = (p.strip() for p in item.split("=", 1))
        cand = [d / name for d in dirs if (d / name).is_dir()]
        cand += [p for d in dirs for p in d.rglob(name) if p.is_dir()]
        if not cand:
            raise SystemExit(f"--generator-map: no directory named {name!r} under the search dirs")
        out[tag] = cand[0]
    return out


def index_sources(dirs: list[Path]) -> dict[str, Path]:
    """basename -> first matching real file found (audio extensions only)."""
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    idx: dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            logger.warning("search dir does not exist: %s", d)
            continue
        n = 0
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                idx.setdefault(p.name, p)
                # also index under a .wav name so a .mp3/.flac source can satisfy
                # a protocol line that asks for .wav
                idx.setdefault(p.stem + ".wav", p)
                n += 1
        logger.info("indexed %6d audio files from %s", n, d)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--search-dir", action="append", required=True, help="repeatable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--generator-map",
        default="",
        help=(
            "TTM01=<dir>,TTM02=<dir>,... mapping their flat TTM0k_<ytid>.wav naming onto our "
            "per-generator directories. REQUIRED for the fakes to resolve. Note: under "
            "--only_real the training set has no fakes, so a PERMUTED map leaves macro AUC/EER "
            "unchanged and only relabels which row is which generator; under supervised training "
            "(no --only_real) the map determines which generator is SEEN, and a wrong map "
            "invalidates the run."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    want = wanted_from_protocols(Path(args.protocol_dir).expanduser())
    idx = index_sources([Path(d).expanduser() for d in args.search_dir])
    out = Path(args.out_dir).expanduser()

    gmap = parse_generator_map(args.generator_map, [Path(d).expanduser() for d in args.search_dir])
    if gmap:
        logger.info("generator map: %s", {k: v.name for k, v in sorted(gmap.items())})

    found, missing = {}, []
    for name in want:
        src = idx.get(name)
        if src is None:
            # their flat "TTM0k_<ytid>.wav" -> our "<generator dir>/<ytid>.<ext>"
            m = TTM_RE.match(Path(name).stem)
            if m and m.group(1) in gmap:
                stem = m.group(2)
                for cand in sorted(gmap[m.group(1)].rglob(stem + ".*")):
                    if cand.is_file():
                        src = cand
                        break
        (found.setdefault(name, src) if src else missing.append(name))

    found = {k: v for k, v in found.items() if v is not None}
    cov = 100.0 * len(found) / max(1, len(want))
    logger.info("resolved %d / %d required files (%.1f%% coverage)", len(found), len(want), cov)

    if missing:
        by_label = Counter(want[m] for m in missing)
        logger.warning("MISSING %d files, by label: %s", len(missing), dict(by_label))
        for m in missing[:10]:
            logger.warning("  e.g. %s (%s)", m, want[m])

    if args.dry_run:
        logger.info("--dry-run: nothing written")
    else:
        out.mkdir(parents=True, exist_ok=True)
        n_new = 0
        for name, src in found.items():
            dst = out / name
            if dst.is_symlink() or dst.exists():
                continue
            dst.symlink_to(src.resolve())
            n_new += 1
        logger.info("created %d new symlinks in %s (%d already present)", n_new, out, len(found) - n_new)

    if cov < 99.5:
        logger.error(
            "COVERAGE %.1f%% < 99.5%%: rung 0 would train/evaluate on a SUBSET of their protocol, "
            "so its numbers are NOT comparable to their published table. Fix the missing files "
            "(or report the run as a subset) before drawing any conclusion.",
            cov,
        )
        return 1
    logger.info("coverage OK — rung 0 can be run against their unmodified protocol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
