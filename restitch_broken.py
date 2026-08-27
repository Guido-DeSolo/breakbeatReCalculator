#!/usr/bin/env python3
"""Run-level reconstruction of unresolved breakbeat fragments (dry-run by default)."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from breakbeat_organizer import Analysis, analyze_signal, formatted_name, safe_target, seam_score, tag_wav


ORIGINAL_RE = re.compile(r"^(.*?)-(\d+)\.wav$", re.I)
POW2_BARS = (1, 2, 4, 8)


@dataclass
class Fragment:
    original: str
    current: Path
    series: str
    number: int
    analysis: Analysis


@dataclass
class Candidate:
    start: int
    stop: int
    analysis: Analysis
    bars: int
    score: float
    seam_mean: float


def load_fragments(root: Path, report: Path, bpm_min: float, bpm_max: float) -> list[Fragment]:
    result = []
    with report.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["status"].startswith("unresolved"):
                continue
            match = ORIGINAL_RE.match(row["sources"])
            current = root / row["target"]
            if not match or not current.is_file():
                continue
            y, sr = librosa.load(current, sr=22050, mono=True)
            analysis = analyze_signal(Path(row["sources"]), y, sr, bpm_min, bpm_max)
            result.append(Fragment(row["sources"], current, match.group(1).casefold(),
                                   int(match.group(2)), analysis))
    return sorted(result, key=lambda x: (x.series, x.number))


def consecutive_runs(items: list[Fragment]) -> list[list[Fragment]]:
    runs: list[list[Fragment]] = []
    for item in items:
        if runs and runs[-1][-1].series == item.series and runs[-1][-1].number + 1 == item.number:
            runs[-1].append(item)
        else:
            runs.append([item])
    return [run for run in runs if len(run) >= 2]


def candidate_for(run: list[Fragment], start: int, stop: int, args: argparse.Namespace) -> Candidate | None:
    pieces = run[start:stop]
    y = np.concatenate([x.analysis.y for x in pieces])
    duration = len(y) / pieces[0].analysis.sr
    if duration < 1.25 or duration > 24.0:
        return None
    analysis = analyze_signal(Path(pieces[0].original), y, pieces[0].analysis.sr,
                              args.bpm_min, args.bpm_max)
    if analysis.bpm is None or analysis.bpm_confidence < args.min_bpm_confidence:
        return None
    bars_float = duration * analysis.bpm / 240.0
    bars = min(POW2_BARS, key=lambda x: abs(x - bars_float))
    bar_error = abs(bars_float - bars) / bars
    if bar_error > args.max_bar_error:
        return None
    if analysis.camelot is None or analysis.key_confidence < args.min_key_confidence:
        return None
    seams = [seam_score(a.analysis, b.analysis) for a, b in zip(pieces, pieces[1:])]
    seam_mean = float(np.mean(seams)) if seams else 0.0
    if seam_mean > args.max_internal_seam:
        return None
    score = (4.0 * analysis.bpm_confidence + 1.5 * analysis.key_confidence
             + 0.08 * len(pieces) + 0.25 * math.log2(bars + 1)
             - 0.18 * seam_mean - 3.0 * bar_error)
    return Candidate(start, stop, analysis, bars, score, seam_mean)


def best_partition(run: list[Fragment], args: argparse.Namespace) -> list[Candidate]:
    n = len(run)
    candidates: dict[int, list[Candidate]] = {i: [] for i in range(n)}
    for start in range(n):
        for stop in range(start + 2, min(n, start + args.max_pieces) + 1):
            candidate = candidate_for(run, start, stop, args)
            if candidate is not None:
                candidates[start].append(candidate)
    dp = [0.0] * (n + 1)
    choice: list[Candidate | None] = [None] * n
    for i in range(n - 1, -1, -1):
        dp[i] = dp[i + 1]
        for candidate in candidates[i]:
            value = candidate.score + dp[candidate.stop]
            if value > dp[i] and candidate.score >= args.min_candidate_score:
                dp[i], choice[i] = value, candidate
    selected = []
    i = 0
    while i < n:
        if choice[i] is None:
            i += 1
        else:
            selected.append(choice[i])
            i = choice[i].stop
    return selected


def write_join(pieces: list[Fragment], destination: Path) -> None:
    arrays = []
    sample_rate = channels = None
    subtype = None
    for piece in pieces:
        info = sf.info(piece.current)
        data, sr = sf.read(piece.current, always_2d=True)
        if sample_rate is None:
            sample_rate, channels, subtype = sr, data.shape[1], info.subtype
        if sr != sample_rate or data.shape[1] != channels:
            raise ValueError("source format changes inside candidate")
        arrays.append(data)
    sf.write(destination, np.concatenate(arrays), sample_rate, subtype=subtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("breaks"))
    parser.add_argument("--report", type=Path, help="default: DIRECTORY/breakbeat_report.csv")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--bpm-min", type=float, default=60.0)
    parser.add_argument("--bpm-max", type=float, default=190.0)
    parser.add_argument("--min-bpm-confidence", type=float, default=0.20)
    parser.add_argument("--min-key-confidence", type=float, default=0.08)
    parser.add_argument("--max-bar-error", type=float, default=0.06)
    parser.add_argument("--max-internal-seam", type=float, default=2.6)
    parser.add_argument("--min-candidate-score", type=float, default=1.35)
    parser.add_argument("--max-pieces", type=int, default=32)
    args = parser.parse_args()
    root = args.directory.resolve()
    report = args.report.resolve() if args.report else root / "breakbeat_report.csv"
    if not root.is_dir() or not report.is_file():
        parser.error("directory or breakbeat report not found")

    print("Loading unresolved files and restoring their original order...")
    fragments = load_fragments(root, report, args.bpm_min, args.bpm_max)
    runs = consecutive_runs(fragments)
    print(f"Found {len(fragments)} unresolved files in {len(runs)} consecutive multi-file runs")
    proposals: list[tuple[list[Fragment], Candidate]] = []
    for run in runs:
        for candidate in best_partition(run, args):
            pieces = run[candidate.start:candidate.stop]
            proposals.append((pieces, candidate))
            print(f"PROPOSE: {pieces[0].original}..{pieces[-1].original} ({len(pieces)} pieces) -> "
                  f"{candidate.analysis.camelot}, {candidate.analysis.bpm:.1f} BPM, "
                  f"{candidate.bars} bars; score={candidate.score:.2f}, "
                  f"tempo_conf={candidate.analysis.bpm_confidence:.2f}, "
                  f"key_conf={candidate.analysis.key_confidence:.2f}, seam={candidate.seam_mean:.2f}")
    consumed = sum(len(x[0]) for x in proposals)
    print(f"Summary: {len(proposals)} proposed loops using {consumed}/{len(fragments)} unresolved files")
    if not args.apply:
        print("Dry run only. Review proposals, then rerun with --apply.")
        return 0

    archive = root / "_restitch_sources"
    archive.mkdir(exist_ok=True)
    reserved: set[Path] = set()
    rows = []
    work = Path(tempfile.mkdtemp(prefix=".restitch-work-", dir=root))
    try:
        for index, (pieces, candidate) in enumerate(proposals, 1):
            target = safe_target(root, formatted_name(candidate.analysis, candidate.bars,
                                                       pieces[0].number), reserved)
            staged = work / f"joined-{index}.wav"
            write_join(pieces, staged)
            tag_wav(staged, candidate.analysis, candidate.bars, target.stem)
            for piece in pieces:
                os.replace(piece.current, safe_target(archive, piece.current.name, reserved))
            os.replace(staged, target)
            rows.append({"sources": ";".join(x.original for x in pieces), "target": target.name,
                         "bars": candidate.bars, "bpm": f"{candidate.analysis.bpm:.1f}",
                         "camelot": candidate.analysis.camelot, "score": f"{candidate.score:.3f}",
                         "bpm_confidence": f"{candidate.analysis.bpm_confidence:.3f}",
                         "key_confidence": f"{candidate.analysis.key_confidence:.3f}",
                         "mean_internal_seam": f"{candidate.seam_mean:.3f}"})
        output_report = root / "restitch_report.csv"
        fields = ["sources", "target", "bars", "bpm", "camelot", "score",
                  "bpm_confidence", "key_confidence", "mean_internal_seam"]
        with output_report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Applied {len(rows)} reconstructions; wrote {output_report.name}")
    finally:
        if work.exists():
            shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
