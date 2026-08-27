#!/usr/bin/env python3
"""Analyze, tag, optionally stitch, and rename breakbeat WAV files.

The default mode is a read-only preview. Pass --apply to make changes.
All temporary and final files are kept beneath the selected directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from mutagen.id3 import TBPM, TKEY, TIT2, TXXX
from mutagen.wave import WAVE


AUDIO_SUFFIXES = {".wav", ".wave"}
NAME_RE = re.compile(r"(\d+)(?=\D*$)")
FINAL_RE = re.compile(r"^\d{2}[AB] - \d{3}\.\d - \d+ Bars? - \d+\.wav$", re.I)

# Krumhansl-Schmuckler pitch-class profiles, C through B.
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CAM_MAJOR = {0: "08B", 1: "03B", 2: "10B", 3: "05B", 4: "12B", 5: "07B", 6: "02B", 7: "09B", 8: "04B", 9: "11B", 10: "06B", 11: "01B"}
CAM_MINOR = {0: "05A", 1: "12A", 2: "07A", 3: "02A", 4: "09A", 5: "04A", 6: "11A", 7: "06A", 8: "01A", 9: "08A", 10: "03A", 11: "10A"}


@dataclass
class Analysis:
    path: Path
    source_number: int | None
    duration: float
    bpm: float | None
    bpm_confidence: float
    key: str | None
    camelot: str | None
    key_confidence: float
    y: np.ndarray
    sr: int

    @property
    def usable(self) -> bool:
        return self.bpm is not None and self.camelot is not None


def source_number(path: Path) -> int | None:
    match = NAME_RE.search(path.stem)
    return int(match.group(1)) if match else None


def source_series(path: Path) -> str:
    """Filename portion before its trailing sequence number."""
    return NAME_RE.sub("", path.stem).rstrip(" -_").casefold()


def normalize_bpm(bpm: float, minimum: float, maximum: float) -> float:
    while bpm < minimum:
        bpm *= 2.0
    while bpm > maximum:
        bpm /= 2.0
    return bpm


def estimate_key(y: np.ndarray, sr: int) -> tuple[str | None, str | None, float]:
    # Harmonic separation stops kick/snare energy from dominating key detection.
    harmonic = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    if chroma.size == 0:
        return None, None, 0.0
    vector = np.median(chroma, axis=1)
    if not np.isfinite(vector).all() or float(vector.sum()) < 1e-6:
        return None, None, 0.0
    vector = (vector - vector.mean()) / (vector.std() + 1e-9)
    candidates: list[tuple[float, int, bool]] = []
    for tonic in range(12):
        for profile, is_minor in ((MAJOR, False), (MINOR, True)):
            p = np.roll(profile, tonic)
            p = (p - p.mean()) / p.std()
            candidates.append((float(np.corrcoef(vector, p)[0, 1]), tonic, is_minor))
    candidates.sort(reverse=True)
    score, tonic, is_minor = candidates[0]
    runner_up = candidates[1][0]
    confidence = max(0.0, min(1.0, (score - runner_up) * 3.0 + max(score, 0.0) * 0.35))
    key = NOTE_NAMES[tonic] + ("m" if is_minor else "")
    camelot = (CAM_MINOR if is_minor else CAM_MAJOR)[tonic]
    return key, camelot, confidence


def analyze_signal(path: Path, y: np.ndarray, sr: int, bpm_min: float, bpm_max: float) -> Analysis:
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 0.12 or float(np.max(np.abs(y), initial=0.0)) < 1e-5:
        return Analysis(path, source_number(path), duration, None, 0.0, None, None, 0.0, y, sr)

    onset = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.mean)
    tempo_values = librosa.feature.tempo(onset_envelope=onset, sr=sr, aggregate=None)
    bpm = float(np.median(tempo_values)) if tempo_values.size else 0.0
    bpm = normalize_bpm(bpm, bpm_min, bpm_max) if bpm > 0 else 0.0

    # Confidence compares the selected periodicity with all other plausible lags.
    ac = librosa.autocorrelate(onset, max_size=max(2, int(len(onset) * 0.75)))
    # Exported loops normally contain a power-of-two beat count. Compare those
    # exact-duration candidates against the onset autocorrelation; this resolves
    # common short-loop errors where a generic tracker chooses the wrong pulse.
    if duration >= 1.25:
        duration_candidates = [60.0 * beats / duration for beats in (2, 4, 8, 16, 32)]
        duration_candidates = [x for x in duration_candidates if bpm_min <= x <= bpm_max]
        scored: list[tuple[float, float]] = []
        for candidate in duration_candidates:
            candidate_lag = int(round(60.0 * sr / (512 * candidate)))
            if 0 < candidate_lag < len(ac):
                scored.append((float(ac[candidate_lag]), candidate))
        if scored and max(scored)[0] >= float(ac[0]) * 0.10:
            bpm = max(scored)[1]

    lag = int(round(60.0 * sr / (512 * bpm))) if bpm else 0
    peak = float(ac[lag]) if 0 < lag < len(ac) else 0.0
    baseline = float(np.percentile(ac[1:], 90)) if len(ac) > 2 else 0.0
    bpm_conf = max(0.0, min(1.0, peak / (float(ac[0]) + 1e-9) * 1.8))
    if peak < baseline * 0.65:
        bpm_conf *= 0.5

    # Sub-half-second hits do not contain enough harmonic context for a defensible key.
    key, camelot, key_conf = estimate_key(y, sr) if duration >= 0.5 else (None, None, 0.0)
    return Analysis(path, source_number(path), duration, bpm or None, bpm_conf, key, camelot, key_conf, y, sr)


def analyze(path: Path, target_sr: int, bpm_min: float, bpm_max: float) -> Analysis:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return analyze_signal(path, y, sr, bpm_min, bpm_max)


def adjacent(a: Analysis, b: Analysis) -> bool:
    return (source_series(a.path) == source_series(b.path)
            and a.source_number is not None and b.source_number == a.source_number + 1)


def seam_score(a: Analysis, b: Analysis) -> float:
    """Lower is better: compare timbre/level immediately around the proposed seam."""
    n = min(int(0.08 * a.sr), len(a.y), len(b.y))
    if n < 64:
        return math.inf
    left, right = a.y[-n:], b.y[:n]
    rms_l = math.sqrt(float(np.mean(left * left)) + 1e-12)
    rms_r = math.sqrt(float(np.mean(right * right)) + 1e-12)
    level = abs(math.log((rms_l + 1e-6) / (rms_r + 1e-6)))
    z_l = float(np.mean(librosa.feature.zero_crossing_rate(y=left)))
    z_r = float(np.mean(librosa.feature.zero_crossing_rate(y=right)))
    return level + 4.0 * abs(z_l - z_r)


def can_stitch(a: Analysis, b: Analysis, fragment_seconds: float) -> bool:
    if not adjacent(a, b) or a.sr != b.sr:
        return False
    if min(a.duration, b.duration) > fragment_seconds:
        return False
    if a.bpm and b.bpm and abs(math.log2(a.bpm / b.bpm)) > 0.08:
        return False
    # Key estimates from sub-second drum fragments are not stable enough to veto a join.
    if (a.duration >= 1.0 and b.duration >= 1.0 and a.key_confidence >= 0.35
            and b.key_confidence >= 0.35 and a.camelot and b.camelot and a.camelot != b.camelot):
        return False
    return seam_score(a, b) <= 1.15


def make_groups(items: list[Analysis], fragment_seconds: float) -> list[list[Analysis]]:
    groups: list[list[Analysis]] = []
    for item in items:
        if groups and can_stitch(groups[-1][-1], item, fragment_seconds):
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def bar_count(a: Analysis, beats_per_bar: int) -> int | None:
    if a.bpm is None or a.duration <= 0:
        return None
    bars = a.duration * a.bpm / (60.0 * beats_per_bar)
    return max(1, int(round(bars)))


def formatted_name(a: Analysis, bars: int, sequence: int) -> str:
    assert a.camelot is not None and a.bpm is not None
    label = "Bar" if bars == 1 else "Bars"
    return f"{a.camelot} - {a.bpm:05.1f} - {bars} {label} - {sequence}.wav"


def tag_wav(path: Path, a: Analysis, bars: int, title: str) -> None:
    audio = WAVE(path)
    if audio.tags is None:
        audio.add_tags()
    audio.tags.delall("TBPM")
    audio.tags.delall("TKEY")
    audio.tags.delall("TIT2")
    for frame in list(audio.tags.getall("TXXX")):
        if frame.desc.upper() == "BARS":
            audio.tags.delall(frame.HashKey)
    audio.tags.add(TBPM(encoding=3, text=[f"{a.bpm:.1f}"]))
    audio.tags.add(TKEY(encoding=3, text=[a.camelot, a.key or ""]))
    audio.tags.add(TIT2(encoding=3, text=[title]))
    audio.tags.add(TXXX(encoding=3, desc="BARS", text=[str(bars)]))
    audio.save()


def safe_target(directory: Path, name: str, reserved: set[Path]) -> Path:
    candidate = directory / name
    if candidate not in reserved and not candidate.exists():
        reserved.add(candidate)
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    i = 2
    while True:
        candidate = directory / f"{stem} ({i}){suffix}"
        if candidate not in reserved and not candidate.exists():
            reserved.add(candidate)
            return candidate
        i += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".", type=Path)
    parser.add_argument("--apply", action="store_true", help="write tags, stitched WAVs, and filenames")
    parser.add_argument("--include-renamed", action="store_true", help="reanalyze files already using the final naming scheme")
    parser.add_argument("--bpm-min", type=float, default=60.0)
    parser.add_argument("--bpm-max", type=float, default=190.0)
    parser.add_argument("--fragment-seconds", type=float, default=1.35)
    parser.add_argument("--beats-per-bar", type=int, default=4,
                        help="meter used to calculate bars (default: 4)")
    parser.add_argument("--min-bpm-confidence", type=float, default=0.12)
    parser.add_argument("--min-key-confidence", type=float, default=0.08)
    args = parser.parse_args()
    root = args.directory.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    if args.bpm_min <= 0 or args.bpm_max <= args.bpm_min:
        parser.error("invalid BPM range")
    if args.beats_per_bar <= 0:
        parser.error("--beats-per-bar must be positive")

    paths = sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
         and (args.include_renamed or not FINAL_RE.match(p.name))),
        key=lambda p: (source_series(p), source_number(p) is None, source_number(p) or 0, p.name.lower()),
    )
    if not paths:
        print(f"No unprocessed WAV files directly inside {root}")
        return 0

    analyses: list[Analysis] = []
    for i, path in enumerate(paths, 1):
        try:
            item = analyze(path, 22050, args.bpm_min, args.bpm_max)
            if item.bpm_confidence < args.min_bpm_confidence:
                item.bpm = None
            if item.key_confidence < args.min_key_confidence:
                item.key = item.camelot = None
            analyses.append(item)
            print(f"[{i}/{len(paths)}] {path.name}: BPM={item.bpm and f'{item.bpm:.1f}'} "
                  f"({item.bpm_confidence:.2f}), key={item.key or '?'} / {item.camelot or '?'} "
                  f"({item.key_confidence:.2f}), {item.duration:.2f}s")
        except Exception as exc:
            print(f"[{i}/{len(paths)}] {path.name}: analysis failed: {exc}", file=sys.stderr)
            analyses.append(Analysis(path, source_number(path), 0, None, 0, None, None, 0, np.array([]), 22050))

    groups = make_groups(analyses, args.fragment_seconds)
    reserved: set[Path] = set()
    broken_count = 0
    successful_count = 0
    rows: list[dict[str, object]] = []

    # Stage every changed file first, so swaps/collisions cannot lose data.
    temp_dir: Path | None = None
    if args.apply:
        temp_dir = Path(tempfile.mkdtemp(prefix=".breakbeat-work-", dir=root))

    try:
        for group in groups:
            stitched = len(group) > 1
            sources = ";".join(x.path.name for x in group)
            if stitched:
                y = np.concatenate([x.y for x in group])
                staged = temp_dir / f"stitched-{group[0].source_number or successful_count}.wav" if temp_dir else None
                result = analyze_signal(group[0].path, y, group[0].sr, args.bpm_min, args.bpm_max)
                if staged:
                    sf.write(staged, y, group[0].sr, subtype="PCM_24")
                if result.bpm_confidence < args.min_bpm_confidence:
                    result.bpm = None
                if result.key_confidence < args.min_key_confidence:
                    result.key = result.camelot = None
            else:
                result = group[0]
                staged = None

            # A failed join is not emitted as an extra file: its original pieces are
            # handled individually, preserving all audio and keeping counts exact.
            if stitched and not result.usable:
                print(f"JOIN REJECTED: {sources}; keeping pieces for unresolved numbering")
                for original in group:
                    broken_count += 1
                    target = safe_target(root, f"{broken_count:03d}.wav", reserved)
                    rows.append({"sources": original.path.name, "target": target.name,
                                 "status": "unresolved-fragment", "bpm": "", "camelot": "", "bars": "",
                                 "bpm_confidence": original.bpm_confidence,
                                 "key_confidence": original.key_confidence})
                    print(f"UNRESOLVED: {original.path.name} -> {target.name}")
                    if args.apply:
                        os.replace(original.path, target)
                continue

            if result.usable:
                successful_count += 1
                seq = result.source_number if result.source_number is not None else successful_count
                bars = bar_count(result, args.beats_per_bar)
                assert bars is not None
                target = safe_target(root, formatted_name(result, bars, seq), reserved)
                status = "stitched" if stitched else "ready"
            else:
                broken_count += 1
                target = safe_target(root, f"{broken_count:03d}.wav", reserved)
                status = "unresolved"

            print(f"{status.upper()}: {sources} -> {target.name}")
            rows.append({"sources": sources, "target": target.name, "status": status,
                         "bpm": result.bpm or "", "camelot": result.camelot or "",
                         "bars": bars if result.usable else "",
                         "bpm_confidence": result.bpm_confidence, "key_confidence": result.key_confidence})

            if args.apply:
                if stitched:
                    assert staged is not None
                    source_for_target = staged
                else:
                    source_for_target = temp_dir / f"source-{len(rows)}.wav"
                    shutil.copy2(group[0].path, source_for_target)
                if result.usable:
                    tag_wav(source_for_target, result, bars, target.stem)
                # Keep source pieces in an in-directory archive; they are not failures.
                if stitched:
                    archive = root / "_stitched_sources"
                    archive.mkdir(exist_ok=True)
                    for original in group:
                        fragment_target = safe_target(archive, original.path.name, reserved)
                        os.replace(original.path, fragment_target)
                        rows.append({"sources": original.path.name, "target": fragment_target.name,
                                     "status": "stitched-source", "bpm": "", "camelot": "", "bars": "",
                                     "bpm_confidence": "", "key_confidence": ""})
                else:
                    group[0].path.unlink()
                os.replace(source_for_target, target)

        report = root / "breakbeat_report.csv"
        if args.apply:
            with report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {report.name}")
        else:
            print("\nDry run only. Run again with --apply to make these changes.")
        print(f"Summary: {successful_count} usable groups, {broken_count} unresolved files")
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
