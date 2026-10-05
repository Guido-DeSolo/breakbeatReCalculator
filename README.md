# Breakbeat ReCalculator

**Version 0.0.1** · Command-line tools for organizing and reconstructing drum-break WAV files.

The organizer estimates tempo and key, converts keys to Camelot notation, writes WAV metadata, and renames usable files with their musical information. It can also join compatible adjacent fragments. The reconstruction tool makes second-pass proposals from unresolved fragments.

Both commands preview changes by default. They modify files only when run with `--apply`.

## Requirements

- Python 3.9 or newer
- `libsndfile` support for WAV input/output
- Python dependencies listed in `requirements-breakbeat.txt`

Create an environment and install the dependencies:

```bash
python3 -m venv .breakbeat-venv
.breakbeat-venv/bin/python -m pip install -r requirements-breakbeat.txt
```

## Organize a directory

Preview WAV files directly inside the selected directory:

```bash
.breakbeat-venv/bin/python breakbeat_organizer.py /path/to/breaks
```

Review the proposed renames, groups, and `breakbeat_report.csv`. Apply the plan only after reviewing it:

```bash
.breakbeat-venv/bin/python breakbeat_organizer.py /path/to/breaks --apply
```

Usable files are named with Camelot key, BPM, bar count, and sample count, for example `04B - 087.0 - 2 Bars - 256.wav`. Bar count assumes 4/4; use `--beats-per-bar` for another meter. BPM limits, confidence thresholds, and fragment length can be adjusted with the corresponding command-line options. Files whose names already match the final format are skipped unless `--include-renamed` is set.

Uncertain BPM or key estimates are kept as numbered fallback files. Adjacent fragments are stitched only when their sequence, tempo, key, level, and timbre are compatible. Original source pieces from a successful stitch are retained in `_stitched_sources`.

## Reconstruct unresolved fragments

After organizing, preview a second pass using the generated report:

```bash
.breakbeat-venv/bin/python restitch_broken.py /path/to/breaks
```

The tool searches consecutive unresolved fragments for non-overlapping complete-loop candidates. It scores tempo and key confidence, 1/2/4/8-bar fit, and continuity at joins. Review `restitch_report.csv`, then apply the proposals if they look right:

```bash
.breakbeat-venv/bin/python restitch_broken.py /path/to/breaks --apply
```

Consumed numbered fallbacks are preserved under `_restitch_sources`. Reconstructed WAVs retain the source sample rate, channel count, and PCM subtype.

Only files directly inside the selected directory are analyzed. Reports, temporary files, and reconstructed audio stay inside that directory. The original WAVs are preserved when fragments are joined.
