# Breakbeat organizer

The tool analyzes WAV files directly inside the chosen directory. It estimates tempo from
the onset pattern, estimates key from harmonic chroma, converts that key to Camelot notation,
writes `TBPM`, `TKEY`, and `TIT2` metadata, and renames usable files like:

    04B - 087.0 - 2 Bars - 256.wav

Bar count assumes 4/4 and is calculated from duration and detected BPM. Use `--beats-per-bar`
to analyze another meter. The bar count is also written as a `BARS` metadata field.

Files whose BPM or key cannot be measured confidently are numbered `001.wav`, `002.wav`, and so
on. Adjacent numeric source files are stitched only when one is short and their tempo, key,
level, and timbre are compatible. The original pieces of a successful stitch are retained in
the `_stitched_sources` folder, so the operation does not discard audio or count them as failures.

Create the local environment (already done in this copy) and preview:

    python3 -m venv .breakbeat-venv
    .breakbeat-venv/bin/python -m pip install -r requirements-breakbeat.txt
    .breakbeat-venv/bin/python breakbeat_organizer.py breaks

Review the preview, then apply it:

    .breakbeat-venv/bin/python breakbeat_organizer.py breaks --apply

Only files directly inside `breaks` are considered. All output and temporary files remain
inside that directory. A successful run writes `breakbeat_report.csv`. Adjust `--bpm-min`,
`--bpm-max`, confidence thresholds, or `--fragment-seconds` if the preview is too strict or
too permissive. Existing final-format filenames are skipped unless `--include-renamed` is used.

## Second-pass reconstruction

After the organizer pass, preview run-level reconstruction of unresolved fragments:

    .breakbeat-venv/bin/python restitch_broken.py breaks

This reads `breakbeat_report.csv`, recovers original ordering, and uses dynamic partitioning
to find non-overlapping complete-loop candidates across consecutive runs. It scores tempo
confidence, 1/2/4/8-bar fit, key confidence, and internal seam continuity. Apply only after
reviewing the proposals:

    .breakbeat-venv/bin/python restitch_broken.py breaks --apply

Consumed numbered fallbacks are preserved under `_restitch_sources`. Reconstructed WAVs retain
the original sample rate, channel count, and PCM subtype. Results go to `restitch_report.csv`.
