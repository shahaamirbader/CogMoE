#!/usr/bin/env python3
"""
CogMoE Preprocessing Script.
Runs the Stage 1 preprocessing pipeline (CWT -> align -> mask -> recover -> iCWT -> features).

Usage:
    python scripts/preprocess.py --config configs/default.yaml --input raw_data/ --output processed/
    python scripts/preprocess.py --config configs/default.yaml --input raw_data/ --output processed/ --fs 128
"""

import argparse
import os
import sys
import glob
import time

import numpy as np
import pandas as pd
import yaml

from cogmoe.preprocessing.pipeline import PreprocessingPipeline
from cogmoe.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def discover_raw_files(input_dir, pattern="*.csv"):
    """
    Discover raw signal CSV files from the input directory.

    Searches recursively for CSV files that contain raw signal data.
    Each CSV is expected to have columns for different modalities
    (e.g., ECG, EEG, Gaze, EDA) with raw time-series values.

    Args:
        input_dir: str - path to directory containing raw CSV files.
        pattern: str - glob pattern for file discovery.

    Returns:
        list of str - sorted list of absolute file paths found.
    """
    search_path = os.path.join(input_dir, "**", pattern)
    files = sorted(glob.glob(search_path, recursive=True))
    return files


def load_raw_segment(filepath, modalities, modality_columns=None):
    """
    Load a single raw signal CSV file and extract per-modality signal arrays.

    Args:
        filepath: str - path to a CSV file with raw signal columns.
        modalities: list of str - modality names (e.g., ['ECG', 'EEG', 'Gaze', 'EDA']).
        modality_columns: dict mapping modality name to list of column names.
            If None, columns are discovered by suffix matching (e.g., '_ECG').

    Returns:
        segments: list of dict {modality_name: 1D np.ndarray} - one dict per row/segment.
        metadata: dict with file-level metadata (participant, num_rows, etc.).
    """
    df = pd.read_csv(filepath)

    # Build column mapping if not provided
    if modality_columns is None:
        modality_columns = {}
        for mod in modalities:
            suffix = f"_{mod}"
            cols = [c for c in df.columns if c.endswith(suffix)]
            if cols:
                modality_columns[mod] = cols

    # Extract per-row signal segments
    segments = []
    for i in range(len(df)):
        row_signals = {}
        for mod in modalities:
            if mod in modality_columns and modality_columns[mod]:
                vals = df.iloc[i][modality_columns[mod]].values.astype(np.float64)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                row_signals[mod] = vals
        if row_signals:
            segments.append(row_signals)

    # Metadata
    metadata = {
        "filepath": filepath,
        "num_rows": len(df),
        "columns": list(df.columns),
        "modalities_found": list(modality_columns.keys()),
    }
    # Try to extract participant ID
    for col_name in ["participant_ID", "participant_id", "subject_id", "SubjectID"]:
        if col_name in df.columns:
            metadata["participant"] = str(df[col_name].iloc[0])
            break

    return segments, metadata


def save_processed_features(recovered_segments, metadata, output_dir, filename):
    """
    Save recovered signal features to a CSV file in the output directory.

    Args:
        recovered_segments: list of dict {modality: 1D np.ndarray}.
        metadata: dict with file-level metadata.
        output_dir: str - output directory path.
        filename: str - output filename.
    """
    rows = []
    for seg in recovered_segments:
        row = {}
        for mod, signal in seg.items():
            for j, val in enumerate(signal):
                row[f"recovered_{j}_{mod}"] = val
        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Add metadata columns if available
    if "participant" in metadata:
        df_out.insert(0, "participant_ID", metadata["participant"])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df_out.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CogMoE Stage 1 Preprocessing Pipeline"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory containing raw signal CSV files."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for processed features."
    )
    parser.add_argument(
        "--fs", type=float, default=128.0,
        help="Sampling frequency for raw signals (default: 128 Hz)."
    )
    parser.add_argument(
        "--pattern", type=str, default="*.csv",
        help="Glob pattern for discovering raw files (default: '*.csv')."
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    modalities = data_cfg.get("modalities", ["ECG", "EEG", "Gaze", "EDA"])
    preproc_cfg = cfg.get("preprocessing", {})

    # Initialize pipeline
    pipeline = PreprocessingPipeline(preproc_cfg)

    # Discover raw files
    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    raw_files = discover_raw_files(input_dir, pattern=args.pattern)
    if not raw_files:
        print(f"Error: No files matching '{args.pattern}' found in '{input_dir}'.")
        sys.exit(1)

    print(f"CogMoE Preprocessing Pipeline")
    print(f"{'='*60}")
    print(f"  Config:      {args.config}")
    print(f"  Input dir:   {input_dir}")
    print(f"  Output dir:  {args.output}")
    print(f"  Sampling fs: {args.fs} Hz")
    print(f"  Modalities:  {modalities}")
    print(f"  Files found: {len(raw_files)}")
    print(f"{'='*60}\n")

    # Processing statistics
    stats = {
        "total_files": len(raw_files),
        "total_segments": 0,
        "processed_files": 0,
        "failed_files": 0,
        "per_modality_segments": {mod: 0 for mod in modalities},
    }

    t_start = time.time()

    for file_idx, filepath in enumerate(raw_files):
        rel_path = os.path.relpath(filepath, input_dir)
        print(f"  [{file_idx + 1}/{len(raw_files)}] Processing: {rel_path}")

        try:
            # Load raw segments
            segments, metadata = load_raw_segment(filepath, modalities)

            if not segments:
                print(f"    Warning: No valid segments found, skipping.")
                stats["failed_files"] += 1
                continue

            # Run the preprocessing pipeline on all segments
            recovered = pipeline.process_dataset(segments, fs=args.fs)

            # Track per-modality stats
            for seg in recovered:
                for mod in seg:
                    if mod in stats["per_modality_segments"]:
                        stats["per_modality_segments"][mod] += 1

            # Save to output directory, preserving subdirectory structure
            out_subdir = os.path.dirname(rel_path)
            out_dir = os.path.join(args.output, out_subdir) if out_subdir else args.output
            out_filename = os.path.splitext(os.path.basename(filepath))[0] + "_processed.csv"
            out_path = save_processed_features(recovered, metadata, out_dir, out_filename)

            stats["total_segments"] += len(recovered)
            stats["processed_files"] += 1
            participant_str = metadata.get("participant", "unknown")
            print(f"    Participant: {participant_str} | "
                  f"Segments: {len(recovered)} | "
                  f"Saved: {out_path}")

        except Exception as e:
            print(f"    Error processing {rel_path}: {e}")
            stats["failed_files"] += 1
            continue

    elapsed = time.time() - t_start

    # Report summary statistics
    print(f"\n{'='*60}")
    print(f"  Preprocessing Complete")
    print(f"{'='*60}")
    print(f"  Total time:         {elapsed:.1f}s")
    print(f"  Files processed:    {stats['processed_files']}/{stats['total_files']}")
    print(f"  Files failed:       {stats['failed_files']}")
    print(f"  Total segments:     {stats['total_segments']}")
    print(f"\n  Per-modality segment counts:")
    for mod, count in stats["per_modality_segments"].items():
        print(f"    {mod:8s}: {count}")
    print(f"\n  Output directory: {os.path.abspath(args.output)}")
    print(f"  Preprocessing complete.")


if __name__ == "__main__":
    main()
