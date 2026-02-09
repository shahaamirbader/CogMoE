"""
CSV input/output utilities and metadata handling for CogMoE.
"""
import os
import glob
import pandas as pd


def find_modality_files(base_dir, pattern="*_combined_resampled.csv"):
    """
    List all modality CSV files matching a glob pattern.

    Args:
        base_dir (str): Directory containing modality files.
        pattern  (str): Glob pattern, e.g. "*_combined_resampled.csv".

    Returns:
        List[str]: Sorted list of file paths.
    """
    search_path = os.path.join(base_dir, pattern)
    files = sorted(glob.glob(search_path))
    return files


def read_modality_csv(path, time_col="Timestamp.1", metadata_cols=None):
    """
    Read a modality CSV, parse the time column, and separate metadata.

    Args:
        path          (str): Path to CSV file.
        time_col      (str): Column name for timestamp.
        metadata_cols (list[str] or None): Columns to treat as metadata.

    Returns:
        df          (pd.DataFrame): Data indexed by time.
        metadata    (dict): Mapping of metadata column to scalar values.
    """
    df = pd.read_csv(path)
    # Parse time column to numeric
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.set_index(time_col).sort_index()

    # Extract metadata
    metadata = {}
    if metadata_cols:
        for col in metadata_cols:
            if col in df.columns:
                metadata[col] = df[col].iloc[0]
                df = df.drop(columns=[col])
    return df, metadata


def save_preprocessed_csv(df, metadata, output_path, time_col="Timestamp"):
    """
    Save a preprocessed DataFrame, reinserting metadata and time column.

    Args:
        df           (pd.DataFrame): Data indexed by time.
        metadata     (dict): Metadata to include as columns.
        output_path  (str): File path to write CSV.
        time_col     (str): Name for the time column in output.
    """
    out = df.copy()
    # Insert time column as first column
    out[time_col] = out.index
    # Add metadata columns
    for k, v in metadata.items():
        out[k] = v
    # Reorder: time first, then metadata, then signals
    cols = [time_col] + list(metadata.keys()) + [c for c in out.columns if c not in [time_col] + list(metadata.keys())]
    out = out[cols]
    # Save
    out.to_csv(output_path, index=False)

