import pandas as pd
import numpy as np
from pathlib import Path
import re

INPUT_DIR = "."
OUTPUT_FILE = "compiled_benzene_data.csv"
MONITOR_LOOKUP_FILE = "monitor_locations.csv"


def clean_monitor_id(name):
    """Strip asterisks and whitespace from monitor IDs like 'BMT-04*', 'BMT-15**'."""
    return re.sub(r"\*+", "", str(name)).strip()


def process_file(file_path):
    raw = pd.read_csv(file_path, header=None)

    # --- Extract monitor IDs from row 0 (columns 1 onward) ---
    monitor_ids = [clean_monitor_id(v) for v in raw.iloc[0, 1:]]

    # --- Extract latitudes (row 1) and longitudes (row 2) ---
    latitudes  = pd.to_numeric(raw.iloc[1, 1:].values, errors="coerce")
    longitudes = pd.to_numeric(raw.iloc[2, 1:].values, errors="coerce")

    # --- Build monitor lookup for this file ---
    monitor_lookup = pd.DataFrame({
        "monitor_id": monitor_ids,
        "latitude":   latitudes,
        "longitude":  longitudes,
    })

    # --- Data rows start at row 4 (0-indexed row 3 = "Date of Retrieval" label row, skip it) ---
    # Find first actual date row: where column 0 looks like a date
    data_start = None
    for i, val in enumerate(raw.iloc[:, 0]):
        try:
            pd.to_datetime(val)
            data_start = i
            break
        except (ValueError, TypeError):
            continue

    if data_start is None:
        raise ValueError(f"Could not find any date rows in {file_path.name}")

    data_rows = raw.iloc[data_start:].copy()

    # Drop footer rows: keep only rows where column 0 parses as a date
    def is_date(val):
        try:
            pd.to_datetime(val)
            return True
        except (ValueError, TypeError):
            return False

    data_rows = data_rows[data_rows.iloc[:, 0].apply(is_date)].copy()

    # --- Melt wide → long ---
    data_rows.columns = ["date"] + monitor_ids
    data_rows["date"] = pd.to_datetime(data_rows["date"])

    long = data_rows.melt(id_vars="date", var_name="monitor_id", value_name="benzene_ug_m3")

    # Convert benzene values to numeric (non-numeric → NaN, kept as empty per spec)
    long["benzene_ug_m3"] = pd.to_numeric(long["benzene_ug_m3"], errors="coerce")

    # --- Join lat/lon ---
    long = long.merge(monitor_lookup, on="monitor_id", how="left")

    # --- Final column order ---
    long = long[["date", "monitor_id", "latitude", "longitude", "benzene_ug_m3"]]
    long = long.sort_values(["date", "monitor_id"]).reset_index(drop=True)

    return long, monitor_lookup


def compile_all_files():
    search_path = Path(INPUT_DIR)
    files = [f for f in search_path.glob("*.csv") if f.name not in (OUTPUT_FILE, MONITOR_LOOKUP_FILE)]

    print(f"Found {len(files)} input file(s)")

    all_data = []
    all_monitor_lookups = []

    for file in files:
        try:
            processed, monitor_lookup = process_file(file)
            all_data.append(processed)
            all_monitor_lookups.append(monitor_lookup)
            print(f"  ✓ {file.name}  ({len(processed)} rows, {processed['monitor_id'].nunique()} monitors)")
        except Exception as e:
            print(f"  ✗ {file.name}: {e}")

    if not all_data:
        raise ValueError("No valid files processed.")

    # --- Compile main output ---
    master = pd.concat(all_data, ignore_index=True)
    master = master.sort_values(["date", "monitor_id"]).reset_index(drop=True)
    master.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(master)} rows → {OUTPUT_FILE}")

    # --- Compile monitor lookup (deduplicated) ---
    monitors = (
        pd.concat(all_monitor_lookups, ignore_index=True)
        .drop_duplicates(subset="monitor_id")
        .sort_values("monitor_id")
        .reset_index(drop=True)
    )
    monitors.to_csv(MONITOR_LOOKUP_FILE, index=False)
    print(f"Saved {len(monitors)} monitors → {MONITOR_LOOKUP_FILE}")


if __name__ == "__main__":
    compile_all_files()
