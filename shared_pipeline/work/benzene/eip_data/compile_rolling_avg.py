import pandas as pd
import re
from pathlib import Path

INPUT_DIR = "."
OUTPUT_FILE = "compiled_rolling_avg.csv"


def rewrite_monitor_id(name):
    """Convert BMRF-01 -> BMT-01 (zero-pads preserved as-is)."""
    return re.sub(r"^BMRF-", "BMT-", str(name).strip())


def process_file(file_path):
    df = pd.read_csv(file_path)

    # Drop Duplicate sampler rows
    df = df[df["sampler_type"] != "Duplicate"].copy()

    # Remap monitor name
    df["monitor_id"] = df["monitor_name"].apply(rewrite_monitor_id)

    # Parse period_end -> date (mm/dd/yyyy)
    df["date"] = pd.to_datetime(df["period_end"], utc=True).dt.strftime("%m/%d/%Y")

    # Rename coordinate and concentration columns
    df = df.rename(columns={
        "monitor_lat":   "latitude",
        "monitor_long":  "longitude",
        "concentration": "rolling_12mo_avg",
    })

    # Return only required columns in order
    return df[["date", "monitor_id", "latitude", "longitude", "rolling_12mo_avg"]]


def compile_all_files():
    search_path = Path(INPUT_DIR)
    files = [f for f in search_path.glob("*.csv") if f.name != OUTPUT_FILE]

    print(f"Found {len(files)} input file(s)")

    all_data = []

    for file in files:
        try:
            processed = process_file(file)
            all_data.append(processed)
            print(f"  ✓ {file.name}  ({len(processed)} rows)")
        except Exception as e:
            print(f"  ✗ {file.name}: {e}")

    if not all_data:
        raise ValueError("No valid files processed.")

    master = pd.concat(all_data, ignore_index=True)
    master = master.sort_values(["date", "monitor_id"]).reset_index(drop=True)
    master.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(master)} rows → {OUTPUT_FILE}")


if __name__ == "__main__":
    compile_all_files()
