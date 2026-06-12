import pandas as pd
import numpy as np
from pathlib import Path
import re

INPUT_DIR = "."
OUTPUT_FILE = "cancer.csv"

def parse_metadata(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata_cell = lines[1].strip().strip('"')

    match = re.match(r"(.+?),\s*(\d{4})", metadata_cell)

    if not match:
        raise ValueError(
            f"Could not parse metadata from {file_path}\n"
            f"Found: {metadata_cell}"
        )

    cancer_type = match.group(1).strip().lower()
    year = int(match.group(2))

    return cancer_type, year


def process_file(file_path):
    cancer_type, year = parse_metadata(file_path)

    df = pd.read_csv(file_path, skiprows=3)
    df = df.dropna(how="all")

    # Build output with correct column names from the start
    output = pd.DataFrame({
        "geography_name": df["County"].astype(str).str.strip(),
        "case_count": df["Cases"],
        "incidence_rate": df["Age-Adjusted Rate"],
        "cancer_site": cancer_type,
        "cancer_year": year,
    })

    # Merge with county lookup to get geography_id and centroids
    county_lookup = pd.read_csv("county_data.csv")
    county_lookup["geography_name"] = (
        county_lookup["geography_name"].astype(str).str.strip()
    )

    output = output.merge(
        county_lookup[["geography_name", "geography_id", "centroid_latitude", "centroid_longitude"]],
        on="geography_name",
        how="left"
    )

    # Check for missing geography matches
    missing = output["geography_id"].isna().sum()
    if missing > 0:
        print(f"  Missing geography matches: {missing}")
        print("  Unmatched:", output[output["geography_id"].isna()]["geography_name"].unique())

    output = output.dropna(subset=["incidence_rate"])

    # Return with exact required column order
    return output[[
        "geography_name",
        "geography_id",
        "cancer_year",
        "cancer_site",
        "incidence_rate",
        "centroid_latitude",
        "centroid_longitude",
        "case_count",
    ]]


def compile_all_files():
    all_data = []

    search_path = Path(INPUT_DIR)
    files = [f for f in search_path.glob("*.csv") if f.name != OUTPUT_FILE]

    print(f"Found {len(files)} input files")

    for file in files:
        try:
            processed = process_file(file)
            all_data.append(processed)
            print(f"✓ {file.name} ({len(processed)} rows)")
        except Exception as e:
            print(f"✗ {file.name}: {e}")

    if not all_data:
        raise ValueError("No valid files processed")

    master = pd.concat(all_data, ignore_index=True)

    # Sort using the correct final column names
    master = master.sort_values(["cancer_site", "cancer_year", "geography_name"])

    master.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(master)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    compile_all_files()