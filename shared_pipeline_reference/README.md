# Beaumont Refinery Benzene-Cancer Analysis

Exploratory ecological analysis scaffold for evaluating whether higher benzene exposure around the ExxonMobil Beaumont Refinery is associated with elevated cancer incidence in nearby areas, with leukemia and hematologic cancers as the primary endpoints.

This project is hypothesis-generating only. County-level or area-level ecological results should not be interpreted as causal evidence.

## Study Focus

- Primary endpoints: leukemia, hematologic cancers, acute myeloid leukemia if available.
- Secondary endpoints: all-cancer incidence and mortality, reported only as context.
- Primary exposure concept: refinery-area benzene measurements combined with distance, wind direction, and geography.
- Primary lag: 5 years, with 10-year and 15-year sensitivity checks where historical data support them.

## Data Inputs

Place source CSVs in `data/raw/`.

Expected inputs:

- `benzene_monitoring.csv`
- `rolling_12mo_avg.csv` optional
- `cancer.csv`
- `wind.csv`

The pipeline accepts custom paths, so filenames can differ.

## Expected Benzene Fields

Minimum:

- `date`
- `monitor_id`
- `latitude`
- `longitude`
- one of `benzene_ppbv` or `benzene_ug_m3`

Optional:

- `source`
- `concentration_units`
- `exceedance_flag`

## Optional Rolling Average Fields

If you have a separate EIP or other third-party rolling-average file, pass it with `--rolling`.

Required columns:

- `date`
- `monitor_id`
- `latitude`
- `longitude`
- `rolling_12mo_avg`

The pipeline treats this as `rolling_12mo_avg_source`, calculates its own `rolling_12mo_avg_calc` from raw `benzene_ug_m3` using a 365-day time window, and exports `rolling_12mo_avg_difference`.

Recommended interpretation:

- Use raw ExxonMobil monitor readings as the primary exposure source.
- Use EIP rolling averages as a validation or comparison series.
- If the calculated and source rolling averages disagree, inspect the date window, missing samples, monitor mapping, and whether EIP used the same underlying monitor data.

## Expected Cancer Fields

Recommended:

- `geography_id`
- `geography_name`
- `cancer_year`
- `cancer_site`
- `incidence_rate`

Optional:

- `mortality_rate`
- `population`
- `centroid_latitude`
- `centroid_longitude`
- `case_count`

## Expected Wind Fields

Recommended:

- `date`
- `wind_direction_degrees`
- `wind_speed`

Wind direction is interpreted as meteorological source direction, meaning the direction the wind comes from.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
python3 -m beaumont_benzene_cancer.pipeline \
  --benzene data/raw/benzene_monitoring.csv \
  --rolling data/raw/rolling_12mo_avg.csv \
  --cancer data/raw/cancer.csv \
  --wind data/raw/wind.csv \
  --out outputs
```

## Outputs

- `cleaned_monitor_dataset.csv`
- `monitor_summary_table.csv`
- `area_level_exposure_table.csv`
- `correlation_table.csv`
- `regression_table.csv`
- `sensitivity_analysis_table.csv`
- `benzene_timeseries.png`
- `monitor_exposure_map.png`

Some outputs require enough input columns to support them. When a result cannot be computed safely, the pipeline writes an empty table with documented columns instead of forcing a conclusion.

## Source Notes

- ATSDR benzene toxicology profile supports prioritizing leukemia and hematopoietic cancers.
- Texas DSHS Texas Cancer Registry / TxCanViz is the preferred cancer source.
- CDC U.S. Cancer Statistics is a backup and comparison source.
- NOAA ISD / Climate Data Online / GHCNh are preferred weather sources.
- EIP-derived series should be treated as derived or validation data, not an independent exposure source.
