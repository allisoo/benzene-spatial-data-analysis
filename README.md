# Beaumont Refinery Benzene-Cancer Analysis
## Data Inputs

### Raw Data

Place raw source CSVs in `work`
- place ExxonMobil's self-reported monitoring in `work/benzene/exxonmobil_data`
	- these will download as a PDF, so use a converting software like [iLovePDF](https://www.ilovepdf.com/pdf_to_excel) to convert the PDF to XLSX, then save as a CSV
	- check headers as converting software does not always convert properly
		- cells A1:A4 should be: `Sampler Name, Sampler Latitude, Sampler Longitude, Date of Retrieval`
- place Environmental Integrity Project (EIP) derived 12 month rolling average in `work/benzene/eip_data`
	- OPTIONAL DATASET but highly recommend including
- place All Texas Cancer Registry (TCR) files in `work/cancer`

### Reformatting/Getting Data

#### `work/wind`
- run `get_wind_data.py` to get wind data
	- input start and end dates as prompted (YYYYMMDD)
- copy resulting `wind.csv` to `data/raw`

#### `work/cancer`
- run `compile_cancer_data.py`
- copy resulting `cancer.csv` to `data/raw`

#### `work/benzene/exxonmobil_data`
- run `compile_benzene_data.csv`
- copy resulting `benzene_monitoring.csv` to `data/raw`

#### `work/benzene/eip_data`
- run `compile_rolling_avg.py`
- copy resulting `rolling_12mo_avg.csv` to `data/raw`

All reformatted CSVs should now be placed in `data/raw/`.
- `benzene_monitoring.csv`
- `rolling_12mo_avg.csv` (optional)
- `cancer.csv`
- `wind.csv`

The pipeline accepts custom paths, so filenames can differ.

---
## Expected Fields

If you used the data compilers, these field should already be filled. If reformatting your own data, use the following:
### Expected Benzene Fields

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

### Optional Rolling Average Fields

If you have a separate EIP or other third-party rolling-average file, pass it with `--rolling`.

Required columns:
- `date`
- `monitor_id`
- `latitude`
- `longitude`
- `rolling_12mo_avg`

The pipeline treats this as `rolling_12mo_avg_source`, calculates its own `rolling_12mo_avg_calc` from raw `benzene_ug_m3` using a 365-day time window, and exports `rolling_12mo_avg_difference`.

Recommended:
- Use raw ExxonMobil monitor readings as the primary exposure source.
- Use EIP rolling averages as a validation or comparison series.
- If the calculated and source rolling averages disagree, inspect the date window, missing samples, monitor mapping, and whether EIP used the same underlying monitor data.

### Expected Cancer Fields

Recommended:
- `geography_id`
- `geography_name`
- `cancer_year`
- `cancer_site`
- `incidence_rate`
- `centroid_latitude`
- `centroid_longitude`

Optional:
- `mortality_rate`
- `population`
- `case_count`

### Expected Wind Fields

Recommended:
- `date`
- `wind_direction_degrees`
- `wind_speed`

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

