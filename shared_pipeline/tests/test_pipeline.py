import numpy as np
import pandas as pd

from beaumont_benzene_cancer.pipeline import (
    BENZENE_MOLECULAR_WEIGHT,
    MOLAR_VOLUME_25C,
    calculate_rolling_means,
    compute_downwind_score,
    convert_units,
    merge_rolling_average_data,
    summarize_by_monitor,
)


def test_convert_units_from_ppbv():
    df = pd.DataFrame({"benzene_ppbv": [1.0]})
    out = convert_units(df)
    assert np.isclose(out.loc[0, "benzene_ug_m3"], BENZENE_MOLECULAR_WEIGHT / MOLAR_VOLUME_25C)


def test_compute_downwind_score_uses_plume_direction():
    wind = pd.DataFrame({"wind_direction_degrees": [270, 90]})
    assert compute_downwind_score(wind, receptor_bearing=90, tolerance_degrees=20) == 0.5


def test_summarize_by_monitor_keeps_monitor_level_rows():
    benzene = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-01-01"]),
            "monitor_id": ["a", "a", "b"],
            "latitude": [30.1, 30.1, 30.2],
            "longitude": [-94.1, -94.1, -94.2],
            "benzene_ug_m3": [1.0, 3.0, 5.0],
            "exceedance_flag": [False, True, True],
        }
    )
    out = summarize_by_monitor(benzene)
    assert set(out["monitor_id"]) == {"a", "b"}
    assert out.loc[out["monitor_id"] == "a", "mean_benzene"].item() == 2.0


def test_merge_rolling_average_data_keeps_source_and_difference():
    dates = pd.date_range("2020-01-01", periods=22, freq="14D")
    raw = pd.DataFrame(
        {
            "date": dates,
            "monitor_id": ["a"] * len(dates),
            "latitude": [30.1] * len(dates),
            "longitude": [-94.1] * len(dates),
            "benzene_ug_m3": np.arange(len(dates), dtype=float),
        }
    )
    rolling = pd.DataFrame(
        {
            "date": dates,
            "monitor_id": ["a"] * len(dates),
            "latitude": [30.1] * len(dates),
            "longitude": [-94.1] * len(dates),
            "rolling_12mo_avg_source": [10.0] * len(dates),
        }
    )
    merged = merge_rolling_average_data(raw, rolling)
    out = calculate_rolling_means(merged)
    assert "rolling_12mo_avg_source" in out.columns
    assert "rolling_12mo_avg_difference" in out.columns
    assert out["rolling_12mo_avg_calc"].notna().any()
