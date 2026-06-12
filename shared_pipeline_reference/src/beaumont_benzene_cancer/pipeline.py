from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BENZENE_MOLECULAR_WEIGHT = 78.11
MOLAR_VOLUME_25C = 24.45
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class RefineryLocation:
    latitude: float = 30.0802
    longitude: float = -94.1266


def clean_benzene_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "monitor_id", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benzene data missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["latitude", "longitude", "benzene_ppbv", "benzene_ug_m3", "rolling_12mo_avg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "source" not in df.columns:
        df["source"] = "unknown"
    if "concentration_units" not in df.columns:
        df["concentration_units"] = np.nan
    if "exceedance_flag" not in df.columns:
        df["exceedance_flag"] = False

    return df.dropna(subset=["date", "monitor_id", "latitude", "longitude"]).sort_values(
        ["monitor_id", "date"]
    )


def load_rolling_average_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "monitor_id", "latitude", "longitude", "rolling_12mo_avg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rolling average data missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["latitude", "longitude", "rolling_12mo_avg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return (
        df.dropna(subset=["date", "monitor_id", "latitude", "longitude", "rolling_12mo_avg"])
        .rename(columns={"rolling_12mo_avg": "rolling_12mo_avg_source"})
        .sort_values(["monitor_id", "date"])
    )


def merge_rolling_average_data(raw: pd.DataFrame | None, rolling: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None and rolling is None:
        raise ValueError("Provide at least one benzene input: raw measurements or rolling averages.")
    if raw is None:
        out = rolling.copy()
        out["source"] = "rolling_average_only"
        out["benzene_ppbv"] = np.nan
        out["benzene_ug_m3"] = np.nan
        out["concentration_units"] = np.nan
        out["exceedance_flag"] = False
        out["rolling_source_latitude_delta"] = 0.0
        out["rolling_source_longitude_delta"] = 0.0
        return out
    if rolling is None:
        return raw

    rolling_cols = [
        "date",
        "monitor_id",
        "rolling_12mo_avg_source",
        "latitude",
        "longitude",
    ]
    merged = raw.merge(
        rolling[rolling_cols],
        on=["date", "monitor_id"],
        how="outer",
        suffixes=("", "_rolling_source"),
        indicator="rolling_merge_status",
    )
    for col in ["latitude", "longitude"]:
        source_col = f"{col}_rolling_source"
        merged[col] = merged[col].fillna(merged[source_col])
    for col, default in [
        ("source", "rolling_average_only"),
        ("benzene_ppbv", np.nan),
        ("benzene_ug_m3", np.nan),
        ("concentration_units", np.nan),
        ("exceedance_flag", False),
    ]:
        if col not in merged.columns:
            merged[col] = default
        elif not pd.isna(default):
            merged[col] = merged[col].fillna(default)
    merged["rolling_source_latitude_delta"] = merged["latitude_rolling_source"] - merged["latitude"]
    merged["rolling_source_longitude_delta"] = merged["longitude_rolling_source"] - merged["longitude"]
    return merged.drop(columns=["latitude_rolling_source", "longitude_rolling_source"])


def ppbv_to_ug_m3(ppbv: pd.Series | float) -> pd.Series | float:
    return ppbv * BENZENE_MOLECULAR_WEIGHT / MOLAR_VOLUME_25C


def ug_m3_to_ppbv(ug_m3: pd.Series | float) -> pd.Series | float:
    return ug_m3 * MOLAR_VOLUME_25C / BENZENE_MOLECULAR_WEIGHT


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "benzene_ppbv" not in out.columns and "benzene_ug_m3" not in out.columns:
        raise ValueError("Benzene data must include benzene_ppbv or benzene_ug_m3")
    if "benzene_ppbv" not in out.columns:
        out["benzene_ppbv"] = ug_m3_to_ppbv(out["benzene_ug_m3"])
    if "benzene_ug_m3" not in out.columns:
        out["benzene_ug_m3"] = ppbv_to_ug_m3(out["benzene_ppbv"])

    out["benzene_ppbv"] = out["benzene_ppbv"].fillna(ug_m3_to_ppbv(out["benzene_ug_m3"]))
    out["benzene_ug_m3"] = out["benzene_ug_m3"].fillna(ppbv_to_ug_m3(out["benzene_ppbv"]))
    return out


def weighted_average_nonmissing(values: pd.Series, weights: pd.Series) -> float:
    clean = pd.DataFrame({"value": values, "weight": weights}).replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean["weight"] > 0]
    if clean.empty or clean["weight"].sum() == 0:
        return np.nan
    return float(np.average(clean["value"], weights=clean["weight"]))


def first_available_metric(df: pd.DataFrame, preferred: str, fallback: str) -> pd.Series:
    if preferred in df.columns and fallback in df.columns:
        return df[preferred].fillna(df[fallback])
    if preferred in df.columns:
        return df[preferred]
    if fallback in df.columns:
        return df[fallback]
    return pd.Series(np.nan, index=df.index)


def calculate_rolling_means(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["monitor_id", "date"]).copy()
    if "benzene_ug_m3" not in out.columns:
        out["benzene_ug_m3"] = np.nan
    rolled = []
    for _, group in out.groupby("monitor_id", sort=False):
        monitor = group.set_index("date").sort_index()
        monitor["rolling_12mo_avg_calc"] = monitor["benzene_ug_m3"].rolling("365D", min_periods=20).mean()
        monitor["rolling_12mo_sample_count"] = monitor["benzene_ug_m3"].rolling("365D", min_periods=1).count()
        rolled.append(monitor.reset_index())
    out = pd.concat(rolled, ignore_index=True).sort_values(["monitor_id", "date"])
    if "rolling_12mo_avg_source" in out.columns:
        out["rolling_12mo_avg_difference"] = out["rolling_12mo_avg_calc"] - out["rolling_12mo_avg_source"]
    out["rolling_12mo_avg_best"] = first_available_metric(
        out, "rolling_12mo_avg_calc", "rolling_12mo_avg_source"
    )
    return out


def haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bearing_degrees(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2.astype(float))
    dlon = np.radians(lon2.astype(float) - lon1)
    x = np.sin(dlon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def angular_difference_degrees(a: pd.Series | np.ndarray, b: float) -> np.ndarray:
    return np.abs((np.asarray(a, dtype=float) - b + 180) % 360 - 180)


def load_wind_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "wind_direction_degrees" not in df.columns:
        raise ValueError("Wind data must include wind_direction_degrees")
    df["wind_direction_degrees"] = pd.to_numeric(df["wind_direction_degrees"], errors="coerce")
    if "wind_speed" in df.columns:
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    return df.dropna(subset=["wind_direction_degrees"])


def compute_downwind_score(
    wind: pd.DataFrame,
    receptor_bearing: float,
    tolerance_degrees: float = 45.0,
) -> float:
    # Meteorological direction is where wind comes from; plume travels 180 degrees from it.
    plume_to_degrees = (wind["wind_direction_degrees"] + 180) % 360
    diffs = angular_difference_degrees(plume_to_degrees, receptor_bearing)
    if "wind_speed" in wind.columns and wind["wind_speed"].notna().any():
        weights = wind["wind_speed"].fillna(0)
        return float(weights[diffs <= tolerance_degrees].sum() / weights.sum()) if weights.sum() else np.nan
    return float(np.mean(diffs <= tolerance_degrees))


def summarize_by_monitor(
    df: pd.DataFrame,
    wind: pd.DataFrame | None = None,
    refinery: RefineryLocation = RefineryLocation(),
    exceedance_threshold_ug_m3: float | None = None,
) -> pd.DataFrame:
    work = df.copy()
    if "rolling_12mo_avg_calc" not in work.columns:
        work["rolling_12mo_avg_calc"] = np.nan
    if "rolling_12mo_avg_source" not in work.columns:
        work["rolling_12mo_avg_source"] = np.nan
    if "rolling_12mo_avg_best" not in work.columns:
        work["rolling_12mo_avg_best"] = first_available_metric(
            work, "rolling_12mo_avg_calc", "rolling_12mo_avg_source"
        )
    if "rolling_12mo_avg_difference" not in work.columns:
        work["rolling_12mo_avg_difference"] = np.nan
    if exceedance_threshold_ug_m3 is not None:
        work["exceedance_flag"] = work["benzene_ug_m3"] > exceedance_threshold_ug_m3

    summary = (
        work.groupby("monitor_id")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            mean_benzene=("benzene_ug_m3", "mean"),
            median_benzene=("benzene_ug_m3", "median"),
            max_benzene=("benzene_ug_m3", "max"),
            p95_benzene=("benzene_ug_m3", lambda s: s.quantile(0.95)),
            exceedance_count=("exceedance_flag", "sum"),
            rolling12_latest=("rolling_12mo_avg_calc", "last"),
            rolling12_source_latest=("rolling_12mo_avg_source", "last"),
            rolling12_best_latest=("rolling_12mo_avg_best", "last"),
            rolling12_best_mean=("rolling_12mo_avg_best", "mean"),
            rolling12_mean_abs_difference=("rolling_12mo_avg_difference", lambda s: s.abs().mean()),
            rolling_mean_last_available_date=("date", "max"),
        )
        .reset_index()
    )
    summary["distance_to_refinery"] = haversine_km(
        refinery.latitude, refinery.longitude, summary["latitude"], summary["longitude"]
    )
    summary["azimuth_from_refinery"] = bearing_degrees(
        refinery.latitude, refinery.longitude, summary["latitude"], summary["longitude"]
    )
    if wind is not None and not wind.empty:
        summary["downwind_score"] = summary["azimuth_from_refinery"].apply(
            lambda bearing: compute_downwind_score(wind, bearing)
        )
    else:
        summary["downwind_score"] = np.nan
    return summary


def load_cancer_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"geography_id", "geography_name", "cancer_year", "cancer_site", "incidence_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cancer data missing required columns: {sorted(missing)}")
    df["cancer_year"] = pd.to_numeric(df["cancer_year"], errors="coerce").astype("Int64")
    for col in ["incidence_rate", "mortality_rate", "population", "case_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["cancer_year", "cancer_site", "incidence_rate"])


def match_exposure_to_outcome(
    cancer: pd.DataFrame,
    monitor_summary: pd.DataFrame,
    lag_years: int = 5,
) -> pd.DataFrame:
    out = cancer.copy()
    out["exposure_lag_years"] = lag_years

    has_centroids = {"centroid_latitude", "centroid_longitude"} <= set(out.columns)
    if not has_centroids:
        exposure_mean = monitor_summary["mean_benzene"].mean()
        exposure_max = monitor_summary["max_benzene"].max()
        exposure_exceedances = monitor_summary["exceedance_count"].sum()
        exposure_rolling12 = (
            monitor_summary["rolling12_best_latest"].mean()
            if "rolling12_best_latest" in monitor_summary
            else np.nan
        )
        exposure_wind_weighted = weighted_average_nonmissing(
            monitor_summary["mean_benzene"], monitor_summary["downwind_score"].fillna(0)
        )
        exposure_rolling12_wind_weighted = weighted_average_nonmissing(
            monitor_summary["rolling12_best_latest"], monitor_summary["downwind_score"].fillna(0)
        ) if "rolling12_best_latest" in monitor_summary else np.nan
        distance_weights = 1 / monitor_summary["distance_to_refinery"].replace(0, np.nan)
        distance_weights = (
            distance_weights.fillna(distance_weights.max())
            if distance_weights.notna().any()
            else pd.Series(np.nan, index=monitor_summary.index)
        )
        exposure_distance_weighted = weighted_average_nonmissing(
            monitor_summary["mean_benzene"], distance_weights
        )
        exposure_rolling12_distance_weighted = (
            weighted_average_nonmissing(monitor_summary["rolling12_best_latest"], distance_weights)
            if "rolling12_best_latest" in monitor_summary
            else np.nan
        )
        out["exposure_mean"] = exposure_mean
        out["exposure_rolling12"] = exposure_rolling12
        out["exposure_max"] = exposure_max
        out["exposure_exceedances"] = exposure_exceedances
        out["exposure_wind_weighted"] = exposure_wind_weighted
        out["exposure_rolling12_wind_weighted"] = exposure_rolling12_wind_weighted
        out["exposure_distance_weighted"] = exposure_distance_weighted
        out["exposure_rolling12_distance_weighted"] = exposure_rolling12_distance_weighted
        return out

    for idx, row in out.iterrows():
        distances = haversine_km(
            row["centroid_latitude"],
            row["centroid_longitude"],
            monitor_summary["latitude"],
            monitor_summary["longitude"],
        )
        distance_weights = 1 / distances.replace(0, np.nan)
        if distance_weights.notna().any():
            distance_weights = distance_weights.fillna(distance_weights.max())
        else:
            distance_weights = pd.Series(np.ones(len(monitor_summary)), index=monitor_summary.index)

        wind_weights = monitor_summary["downwind_score"].fillna(0)
        combined_weights = distance_weights * wind_weights

        out.loc[idx, "exposure_mean"] = weighted_average_nonmissing(
            monitor_summary["mean_benzene"], distance_weights
        )
        out.loc[idx, "exposure_rolling12"] = (
            weighted_average_nonmissing(monitor_summary["rolling12_best_latest"], distance_weights)
            if "rolling12_best_latest" in monitor_summary
            else np.nan
        )
        out.loc[idx, "exposure_max"] = monitor_summary.loc[distances.idxmin(), "max_benzene"]
        out.loc[idx, "exposure_exceedances"] = weighted_average_nonmissing(
            monitor_summary["exceedance_count"], distance_weights
        )
        out.loc[idx, "exposure_wind_weighted"] = (
            weighted_average_nonmissing(monitor_summary["mean_benzene"], combined_weights)
        )
        out.loc[idx, "exposure_rolling12_wind_weighted"] = (
            weighted_average_nonmissing(monitor_summary["rolling12_best_latest"], combined_weights)
            if "rolling12_best_latest" in monitor_summary
            else np.nan
        )
        out.loc[idx, "exposure_distance_weighted"] = out.loc[idx, "exposure_mean"]
        out.loc[idx, "exposure_rolling12_distance_weighted"] = out.loc[idx, "exposure_rolling12"]
    return out


def run_correlations(area_exposure: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError("run_correlations requires scipy. Install dependencies with `pip install -r requirements.txt`.") from exc

    rows = []
    exposure_cols = [
        "exposure_mean",
        "exposure_rolling12",
        "exposure_max",
        "exposure_exceedances",
        "exposure_wind_weighted",
        "exposure_rolling12_wind_weighted",
        "exposure_distance_weighted",
        "exposure_rolling12_distance_weighted",
    ]
    for site, site_df in area_exposure.groupby("cancer_site"):
        for col in exposure_cols:
            clean = site_df[[col, "incidence_rate"]].dropna()
            if len(clean) < 3 or clean[col].nunique() < 2:
                rows.append({"cancer_site": site, "exposure_variable": col, "method": "pearson", "n": len(clean), "r": np.nan, "p_value": np.nan})
                rows.append({"cancer_site": site, "exposure_variable": col, "method": "spearman", "n": len(clean), "r": np.nan, "p_value": np.nan})
                continue
            pearson = stats.pearsonr(clean[col], clean["incidence_rate"])
            spearman = stats.spearmanr(clean[col], clean["incidence_rate"])
            rows.append({"cancer_site": site, "exposure_variable": col, "method": "pearson", "n": len(clean), "r": pearson.statistic, "p_value": pearson.pvalue})
            rows.append({"cancer_site": site, "exposure_variable": col, "method": "spearman", "n": len(clean), "r": spearman.statistic, "p_value": spearman.pvalue})
    return pd.DataFrame(rows)


def run_regressions(area_exposure: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError("run_regressions requires statsmodels. Install dependencies with `pip install -r requirements.txt`.") from exc

    rows = []
    for site, site_df in area_exposure.groupby("cancer_site"):
        for exposure_col in ["exposure_mean", "exposure_rolling12"]:
            if exposure_col not in site_df.columns:
                continue
            clean = site_df[["incidence_rate", exposure_col]].dropna()
            if len(clean) < 3 or clean[exposure_col].nunique() < 2:
                rows.append({"cancer_site": site, "model": "ols_incidence_rate", "n": len(clean), "term": exposure_col, "coef": np.nan, "p_value": np.nan})
                continue
            x = sm.add_constant(clean[exposure_col])
            model = sm.OLS(clean["incidence_rate"], x).fit()
            rows.append(
                {
                    "cancer_site": site,
                    "model": "ols_incidence_rate",
                    "n": int(model.nobs),
                    "term": exposure_col,
                    "coef": model.params.get(exposure_col, np.nan),
                    "p_value": model.pvalues.get(exposure_col, np.nan),
                }
            )
    return pd.DataFrame(rows)


def run_sensitivity_analyses(cancer: pd.DataFrame, monitor_summary: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    rows = []
    for lag in lags:
        area = match_exposure_to_outcome(cancer, monitor_summary, lag_years=lag)
        corr = run_correlations(area)
        leukemia = corr[corr["cancer_site"].str.contains("leuk", case=False, na=False)]
        rows.append(
            {
                "lag_years": lag,
                "leukemia_tests": len(leukemia),
                "min_leukemia_p_value": leukemia["p_value"].min() if not leukemia.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def export_tables_and_figures(
    cleaned: pd.DataFrame,
    monitor_summary: pd.DataFrame,
    area_exposure: pd.DataFrame,
    correlations: pd.DataFrame,
    regressions: pd.DataFrame,
    sensitivity: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path / "cleaned_monitor_dataset.csv", index=False)
    monitor_summary.to_csv(out_path / "monitor_summary_table.csv", index=False)
    area_exposure.to_csv(out_path / "area_level_exposure_table.csv", index=False)
    correlations.to_csv(out_path / "correlation_table.csv", index=False)
    regressions.to_csv(out_path / "regression_table.csv", index=False)
    sensitivity.to_csv(out_path / "sensitivity_analysis_table.csv", index=False)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for monitor_id, group in cleaned.groupby("monitor_id"):
        ax.plot(group["date"], group["benzene_ug_m3"], marker="o", linewidth=1, label=str(monitor_id))
    ax.set_title("Benzene concentrations by monitor")
    ax.set_ylabel("Benzene (ug/m3)")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path / "benzene_timeseries.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    color_metric = (
        monitor_summary["mean_benzene"]
        if monitor_summary["mean_benzene"].notna().any()
        else monitor_summary["rolling12_best_latest"]
    )
    scatter = ax.scatter(
        monitor_summary["longitude"],
        monitor_summary["latitude"],
        c=color_metric,
        s=80,
        cmap="viridis",
    )
    ax.set_title("Monitor locations and exposure")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(scatter, ax=ax, label="Benzene exposure (ug/m3)")
    fig.tight_layout()
    fig.savefig(out_path / "monitor_exposure_map.png", dpi=200)
    plt.close(fig)


def build_pipeline(
    benzene_path: str | Path | None,
    cancer_path: str | Path,
    wind_path: str | Path | None,
    out_dir: str | Path,
    rolling_path: str | Path | None = None,
    refinery: RefineryLocation = RefineryLocation(),
    primary_lag_years: int = 5,
    sensitivity_lags_years: list[int] | None = None,
    exceedance_threshold_ug_m3: float | None = None,
) -> None:
    raw_benzene = convert_units(clean_benzene_data(benzene_path)) if benzene_path else None
    rolling = load_rolling_average_data(rolling_path) if rolling_path else None
    benzene = calculate_rolling_means(merge_rolling_average_data(raw_benzene, rolling))
    wind = load_wind_data(wind_path) if wind_path else None
    cancer = load_cancer_data(cancer_path)
    monitor_summary = summarize_by_monitor(
        benzene,
        wind=wind,
        refinery=refinery,
        exceedance_threshold_ug_m3=exceedance_threshold_ug_m3,
    )
    area_exposure = match_exposure_to_outcome(cancer, monitor_summary, lag_years=primary_lag_years)
    try:
        correlations = run_correlations(area_exposure)
    except ImportError:
        correlations = pd.DataFrame(
            columns=["cancer_site", "exposure_variable", "method", "n", "r", "p_value"]
        )
    try:
        regressions = run_regressions(area_exposure)
    except ImportError:
        regressions = pd.DataFrame(columns=["cancer_site", "model", "n", "term", "coef", "p_value"])
    try:
        sensitivity = run_sensitivity_analyses(
            cancer,
            monitor_summary,
            lags=sensitivity_lags_years or [10, 15],
        )
    except ImportError:
        sensitivity = pd.DataFrame(columns=["lag_years", "leukemia_tests", "min_leukemia_p_value"])
    export_tables_and_figures(
        benzene,
        monitor_summary,
        area_exposure,
        correlations,
        regressions,
        sensitivity,
        out_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benzene", help="Optional path to raw benzene monitoring CSV")
    parser.add_argument("--rolling", help="Optional path to third-party rolling 12-month average CSV")
    parser.add_argument("--cancer", required=True, help="Path to cancer CSV")
    parser.add_argument("--wind", help="Path to wind CSV")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--refinery-latitude", type=float, default=RefineryLocation.latitude)
    parser.add_argument("--refinery-longitude", type=float, default=RefineryLocation.longitude)
    parser.add_argument("--primary-lag-years", type=int, default=5)
    parser.add_argument("--sensitivity-lags-years", type=int, nargs="*", default=[10, 15])
    parser.add_argument("--exceedance-threshold-ug-m3", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.benzene and not args.rolling:
        raise ValueError("Provide --benzene, --rolling, or both.")
    build_pipeline(
        benzene_path=args.benzene,
        cancer_path=args.cancer,
        wind_path=args.wind,
        out_dir=args.out,
        rolling_path=args.rolling,
        refinery=RefineryLocation(args.refinery_latitude, args.refinery_longitude),
        primary_lag_years=args.primary_lag_years,
        sensitivity_lags_years=args.sensitivity_lags_years,
        exceedance_threshold_ug_m3=args.exceedance_threshold_ug_m3,
    )


if __name__ == "__main__":
    main()
