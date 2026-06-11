"""Scenario-based health impact assessment utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

ExposureFramework = Literal["long_term", "short_term"]
CounterfactualMode = Literal["who2021", "zero", "custom"]


@dataclass(frozen=True)
class HIAConfig:
    """Configuration for scenario-based HIA calculations.

    The default counterfactual mode uses the WHO 2021 guideline values by
    exposure framework: annual values for long-term endpoints and 24-hour values
    for short-term hospital-admission endpoints.

    The coefficient-of-variation parameters are sensitivity assumptions for
    input uncertainty where source-specific confidence intervals are not
    available. They are implemented as mean-preserving multiplicative lognormal
    factors.
    """

    n_samples: int = 10_000
    seed: int = 20260430
    counterfactual_mode: CounterfactualMode = "who2021"
    counterfactual_pm25_long_term: float = 5.0
    counterfactual_pm10_long_term: float = 15.0
    counterfactual_pm25_short_term: float = 15.0
    counterfactual_pm10_short_term: float = 45.0
    baseline_rate_cv: float = 0.05
    population_cv: float = 0.01
    ppm_measurement_cv: float = 0.10
    pm25_pm10_ratio_mean: float = 0.65
    pm25_pm10_ratio_sd: float = 0.10
    pm25_pm10_ratio_min: float = 0.40
    pm25_pm10_ratio_max: float = 0.90
    upper_tail_sensitivity_quantile: float = 0.95


def config_to_dataframe(config: HIAConfig) -> pd.DataFrame:
    """Return HIA configuration as an audit table."""
    return pd.DataFrame([{"parameter": k, "value": v} for k, v in asdict(config).items()])


def default_counterfactual_table(config: HIAConfig | None = None) -> pd.DataFrame:
    """Return counterfactual concentrations used by the current HIA run."""
    config = config or HIAConfig()
    rows = [
        ("PM2.5", "long_term", config.counterfactual_pm25_long_term, "annual_mean", "WHO 2021 annual AQG"),
        ("PM10", "long_term", config.counterfactual_pm10_long_term, "annual_mean", "WHO 2021 annual AQG"),
        ("PM2.5", "short_term", config.counterfactual_pm25_short_term, "24_hour_mean", "WHO 2021 24-hour AQG"),
        ("PM10", "short_term", config.counterfactual_pm10_short_term, "24_hour_mean", "WHO 2021 24-hour AQG"),
    ]
    if config.counterfactual_mode == "zero":
        rows = [
            ("PM2.5", "long_term", 0.0, "annual_mean", "zero counterfactual sensitivity"),
            ("PM10", "long_term", 0.0, "annual_mean", "zero counterfactual sensitivity"),
            ("PM2.5", "short_term", 0.0, "24_hour_mean", "zero counterfactual sensitivity"),
            ("PM10", "short_term", 0.0, "24_hour_mean", "zero counterfactual sensitivity"),
        ]
    elif config.counterfactual_mode == "custom":
        rows = [(p, f, c, a, "custom user-defined") for p, f, c, a, _ in rows]
    return pd.DataFrame(
        rows,
        columns=["pollutant", "framework", "counterfactual_ug_m3", "averaging_time", "basis"],
    )


def default_crf_table() -> pd.DataFrame:
    """Return endpoint-specific CRFs used in the revised scenario HIA.

    RR values are expressed per 10 µg/m³ increment. Rows labelled as ``direct``
    use the CRF as reported for that pollutant. Rows labelled as
    ``converted_from_pm10`` are sensitivity rows where a PM10 CRF is translated
    to a PM2.5-equivalent CRF using a sampled PM2.5/PM10 ratio.
    """
    rows = [
        # pollutant, endpoint, rr, ci_low, ci_high, framework, crf_mapping, source_pollutant, reference_note
        ("PM2.5", "Mortality all causes", 1.062, 1.040, 1.083, "long_term", "direct", "PM2.5", "WHO/HRAPIE; Orellano et al."),
        ("PM10", "Mortality all causes", 1.081, 1.052, 1.110, "long_term", "direct", "PM10", "Orellano et al."),
        ("PM2.5", "Cardiovascular", 1.127, 1.102, 1.152, "long_term", "direct", "PM2.5", "Orellano et al."),
        ("PM10", "Cardiovascular", 1.080, 1.042, 1.120, "long_term", "direct", "PM10", "Orellano et al."),
        ("PM2.5", "Chronic bronchitis", 1.117, 1.040, 1.189, "long_term", "converted_from_pm10", "PM10", "WHO/HRAPIE PM10 CRF converted to PM2.5-equivalent sensitivity row"),
        ("PM10", "Chronic bronchitis", 1.117, 1.040, 1.189, "long_term", "direct", "PM10", "WHO/HRAPIE"),
        ("PM2.5", "Cardiac hospital admissions", 1.006, 1.003, 1.009, "short_term", "converted_from_pm10", "PM10", "Hurley et al. PM10 CRF converted to PM2.5-equivalent sensitivity row"),
        ("PM10", "Cardiac hospital admissions", 1.006, 1.003, 1.009, "short_term", "direct", "PM10", "Hurley et al."),
        ("PM2.5", "Respiratory hospital admissions", 1.009, 1.007, 1.010, "short_term", "converted_from_pm10", "PM10", "Hurley et al. PM10 CRF converted to PM2.5-equivalent sensitivity row"),
        ("PM10", "Respiratory hospital admissions", 1.009, 1.007, 1.010, "short_term", "direct", "PM10", "Hurley et al."),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "pollutant",
            "endpoint",
            "rr_per10",
            "rr_ci_low",
            "rr_ci_high",
            "framework",
            "crf_mapping",
            "source_pollutant",
            "reference_note",
        ],
    )


def default_baseline_rates() -> pd.DataFrame:
    """Return annual baseline rates per person for each city and endpoint."""
    rows = [
        ("Thessaloniki", "Mortality all causes", 828),
        ("Thessaloniki", "Cardiovascular", 272.6),
        ("Thessaloniki", "Chronic bronchitis", 70),
        ("Thessaloniki", "Cardiac hospital admissions", 1870.9),
        ("Thessaloniki", "Respiratory hospital admissions", 1017.9),
        ("Milan", "Mortality all causes", 840),
        ("Milan", "Cardiovascular", 270.3),
        ("Milan", "Chronic bronchitis", 86),
        ("Milan", "Cardiac hospital admissions", 1566.1),
        ("Milan", "Respiratory hospital admissions", 1105.1),
    ]
    df = pd.DataFrame(rows, columns=["city", "endpoint", "annual_rate_per_100k"])
    df["annual_rate_per_person"] = df["annual_rate_per_100k"] / 100_000.0
    df["daily_rate_per_person"] = df["annual_rate_per_person"] / 365.0
    return df


def default_populations() -> pd.DataFrame:
    return pd.DataFrame(
        [("Thessaloniki", 1_092_919), ("Milan", 3_606_653)],
        columns=["city", "population"],
    )


def sample_rr_lognormal(
    mean_rr: float,
    ci_low: float,
    ci_high: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample RR values from a lognormal distribution defined by a 95% CI."""
    if mean_rr <= 0 or ci_low <= 0 or ci_high <= 0:
        raise ValueError("RR and confidence interval bounds must be positive.")
    z = stats.norm.ppf(0.975)
    sigma = (np.log(ci_high) - np.log(ci_low)) / (2 * z)
    mu = np.log(mean_rr)
    samples = rng.lognormal(mean=mu, sigma=sigma, size=n_samples)
    return np.maximum(samples, 1.0)


def sample_lognormal_from_cv(
    mean_value: float,
    cv: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample positive values from a lognormal distribution with target mean/CV."""
    if mean_value < 0:
        raise ValueError("Mean value must be non-negative.")
    if cv <= 0 or mean_value == 0:
        return np.full(n_samples, mean_value, dtype=float)
    sigma = np.sqrt(np.log1p(cv**2))
    mu = np.log(mean_value) - 0.5 * sigma**2
    return rng.lognormal(mean=mu, sigma=sigma, size=n_samples)


def sample_truncated_normal(
    mean_value: float,
    sd: float,
    min_value: float,
    max_value: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a truncated normal distribution."""
    if sd <= 0:
        return np.full(n_samples, mean_value, dtype=float)
    a = (min_value - mean_value) / sd
    b = (max_value - mean_value) / sd
    return stats.truncnorm.rvs(a, b, loc=mean_value, scale=sd, size=n_samples, random_state=rng)


def bootstrap_mean(values: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap the mean of a 1-D exposure distribution."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full(n_samples, np.nan)
    draw_idx = rng.integers(0, values.size, size=(n_samples, values.size))
    return values[draw_idx].mean(axis=1)


def hierarchical_bootstrap_participant_day(
    group: pd.DataFrame,
    pm_col: str,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample participant-day PM values while giving participants equal weight."""
    values_by_participant = [
        values[np.isfinite(values)]
        for values in group.groupby("participant_uid")[pm_col].apply(lambda s: s.to_numpy(dtype=float)).to_list()
    ]
    values_by_participant = [values for values in values_by_participant if values.size > 0]
    if not values_by_participant:
        return np.full(n_samples, np.nan)
    participant_idx = rng.integers(0, len(values_by_participant), size=n_samples)
    out = np.empty(n_samples, dtype=float)
    for i, idx in enumerate(participant_idx):
        vals = values_by_participant[idx]
        out[i] = vals[rng.integers(0, vals.size)]
    return out


def resolve_counterfactual(
    pollutant: str,
    framework: str,
    config: HIAConfig,
) -> tuple[float, str]:
    """Return endpoint-specific counterfactual concentration and basis label."""
    if config.counterfactual_mode == "zero":
        return 0.0, "zero_counterfactual_sensitivity"
    if pollutant == "PM2.5" and framework == "long_term":
        value = config.counterfactual_pm25_long_term
    elif pollutant == "PM10" and framework == "long_term":
        value = config.counterfactual_pm10_long_term
    elif pollutant == "PM2.5" and framework == "short_term":
        value = config.counterfactual_pm25_short_term
    elif pollutant == "PM10" and framework == "short_term":
        value = config.counterfactual_pm10_short_term
    else:
        raise ValueError(f"Unsupported pollutant/framework: {pollutant}/{framework}")

    if config.counterfactual_mode == "who2021":
        return value, "WHO_2021_AQG_annual_or_24h_by_framework"
    if config.counterfactual_mode == "custom":
        return value, "custom_user_defined"
    raise ValueError(f"Unsupported counterfactual mode: {config.counterfactual_mode}")


def apply_crf_mapping(
    base_rr_samples: np.ndarray,
    crf_row: pd.Series,
    config: HIAConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pollutant-specific RR samples and conversion-ratio samples."""
    mapping = crf_row.get("crf_mapping", "direct")
    if mapping == "direct":
        return base_rr_samples, np.full(config.n_samples, np.nan)

    ratio_samples = sample_truncated_normal(
        config.pm25_pm10_ratio_mean,
        config.pm25_pm10_ratio_sd,
        config.pm25_pm10_ratio_min,
        config.pm25_pm10_ratio_max,
        config.n_samples,
        rng,
    )
    if mapping == "converted_from_pm10" and crf_row["pollutant"] == "PM2.5":
        return base_rr_samples ** (1.0 / ratio_samples), ratio_samples
    if mapping == "converted_from_pm25" and crf_row["pollutant"] == "PM10":
        return base_rr_samples ** ratio_samples, ratio_samples
    raise ValueError(f"Unsupported CRF mapping: {mapping} for {crf_row['pollutant']}")


def apply_measurement_uncertainty(
    pm_samples: np.ndarray,
    config: HIAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply optional multiplicative measurement-error uncertainty to PM samples."""
    if config.ppm_measurement_cv <= 0:
        return pm_samples
    multiplier = sample_lognormal_from_cv(1.0, config.ppm_measurement_cv, config.n_samples, rng)
    return np.maximum(pm_samples * multiplier, 0.0)


def attributable_fraction(pm: np.ndarray, rr_per10: np.ndarray, counterfactual: float) -> np.ndarray:
    """Compute AF for a PM contrast against a counterfactual concentration."""
    contrast = np.maximum(pm - counterfactual, 0.0)
    rr = rr_per10 ** (contrast / 10.0)
    return (rr - 1.0) / rr



def summarize_exposure_distribution(
    daily_pm: pd.DataFrame,
    config: HIAConfig | None = None,
) -> pd.DataFrame:
    """Summarize retained daily personal PM distributions for HIA audit purposes."""
    config = config or HIAConfig()
    cf = default_counterfactual_table(config)
    rows: list[dict[str, object]] = []
    for (city, season), group in daily_pm.groupby(["city", "season"]):
        for pollutant, pm_col in [("PM2.5", "pm25"), ("PM10", "pm10")]:
            values = group[pm_col].dropna().astype(float)
            if values.empty:
                continue
            long_cf = cf.loc[(cf["pollutant"] == pollutant) & (cf["framework"] == "long_term"), "counterfactual_ug_m3"].iloc[0]
            short_cf = cf.loc[(cf["pollutant"] == pollutant) & (cf["framework"] == "short_term"), "counterfactual_ug_m3"].iloc[0]
            rows.append({
                "city": city,
                "season": season,
                "pollutant": pollutant,
                "participant_days": int(values.size),
                "participants": int(group["participant_uid"].nunique()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "sd": float(values.std(ddof=1)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
                "long_term_counterfactual": float(long_cf),
                "short_term_counterfactual": float(short_cf),
                "share_above_long_term_counterfactual": float((values > long_cf).mean()),
                "share_above_short_term_counterfactual": float((values > short_cf).mean()),
            })
    return pd.DataFrame(rows).sort_values(["city", "season", "pollutant"]).reset_index(drop=True)


def cap_daily_exposures_by_city_season(
    daily_pm: pd.DataFrame,
    quantile: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap daily PM values at a city-season-specific upper quantile."""
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in the interval (0, 1].")
    capped = daily_pm.copy()
    audit_rows: list[dict[str, object]] = []
    for (city, season), idx in capped.groupby(["city", "season"]).groups.items():
        idx = list(idx)
        for pollutant, pm_col in [("PM2.5", "pm25"), ("PM10", "pm10")]:
            values = capped.loc[idx, pm_col].astype(float)
            cap_value = float(values.quantile(quantile))
            above_cap = values > cap_value
            capped.loc[idx, pm_col] = values.clip(upper=cap_value)
            audit_rows.append({
                "city": city,
                "season": season,
                "pollutant": pollutant,
                "cap_quantile": quantile,
                "cap_value_ug_m3": cap_value,
                "participant_days": int(values.notna().sum()),
                "values_above_cap": int(above_cap.sum()),
                "share_values_above_cap": float(above_cap.mean()),
                "mean_before_cap": float(values.mean()),
                "mean_after_cap": float(capped.loc[idx, pm_col].astype(float).mean()),
                "max_before_cap": float(values.max()),
                "max_after_cap": float(capped.loc[idx, pm_col].astype(float).max()),
            })
    return capped, pd.DataFrame(audit_rows).sort_values(["city", "season", "pollutant"]).reset_index(drop=True)


def calculate_upper_tail_sensitivity(
    daily_pm: pd.DataFrame,
    config: HIAConfig | None = None,
    primary_summary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare primary HIA estimates with upper-tail capped exposure estimates."""
    config = config or HIAConfig()
    if primary_summary is None:
        _, primary_summary, _, _ = calculate_hia_scenarios(daily_pm, config=config, keep_samples=False)
    capped_daily, cap_audit = cap_daily_exposures_by_city_season(
        daily_pm,
        quantile=config.upper_tail_sensitivity_quantile,
    )
    _, capped_summary, _, _ = calculate_hia_scenarios(capped_daily, config=config, keep_samples=False)
    keep_cols = [
        "scenario_key",
        "cases_mean",
        "cases_median",
        "cases_p025",
        "cases_p975",
        "pm_mean_of_samples",
        "exposure_contrast_mean",
        "share_samples_above_counterfactual",
    ]
    merged = primary_summary.merge(
        capped_summary[keep_cols],
        on="scenario_key",
        how="left",
        suffixes=("_primary", "_upper_tail_capped"),
    )
    merged["upper_tail_cap_quantile"] = config.upper_tail_sensitivity_quantile
    for metric in ["cases_mean", "cases_median", "pm_mean_of_samples", "exposure_contrast_mean"]:
        primary = merged[f"{metric}_primary"].astype(float)
        capped = merged[f"{metric}_upper_tail_capped"].astype(float)
        merged[f"{metric}_absolute_change"] = capped - primary
        merged[f"{metric}_percent_change"] = np.where(primary != 0, 100.0 * (capped - primary) / primary, np.nan)
    return merged, cap_audit

def calculate_hia_scenarios(
    daily_pm: pd.DataFrame,
    config: HIAConfig | None = None,
    crf_table: pd.DataFrame | None = None,
    baseline_rates: pd.DataFrame | None = None,
    populations: pd.DataFrame | None = None,
    keep_samples: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate scenario-based HIA samples and summary tables.

    Long-term endpoints use bootstrapped participant-level city-season mean
    exposure values and annual baseline rates. Short-term hospital-admission
    endpoints use hierarchical sampling of participant-day exposures and daily
    baseline rates. CRF uncertainty, exposure-distribution uncertainty, baseline
    rate uncertainty, population-count uncertainty, PPM measurement uncertainty,
    and PM2.5/PM10 conversion-ratio uncertainty for converted CRFs are propagated
    when configured.
    """
    config = config or HIAConfig()
    crf_table = default_crf_table() if crf_table is None else crf_table.copy()
    baseline_rates = default_baseline_rates() if baseline_rates is None else baseline_rates.copy()
    populations = default_populations() if populations is None else populations.copy()
    counterfactuals = default_counterfactual_table(config)
    rng = np.random.default_rng(config.seed)

    daily = daily_pm.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    participant_means = daily.groupby(["city", "season", "participant_uid"], as_index=False)[["pm25", "pm10"]].mean()

    sample_frames = []
    summary_rows = []
    exposure_rows = []

    for (city, season), group in daily.groupby(["city", "season"]):
        pgroup = participant_means[(participant_means["city"] == city) & (participant_means["season"] == season)]
        days_per_participant = group.groupby("participant_uid")["date"].nunique()
        exposure_rows.append(
            {
                "city": city,
                "season": season,
                "date_min": group["date"].min(),
                "date_max": group["date"].max(),
                "participant_days": len(group),
                "participants": group["participant_uid"].nunique(),
                "median_days_per_participant": float(days_per_participant.median()) if len(days_per_participant) else 0,
                "max_days_per_participant": int(days_per_participant.max()) if len(days_per_participant) else 0,
                "pm25_participant_day_mean": group["pm25"].mean(),
                "pm25_participant_weighted_mean": pgroup["pm25"].mean(),
                "pm25_median": group["pm25"].median(),
                "pm25_sd": group["pm25"].std(),
                "pm10_participant_day_mean": group["pm10"].mean(),
                "pm10_participant_weighted_mean": pgroup["pm10"].mean(),
                "pm10_median": group["pm10"].median(),
                "pm10_sd": group["pm10"].std(),
                "city_season_dates": group["date"].nunique(),
            }
        )

    merged = crf_table.merge(baseline_rates, on="endpoint", how="left").merge(populations, on="city", how="left")
    if merged[["annual_rate_per_person", "population"]].isna().any().any():
        raise ValueError("Missing baseline-rate or population values after merging HIA inputs.")

    for _, crf_row in crf_table.iterrows():
        pollutant = crf_row["pollutant"]
        pm_col = "pm25" if pollutant == "PM2.5" else "pm10"
        counterfactual, counterfactual_basis = resolve_counterfactual(pollutant, crf_row["framework"], config)

        base_rr_samples = sample_rr_lognormal(
            crf_row["rr_per10"], crf_row["rr_ci_low"], crf_row["rr_ci_high"], config.n_samples, rng
        )
        rr_samples, ratio_samples = apply_crf_mapping(base_rr_samples, crf_row, config, rng)

        for (city, season), group in daily.groupby(["city", "season"]):
            rate_row = baseline_rates[(baseline_rates["city"] == city) & (baseline_rates["endpoint"] == crf_row["endpoint"])]
            pop_point = populations.loc[populations["city"] == city, "population"].iloc[0]
            if rate_row.empty:
                raise ValueError(f"Missing baseline rate for {city} / {crf_row['endpoint']}")

            if crf_row["framework"] == "long_term":
                pgroup = participant_means[(participant_means["city"] == city) & (participant_means["season"] == season)]
                pm_samples = bootstrap_mean(pgroup[pm_col].to_numpy(dtype=float), config.n_samples, rng)
                baseline_rate_point = rate_row["annual_rate_per_person"].iloc[0]
                output_scale = "annual_persistent_exposure_scenario"
            else:
                pm_samples = hierarchical_bootstrap_participant_day(group, pm_col, config.n_samples, rng)
                baseline_rate_point = rate_row["daily_rate_per_person"].iloc[0]
                output_scale = "daily_short_term_scenario"

            pm_samples = apply_measurement_uncertainty(pm_samples, config, rng)
            baseline_rate_samples = sample_lognormal_from_cv(baseline_rate_point, config.baseline_rate_cv, config.n_samples, rng)
            pop_samples = sample_lognormal_from_cv(pop_point, config.population_cv, config.n_samples, rng)

            af = attributable_fraction(pm_samples, rr_samples, counterfactual)
            attributable_cases = af * baseline_rate_samples * pop_samples
            exposure_contrast = np.maximum(pm_samples - counterfactual, 0.0)

            scenario_key = f"{city}_{season}_{pollutant}_{crf_row['endpoint']}"
            if keep_samples:
                sample_frames.append(
                    pd.DataFrame(
                        {
                            "scenario_key": scenario_key,
                            "city": city,
                            "season": season,
                            "pollutant": pollutant,
                            "endpoint": crf_row["endpoint"],
                            "framework": crf_row["framework"],
                            "output_scale": output_scale,
                            "iteration": np.arange(config.n_samples),
                            "pm_sample": pm_samples,
                            "exposure_contrast": exposure_contrast,
                            "rr_per10_sample": rr_samples,
                            "pm25_pm10_ratio_sample": ratio_samples,
                            "baseline_rate_sample": baseline_rate_samples,
                            "population_sample": pop_samples,
                            "attributable_fraction": af,
                            "attributable_cases": attributable_cases,
                            "counterfactual": counterfactual,
                            "counterfactual_basis": counterfactual_basis,
                            "crf_mapping": crf_row["crf_mapping"],
                        }
                    )
                )
            summary_rows.append(
                {
                    "scenario_key": scenario_key,
                    "city": city,
                    "season": season,
                    "pollutant": pollutant,
                    "endpoint": crf_row["endpoint"],
                    "framework": crf_row["framework"],
                    "output_scale": output_scale,
                    "counterfactual": counterfactual,
                    "counterfactual_basis": counterfactual_basis,
                    "crf_mapping": crf_row["crf_mapping"],
                    "source_pollutant": crf_row["source_pollutant"],
                    "reference_note": crf_row["reference_note"],
                    "n_samples": config.n_samples,
                    "baseline_rate_cv": config.baseline_rate_cv,
                    "population_cv": config.population_cv,
                    "ppm_measurement_cv": config.ppm_measurement_cv,
                    "pm25_pm10_ratio_mean": config.pm25_pm10_ratio_mean if crf_row["crf_mapping"] != "direct" else np.nan,
                    "pm25_pm10_ratio_sd": config.pm25_pm10_ratio_sd if crf_row["crf_mapping"] != "direct" else np.nan,
                    "pm_mean_of_samples": np.nanmean(pm_samples),
                    "exposure_contrast_mean": np.nanmean(exposure_contrast),
                    "share_samples_above_counterfactual": np.nanmean(pm_samples > counterfactual),
                    "cases_mean": np.nanmean(attributable_cases),
                    "cases_median": np.nanmedian(attributable_cases),
                    "cases_p025": np.nanpercentile(attributable_cases, 2.5),
                    "cases_p975": np.nanpercentile(attributable_cases, 97.5),
                }
            )

    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    exposure_inputs = pd.DataFrame(exposure_rows).sort_values(["city", "season"]).reset_index(drop=True)
    counterfactuals = counterfactuals.sort_values(["framework", "pollutant"]).reset_index(drop=True)
    return samples, summary, exposure_inputs, counterfactuals

# Manuscript reproducibility descriptive stage contract.
try:
    from pathlib import Path as _RI_Path
    from raise_icarus.stage_contracts import StageDefinition as _RI_StageDefinition
    from raise_icarus.stage_contracts import StageResult as _RI_StageResult
    from raise_icarus.stage_contracts import definition_for as _ri_definition_for
    from raise_icarus.stage_contracts import run_contract_stage as _ri_run_contract_stage

    _RI_STAGE = _ri_definition_for(__name__)

    def stage_definition() -> _RI_StageDefinition:
        return _RI_STAGE

    def run_stage(
        harmonized_zip: str | _RI_Path | None,
        run_dir: str | _RI_Path,
        n_samples: int = 10000,
        dry_run: bool = False,
    ) -> _RI_StageResult:
        return _ri_run_contract_stage(_RI_STAGE, harmonized_zip=harmonized_zip, run_dir=run_dir, n_samples=n_samples, dry_run=dry_run)
except Exception:
    pass

# Descriptive controlled-data analytical exports.

from pathlib import Path as _RI_Path2
from raise_icarus.controlled_runtime import copy_alias as _ri_copy_alias2
from raise_icarus.controlled_runtime import config_value as _ri_config_value2
from raise_icarus.controlled_runtime import domain_dir as _ri_domain_dir2
from raise_icarus.controlled_runtime import reports_dir as _ri_reports_dir2
from raise_icarus.controlled_runtime import write_text_report as _ri_write_text_report2
from raise_icarus.stage_contracts import StageResult as _RI_StageResult3
from raise_icarus.stage_contracts import dry_run_stage_result as _ri_dry_run_stage_result2


def run_hia_primary_scenarios(harmonized_zip: str | _RI_Path2, out_dir: str | _RI_Path2, n_samples: int = 10000, config: object = None) -> dict[str, _RI_Path2]:
    from raise_icarus.phase3_hia_primary import write_primary_hia_outputs

    output_dir = _ri_domain_dir2(out_dir, "hia")
    phase2_dir = _ri_domain_dir2(out_dir, "exposure")
    outputs = write_primary_hia_outputs(
        harmonized_zip,
        phase2_dir=phase2_dir,
        outdir=output_dir,
        n_samples=n_samples,
        seed=_ri_config_value2(config, "seed", 20260430),
        date_filter_mode=_ri_config_value2(config, "date_filter_mode", "campaign"),
    )
    _ri_copy_alias2(outputs["hia_primary_scenario_summary"], output_dir / "HIA primary scenario summary.csv")
    _ri_copy_alias2(outputs["hia_primary_scenario_summary"], output_dir / "Figure 5 data - HIA attributable cases.csv")
    _ri_copy_alias2(outputs["hia_primary_scenario_summary"], output_dir / "Figure 11 data - HIA attributable cases.csv")
    _ri_copy_alias2(outputs["hia_counterfactual_audit"], output_dir / "HIA counterfactual audit.csv")
    return outputs


def run_hia_upper_tail_sensitivity(harmonized_zip: str | _RI_Path2, out_dir: str | _RI_Path2, n_samples: int = 10000, config: object = None) -> dict[str, _RI_Path2]:
    from raise_icarus.phase4_hia_upper_tail import run_upper_tail_sensitivity

    output_dir = _ri_domain_dir2(out_dir, "hia")
    outputs = run_upper_tail_sensitivity(
        harmonized_zip,
        phase2_dir=_ri_domain_dir2(out_dir, "exposure"),
        phase3_dir=output_dir,
        outdir=output_dir,
        n_samples=n_samples,
        cap_quantile=_ri_config_value2(config, "cap_quantile", 0.95),
        seed=_ri_config_value2(config, "seed", 20260430),
        date_filter_mode=_ri_config_value2(config, "date_filter_mode", "campaign"),
    )
    _ri_copy_alias2(outputs["hia_upper_tail_sensitivity_summary"], output_dir / "HIA upper-tail sensitivity summary.csv")
    _ri_copy_alias2(outputs["hia_upper_tail_percent_change"], output_dir / "HIA upper-tail percent change.csv")
    return outputs


def validate_hia_outputs(out_dir: str | _RI_Path2, expected_manifest: str | _RI_Path2 | None = None) -> _RI_Path2:
    del expected_manifest
    output_dir = _ri_domain_dir2(out_dir, "hia")
    checks = [output_dir / "HIA primary scenario summary.csv", output_dir / "HIA counterfactual audit.csv", output_dir / "HIA upper-tail sensitivity summary.csv"]
    return _ri_write_text_report2(_ri_reports_dir2(out_dir) / "HIA validation report.txt", "HIA Validation Report", [f"{path.name}: {'PASS' if path.exists() else 'FAIL'}" for path in checks])


def run_stage(harmonized_zip: str | _RI_Path2 | None, run_dir: str | _RI_Path2, n_samples: int = 10000, dry_run: bool = False) -> _RI_StageResult3:
    if dry_run:
        return _ri_dry_run_stage_result2(_RI_STAGE, run_dir)
    if harmonized_zip is None:
        return _RI_StageResult3(_RI_STAGE.stage_name, _RI_STAGE.module_name, "FAIL", _RI_STAGE.output_domain, (), "harmonized archive path is required")
    run_hia_primary_scenarios(harmonized_zip, run_dir, n_samples=n_samples)
    run_hia_upper_tail_sensitivity(harmonized_zip, run_dir, n_samples=n_samples)
    report = validate_hia_outputs(run_dir)
    return _RI_StageResult3(_RI_STAGE.stage_name, _RI_STAGE.module_name, "PASS", _RI_STAGE.output_domain, (str(report),), "HIA outputs generated.")

