from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _get_feature_names(pipeline, fallback_n: int) -> list[str]:
    pre = pipeline.named_steps.get("preprocess")
    if pre is not None:
        try:
            names = list(pre.get_feature_names_out())
            if names:
                return names
        except Exception:
            pass
    return [f"feature_{i}" for i in range(fallback_n)]


def _to_base_feature(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def _feature_group(base_name: str) -> str:
    market_location = {"postal_fsa_te", "region_name_te", "subarea_name_te"}
    market_history = {
        "prop_prev_list_price", "prop_years_since_prev_list", "prop_has_prev_listing",
        "area_prev_year_median_list", "area_prev_year_listing_count", "area_prev_year_growth",
    }
    market_structure = {
        "p_yearbuilt", "p_bedinbase", "p_bednobase", "p_bedplus", "p_totalbed",
        "p_dwellclass", "p_dwelltype", "p_grandtotalfloorarea", "p_floorareamain",
        "p_totalfloorarea", "sqft_best", "p_fullbath", "p_halfbath", "p_totalbath",
        "p_totalparking", "p_propertytype", "p_style", "p_view", "p_type",
        "p_constructiontype", "property_age",
    }
    if base_name in market_location:
        return "market_location"
    if base_name in market_history:
        return "market_history"
    if base_name in market_structure:
        return "property_structure"
    if base_name == "list_month":
        return "market_timing"
    if base_name.startswith("neigh_prev_year_") or base_name.startswith("fsa_prev_year_"):
        return "local_history"
    if (
        base_name.startswith("mortgage_")
        or base_name.startswith("ircc_")
        or base_name.startswith("study_permits_")
        or base_name.startswith("cmhc_")
        or base_name.startswith("census_")
        or base_name.endswith("_lag1")
        or base_name.endswith("_yoy_change")
    ):
        return "macro_temporal"

    property_like = {
        "LEGAL_TYPE",
        "ZONING_DISTRICT",
        "ZONING_CLASSIFICATION",
        "NEIGHBOURHOOD_CODE",
        "PROPERTY_POSTAL_CODE",
        "POSTAL_FSA",
        "YEAR_BUILT",
        "BIG_IMPROVEMENT_YEAR",
        "LAND_COORDINATE",
        "PROPERTY_AGE",
        "YEARS_SINCE_IMPROVEMENT",
        "HAS_BIG_IMPROVEMENT",
        "BUILDING_AGE_BIN",
    }
    if base_name in property_like:
        return "property_location_structure"
    return "other"


def _plain_language_mapping(base_feature: str) -> tuple[str, str]:
    mapping = {
        "LEGAL_TYPE": (
            "Legal form of the property (for example, strata/condo-related legal type versus other forms).",
            "property-tax-report.csv (City of Vancouver property tax data)",
        ),
        "PROPERTY_POSTAL_CODE": (
            "Full postal code of the property.",
            "property-tax-report.csv (City of Vancouver property tax data)",
        ),
        "ZONING_DISTRICT": (
            "City zoning district assigned to the property.",
            "property-tax-report.csv (City of Vancouver property tax data)",
        ),
        "ZONING_CLASSIFICATION": (
            "Detailed zoning classification category.",
            "property-tax-report.csv (City of Vancouver property tax data)",
        ),
        "NEIGHBOURHOOD_CODE": (
            "Neighbourhood code used by the City dataset.",
            "property-tax-report.csv (City of Vancouver property tax data)",
        ),
        "YEAR_BUILT": (
            "Year when the property was built.",
            "property-tax-report.csv (City of Vancouver property tax data)",
        ),
        "BUILDING_AGE_BIN": (
            "Building age grouped into broad ranges (for example pre-1950, 1950-1979).",
            "Engineered from property-tax-report.csv",
        ),
        "POSTAL_FSA": (
            "First 3 characters of postal code (Forward Sortation Area), used as a rough location group.",
            "Engineered from property-tax-report.csv",
        ),
        "HAS_BIG_IMPROVEMENT": (
            "Whether a major improvement year is recorded (1=yes, 0=no).",
            "Engineered from property-tax-report.csv",
        ),
        "BIG_IMPROVEMENT_YEAR": (
            "Year of major building improvement, if available.",
            "property-tax-report.csv (City of Vancouver property tax data)",
        ),
        "neigh_prev_year_mean_land_value": (
            "Average land value in the same neighbourhood in the previous year.",
            "Engineered from historical property-tax data (t-1 only)",
        ),
        "neigh_prev_year_median_land_value": (
            "Median land value in the same neighbourhood in the previous year.",
            "Engineered from historical property-tax data (t-1 only)",
        ),
        "neigh_prev_year_property_count": (
            "Number of properties observed in the same neighbourhood in the previous year.",
            "Engineered from historical property-tax data (t-1 only)",
        ),
        "neigh_prev_year_growth_rate": (
            "Previous-year neighbourhood growth rate based on median land value.",
            "Engineered from historical property-tax data (t-1 only)",
        ),
        "fsa_prev_year_mean_land_value": (
            "Average land value in the same postal FSA area in the previous year.",
            "Engineered from historical property-tax data (t-1 only)",
        ),
        "fsa_prev_year_median_land_value": (
            "Median land value in the same postal FSA area in the previous year.",
            "Engineered from historical property-tax data (t-1 only)",
        ),
        "fsa_prev_year_property_count": (
            "Number of properties in the same postal FSA area in the previous year.",
            "Engineered from historical property-tax data (t-1 only)",
        ),
        "REPORT_YEAR_NUM": (
            "Numeric version of the report year.",
            "Engineered from property-tax-report.csv",
        ),
        "REPORT_YEAR_CENTERED": (
            "Report year shifted to center around a reference year for model stability.",
            "Engineered from property-tax-report.csv",
        ),
        "cmhc_rental_supply_existing_converted_lag1": (
            "Previous-year count of existing rental units converted, from CMHC rental supply data.",
            "cmhc_vancouver_rental_supply_change_20260228.csv",
        ),
    }

    if base_feature in mapping:
        return mapping[base_feature]
    if base_feature.startswith("mortgage_"):
        return (
            "Year-level mortgage-rate signal.",
            "statcan_mortgage_rate_5yr_20260228.csv",
        )
    if base_feature.startswith("ircc_pr_"):
        return (
            "Year-level permanent resident admissions signal.",
            "ircc_pr_cma_20260228.xlsx",
        )
    if base_feature.startswith("study_permits_"):
        return (
            "Year-level study permit signal.",
            "ircc_studypermits_pt_studylevel_20260228.xlsx",
        )
    if base_feature.startswith("cmhc_"):
        return (
            "Year-level rental supply/change signal.",
            "cmhc_vancouver_rental_supply_change_20260228.csv",
        )
    if base_feature.startswith("census_"):
        return (
            "Census profile signal (only used when census merge is enabled).",
            "statcan_censusprofile2021_* files",
        )
    return (
        "Additional engineered or transformed model input.",
        "Derived during table building",
    )


def save_feature_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_csv_path: Path,
    out_png_path: Path,
    grouped_csv_path: Path | None = None,
    summary_csv_path: Path | None = None,
    explained_csv_path: Path | None = None,
    max_permutation_rows: int = 5000,
) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]
    Xt = pre.transform(X_test)
    feature_names = _get_feature_names(pipeline, Xt.shape[1])

    importance_values = None

    # Native feature importance for tree models.
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_).reshape(-1)
        if len(vals) == len(feature_names):
            importance_values = vals

    # Coefficient-based fallback for linear models.
    if importance_values is None and hasattr(model, "coef_"):
        vals = np.asarray(model.coef_).reshape(-1)
        if len(vals) == len(feature_names):
            importance_values = np.abs(vals)

    # Permutation fallback.
    if importance_values is None:
        if len(X_test) > max_permutation_rows:
            X_eval = X_test.sample(n=max_permutation_rows, random_state=42)
            y_eval = y_test.loc[X_eval.index]
        else:
            X_eval = X_test
            y_eval = y_test
        perm = permutation_importance(
            pipeline,
            X_eval,
            y_eval,
            n_repeats=5,
            random_state=42,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        vals = np.asarray(perm.importances_mean).reshape(-1)
        if len(vals) == len(feature_names):
            importance_values = np.abs(vals)

    if importance_values is None:
        raise ValueError("Could not compute feature importance with available model outputs.")

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "base_feature": [_to_base_feature(f) for f in feature_names],
            "importance": importance_values,
        }
    )
    importance_df["importance_abs"] = importance_df["importance"].abs()
    importance_df["feature_group"] = importance_df["base_feature"].map(_feature_group)
    importance_df = importance_df.sort_values("importance_abs", ascending=False).reset_index(drop=True)
    importance_df["importance_rank"] = np.arange(1, len(importance_df) + 1)
    denom = float(importance_df["importance_abs"].sum())
    importance_df["importance_share"] = (
        importance_df["importance_abs"] / denom if denom > 0 else np.nan
    )

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [
        "feature",
        "base_feature",
        "importance",
        "importance_abs",
        "importance_rank",
        "importance_share",
        "feature_group",
    ]
    importance_df[export_cols].to_csv(out_csv_path, index=False)

    top20 = importance_df.head(20).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top20["base_feature"], top20["importance_abs"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Final Model Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=180)
    plt.close()

    if grouped_csv_path is not None:
        grouped_csv_path.parent.mkdir(parents=True, exist_ok=True)
        grouped = (
            importance_df.groupby("feature_group", as_index=False)["importance_abs"]
            .sum()
            .sort_values("importance_abs", ascending=False)
        )
        grouped = grouped.rename(columns={"importance_abs": "group_importance"})
        gden = float(grouped["group_importance"].sum())
        grouped["group_importance_share"] = (
            grouped["group_importance"] / gden if gden > 0 else np.nan
        )
        grouped.to_csv(grouped_csv_path, index=False)

    if summary_csv_path is not None:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        k = min(10, len(importance_df))
        top_k = importance_df.head(k).copy()
        top_k["importance_bucket"] = "most_important"
        bottom_k = importance_df.tail(k).copy()
        bottom_k["importance_bucket"] = "least_important"
        summary = pd.concat([top_k, bottom_k], ignore_index=True)
        summary = summary[
            [
                "importance_bucket",
                "importance_rank",
                "feature",
                "base_feature",
                "feature_group",
                "importance_abs",
                "importance_share",
            ]
        ]
        summary.to_csv(summary_csv_path, index=False)

    if explained_csv_path is not None:
        explained_csv_path.parent.mkdir(parents=True, exist_ok=True)
        explained = importance_df.copy()
        meanings = explained["base_feature"].map(_plain_language_mapping)
        explained["plain_language_meaning"] = meanings.map(lambda x: x[0])
        explained["source_dataset"] = meanings.map(lambda x: x[1])
        explained = explained[
            [
                "feature",
                "importance",
                "importance_rank",
                "feature_group",
                "plain_language_meaning",
                "source_dataset",
            ]
        ]
        explained.to_csv(explained_csv_path, index=False)

    return importance_df
