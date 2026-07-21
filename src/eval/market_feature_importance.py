"""Generate feature-importance reports for the deployed segmented market model.

The bundle routes detached homes to a direct-price model and attached homes to either a
price-per-square-foot or direct fallback model.  A single pipeline's importance is therefore
misleading.  This report normalises each routed model's native importance and weights it by the
number of out-of-time rows that actually use that route.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.feature_importance import _feature_group, _get_feature_names, _to_base_feature


def _route_weights(bundle: dict, test: pd.DataFrame) -> dict[tuple[str, str], int]:
    detached_types = set(bundle["detached_types"])
    detached = test["p_propertytype"].astype(str).isin(detached_types)
    sqft = pd.to_numeric(test[bundle["sqft_col"]], errors="coerce")
    return {
        ("detached", "direct"): int(detached.sum()),
        ("attached", "ppsf"): int((~detached & sqft.notna()).sum()),
        ("attached", "direct"): int((~detached & sqft.isna()).sum()),
    }


def build(model_path: Path, data_path: Path, out_dir: Path) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    if bundle.get("model_type") != "segmented_ppsf_v2":
        raise ValueError(f"Unsupported market model type: {bundle.get('model_type')}")

    data = pd.read_parquet(data_path, columns=[
        bundle["year_col"], "p_propertytype", bundle["sqft_col"]
    ])
    split_year = 2025
    test = data[pd.to_numeric(data[bundle["year_col"]], errors="coerce") >= split_year]
    weights = _route_weights(bundle, test)

    combined: dict[str, float] = {}
    total_weight = 0
    route_rows = []
    for (segment, route), weight in weights.items():
        pipeline = bundle["segments"][segment].get(route)
        if pipeline is None or weight <= 0:
            continue
        model = pipeline.named_steps["model"]
        values = np.asarray(model.feature_importances_, dtype=float)
        names = _get_feature_names(pipeline, len(values))
        if len(names) != len(values) or values.sum() <= 0:
            raise ValueError(f"Invalid importance vector for {segment}/{route}")
        shares = values / values.sum()
        for name, share in zip(names, shares, strict=True):
            combined[name] = combined.get(name, 0.0) + float(share) * weight
        total_weight += weight
        route_rows.append({"segment": segment, "route": route, "test_rows": weight})

    if total_weight != len(test):
        raise ValueError(f"Route weights cover {total_weight} rows, expected {len(test)}")

    report = pd.DataFrame({"feature": list(combined), "importance": list(combined.values())})
    report["importance"] /= total_weight
    report["base_feature"] = report["feature"].map(_to_base_feature)
    report["importance_abs"] = report["importance"].abs()
    report["feature_group"] = report["base_feature"].map(_feature_group)
    report = report.sort_values("importance_abs", ascending=False).reset_index(drop=True)
    report["importance_rank"] = np.arange(1, len(report) + 1)
    report["importance_share"] = report["importance_abs"] / report["importance_abs"].sum()

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["feature", "base_feature", "importance", "importance_abs", "importance_rank",
            "importance_share", "feature_group"]
    report[cols].to_csv(out_dir / "market_feature_importance.csv", index=False)
    pd.DataFrame(route_rows).to_csv(out_dir / "market_feature_importance_routes.csv", index=False)

    grouped = report.groupby("feature_group", as_index=False)["importance_abs"].sum()
    grouped = grouped.rename(columns={"importance_abs": "group_importance"})
    grouped["group_importance_share"] = grouped["group_importance"] / grouped["group_importance"].sum()
    grouped.sort_values("group_importance", ascending=False).to_csv(
        out_dir / "market_feature_importance_grouped.csv", index=False
    )

    top = report.head(20).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["base_feature"], top["importance_share"])
    plt.xlabel("Weighted importance share")
    plt.ylabel("Feature")
    plt.title("Market Model Feature Importance (Top 20, Route-Weighted)")
    plt.tight_layout()
    plt.savefig(out_dir / "market_feature_importance_top20.png", dpi=180)
    plt.close()
    print(f"[market_importance] {len(report)} features; route rows={route_rows}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Build route-weighted market-model importance reports.")
    p.add_argument("--model_path", default="artifacts/market_price_model.joblib")
    p.add_argument("--data_path", default="data/interim/market_model_table.parquet")
    p.add_argument("--out_dir", default="reports/figures")
    a = p.parse_args()
    build(Path(a.model_path), Path(a.data_path), Path(a.out_dir))


if __name__ == "__main__":
    main()
