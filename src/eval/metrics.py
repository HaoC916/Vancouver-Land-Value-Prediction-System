import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred, y_train):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    mask = y_true > 0
    if np.any(mask):
        ape = np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]
        median_ape = float(np.median(ape))
    else:
        median_ape = float("nan")

    cap = float(np.percentile(y_train, 99.5))
    y_true_cap = np.minimum(y_true, cap)
    y_pred_cap = np.minimum(y_pred, cap)
    robust_rmse = np.sqrt(mean_squared_error(y_true_cap, y_pred_cap))
    robust_mae = mean_absolute_error(y_true_cap, y_pred_cap)

    return {
        "rmse": rmse,
        "mae": mae,
        "median_ape": median_ape,
        "robust_cap_p99_5": cap,
        "robust_rmse": robust_rmse,
        "robust_mae": robust_mae,
    }


def prediction_sanity_stats(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    def _stats(arr):
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    return _stats(y_true), _stats(y_pred)


def scale_warning(y_true_stats, y_pred_stats):
    p99_true = y_true_stats["p99"]
    p99_pred = y_pred_stats["p99"]
    if p99_pred <= 0:
        return "WARNING: y_pred p99 <= 0; predictions may be on the wrong scale."
    if p99_true > 0:
        ratio = p99_pred / p99_true
        if ratio < 0.1 or ratio > 10.0:
            return (
                "WARNING: y_pred scale seems off (p99_pred / p99_true = "
                f"{ratio:.3f})."
            )
    return ""


def neighbourhood_summary(y_true, y_pred, neighbourhood_code):
    df = pd.DataFrame(
        {
            "NEIGHBOURHOOD_CODE": neighbourhood_code,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df["abs_error"] = (df["y_pred"] - df["y_true"]).abs()
    df["sq_error"] = (df["y_pred"] - df["y_true"]) ** 2
    mask = df["y_true"] > 0
    df["ape"] = np.where(mask, df["abs_error"] / df["y_true"], np.nan)

    summary = (
        df.groupby("NEIGHBOURHOOD_CODE", dropna=False)
        .agg(
            count=("y_true", "size"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
            median_ape=("ape", "median"),
        )
        .reset_index()
    )
    return summary
