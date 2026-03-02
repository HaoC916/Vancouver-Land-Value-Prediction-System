from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _sample_points(x: np.ndarray, y: np.ndarray, max_points: int = 60000, seed: int = 42):
    """Downsample points for readability and speed."""
    n = len(x)
    if n <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def save_model_plots(y_true, y_pred, summary_df, out_dir: Path, model_name: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    # ===== Scatter plot (log1p axes) + y=x reference line =====
    y_true_log = np.log1p(np.maximum(y_true, 0))
    y_pred_log = np.log1p(np.maximum(y_pred, 0))

    x, y = _sample_points(y_true_log, y_pred_log, max_points=60000, seed=42)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=3, alpha=0.18)

    # y=x line (on log scale)
    lo = np.nanpercentile(np.concatenate([x, y]), 1)
    hi = np.nanpercentile(np.concatenate([x, y]), 99)
    plt.plot([lo, hi], [lo, hi], linewidth=2, alpha=0.8)

    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.xlabel("log1p(Actual CURRENT_LAND_VALUE)")
    plt.ylabel("log1p(Predicted CURRENT_LAND_VALUE)")
    plt.title(f"{model_name}: Actual vs Predicted (log1p)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_scatter_log.png", dpi=180)
    plt.close()

    # ===== Residual histogram (clipped for readability) =====
    lo_r = np.percentile(residuals, 1)
    hi_r = np.percentile(residuals, 99)
    residuals_clip = np.clip(residuals, lo_r, hi_r)

    plt.figure(figsize=(6, 4))
    plt.hist(residuals_clip, bins=60, alpha=0.85)
    plt.xlabel("Residual (y_pred - y_true), clipped (p1–p99)")
    plt.ylabel("Count")
    plt.title(f"{model_name}: Residuals (clipped)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_residuals_clip.png", dpi=180)
    plt.close()

    # ===== Top 20 neighbourhoods by MAE =====
    top20 = summary_df.sort_values("mean_abs_error", ascending=False).head(20)
    plt.figure(figsize=(8, 5))
    plt.bar(top20["NEIGHBOURHOOD_CODE"].astype(str), top20["mean_abs_error"])
    plt.xlabel("NEIGHBOURHOOD_CODE")
    plt.ylabel("Mean Absolute Error")
    plt.title(f"{model_name}: Top 20 Neighbourhoods by MAE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_neighbourhood_mae_top20.png", dpi=180)
    plt.close()


def save_baseline_plots(y_true, y_pred, summary_df, out_dir: Path):
    save_model_plots(y_true, y_pred, summary_df, out_dir, "baseline")