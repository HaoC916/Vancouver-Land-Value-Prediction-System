from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_model_plots(y_true, y_pred, summary_df, out_dir: Path, model_name: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_pred - y_true

    # Scatter plot (log1p axes)
    y_true_log = np.log1p(np.maximum(y_true, 0))
    y_pred_log = np.log1p(np.maximum(y_pred, 0))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_log, y_pred_log, s=4, alpha=0.3)
    plt.xlabel("log1p(Actual CURRENT_LAND_VALUE)")
    plt.ylabel("log1p(Predicted CURRENT_LAND_VALUE)")
    plt.title(f"{model_name}: Actual vs Predicted (log1p)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_scatter_log.png", dpi=150)
    plt.close()

    # Residual histogram (clipped for readability)
    lo = np.percentile(residuals, 1)
    hi = np.percentile(residuals, 99)
    residuals_clip = np.clip(residuals, lo, hi)
    plt.figure(figsize=(6, 4))
    plt.hist(residuals_clip, bins=50, alpha=0.8)
    plt.xlabel("Residual (y_pred - y_true), clipped")
    plt.ylabel("Count")
    plt.title(f"{model_name}: Residuals (clipped)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_residuals_clip.png", dpi=150)
    plt.close()

    # Top 20 neighbourhoods by MAE
    top20 = summary_df.sort_values("mean_abs_error", ascending=False).head(20)
    plt.figure(figsize=(8, 5))
    plt.bar(top20["NEIGHBOURHOOD_CODE"].astype(str), top20["mean_abs_error"])
    plt.xlabel("NEIGHBOURHOOD_CODE")
    plt.ylabel("Mean Absolute Error")
    plt.title(f"{model_name}: Top 20 Neighbourhoods by MAE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_neighbourhood_mae_top20.png", dpi=150)
    plt.close()


def save_baseline_plots(y_true, y_pred, summary_df, out_dir: Path):
    save_model_plots(y_true, y_pred, summary_df, out_dir, "baseline")
