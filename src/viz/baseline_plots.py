from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def _sample_points(x: np.ndarray, y: np.ndarray, max_points: int = 60000, seed: int = 42):
    n = len(x)
    if n <= max_points:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return x[idx], y[idx]


def _fmt_dollars_millions(x, pos):
    # show ticks like $0.5M, $2M
    return f"${x/1_000_000:.1f}M"


def save_model_plots(y_true, y_pred, summary_df, out_dir: Path, model_name: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_pred - y_true

    # =========================
    # Public-friendly Plot 1: Prediction vs Official (log scale, but human labels)
    # =========================
    x, y = _sample_points(y_true, y_pred, max_points=60000, seed=42)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=3, alpha=0.18)

    # Use log scale to handle heavy tail, but keep labels human-friendly
    plt.xscale("log")
    plt.yscale("log")

    # Perfect match line
    lo = np.nanpercentile(np.concatenate([x, y]), 1)
    hi = np.nanpercentile(np.concatenate([x, y]), 99)
    plt.plot([lo, hi], [lo, hi], linewidth=2, alpha=0.8, color="red")

    plt.xlabel("Official land value (CAD)  — log scale")
    plt.ylabel("Model estimate (CAD)  — log scale")
    plt.title("How close are our predictions to the official land values?")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_scatter_public.png", dpi=180)
    plt.close()

    # =========================
    # Public-friendly Plot 2: Error distribution (clipped) in $M
    # =========================
    lo_r = np.percentile(residuals, 1)
    hi_r = np.percentile(residuals, 99)
    residuals_clip = np.clip(residuals, lo_r, hi_r)

    plt.figure(figsize=(7, 4))
    plt.hist(residuals_clip, bins=60, alpha=0.85)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(_fmt_dollars_millions))
    plt.xlabel("Error (estimate − official value)")
    plt.ylabel("Number of properties")
    plt.title("How big are the errors? (Most properties, extreme cases clipped)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_error_distribution_public.png", dpi=180)
    plt.close()

    # =========================
    # Public-friendly Plot 3: Neighbourhood difficulty (Top 20)
    # =========================
    top20 = summary_df.sort_values("mean_abs_error", ascending=False).head(20)

    plt.figure(figsize=(9, 5))
    plt.bar(top20["NEIGHBOURHOOD_CODE"].astype(str), top20["mean_abs_error"])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(_fmt_dollars_millions))
    plt.xlabel("Neighbourhood (code)")
    plt.ylabel("Typical error (CAD)")
    plt.title("Which neighbourhoods are harder to estimate? (Top 20)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_neighbourhood_difficulty_public.png", dpi=180)
    plt.close()

    # =========================
    # Keep your existing technical plots (optional)
    # If you still want them, keep your old code below.
    # =========================