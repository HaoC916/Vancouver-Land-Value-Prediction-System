from pathlib import Path

from src.eval.metrics import neighbourhood_summary


def write_baseline_reports(y_true, y_pred, test_df, out_path):
    summary = neighbourhood_summary(
        y_true, y_pred, test_df["NEIGHBOURHOOD_CODE"].values
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    return summary
