from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TargetEncoderSpec:
    mapping: dict[str, float]
    global_mean: float


def fit_target_encoders(
    X_train: pd.DataFrame,
    y_train_target: pd.Series,
    cols: list[str],
) -> dict[str, TargetEncoderSpec]:
    """
    Fit simple mean target encoders using TRAIN ONLY.

    Parameters
    ----------
    X_train:
        Training feature frame.
    y_train_target:
        Target series aligned with X_train (e.g., log1p target).
    cols:
        Categorical columns to encode.
    """
    out: dict[str, TargetEncoderSpec] = {}
    global_mean = float(y_train_target.mean())

    for col in cols:
        if col not in X_train.columns:
            continue
        key = X_train[col].fillna("Unknown").astype(str)
        mapping = (
            pd.DataFrame({"_key": key, "_target": y_train_target.values})
            .groupby("_key", dropna=False)["_target"]
            .mean()
            .to_dict()
        )
        out[col] = TargetEncoderSpec(mapping=mapping, global_mean=global_mean)

    return out


def apply_target_encoders(
    X: pd.DataFrame,
    encoders: dict[str, TargetEncoderSpec],
    suffix: str = "_te",
) -> pd.DataFrame:
    """
    Apply fitted target encoders to a dataframe.
    Unseen categories fall back to the training global mean.
    """
    out = X.copy()
    for col, spec in encoders.items():
        if col not in out.columns:
            continue
        key = out[col].fillna("Unknown").astype(str)
        out[f"{col}{suffix}"] = key.map(spec.mapping).fillna(spec.global_mean).astype(float)
    return out
