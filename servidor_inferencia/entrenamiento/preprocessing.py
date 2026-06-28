from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TFM_COLUMNS = ["seq", "timestamp_ms", "acc_x_g", "acc_y_g", "acc_z_g"]
MLOPS_COLUMNS = ["timestamp", "accel_x", "accel_y", "accel_z"]


def read_vibration_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = set(df.columns)

    if set(TFM_COLUMNS).issubset(columns):
        return df[TFM_COLUMNS].copy()

    if set(MLOPS_COLUMNS).issubset(columns):
        converted = pd.DataFrame()
        converted["seq"] = np.arange(len(df), dtype=np.int64)
        converted["timestamp_ms"] = np.rint(df["timestamp"].astype(float) * 1000.0).astype(np.int64)
        converted["acc_x_g"] = df["accel_x"].astype(float)
        converted["acc_y_g"] = df["accel_y"].astype(float)
        converted["acc_z_g"] = df["accel_z"].astype(float)
        return converted

    raise ValueError(
        "CSV must include either TFM columns "
        f"{TFM_COLUMNS} or MLOps columns {MLOPS_COLUMNS}. Found: {list(df.columns)}"
    )


def make_windows(values: np.ndarray, window_size: int = 100, window_step: int = 50) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"Expected values shape [samples, 3], got {values.shape}")
    if len(values) < window_size:
        return np.empty((0, window_size, 3), dtype=np.float32)

    windows = [values[start : start + window_size] for start in range(0, len(values) - window_size + 1, window_step)]
    return np.asarray(windows, dtype=np.float32)


def minmax_fit(values: np.ndarray) -> dict[str, list[float]]:
    min_vals = np.min(values, axis=(0, 1)).astype(float)
    max_vals = np.max(values, axis=(0, 1)).astype(float)
    return {"min_vals": min_vals.tolist(), "max_vals": max_vals.tolist()}


def minmax_transform(values: np.ndarray, normalization: dict[str, list[float]]) -> np.ndarray:
    min_vals = np.asarray(normalization["min_vals"], dtype=np.float32)
    max_vals = np.asarray(normalization["max_vals"], dtype=np.float32)
    denominator = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
    return ((values.astype(np.float32) - min_vals) / denominator).astype(np.float32)
