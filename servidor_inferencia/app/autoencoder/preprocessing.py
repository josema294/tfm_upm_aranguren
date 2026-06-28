from __future__ import annotations

import numpy as np


FEATURE_COLUMNS = ["acc_x_g", "acc_y_g", "acc_z_g"]


def normalize_features(features: np.ndarray, min_vals: list[float], max_vals: list[float]) -> np.ndarray:
    min_arr = np.asarray(min_vals, dtype=np.float32)
    max_arr = np.asarray(max_vals, dtype=np.float32)
    denominator = np.where((max_arr - min_arr) == 0, 1.0, max_arr - min_arr)
    return (features.astype(np.float32) - min_arr) / denominator


def samples_to_feature_matrix(samples: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["acc_x_g"], row["acc_y_g"], row["acc_z_g"]] for row in samples],
        dtype=np.float32,
    )


def feature_matrix_to_model_input(features: np.ndarray) -> np.ndarray:
    """Convert [window_size, channels] to [1, channels, window_size]."""
    if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected feature matrix shape [n, {len(FEATURE_COLUMNS)}], got {features.shape}")
    return np.transpose(features[None, :, :], (0, 2, 1)).astype(np.float32)
