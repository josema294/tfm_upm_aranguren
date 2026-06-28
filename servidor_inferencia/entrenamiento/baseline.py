from __future__ import annotations

import numpy as np


def window_features(window: np.ndarray) -> dict[str, float]:
    """Compute baseline statistical features for one [samples, channels] window."""
    if window.ndim != 2 or window.shape[1] != 3:
        raise ValueError(f"Expected window shape [samples, 3], got {window.shape}")

    magnitude = np.linalg.norm(window, axis=1)
    centered = window - np.mean(window, axis=0, keepdims=True)

    features: dict[str, float] = {
        "mag_rms": float(np.sqrt(np.mean(magnitude**2))),
        "mag_mean": float(np.mean(magnitude)),
        "mag_std": float(np.std(magnitude)),
        "mag_peak": float(np.max(np.abs(magnitude))),
        "energy": float(np.mean(np.sum(centered**2, axis=1))),
    }

    for axis, name in enumerate(("x", "y", "z")):
        values = window[:, axis]
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))
        features[f"{name}_rms"] = float(np.sqrt(np.mean(values**2)))
        features[f"{name}_peak"] = float(np.max(np.abs(values)))

    return features


def percentile_threshold(scores: np.ndarray, percentile: float = 99.0) -> float:
    if scores.size == 0:
        raise ValueError("Cannot compute threshold from an empty score array")
    return float(np.percentile(scores, percentile))
