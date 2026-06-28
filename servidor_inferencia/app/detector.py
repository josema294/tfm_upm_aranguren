from __future__ import annotations

import numpy as np


class PlaceholderDetector:
    """Temporary detector until the autoencoder is trained.

    It returns a simple vibration energy score so the VPS -> PC -> VPS loop can
    be tested end to end before deep learning is introduced.
    """

    def __init__(self, anomaly_threshold: float = 0.25):
        self.anomaly_threshold = anomaly_threshold

    def predict(self, window: np.ndarray, quality: dict) -> dict:
        centered = window - np.mean(window, axis=0, keepdims=True)
        vibration_energy = float(np.sqrt(np.mean(centered**2)))
        status = "anomaly" if vibration_energy >= self.anomaly_threshold else "normal"

        if quality["lost_samples"] > 0 or quality["samples_received"] < quality["samples_expected"]:
            status = "unreliable"

        return {
            "status": status,
            "anomaly_score": vibration_energy,
            "reconstruction_error": None,
        }
