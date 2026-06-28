from __future__ import annotations

from .types import DetectorLike


class HybridVaeSlipDetector:
    def __init__(self, vae_detector: DetectorLike, slip_detector: DetectorLike):
        self.vae_detector = vae_detector
        self.slip_detector = slip_detector

    def predict(self, samples: list[dict], quality: dict) -> dict:
        vae_prediction = self.vae_detector.predict(samples, quality)
        slip_prediction = self.slip_detector.predict(samples, quality)

        vae_status = vae_prediction["status"]
        slip_status = slip_prediction["status"]
        if "unreliable" in {vae_status, slip_status}:
            status = "unreliable"
            anomaly_type = "unreliable_window"
        elif vae_status == "anomaly":
            status = "anomaly"
            anomaly_type = "impact_or_general"
        elif slip_status == "anomaly":
            status = "anomaly"
            anomaly_type = "slip"
        else:
            status = "normal"
            anomaly_type = "none"

        vae_score = float(vae_prediction.get("anomaly_score") or 0.0)
        slip_score = float(slip_prediction.get("anomaly_score") or 0.0)
        primary_score = max(vae_score, slip_score)
        reconstruction_error = float(vae_prediction.get("reconstruction_error") or 0.0)

        vae_metadata = dict(vae_prediction.get("metadata", {}))
        # The async viewer uses these arrays for reconstruction plots, but the
        # hybrid live detector should keep result payloads small.
        vae_metadata.pop("input_norm", None)
        vae_metadata.pop("reconstruction_norm", None)

        return {
            "status": status,
            "anomaly_score": primary_score,
            "reconstruction_error": reconstruction_error,
            "metadata": {
                "anomaly_type": anomaly_type,
                "vae": {
                    "status": vae_status,
                    "anomaly_score": vae_score,
                    **vae_metadata,
                },
                "slip": {
                    "status": slip_status,
                    "anomaly_score": slip_score,
                    **slip_prediction.get("metadata", {}),
                },
            },
        }
