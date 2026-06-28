from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import Conv1DAutoencoder
from .preprocessing import feature_matrix_to_model_input, normalize_features, samples_to_feature_matrix


class AutoencoderDetector:
    def __init__(self, model_path: str | Path, device: str = "auto"):
        self.model_path = Path(model_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model: Conv1DAutoencoder | None = None
        self.threshold: float | None = None
        self.normalization: dict | None = None
        self.config: dict = {}
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Autoencoder model not found: {self.model_path}")

        artifact = torch.load(self.model_path, map_location=self.device)
        config = artifact.get("config", {}) if isinstance(artifact, dict) else {}
        model_config = config.get("model", {})

        self.model = Conv1DAutoencoder(
            in_channels=model_config.get("in_channels", 3),
            filters=model_config.get("encoder_filters", [16, 32, 64]),
            kernel_size=model_config.get("kernel_size", 5),
        ).to(self.device)

        if isinstance(artifact, dict) and "model_state_dict" in artifact:
            self.model.load_state_dict(artifact["model_state_dict"])
            self.normalization = artifact.get("normalization")
            self.threshold = artifact.get("threshold")
            self.config = config
        else:
            self.model.load_state_dict(artifact)

        if self.normalization is None:
            raise ValueError("Autoencoder artifact must include normalization min_vals/max_vals")
        if self.threshold is None:
            raise ValueError("Autoencoder artifact must include threshold")

        self.model.eval()

    def predict(self, samples: list[dict], quality: dict) -> dict:
        assert self.model is not None
        assert self.normalization is not None
        assert self.threshold is not None

        features = samples_to_feature_matrix(samples)
        features_norm = normalize_features(
            features,
            self.normalization["min_vals"],
            self.normalization["max_vals"],
        )
        model_input = feature_matrix_to_model_input(features_norm)
        tensor = torch.as_tensor(model_input, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            reconstruction = self.model(tensor)
            mse = torch.mean((tensor - reconstruction) ** 2, dim=(1, 2))

        reconstruction_error = float(mse.detach().cpu().numpy()[0])
        input_norm = np.transpose(tensor.detach().cpu().numpy()[0], (1, 0))
        reconstruction_norm = np.transpose(reconstruction.detach().cpu().numpy()[0], (1, 0))
        status = "anomaly" if reconstruction_error > float(self.threshold) else "normal"

        if quality["lost_samples"] > 0 or quality["samples_received"] < quality["samples_expected"]:
            status = "unreliable"

        return {
            "status": status,
            "anomaly_score": reconstruction_error / float(self.threshold) if self.threshold else None,
            "reconstruction_error": reconstruction_error,
            "metadata": {
                "input_norm": np.round(input_norm, 6).tolist(),
                "reconstruction_norm": np.round(reconstruction_norm, 6).tolist(),
            },
        }
