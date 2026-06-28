from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import ConvVAE


BASE_FEATURE_COLUMNS = ["acc_x_g", "acc_y_g", "acc_z_g"]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS


def build_feature_array(samples: list[dict], feature_columns: list[str]) -> np.ndarray:
    base = np.asarray(
        [[row[col] for col in BASE_FEATURE_COLUMNS] for row in samples],
        dtype=np.float32,
    )
    derived = {
        "acc_x_g": base[:, 0],
        "acc_y_g": base[:, 1],
        "acc_z_g": base[:, 2],
        "acc_magnitude_g": np.sqrt(np.sum(base**2, axis=1)),
        "delta_x_g": np.diff(base[:, 0], prepend=base[0, 0]),
        "delta_y_g": np.diff(base[:, 1], prepend=base[0, 1]),
        "delta_z_g": np.diff(base[:, 2], prepend=base[0, 2]),
    }
    missing = [column for column in feature_columns if column not in derived]
    if missing:
        raise ValueError(f"Unsupported VAE feature columns: {missing}")
    return np.stack([derived[column] for column in feature_columns], axis=1).astype(np.float32)


class VaeDetector:
    def __init__(self, model_path: str | Path, device: str = "auto"):
        self.model_path = Path(model_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model: ConvVAE | None = None
        self.threshold: float | None = None
        self.normalization: dict | None = None
        self.config: dict = {}
        self.beta = 1e-3
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"VAE model not found: {self.model_path}")

        artifact = torch.load(self.model_path, map_location=self.device)
        if not isinstance(artifact, dict) or "model_state_dict" not in artifact:
            raise ValueError("VAE artifact must include model_state_dict")

        self.config = artifact.get("config", {})
        self.normalization = artifact.get("normalization")
        self.threshold = artifact.get("threshold")
        self.beta = float(self.config.get("beta", self.beta))

        if self.normalization is None:
            raise ValueError("VAE artifact must include normalization mean/std")
        if self.threshold is None:
            raise ValueError("VAE artifact must include threshold")

        self.feature_columns = list(self.config.get("feature_columns", FEATURE_COLUMNS))

        self.model = ConvVAE(
            in_channels=len(self.feature_columns),
            window_size=int(self.config.get("window_size", 100)),
            latent_dim=int(self.config.get("latent_dim", 16)),
        ).to(self.device)
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def _samples_to_tensor(self, samples: list[dict]) -> torch.Tensor:
        features = build_feature_array(samples, self.feature_columns)
        mean = np.asarray(self.normalization["mean"], dtype=np.float32)
        std = np.asarray(self.normalization["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        normalized = ((features - mean) / std).astype(np.float32)
        model_input = np.transpose(normalized[None, :, :], (0, 2, 1)).astype(np.float32)
        return torch.as_tensor(model_input, dtype=torch.float32, device=self.device)

    def predict(self, samples: list[dict], quality: dict) -> dict:
        assert self.model is not None
        assert self.threshold is not None

        tensor = self._samples_to_tensor(samples)
        with torch.no_grad():
            reconstruction, mu, logvar = self.model(tensor)
            reconstruction_error_t = torch.mean((tensor - reconstruction) ** 2, dim=(1, 2))
            kl_t = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            score_t = reconstruction_error_t + self.beta * kl_t

        reconstruction_error = float(reconstruction_error_t.detach().cpu().numpy()[0])
        kl = float(kl_t.detach().cpu().numpy()[0])
        vae_score = float(score_t.detach().cpu().numpy()[0])
        threshold = float(self.threshold)
        status = "anomaly" if vae_score > threshold else "normal"

        if quality["lost_samples"] > 0 or quality["samples_received"] < quality["samples_expected"]:
            status = "unreliable"

        input_norm = np.transpose(tensor.detach().cpu().numpy()[0], (1, 0))
        reconstruction_norm = np.transpose(reconstruction.detach().cpu().numpy()[0], (1, 0))

        return {
            "status": status,
            "anomaly_score": vae_score / threshold if threshold else None,
            "reconstruction_error": reconstruction_error,
            "metadata": {
                "vae_score": vae_score,
                "threshold": threshold,
                "kl": kl,
                "beta": self.beta,
                "input_norm": np.round(input_norm, 6).tolist(),
                "reconstruction_norm": np.round(reconstruction_norm, 6).tolist(),
            },
        }
