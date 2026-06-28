from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BASE_COLUMNS = ["acc_x_g", "acc_y_g", "acc_z_g"]
DEFAULT_FEATURE_COLUMNS = [
    "acc_x_g",
    "acc_y_g",
    "acc_z_g",
    "acc_magnitude_g",
    "delta_x_g",
    "delta_y_g",
    "delta_z_g",
]


class SlipCNN(nn.Module):
    def __init__(self, in_channels: int = len(DEFAULT_FEATURE_COLUMNS), dropout: float = 0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        pooled = torch.cat([F.adaptive_avg_pool1d(h, 1), F.adaptive_max_pool1d(h, 1)], dim=1).squeeze(-1)
        return self.head(pooled).squeeze(-1)


class MultiScaleSlipCNN(nn.Module):
    def __init__(self, scales: tuple[int, ...], in_channels: int = len(DEFAULT_FEATURE_COLUMNS), dropout: float = 0.20):
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.GELU(),
                    nn.Conv1d(32, 48, kernel_size=3, padding=1),
                    nn.BatchNorm1d(48),
                    nn.GELU(),
                )
                for _ in scales
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(len(scales) * 96, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        pooled = []
        for branch, x in zip(self.branches, xs):
            h = branch(x)
            pooled.append(torch.cat([F.adaptive_avg_pool1d(h, 1), F.adaptive_max_pool1d(h, 1)], dim=1).squeeze(-1))
        return self.head(torch.cat(pooled, dim=1)).squeeze(-1)


def build_feature_array(samples: list[dict], feature_columns: list[str]) -> np.ndarray:
    base = np.asarray([[row[column] for column in BASE_COLUMNS] for row in samples], dtype=np.float32)
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
        raise ValueError(f"Unsupported slip feature columns: {missing}")
    return np.stack([derived[column] for column in feature_columns], axis=0).astype(np.float32)


class SlipDetector:
    def __init__(self, model_path: str | Path, device: str = "auto", threshold_override: float | None = None):
        self.model_path = Path(model_path)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold_override = threshold_override
        self.model: SlipCNN | MultiScaleSlipCNN | None = None
        self.threshold: float | None = None
        self.window_size: int | None = None
        self.scales: tuple[int, ...] | None = None
        self.model_kind = "single_scale"
        self.feature_columns: list[str] = DEFAULT_FEATURE_COLUMNS.copy()
        self.normalization: dict | None = None
        self.artifact_metadata: dict = {}
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Slip model not found: {self.model_path}")

        artifact = torch.load(self.model_path, map_location=self.device)
        if not isinstance(artifact, dict) or "model_state_dict" not in artifact:
            raise ValueError("Slip artifact must include model_state_dict")

        self.normalization = artifact.get("normalization")
        if self.normalization is None:
            raise ValueError("Slip artifact must include normalization mean/std")

        self.feature_columns = list(artifact.get("feature_columns", DEFAULT_FEATURE_COLUMNS))
        scales = artifact.get("scales")
        self.scales = tuple(int(scale) for scale in scales) if scales else None
        self.model_kind = "multi_scale" if self.scales else "single_scale"
        self.window_size = max(self.scales) if self.scales else int(artifact.get("window_size", 50))
        self.threshold = float(self.threshold_override if self.threshold_override is not None else artifact.get("threshold", 0.5))
        self.artifact_metadata = {
            "architecture": artifact.get("architecture", "MultiScaleSlipCNN" if self.scales else "SlipCNN"),
            "train_run": artifact.get("train_run"),
            "test_run": artifact.get("test_run"),
            "default_threshold": artifact.get("default_threshold"),
            "training_mode": artifact.get("training_mode"),
            "slip_scales": list(self.scales) if self.scales else None,
        }

        if self.scales:
            self.model = MultiScaleSlipCNN(scales=self.scales, in_channels=len(self.feature_columns)).to(self.device)
        else:
            self.model = SlipCNN(in_channels=len(self.feature_columns)).to(self.device)
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def _normalize_features(self, features: np.ndarray, scale: int | None = None) -> np.ndarray:
        if self.normalization is None:
            raise RuntimeError("Slip model is not loaded")
        if scale is not None and str(scale) in self.normalization:
            params = self.normalization[str(scale)]
            mean = np.asarray(params["mean"], dtype=np.float32)[:, None]
            std = np.asarray(params["std"], dtype=np.float32)[:, None]
        else:
            mean = np.asarray(self.normalization["mean"], dtype=np.float32)[:, None]
            std = np.asarray(self.normalization["std"], dtype=np.float32)[:, None]
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        return ((features - mean) / std).astype(np.float32)

    def _samples_to_tensor(self, samples: list[dict]) -> torch.Tensor:
        if self.window_size is None:
            raise RuntimeError("Slip model is not loaded")
        if len(samples) < self.window_size:
            raise ValueError(f"Slip detector needs at least {self.window_size} samples, received {len(samples)}")

        recent_samples = samples[-self.window_size :]
        features = build_feature_array(recent_samples, self.feature_columns)
        normalized = self._normalize_features(features)
        return torch.as_tensor(normalized[None, :, :], dtype=torch.float32, device=self.device)

    def _samples_to_multiscale_tensors(self, samples: list[dict]) -> list[torch.Tensor]:
        if self.scales is None or self.window_size is None:
            raise RuntimeError("Slip multi-scale model is not loaded")
        if len(samples) < self.window_size:
            raise ValueError(f"Slip detector needs at least {self.window_size} samples, received {len(samples)}")

        recent_samples = samples[-self.window_size :]
        tensors = []
        for scale in self.scales:
            left = (self.window_size - scale) // 2
            scale_samples = recent_samples[left : left + scale]
            features = build_feature_array(scale_samples, self.feature_columns)
            normalized = self._normalize_features(features, scale)
            tensors.append(torch.as_tensor(normalized[None, :, :], dtype=torch.float32, device=self.device))
        return tensors

    def predict(self, samples: list[dict], quality: dict) -> dict:
        assert self.model is not None
        assert self.threshold is not None
        assert self.window_size is not None

        with torch.no_grad():
            if self.model_kind == "multi_scale":
                tensors = self._samples_to_multiscale_tensors(samples)
                probability = float(torch.sigmoid(self.model(*tensors)).detach().cpu().numpy()[0])
            else:
                tensor = self._samples_to_tensor(samples)
                probability = float(torch.sigmoid(self.model(tensor)).detach().cpu().numpy()[0])

        threshold = float(self.threshold)
        score = probability / threshold if threshold else probability
        status = "anomaly" if probability >= threshold else "normal"
        if quality["lost_samples"] > 0 or quality["samples_received"] < quality["samples_expected"]:
            status = "unreliable"

        recent_samples = samples[-self.window_size :]
        return {
            "status": status,
            "anomaly_score": score,
            "reconstruction_error": 0.0,
            "metadata": {
                "slip_probability": probability,
                "slip_threshold": threshold,
                "slip_window_size": self.window_size,
                "slip_seq_start": int(recent_samples[0]["seq"]),
                "slip_seq_end": int(recent_samples[-1]["seq"]),
                "slip_timestamp_start_ms": int(recent_samples[0]["timestamp_ms"]),
                "slip_timestamp_end_ms": int(recent_samples[-1]["timestamp_ms"]),
                **self.artifact_metadata,
            },
        }
