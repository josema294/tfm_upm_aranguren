#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASE_COLUMNS = ("acc_x_g", "acc_y_g", "acc_z_g")
STARTUP_IGNORE_S = 15.0
SLIP_CORE_HALF_WIDTH_S = 0.30
SLIP_TRANSITION_HALF_WIDTH_S = 0.65
OUTLIER_PEAK_ABS_X_G = 8.0
OUTLIER_X_RANGE_G = 8.0


class SlipCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
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
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        pooled = torch.cat([F.adaptive_avg_pool1d(h, 1), F.adaptive_max_pool1d(h, 1)], dim=1).squeeze(-1)
        return self.head(pooled).squeeze(-1)


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    rows: list[dict]
    feature_columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena una CNN de ventana corta para la detección de anomalías por deslizamiento.")
    parser.add_argument("--normal-csv", default="../datos/brutos/real_movement_004.csv")
    parser.add_argument("--slip-csv", default="../datos/brutos/real_slip_only_001.csv")
    parser.add_argument("--labels-csv", default="../datos/etiquetas/real_slip_only_001_windows.csv")
    parser.add_argument("--output-dir", default="../datos/analisis/slip_only/cnn_classifier")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--window-sizes", default="30,50,75")
    parser.add_argument("--window-step", type=int, default=10)
    parser.add_argument("--phase-s", type=float, default=None)
    parser.add_argument("--period-s", type=float, default=None)
    parser.add_argument("--train-cycle-ratio", type=float, default=0.6)
    parser.add_argument("--max-normal-windows", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def read_samples(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        return [
            {
                "seq": int(row["seq"]),
                "timestamp_ms": int(row["timestamp_ms"]),
                "acc_x_g": float(row["acc_x_g"]),
                "acc_y_g": float(row["acc_y_g"]),
                "acc_z_g": float(row["acc_z_g"]),
            }
            for row in csv.DictReader(file)
        ]


def read_timing(labels_csv: Path, phase_s: float | None, period_s: float | None) -> tuple[float, float]:
    if phase_s is not None and period_s is not None:
        return phase_s, period_s
    with labels_csv.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("phase_s") and row.get("period_s"):
                return float(row["phase_s"]), float(row["period_s"])
    raise ValueError("Se requieren 'phase_s' y 'period_s' si el archivo CSV de etiquetas no incluye metadatos temporales.")


def phase_offset(relative_center_s: float, phase_s: float, period_s: float) -> float:
    return ((relative_center_s - phase_s + period_s / 2) % period_s) - period_s / 2


def base_array(samples: list[dict]) -> np.ndarray:
    return np.asarray([[row[col] for col in BASE_COLUMNS] for row in samples], dtype=np.float32)


def derived_features(window: np.ndarray) -> np.ndarray:
    mag = np.sqrt(np.sum(window**2, axis=1, keepdims=True))
    delta = np.diff(window, axis=0, prepend=window[:1])
    # Vector de características: [x, y, z, magnitud, dx, dy, dz]
    return np.concatenate([window, mag, delta], axis=1).astype(np.float32)


def label_slip_window(
    window_rows: list[dict],
    first_timestamp_ms: int,
    phase_s: float,
    period_s: float,
) -> tuple[int, str, float, int]:
    center_timestamp_ms = (int(window_rows[0]["timestamp_ms"]) + int(window_rows[-1]["timestamp_ms"])) / 2
    relative_center_s = (center_timestamp_ms - first_timestamp_ms) / 1000
    offset_s = phase_offset(relative_center_s, phase_s, period_s)
    cycle_index = int(round((relative_center_s - phase_s) / period_s))
    x = np.asarray([row["acc_x_g"] for row in window_rows], dtype=np.float32)
    if relative_center_s < STARTUP_IGNORE_S:
        return -1, "ignore_startup", relative_center_s, cycle_index
    if float(np.max(np.abs(x))) >= OUTLIER_PEAK_ABS_X_G or float(np.ptp(x)) >= OUTLIER_X_RANGE_G:
        return -1, "ignore_outlier_impact", relative_center_s, cycle_index
    if abs(offset_s) <= SLIP_CORE_HALF_WIDTH_S:
        return 1, "slip", relative_center_s, cycle_index
    if abs(offset_s) <= SLIP_TRANSITION_HALF_WIDTH_S:
        return -1, "ignore_transition", relative_center_s, cycle_index
    return 0, "normal_slip_run", relative_center_s, cycle_index


def build_slip_dataset(samples: list[dict], window_size: int, window_step: int, phase_s: float, period_s: float) -> Dataset:
    values = base_array(samples)
    first_timestamp_ms = int(samples[0]["timestamp_ms"])
    x: list[np.ndarray] = []
    y: list[int] = []
    rows: list[dict] = []
    for start in range(0, len(samples) - window_size + 1, window_step):
        window_rows = samples[start : start + window_size]
        binary, label, relative_center_s, cycle_index = label_slip_window(window_rows, first_timestamp_ms, phase_s, period_s)
        if binary < 0:
            continue
        x.append(derived_features(values[start : start + window_size]).T)
        y.append(binary)
        rows.append(
            {
                "source": "slip",
                "label": label,
                "relative_center_s": relative_center_s,
                "cycle_index": cycle_index,
                "seq_start": int(window_rows[0]["seq"]),
                "seq_end": int(window_rows[-1]["seq"]),
            }
        )
    return Dataset(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64), rows, feature_columns())


def build_normal_dataset(samples: list[dict], window_size: int, window_step: int, max_windows: int, seed: int) -> Dataset:
    if max_windows <= 0:
        return Dataset(
            x=np.empty((0, len(feature_columns()), window_size), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            rows=[],
            feature_columns=feature_columns(),
        )
    values = base_array(samples)
    starts = list(range(0, len(samples) - window_size + 1, window_step))
    rng = np.random.default_rng(seed)
    if len(starts) > max_windows:
        starts = sorted(rng.choice(starts, size=max_windows, replace=False).tolist())
    x: list[np.ndarray] = []
    rows: list[dict] = []
    for start in starts:
        x.append(derived_features(values[start : start + window_size]).T)
        window_rows = samples[start : start + window_size]
        rows.append(
            {
                "source": "normal_reference",
                "label": "normal_reference",
                "relative_center_s": math.nan,
                "cycle_index": -999,
                "seq_start": int(window_rows[0]["seq"]),
                "seq_end": int(window_rows[-1]["seq"]),
            }
        )
    return Dataset(np.asarray(x, dtype=np.float32), np.zeros(len(rows), dtype=np.int64), rows, feature_columns())


def feature_columns() -> list[str]:
    return ["acc_x_g", "acc_y_g", "acc_z_g", "acc_magnitude_g", "delta_x_g", "delta_y_g", "delta_z_g"]


def combine(a: Dataset, b: Dataset) -> Dataset:
    if len(b.rows) == 0:
        return a
    if len(a.rows) == 0:
        return b
    return Dataset(
        x=np.concatenate([a.x, b.x], axis=0),
        y=np.concatenate([a.y, b.y], axis=0),
        rows=[*a.rows, *b.rows],
        feature_columns=a.feature_columns,
    )


def split_indices(dataset: Dataset, train_cycle_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cycles = sorted({row["cycle_index"] for row in dataset.rows if row["source"] == "slip" and row["cycle_index"] >= 0})
    split_at = int(len(cycles) * train_cycle_ratio)
    train_cycles = set(cycles[:split_at])

    normal_indices = [idx for idx, row in enumerate(dataset.rows) if row["source"] == "normal_reference"]
    rng = np.random.default_rng(seed)
    rng.shuffle(normal_indices)
    normal_train = set(normal_indices[: int(len(normal_indices) * train_cycle_ratio)])

    train: list[int] = []
    test: list[int] = []
    for idx, row in enumerate(dataset.rows):
        if row["source"] == "normal_reference":
            (train if idx in normal_train else test).append(idx)
        elif row["cycle_index"] in train_cycles:
            train.append(idx)
        else:
            test.append(idx)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def normalize(dataset: Dataset, train_idx: np.ndarray) -> tuple[np.ndarray, dict]:
    train = dataset.x[train_idx]
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((dataset.x - mean) / std).astype(np.float32), {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, balanced: bool) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    if not balanced:
        return DataLoader(tensors, batch_size=batch_size, shuffle=False)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    weights = np.where(y == 1, 1.0 / max(counts[1], 1), 1.0 / max(counts[0], 1))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
    return DataLoader(tensors, batch_size=batch_size, sampler=sampler)


def predict_scores(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size, balanced=False)
    with torch.no_grad():
        for xb, _ in loader:
            scores.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(scores)


def metrics_at(scores: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (scores >= threshold).astype(np.int64)
    return {
        "threshold": threshold,
        "total": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "false_positives": int(np.sum((y == 0) & (pred == 1))),
        "false_negatives": int(np.sum((y == 1) & (pred == 0))),
    }


def best_threshold(scores: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.99, 100)))
    best = metrics_at(scores, y, 0.5)
    for threshold in candidates:
        current = metrics_at(scores, y, float(threshold))
        if (current["f1"], current["recall"], -current["false_positives"]) > (
            best["f1"],
            best["recall"],
            -best["false_positives"],
        ):
            best = current
    return best


def event_metrics(scores: np.ndarray, dataset: Dataset, indices: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = scores >= threshold
    expected: set[int] = set()
    detected: set[int] = set()
    fp_windows = 0
    for local_idx, global_idx in enumerate(indices.tolist()):
        row = dataset.rows[global_idx]
        if row["source"] != "slip":
            if pred[local_idx]:
                fp_windows += 1
            continue
        if dataset.y[global_idx] == 1:
            expected.add(int(row["cycle_index"]))
            if pred[local_idx]:
                detected.add(int(row["cycle_index"]))
        elif pred[local_idx]:
            fp_windows += 1
    return {
        "expected_events": len(expected),
        "detected_events": len(detected),
        "event_recall": len(detected) / len(expected) if expected else 0.0,
        "event_false_positive_windows": fp_windows,
    }


def train_one(dataset: Dataset, train_idx: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace) -> tuple[SlipCNN, dict, dict]:
    x_norm, normalization = normalize(dataset, train_idx)
    model = SlipCNN(in_channels=x_norm.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_loader = make_loader(x_norm[train_idx], dataset.y[train_idx], args.batch_size, balanced=True)

    for _epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

    test_scores = predict_scores(model, x_norm[test_idx], args.batch_size)
    train_scores = predict_scores(model, x_norm[train_idx], args.batch_size)
    test_default = metrics_at(test_scores, dataset.y[test_idx], 0.5)
    test_best = best_threshold(test_scores, dataset.y[test_idx])
    train_default = metrics_at(train_scores, dataset.y[train_idx], 0.5)
    event_default = event_metrics(test_scores, dataset, test_idx, 0.5)
    event_best = event_metrics(test_scores, dataset, test_idx, float(test_best["threshold"]))
    summary = {
        "train_default": train_default,
        "test_default": test_default,
        "test_best": test_best,
        "event_default": event_default,
        "event_best": event_best,
    }
    artifact = {"normalization": normalization}
    return model, artifact, summary


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    normal_samples = read_samples(resolve(args.normal_csv))
    slip_samples = read_samples(resolve(args.slip_csv))
    phase_s, period_s = read_timing(resolve(args.labels_csv), args.phase_s, args.period_s)
    output_dir = resolve(args.output_dir)
    models_dir = resolve(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    experiments: list[dict] = []
    for window_size in [int(value.strip()) for value in args.window_sizes.split(",") if value.strip()]:
        slip_dataset = build_slip_dataset(slip_samples, window_size, args.window_step, phase_s, period_s)
        normal_dataset = build_normal_dataset(normal_samples, window_size, args.window_step, args.max_normal_windows, args.random_seed)
        dataset = combine(slip_dataset, normal_dataset)
        train_idx, test_idx = split_indices(dataset, args.train_cycle_ratio, args.random_seed)
        model, artifact, summary = train_one(dataset, train_idx, test_idx, args)

        run_name = f"slip_cnn_w{window_size}_s{args.window_step}"
        model_path = models_dir / f"{run_name}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_columns": dataset.feature_columns,
                "window_size": window_size,
                "window_step": args.window_step,
                "phase_s": phase_s,
                "period_s": period_s,
                "normalization": artifact["normalization"],
                "architecture": "SlipCNN",
            },
            model_path,
        )

        row = {
            "run_name": run_name,
            "model_path": str(model_path),
            "window_size": window_size,
            "window_step": args.window_step,
            "train_windows": int(len(train_idx)),
            "test_windows": int(len(test_idx)),
            **{f"train_{key}": value for key, value in summary["train_default"].items()},
            **{f"test_{key}": value for key, value in summary["test_default"].items()},
            **{f"best_{key}": value for key, value in summary["test_best"].items()},
            **{f"default_{key}": value for key, value in summary["event_default"].items()},
            **{f"best_event_{key}": value for key, value in summary["event_best"].items()},
        }
        experiments.append(row)
        print(
            f"{run_name:18} F1_test={row['test_f1']:.3f} Precision={row['test_precision']:.3f} "
            f"Recall={row['test_recall']:.3f} Eventos={row['default_detected_events']}/{row['default_expected_events']} "
            f"F1_best={row['best_f1']:.3f} Recall_best={row['best_recall']:.3f} "
            f"Eventos_best={row['best_event_detected_events']}/{row['best_event_expected_events']}",
            flush=True,
        )

    summary_csv = output_dir / "slip_cnn_classifier_summary.csv"
    with summary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(experiments[0].keys()))
        writer.writeheader()
        writer.writerows(experiments)

    summary_json = output_dir / "slip_cnn_classifier_summary.json"
    summary_json.write_text(json.dumps({"experiments": experiments}, indent=2), encoding="utf-8")
    print(f"Resumen de métricas guardado en: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
