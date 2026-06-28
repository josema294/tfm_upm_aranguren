#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASE_COLUMNS = ["acc_x_g", "acc_y_g", "acc_z_g"]
FEATURE_COLUMNS = ["acc_x_g", "acc_y_g", "acc_z_g", "acc_magnitude_g", "delta_x_g", "delta_y_g", "delta_z_g"]
POSITIVE_LABEL = "manual_slip_core"
NEGATIVE_LABEL = "manual_normal_between"


class SlipCNN(nn.Module):
    def __init__(self, in_channels: int = len(FEATURE_COLUMNS), dropout: float = 0.15):
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


@dataclass
class ManualRun:
    run_id: str
    raw_csv: Path
    labels_csv: Path
    x: np.ndarray
    y: np.ndarray
    rows: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y evalúa una rama CNN supervisada utilizando etiquetas manuales de deslizamiento.")
    parser.add_argument("--run-001-raw", default="../datos/brutos/real_slip_manual_001.csv")
    parser.add_argument("--run-001-labels", default="../datos/etiquetas/real_slip_manual_001_windows.csv")
    parser.add_argument("--run-002-raw", default="../datos/brutos/real_slip_manual_002.csv")
    parser.add_argument("--run-002-labels", default="../datos/etiquetas/real_slip_manual_002_windows.csv")
    parser.add_argument("--output-dir", default="../datos/analisis/slip_manual/supervised_cnn")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model-name", default="slip_cnn_supervised_w50_s10_v1")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="tfm-railway-anomaly")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="supervised-slip")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def read_raw(path: Path) -> dict[int, dict]:
    with path.open(newline="") as file:
        return {
            int(row["seq"]): {
                "seq": int(row["seq"]),
                "timestamp_ms": int(row["timestamp_ms"]),
                "acc_x_g": float(row["acc_x_g"]),
                "acc_y_g": float(row["acc_y_g"]),
                "acc_z_g": float(row["acc_z_g"]),
            }
            for row in csv.DictReader(file)
        }


def derived_features(window: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(np.sum(window**2, axis=1, keepdims=True))
    delta = np.diff(window, axis=0, prepend=window[:1])
    return np.concatenate([window, magnitude, delta], axis=1).astype(np.float32).T


def load_manual_run(run_id: str, raw_csv: Path, labels_csv: Path) -> ManualRun:
    raw_by_seq = read_raw(raw_csv)
    x: list[np.ndarray] = []
    y: list[int] = []
    rows: list[dict] = []
    with labels_csv.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["label"] not in {POSITIVE_LABEL, NEGATIVE_LABEL}:
                continue
            seq_start = int(row["seq_start"])
            seq_end = int(row["seq_end"])
            window = [raw_by_seq[seq] for seq in range(seq_start, seq_end + 1) if seq in raw_by_seq]
            if len(window) != seq_end - seq_start + 1:
                continue
            values = np.asarray([[sample[col] for col in BASE_COLUMNS] for sample in window], dtype=np.float32)
            x.append(derived_features(values))
            y.append(1 if row["label"] == POSITIVE_LABEL else 0)
            rows.append(
                {
                    "run_id": run_id,
                    "label": row["label"],
                    "relative_center_s": float(row["relative_center_s"]),
                    "seq_start": seq_start,
                    "seq_end": seq_end,
                }
            )
    if not x:
        raise ValueError(f"No se encontraron ventanas manuales válidas en el archivo {labels_csv}.")
    return ManualRun(
        run_id=run_id,
        raw_csv=raw_csv,
        labels_csv=labels_csv,
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        rows=rows,
    )


def normalize(train_x: np.ndarray, *arrays: np.ndarray) -> tuple[list[np.ndarray], dict]:
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = [((array - mean) / std).astype(np.float32) for array in arrays]
    return normalized, {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, balanced: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    weights = np.where(y == 1, 1.0 / max(counts[1], 1), 1.0 / max(counts[0], 1))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_model(train_x: np.ndarray, train_y: np.ndarray, args: argparse.Namespace, device: torch.device) -> SlipCNN:
    model = SlipCNN(in_channels=train_x.shape[1], dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = make_loader(train_x, train_y, args.batch_size, balanced=True)
    for _epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


def predict_scores(model: SlipCNN, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    loader = make_loader(x, np.zeros(len(x), dtype=np.int64), batch_size, balanced=False)
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def metrics_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "total": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "score_mean_positive": float(np.mean(scores[y_true == 1])) if np.any(y_true == 1) else 0.0,
        "score_mean_negative": float(np.mean(scores[y_true == 0])) if np.any(y_true == 0) else 0.0,
    }


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 160)))
    best = metrics_at(y_true, scores, 0.5)
    for threshold in candidates:
        current = metrics_at(y_true, scores, float(threshold))
        if (current["f1"], current["recall"], -current["false_positive"]) > (
            best["f1"],
            best["recall"],
            -best["false_positive"],
        ):
            best = current
    return best


def run_fold(train_run: ManualRun, test_run: ManualRun, args: argparse.Namespace, device: torch.device) -> tuple[dict, dict]:
    (train_x, test_x), normalization = normalize(train_run.x, train_run.x, test_run.x)
    model = train_model(train_x, train_run.y, args, device)
    train_scores = predict_scores(model, train_x, args.batch_size, device)
    test_scores = predict_scores(model, test_x, args.batch_size, device)
    default_metrics = metrics_at(test_run.y, test_scores, args.threshold)
    best_metrics = best_threshold(test_run.y, test_scores)
    train_metrics = metrics_at(train_run.y, train_scores, args.threshold)
    artifact = {
        "model_state_dict": model.state_dict(),
        "normalization": normalization,
        "feature_columns": FEATURE_COLUMNS,
        "window_size": int(train_run.x.shape[2]),
        "architecture": "SlipCNN",
        "threshold": float(best_metrics["threshold"]),
        "default_threshold": float(args.threshold),
        "train_run": train_run.run_id,
        "test_run": test_run.run_id,
    }
    summary = {
        "train_run": train_run.run_id,
        "test_run": test_run.run_id,
        "window_size": int(train_run.x.shape[2]),
        "train_windows": int(len(train_run.y)),
        "train_positives": int(np.sum(train_run.y == 1)),
        "test_windows": int(len(test_run.y)),
        "test_positives": int(np.sum(test_run.y == 1)),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"default_{key}": value for key, value in default_metrics.items()},
        **{f"best_{key}": value for key, value in best_metrics.items()},
    }
    return summary, artifact


def maybe_log_wandb(args: argparse.Namespace, rows: list[dict]) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        print("La biblioteca wandb no está instalada; se omitirá el registro en Weights & Biases.", file=sys.stderr)
        return
    for row in rows:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=f"{args.model_name}-{row['train_run']}-to-{row['test_run']}",
            config={
                "model_name": args.model_name,
                "architecture": "SlipCNN",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "threshold": args.threshold,
                "train_run": row["train_run"],
                "test_run": row["test_run"],
                "window_size": row["window_size"],
                "feature_columns": FEATURE_COLUMNS,
            },
        )
        wandb.log(
            {
                "slip/default_precision": row["default_precision"],
                "slip/default_recall": row["default_recall"],
                "slip/default_f1": row["default_f1"],
                "slip/default_false_positive": row["default_false_positive"],
                "slip/default_false_negative": row["default_false_negative"],
                "slip/best_threshold": row["best_threshold"],
                "slip/best_precision": row["best_precision"],
                "slip/best_recall": row["best_recall"],
                "slip/best_f1": row["best_f1"],
                "slip/best_false_positive": row["best_false_positive"],
                "slip/best_false_negative": row["best_false_negative"],
            }
        )
        run.finish()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)

    run_001 = load_manual_run("001", resolve(args.run_001_raw), resolve(args.run_001_labels))
    run_002 = load_manual_run("002", resolve(args.run_002_raw), resolve(args.run_002_labels))
    if run_001.x.shape[1:] != run_002.x.shape[1:]:
        raise ValueError(f"Las dimensiones de las ventanas en las ejecuciones son incompatibles: {run_001.x.shape[1:]} frente a {run_002.x.shape[1:]}.")

    output_dir = resolve(args.output_dir)
    models_dir = resolve(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    artifacts: list[tuple[str, dict]] = []
    for train_run, test_run in [(run_001, run_002), (run_002, run_001)]:
        row, artifact = run_fold(train_run, test_run, args, device)
        rows.append(row)
        artifacts.append((f"{args.model_name}_{train_run.run_id}_to_{test_run.run_id}.pth", artifact))
        print(
            f"Entrenamiento={row['train_run']} Prueba={row['test_run']} "
            f"F1_default={row['default_f1']:.3f} Precision_default={row['default_precision']:.3f} "
            f"Recall_default={row['default_recall']:.3f} F1_best={row['best_f1']:.3f} "
            f"Threshold_best={row['best_threshold']:.3f}"
        )

    for filename, artifact in artifacts:
        torch.save(artifact, models_dir / filename)

    summary_csv = output_dir / f"{args.model_name}_crossrun_summary.csv"
    with summary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json = output_dir / f"{args.model_name}_crossrun_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "feature_columns": FEATURE_COLUMNS,
                "device": str(device),
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    maybe_log_wandb(args, rows)
    print(f"Resumen de resultados guardado en: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
