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
SCALES = (30, 50, 100)


class MultiScaleSlipCNN(nn.Module):
    def __init__(self, in_channels: int = len(FEATURE_COLUMNS), dropout: float = 0.20):
        super().__init__()
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
                for _ in SCALES
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(len(SCALES) * 96, 128),
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


@dataclass
class ManualRun:
    run_id: str
    raw_csv: Path
    labels_csv: Path
    x_by_scale: dict[int, np.ndarray]
    y: np.ndarray
    rows: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y evalúa una red neuronal convolucional (CNN) multiescala para la detección de deslizamiento.")
    parser.add_argument("--run-001-raw", default="../datos/brutos/real_slip_manual_001.csv")
    parser.add_argument("--run-001-labels", default="../datos/etiquetas/real_slip_manual_001_windows.csv")
    parser.add_argument("--run-002-raw", default="../datos/brutos/real_slip_manual_002.csv")
    parser.add_argument("--run-002-labels", default="../datos/etiquetas/real_slip_manual_002_windows.csv")
    parser.add_argument("--background-normal-csv", help="Ejecución normal prolongada opcional, empleada como fondo negativo adicional.")
    parser.add_argument("--background-max-train-windows", type=int, default=3000)
    parser.add_argument("--background-max-val-windows", type=int, default=1500)
    parser.add_argument("--background-train-fraction", type=float, default=0.70)
    parser.add_argument("--max-background-fp-rate", type=float, default=0.02)
    parser.add_argument("--output-dir", default="../datos/analisis/slip_manual/multiscale_cnn")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model-name", default="slip_multiscale_cnn_v1")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="tfm-railway-anomaly")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default="supervised-slip-multiscale")
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


def centered_window(raw_by_seq: dict[int, dict], center_seq: int, size: int) -> np.ndarray | None:
    left = size // 2
    start = center_seq - left
    end = start + size - 1
    rows = [raw_by_seq.get(seq) for seq in range(start, end + 1)]
    if any(row is None for row in rows):
        return None
    return np.asarray([[row[column] for column in BASE_COLUMNS] for row in rows if row is not None], dtype=np.float32)


def load_manual_run(run_id: str, raw_csv: Path, labels_csv: Path) -> ManualRun:
    raw_by_seq = read_raw(raw_csv)
    x_by_scale: dict[int, list[np.ndarray]] = {scale: [] for scale in SCALES}
    y: list[int] = []
    rows: list[dict] = []

    with labels_csv.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["label"] not in {POSITIVE_LABEL, NEGATIVE_LABEL}:
                continue
            center_seq = int(row.get("seq_center") or round((int(row["seq_start"]) + int(row["seq_end"])) / 2))
            windows = {scale: centered_window(raw_by_seq, center_seq, scale) for scale in SCALES}
            if any(window is None for window in windows.values()):
                continue
            for scale, window in windows.items():
                assert window is not None
                x_by_scale[scale].append(derived_features(window))
            y.append(1 if row["label"] == POSITIVE_LABEL else 0)
            rows.append(
                {
                    "run_id": run_id,
                    "label": row["label"],
                    "relative_center_s": float(row["relative_center_s"]),
                    "seq_center": center_seq,
                }
            )

    if not y:
        raise ValueError(f"No se encontraron ventanas manuales válidas en el archivo {labels_csv}.")

    return ManualRun(
        run_id=run_id,
        raw_csv=raw_csv,
        labels_csv=labels_csv,
        x_by_scale={scale: np.asarray(values, dtype=np.float32) for scale, values in x_by_scale.items()},
        y=np.asarray(y, dtype=np.int64),
        rows=rows,
    )


def clean_values(values: np.ndarray) -> np.ndarray:
    cleaned = values.astype(np.float32).copy()
    bad_rows = np.any(np.abs(cleaned) > 8.5, axis=1)
    if not np.any(bad_rows):
        return cleaned
    for column in range(cleaned.shape[1]):
        series = cleaned[:, column]
        good = ~bad_rows
        if np.any(good):
            series[bad_rows] = np.interp(np.flatnonzero(bad_rows), np.flatnonzero(good), series[good])
        cleaned[:, column] = series
    return cleaned


def load_background_run(
    run_id: str,
    raw_csv: Path,
    max_windows: int,
    start_fraction: float,
    end_fraction: float,
) -> ManualRun:
    raw_by_seq = read_raw(raw_csv)
    ordered = [raw_by_seq[seq] for seq in sorted(raw_by_seq)]
    n = len(ordered)
    min_margin = max(SCALES) // 2
    start_index = max(min_margin, int(n * start_fraction))
    end_index = min(n - min_margin - 1, int(n * end_fraction))
    if end_index <= start_index:
        raise ValueError(f"Rango de fondo (background) no válido para {raw_csv}: {start_index}..{end_index}.")

    candidate_centers = np.arange(start_index, end_index + 1, 25, dtype=np.int64)
    if len(candidate_centers) > max_windows:
        selected = np.linspace(0, len(candidate_centers) - 1, max_windows, dtype=np.int64)
        candidate_centers = candidate_centers[selected]

    x_by_scale: dict[int, list[np.ndarray]] = {scale: [] for scale in SCALES}
    rows = []
    for center_index in candidate_centers:
        valid = True
        for scale in SCALES:
            left = scale // 2
            values = np.asarray(
                [[ordered[index][column] for column in BASE_COLUMNS] for index in range(center_index - left, center_index - left + scale)],
                dtype=np.float32,
            )
            values = clean_values(values)
            if values.shape[0] != scale:
                valid = False
                break
            x_by_scale[scale].append(derived_features(values))
        if valid:
            rows.append(
                {
                    "run_id": run_id,
                    "label": NEGATIVE_LABEL,
                    "relative_center_s": (ordered[center_index]["timestamp_ms"] - ordered[0]["timestamp_ms"]) / 1000,
                    "seq_center": int(ordered[center_index]["seq"]),
                }
            )
    y = np.zeros(len(rows), dtype=np.int64)
    return ManualRun(
        run_id=run_id,
        raw_csv=raw_csv,
        labels_csv=raw_csv,
        x_by_scale={scale: np.asarray(values, dtype=np.float32) for scale, values in x_by_scale.items()},
        y=y,
        rows=rows,
    )


def merge_runs(run_id: str, *runs: ManualRun) -> ManualRun:
    x_by_scale = {
        scale: np.concatenate([run.x_by_scale[scale] for run in runs], axis=0)
        for scale in SCALES
    }
    return ManualRun(
        run_id=run_id,
        raw_csv=runs[0].raw_csv,
        labels_csv=runs[0].labels_csv,
        x_by_scale=x_by_scale,
        y=np.concatenate([run.y for run in runs], axis=0),
        rows=[row for run in runs for row in run.rows],
    )


def normalize(train: ManualRun, *runs: ManualRun) -> tuple[list[dict[int, np.ndarray]], dict]:
    normalization = {}
    normalized_runs: list[dict[int, np.ndarray]] = []
    for scale in SCALES:
        train_x = train.x_by_scale[scale]
        mean = train_x.mean(axis=(0, 2), keepdims=True)
        std = train_x.std(axis=(0, 2), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        normalization[str(scale)] = {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}

    for run in runs:
        normalized = {}
        for scale in SCALES:
            mean = np.asarray(normalization[str(scale)]["mean"], dtype=np.float32)[None, :, None]
            std = np.asarray(normalization[str(scale)]["std"], dtype=np.float32)[None, :, None]
            normalized[scale] = ((run.x_by_scale[scale] - mean) / std).astype(np.float32)
        normalized_runs.append(normalized)
    return normalized_runs, normalization


def make_loader(x_by_scale: dict[int, np.ndarray], y: np.ndarray, batch_size: int, balanced: bool) -> DataLoader:
    tensors = [torch.from_numpy(x_by_scale[scale]) for scale in SCALES]
    dataset = TensorDataset(*tensors, torch.from_numpy(y.astype(np.float32)))
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    weights = np.where(y == 1, 1.0 / max(counts[1], 1), 1.0 / max(counts[0], 1))
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_model(x_by_scale: dict[int, np.ndarray], y: np.ndarray, args: argparse.Namespace, device: torch.device) -> MultiScaleSlipCNN:
    model = MultiScaleSlipCNN(in_channels=x_by_scale[SCALES[0]].shape[1], dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = make_loader(x_by_scale, y, args.batch_size, balanced=True)
    for _epoch in range(args.epochs):
        model.train()
        for batch in loader:
            *xs, yb = batch
            xs = [x.to(device) for x in xs]
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(*xs), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


def predict_scores(model: MultiScaleSlipCNN, x_by_scale: dict[int, np.ndarray], batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    y_dummy = np.zeros(len(x_by_scale[SCALES[0]]), dtype=np.int64)
    loader = make_loader(x_by_scale, y_dummy, batch_size, balanced=False)
    with torch.no_grad():
        for batch in loader:
            *xs, _yb = batch
            logits = model(*[x.to(device) for x in xs])
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
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 180)))
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


def constrained_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    background_scores: np.ndarray | None,
    max_fp_rate: float,
) -> dict[str, float | int]:
    if background_scores is None or len(background_scores) == 0:
        return best_threshold(y_true, scores)
    min_threshold = float(np.quantile(background_scores, max(0.0, min(1.0, 1.0 - max_fp_rate))))
    candidates = np.unique(np.concatenate([np.quantile(scores, np.linspace(0.01, 0.99, 180)), [min_threshold, 0.5]]))
    best = None
    for threshold in candidates:
        if threshold < min_threshold:
            continue
        current = metrics_at(y_true, scores, float(threshold))
        current["background_fp_rate"] = float(np.mean(background_scores >= threshold))
        if best is None or (current["f1"], current["recall"], -current["false_positive"]) > (
            best["f1"],
            best["recall"],
            -best["false_positive"],
        ):
            best = current
    assert best is not None
    return best


def run_fold(
    train_run: ManualRun,
    test_run: ManualRun,
    args: argparse.Namespace,
    device: torch.device,
    background_val: ManualRun | None = None,
) -> tuple[dict, dict]:
    runs_to_normalize = [train_run, test_run]
    if background_val is not None:
        runs_to_normalize.append(background_val)
    normalized_runs, normalization = normalize(train_run, *runs_to_normalize)
    train_x = normalized_runs[0]
    test_x = normalized_runs[1]
    background_x = normalized_runs[2] if background_val is not None else None
    model = train_model(train_x, train_run.y, args, device)
    train_scores = predict_scores(model, train_x, args.batch_size, device)
    test_scores = predict_scores(model, test_x, args.batch_size, device)
    background_scores = (
        predict_scores(model, background_x, args.batch_size, device)
        if background_val is not None and background_x is not None
        else None
    )
    train_metrics = metrics_at(train_run.y, train_scores, args.threshold)
    default_metrics = metrics_at(test_run.y, test_scores, args.threshold)
    best_metrics = best_threshold(test_run.y, test_scores)
    constrained_metrics = constrained_threshold(test_run.y, test_scores, background_scores, args.max_background_fp_rate)
    artifact = {
        "model_state_dict": model.state_dict(),
        "normalization": normalization,
        "feature_columns": FEATURE_COLUMNS,
        "scales": list(SCALES),
        "architecture": "MultiScaleSlipCNN",
        "threshold": float(constrained_metrics["threshold"]),
        "default_threshold": float(args.threshold),
        "best_unconstrained_threshold": float(best_metrics["threshold"]),
        "max_background_fp_rate": float(args.max_background_fp_rate),
        "train_run": train_run.run_id,
        "test_run": test_run.run_id,
    }
    summary = {
        "train_run": train_run.run_id,
        "test_run": test_run.run_id,
        "scales": ",".join(str(scale) for scale in SCALES),
        "train_windows": int(len(train_run.y)),
        "train_positives": int(np.sum(train_run.y == 1)),
        "test_windows": int(len(test_run.y)),
        "test_positives": int(np.sum(test_run.y == 1)),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"default_{key}": value for key, value in default_metrics.items()},
        **{f"best_{key}": value for key, value in best_metrics.items()},
        **{f"constrained_{key}": value for key, value in constrained_metrics.items()},
    }
    if background_scores is not None:
        summary["background_val_windows"] = int(len(background_scores))
        summary["background_default_fp_rate"] = float(np.mean(background_scores >= args.threshold))
        summary["background_best_fp_rate"] = float(np.mean(background_scores >= best_metrics["threshold"]))
        summary["background_constrained_fp_rate"] = float(np.mean(background_scores >= constrained_metrics["threshold"]))
        summary["background_score_p95"] = float(np.quantile(background_scores, 0.95))
        summary["background_score_p98"] = float(np.quantile(background_scores, 0.98))
        summary["background_score_p99"] = float(np.quantile(background_scores, 0.99))
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
            config={**row, "model_name": args.model_name, "architecture": "MultiScaleSlipCNN"},
        )
        wandb.log(
            {
                "slip/default_precision": row["default_precision"],
                "slip/default_recall": row["default_recall"],
                "slip/default_f1": row["default_f1"],
                "slip/best_threshold": row["best_threshold"],
                "slip/best_precision": row["best_precision"],
                "slip/best_recall": row["best_recall"],
                "slip/best_f1": row["best_f1"],
                "slip/constrained_threshold": row["constrained_threshold"],
                "slip/constrained_precision": row["constrained_precision"],
                "slip/constrained_recall": row["constrained_recall"],
                "slip/constrained_f1": row["constrained_f1"],
                "slip/background_constrained_fp_rate": row.get("background_constrained_fp_rate", 0.0),
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
    background_train = None
    background_val = None
    if args.background_normal_csv:
        background_csv = resolve(args.background_normal_csv)
        background_train = load_background_run(
            "normal_background_train",
            background_csv,
            args.background_max_train_windows,
            0.0,
            args.background_train_fraction,
        )
        background_val = load_background_run(
            "normal_background_val",
            background_csv,
            args.background_max_val_windows,
            args.background_train_fraction,
            1.0,
        )
        print(
            f"Ventanas de fondo para entrenamiento: {len(background_train.y)} | "
            f"Ventanas de validación: {len(background_val.y)} | Origen: {background_csv}"
        )

    output_dir = resolve(args.output_dir)
    models_dir = resolve(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    artifacts = []
    for train_run, test_run in [(run_001, run_002), (run_002, run_001)]:
        effective_train = (
            merge_runs(f"{train_run.run_id}+normal_bg", train_run, background_train)
            if background_train is not None
            else train_run
        )
        row, artifact = run_fold(effective_train, test_run, args, device, background_val=background_val)
        row["manual_train_run"] = train_run.run_id
        rows.append(row)
        artifacts.append((f"{args.model_name}_{train_run.run_id}_to_{test_run.run_id}.pth", artifact))
        print(
            f"Entrenamiento={row['train_run']} Prueba={row['test_run']} "
            f"F1_default={row['default_f1']:.3f} Precision_default={row['default_precision']:.3f} "
            f"Recall_default={row['default_recall']:.3f} F1_best={row['best_f1']:.3f} "
            f"Threshold_best={row['best_threshold']:.3f} F1_constrained={row['constrained_f1']:.3f} "
            f"Recall_constrained={row['constrained_recall']:.3f} Threshold_constrained={row['constrained_threshold']:.3f} "
            f"FP_background={row.get('background_constrained_fp_rate', 0.0):.3f}"
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
        json.dumps({"model_name": args.model_name, "feature_columns": FEATURE_COLUMNS, "scales": SCALES, "device": str(device), "results": rows}, indent=2),
        encoding="utf-8",
    )
    maybe_log_wandb(args, rows)
    print(f"Resumen de resultados guardado en: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
