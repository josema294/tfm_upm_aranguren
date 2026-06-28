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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FEATURE_COLUMNS = ("acc_x_g", "acc_y_g", "acc_z_g")
STARTUP_IGNORE_S = 15.0
SLIP_CORE_HALF_WIDTH_S = 0.30
SLIP_TRANSITION_HALF_WIDTH_S = 0.65
OUTLIER_PEAK_ABS_X_G = 8.0
OUTLIER_X_RANGE_G = 8.0


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    rows: list[dict]
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena clasificadores supervisados de ventana corta para la detección de anomalías por deslizamiento.")
    parser.add_argument("--normal-csv", default="../datos/brutos/real_movement_004.csv")
    parser.add_argument("--slip-csv", default="../datos/brutos/real_slip_only_001.csv")
    parser.add_argument("--labels-csv", default="../datos/etiquetas/real_slip_only_001_windows.csv")
    parser.add_argument("--output-dir", default="../datos/analisis/slip_only/feature_classifier")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--phase-s", type=float, default=None)
    parser.add_argument("--period-s", type=float, default=None)
    parser.add_argument("--sample-rate-hz", type=int, default=100)
    parser.add_argument("--train-cycle-ratio", type=float, default=0.6)
    parser.add_argument("--max-normal-windows", type=int, default=2500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--window-sizes", default="30,50,75")
    parser.add_argument("--model-types", default="logreg_balanced,histgb")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def read_samples(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        return [
            {
                "seq": int(row["seq"]),
                "timestamp_ms": int(row["timestamp_ms"]),
                "acc_x_g": float(row["acc_x_g"]),
                "acc_y_g": float(row["acc_y_g"]),
                "acc_z_g": float(row["acc_z_g"]),
            }
            for row in reader
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


def robust_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.linspace(-0.5, 0.5, len(values), dtype=np.float32)
    x_var = float(np.sum((x - x.mean()) ** 2))
    if x_var <= 1e-12:
        return 0.0
    return float(np.sum((x - x.mean()) * (values - values.mean())) / x_var)


def feature_dict(window: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    magnitude = np.sqrt(np.sum(window**2, axis=1))
    centered = window - window.mean(axis=0, keepdims=True)
    diffs = np.diff(window, axis=0, prepend=window[:1])

    features["mag_mean"] = float(np.mean(magnitude))
    features["mag_std"] = float(np.std(magnitude))
    features["mag_range"] = float(np.ptp(magnitude))
    features["mag_energy_dyn"] = float(np.mean(np.sum(centered**2, axis=1)))
    features["mag_diff_std"] = float(np.std(np.diff(magnitude, prepend=magnitude[0])))
    features["mag_diff_peak"] = float(np.max(np.abs(np.diff(magnitude, prepend=magnitude[0]))))

    for axis, name in enumerate(("x", "y", "z")):
        values = window[:, axis]
        delta = diffs[:, axis]
        centered_axis = values - values.mean()
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))
        features[f"{name}_min"] = float(np.min(values))
        features[f"{name}_max"] = float(np.max(values))
        features[f"{name}_range"] = float(np.ptp(values))
        features[f"{name}_rms_dyn"] = float(np.sqrt(np.mean(centered_axis**2)))
        features[f"{name}_q10"] = float(np.quantile(values, 0.10))
        features[f"{name}_q90"] = float(np.quantile(values, 0.90))
        features[f"{name}_iqr"] = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
        features[f"{name}_slope"] = robust_slope(values)
        features[f"{name}_diff_mean"] = float(np.mean(delta))
        features[f"{name}_diff_std"] = float(np.std(delta))
        features[f"{name}_diff_peak"] = float(np.max(np.abs(delta)))

    # Los cocientes facilitan la discriminación entre el frenado/deslizamiento longitudinal y los impactos verticales.
    eps = 1e-6
    features["x_to_z_range"] = features["x_range"] / (features["z_range"] + eps)
    features["x_to_mag_energy"] = features["x_rms_dyn"] / (math.sqrt(features["mag_energy_dyn"]) + eps)
    features["x_slope_abs"] = abs(features["x_slope"])
    return features


def label_slip_window(
    window: list[dict],
    first_timestamp_ms: int,
    phase_s: float,
    period_s: float,
) -> tuple[str, int, float, int]:
    center_timestamp_ms = (int(window[0]["timestamp_ms"]) + int(window[-1]["timestamp_ms"])) / 2
    relative_center_s = (center_timestamp_ms - first_timestamp_ms) / 1000
    offset_s = phase_offset(relative_center_s, phase_s, period_s)
    cycle_index = int(round((relative_center_s - phase_s) / period_s))
    x_values = np.asarray([float(row["acc_x_g"]) for row in window], dtype=np.float32)
    peak_abs_x = float(np.max(np.abs(x_values)))
    x_range = float(np.ptp(x_values))

    if relative_center_s < STARTUP_IGNORE_S:
        return "ignore_startup", -1, relative_center_s, cycle_index
    if peak_abs_x >= OUTLIER_PEAK_ABS_X_G or x_range >= OUTLIER_X_RANGE_G:
        return "ignore_outlier_impact", -1, relative_center_s, cycle_index
    if abs(offset_s) <= SLIP_CORE_HALF_WIDTH_S:
        return "slip", 1, relative_center_s, cycle_index
    if abs(offset_s) <= SLIP_TRANSITION_HALF_WIDTH_S:
        return "ignore_transition", -1, relative_center_s, cycle_index
    return "normal_slip_run", 0, relative_center_s, cycle_index


def build_slip_dataset(
    samples: list[dict],
    window_size: int,
    window_step: int,
    phase_s: float,
    period_s: float,
) -> Dataset:
    first_timestamp_ms = int(samples[0]["timestamp_ms"])
    rows: list[dict] = []
    feature_rows: list[dict[str, float]] = []
    y: list[int] = []
    for start in range(0, len(samples) - window_size + 1, window_step):
        window_rows = samples[start : start + window_size]
        label, binary, relative_center_s, cycle_index = label_slip_window(
            window_rows, first_timestamp_ms, phase_s, period_s
        )
        if binary < 0:
            continue
        features = feature_dict(np.asarray([[row[col] for col in FEATURE_COLUMNS] for row in window_rows], dtype=np.float32))
        feature_rows.append(features)
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
    feature_names = sorted(feature_rows[0]) if feature_rows else []
    x = np.asarray([[row[name] for name in feature_names] for row in feature_rows], dtype=np.float32)
    return Dataset(x=x, y=np.asarray(y, dtype=np.int64), rows=rows, feature_names=feature_names)


def build_normal_dataset(samples: list[dict], window_size: int, window_step: int, max_windows: int, seed: int) -> Dataset:
    names = sorted(feature_dict(np.zeros((window_size, 3), dtype=np.float32)))
    if max_windows <= 0:
        return Dataset(
            x=np.empty((0, len(names)), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            rows=[],
            feature_names=names,
        )
    feature_rows: list[dict[str, float]] = []
    rows: list[dict] = []
    starts = list(range(0, len(samples) - window_size + 1, window_step))
    rng = np.random.default_rng(seed)
    if len(starts) > max_windows:
        starts = sorted(rng.choice(starts, size=max_windows, replace=False).tolist())
    for start in starts:
        window_rows = samples[start : start + window_size]
        feature_rows.append(
            feature_dict(np.asarray([[row[col] for col in FEATURE_COLUMNS] for row in window_rows], dtype=np.float32))
        )
        rows.append(
            {
                "source": "normal_reference",
                "label": "normal_reference",
                "relative_center_s": np.nan,
                "cycle_index": -999,
                "seq_start": int(window_rows[0]["seq"]),
                "seq_end": int(window_rows[-1]["seq"]),
            }
        )
    feature_names = sorted(feature_rows[0]) if feature_rows else []
    x = np.asarray([[row[name] for name in feature_names] for row in feature_rows], dtype=np.float32)
    return Dataset(x=x, y=np.zeros(len(rows), dtype=np.int64), rows=rows, feature_names=feature_names)


def combine_datasets(datasets: list[Dataset]) -> Dataset:
    feature_names = datasets[0].feature_names
    for dataset in datasets:
        if dataset.feature_names != feature_names:
            raise ValueError("Incongruencia en los nombres de las características (feature names).")
    non_empty = [dataset for dataset in datasets if len(dataset.rows) > 0]
    if not non_empty:
        return Dataset(
            x=np.empty((0, len(feature_names)), dtype=np.float32),
            y=np.empty((0,), dtype=np.int64),
            rows=[],
            feature_names=feature_names,
        )
    return Dataset(
        x=np.concatenate([dataset.x for dataset in non_empty], axis=0),
        y=np.concatenate([dataset.y for dataset in non_empty], axis=0),
        rows=[row for dataset in non_empty for row in dataset.rows],
        feature_names=feature_names,
    )


def split_indices(dataset: Dataset, train_cycle_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    cycles = sorted({row["cycle_index"] for row in dataset.rows if row["source"] == "slip" and row["cycle_index"] >= 0})
    split_at = int(len(cycles) * train_cycle_ratio)
    train_cycles = set(cycles[:split_at])

    train: list[int] = []
    test: list[int] = []
    normal_indices = [idx for idx, row in enumerate(dataset.rows) if row["source"] == "normal_reference"]
    rng = np.random.default_rng(seed)
    rng.shuffle(normal_indices)
    normal_split = int(len(normal_indices) * train_cycle_ratio)
    normal_train = set(normal_indices[:normal_split])

    for idx, row in enumerate(dataset.rows):
        if row["source"] == "normal_reference":
            (train if idx in normal_train else test).append(idx)
        elif row["cycle_index"] in train_cycles:
            train.append(idx)
        else:
            test.append(idx)
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def evaluate_model(model, x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    pred = model.predict(x)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    else:
        score = pred.astype(np.float32)
    return {
        "total": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "false_positives": int(np.sum((y == 0) & (pred == 1))),
        "false_negatives": int(np.sum((y == 1) & (pred == 0))),
        "score_mean_positive": float(np.mean(score[y == 1])) if np.any(y == 1) else 0.0,
        "score_mean_negative": float(np.mean(score[y == 0])) if np.any(y == 0) else 0.0,
    }


def event_recall(model, dataset: Dataset, test_idx: np.ndarray) -> dict[str, float | int]:
    x = dataset.x[test_idx]
    pred = model.predict(x)
    detected_cycles: set[int] = set()
    expected_cycles: set[int] = set()
    false_positive_windows = 0
    for local_idx, global_idx in enumerate(test_idx.tolist()):
        row = dataset.rows[global_idx]
        if row["source"] != "slip":
            if pred[local_idx] == 1:
                false_positive_windows += 1
            continue
        if dataset.y[global_idx] == 1:
            expected_cycles.add(int(row["cycle_index"]))
            if pred[local_idx] == 1:
                detected_cycles.add(int(row["cycle_index"]))
        elif pred[local_idx] == 1:
            false_positive_windows += 1
    return {
        "expected_events": len(expected_cycles),
        "detected_events": len(detected_cycles),
        "event_recall": len(detected_cycles) / len(expected_cycles) if expected_cycles else 0.0,
        "false_positive_windows": false_positive_windows,
    }


def main() -> int:
    args = parse_args()
    normal_samples = read_samples(resolve(args.normal_csv))
    slip_samples = read_samples(resolve(args.slip_csv))
    phase_s, period_s = read_timing(resolve(args.labels_csv), args.phase_s, args.period_s)
    output_dir = resolve(args.output_dir)
    models_dir = resolve(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    requested_window_sizes = [int(value.strip()) for value in args.window_sizes.split(",") if value.strip()]
    requested_model_types = {value.strip() for value in args.model_types.split(",") if value.strip()}

    experiments = []
    for window_size in requested_window_sizes:
        window_step = 10
        slip_dataset = build_slip_dataset(slip_samples, window_size, window_step, phase_s, period_s)
        normal_dataset = build_normal_dataset(
            normal_samples, window_size, window_step, args.max_normal_windows, args.random_seed
        )
        dataset = combine_datasets([slip_dataset, normal_dataset])
        train_idx, test_idx = split_indices(dataset, args.train_cycle_ratio, args.random_seed)

        models = {
            "logreg_balanced": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.random_seed)),
                ]
            ),
            "histgb": HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=120,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=args.random_seed,
            ),
            "rf_balanced": RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=args.random_seed,
                n_jobs=-1,
            ),
        }

        for model_name, model in models.items():
            if model_name not in requested_model_types:
                continue
            model.fit(dataset.x[train_idx], dataset.y[train_idx])
            train_metrics = evaluate_model(model, dataset.x[train_idx], dataset.y[train_idx])
            test_metrics = evaluate_model(model, dataset.x[test_idx], dataset.y[test_idx])
            events = event_recall(model, dataset, test_idx)
            run_name = f"slip_{model_name}_w{window_size}_s{window_step}"
            model_path = models_dir / f"{run_name}.pkl"
            with model_path.open("wb") as file:
                pickle.dump(
                    {
                        "model": model,
                        "feature_names": dataset.feature_names,
                        "window_size": window_size,
                        "window_step": window_step,
                        "phase_s": phase_s,
                        "period_s": period_s,
                    },
                    file,
                )

            row = {
                "run_name": run_name,
                "model": model_name,
                "model_path": str(model_path),
                "window_size": window_size,
                "window_step": window_step,
                "features": len(dataset.feature_names),
                "train_windows": int(len(train_idx)),
                "test_windows": int(len(test_idx)),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **events,
            }
            experiments.append(row)
            print(
                f"{run_name:32} F1_test={row['test_f1']:.3f} "
                f"Precision={row['test_precision']:.3f} Recall={row['test_recall']:.3f} "
                f"Eventos={row['detected_events']}/{row['expected_events']} "
                f"Ventanas_FP={row['test_false_positives']}",
                flush=True,
            )

    summary_csv = output_dir / "slip_feature_classifier_summary.csv"
    with summary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(experiments[0].keys()))
        writer.writeheader()
        writer.writerows(experiments)

    summary_json = output_dir / "slip_feature_classifier_summary.json"
    summary_json.write_text(json.dumps({"experiments": experiments}, indent=2), encoding="utf-8")

    for row in sorted(experiments, key=lambda item: (item["test_f1"], item["event_recall"], -item["test_false_positives"]), reverse=True):
        print(
            f"{row['run_name']:32} F1_test={row['test_f1']:.3f} "
            f"Precision={row['test_precision']:.3f} Recall={row['test_recall']:.3f} "
            f"Eventos={row['detected_events']}/{row['expected_events']} "
            f"Ventanas_FP={row['test_false_positives']}"
        )
    print(f"Resumen general guardado en: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
