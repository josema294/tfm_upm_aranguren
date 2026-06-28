#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from herramientas.train_multiscale_slip_cnn import BASE_COLUMNS, FEATURE_COLUMNS, MultiScaleSlipCNN, SCALES, derived_features


@dataclass
class Bag:
    name: str
    target: int
    x_by_scale: dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena una CNN para deslizamiento con supervisión débil, empleando vueltas positivas como bolsas MIL (Multiple Instance Learning).")
    parser.add_argument(
        "--slip-csv",
        action="append",
        default=None,
        help="Archivo CSV con casos positivos de deslizamiento. Se puede especificar múltiples veces. Por defecto: ../datos/test_performance_2.csv.",
    )
    parser.add_argument("--normal-csv", default="../datos/brutos/real_movement_004.csv")
    parser.add_argument("--slip-end-s", type=float, default=600.0)
    parser.add_argument("--lap-period-s", type=float, default=10.0)
    parser.add_argument("--window-step", type=int, default=10)
    parser.add_argument("--normal-max-bags", type=int, default=240)
    parser.add_argument("--normal-train-fraction", type=float, default=0.70)
    parser.add_argument("--output-dir", default="../datos/analisis/slip_manual/mil_cnn")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model-name", default="slip_mil_w30_50_100_v1")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-normal-fp-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="tfm-railway-anomaly")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb-dataset-artifact", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bad = (df[BASE_COLUMNS].abs() > 8.5).any(axis=1)
    for column in BASE_COLUMNS:
        df.loc[bad, column] = np.nan
        df[column] = df[column].interpolate().ffill().bfill()
    df["relative_s"] = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
    return df


def features_for_window(values: np.ndarray, start: int, scale: int) -> np.ndarray | None:
    end = start + scale
    if start < 0 or end > len(values):
        return None
    return derived_features(values[start:end])


def make_bag(df: pd.DataFrame, name: str, target: int, start_s: float, end_s: float, step: int) -> Bag | None:
    segment = df[(df.relative_s >= start_s) & (df.relative_s < end_s)].reset_index(drop=True)
    if len(segment) < max(SCALES):
        return None
    values = segment[BASE_COLUMNS].to_numpy(dtype=np.float32)
    x_by_scale: dict[int, list[np.ndarray]] = {scale: [] for scale in SCALES}
    for center in range(max(SCALES) // 2, len(segment) - max(SCALES) // 2, step):
        valid = True
        for scale in SCALES:
            start = center - scale // 2
            features = features_for_window(values, start, scale)
            if features is None:
                valid = False
                break
            x_by_scale[scale].append(features)
        if not valid:
            continue
    if not x_by_scale[SCALES[0]]:
        return None
    return Bag(
        name=name,
        target=target,
        x_by_scale={scale: np.asarray(values, dtype=np.float32) for scale, values in x_by_scale.items()},
    )


def make_positive_bags(df: pd.DataFrame, slip_end_s: float, lap_period_s: float, step: int) -> list[Bag]:
    bags = []
    lap_count = int(slip_end_s // lap_period_s)
    for lap in range(lap_count):
        bag = make_bag(df, f"slip_lap_{lap:03d}", 1, lap * lap_period_s, (lap + 1) * lap_period_s, step)
        if bag is not None:
            bags.append(bag)
    return bags


def make_normal_bags(df: pd.DataFrame, lap_period_s: float, step: int, max_bags: int) -> list[Bag]:
    duration_s = float(df.relative_s.iloc[-1])
    candidate_starts = np.arange(0, duration_s - lap_period_s, lap_period_s)
    if len(candidate_starts) > max_bags:
        selected = np.linspace(0, len(candidate_starts) - 1, max_bags, dtype=np.int64)
        candidate_starts = candidate_starts[selected]
    bags = []
    for index, start_s in enumerate(candidate_starts):
        bag = make_bag(df, f"normal_lap_{index:03d}", 0, float(start_s), float(start_s + lap_period_s), step)
        if bag is not None:
            bags.append(bag)
    return bags


def fit_normalization(bags: list[Bag]) -> dict[str, dict[str, list[float]]]:
    normalization = {}
    for scale in SCALES:
        x = np.concatenate([bag.x_by_scale[scale] for bag in bags], axis=0)
        mean = x.mean(axis=(0, 2))
        std = x.std(axis=(0, 2))
        std = np.where(std < 1e-6, 1.0, std)
        normalization[str(scale)] = {"mean": mean.tolist(), "std": std.tolist()}
    return normalization


def normalize_bag(bag: Bag, normalization: dict[str, dict[str, list[float]]]) -> Bag:
    x_by_scale = {}
    for scale in SCALES:
        mean = np.asarray(normalization[str(scale)]["mean"], dtype=np.float32)[None, :, None]
        std = np.asarray(normalization[str(scale)]["std"], dtype=np.float32)[None, :, None]
        x_by_scale[scale] = ((bag.x_by_scale[scale] - mean) / std).astype(np.float32)
    return Bag(name=bag.name, target=bag.target, x_by_scale=x_by_scale)


def bag_tensors(bag: Bag, device: torch.device) -> list[torch.Tensor]:
    return [torch.as_tensor(bag.x_by_scale[scale], dtype=torch.float32, device=device) for scale in SCALES]


def bag_logit(model: MultiScaleSlipCNN, bag: Bag, device: torch.device, tau: float = 0.20) -> torch.Tensor:
    logits = model(*bag_tensors(bag, device))
    return torch.logsumexp(logits / tau, dim=0) * tau


def score_bags(model: MultiScaleSlipCNN, bags: list[Bag], device: torch.device) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for bag in bags:
            scores.append(float(torch.sigmoid(bag_logit(model, bag, device)).detach().cpu().item()))
    return np.asarray(scores, dtype=np.float32)


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def calibrate_threshold(val_scores: np.ndarray, val_y: np.ndarray, max_normal_fp_rate: float) -> float:
    normal_scores = val_scores[val_y == 0]
    min_threshold = float(np.quantile(normal_scores, 1.0 - max_normal_fp_rate)) if len(normal_scores) else 0.5
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), val_scores, [min_threshold]]))
    candidates = candidates[candidates >= min_threshold]
    best_threshold = float(min_threshold)
    best_f1 = -1.0
    for threshold in candidates:
        f1 = metrics(val_y, val_scores, float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.parent))
    except ValueError:
        return str(path.resolve())


def maybe_log_wandb(args: argparse.Namespace, summary: dict, artifact_path: Path, dataset_paths: dict[str, object]) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        print("La biblioteca wandb no está instalada; se omitirá el registro remoto.")
        return
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.model_name,
        group="supervised-slip-mil",
        config=vars(args),
    )
    run.log(summary)
    if not args.no_wandb_dataset_artifact:
        dataset = wandb.Artifact(
            f"{args.model_name}_datasets",
            type="dataset",
            metadata={
                "slip_csvs": [rel(path) for path in dataset_paths["slip_csvs"]],
                "normal_csv": rel(dataset_paths["normal_csv"]),
                "dataset_manifest": rel(dataset_paths["dataset_manifest"]),
                "slip_end_s": args.slip_end_s,
                "lap_period_s": args.lap_period_s,
                "normal_max_bags": args.normal_max_bags,
            },
        )
        paths_to_add = [*dataset_paths["slip_csvs"], dataset_paths["normal_csv"], dataset_paths["dataset_manifest"]]
        for path in paths_to_add:
            if path.exists():
                dataset.add_file(str(path))
        run.log_artifact(dataset)
    artifact = wandb.Artifact(args.model_name, type="model")
    artifact.add_file(str(artifact_path))
    run.log_artifact(artifact)
    run.finish()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)

    slip_csvs = [resolve(path) for path in (args.slip_csv or ["../datos/test_performance_2.csv"])]
    normal_csv = resolve(args.normal_csv)
    dataset_manifest = REPO_ROOT.parent / "datos" / "DATASET_MANIFEST.md"
    normal_df = load_frame(normal_csv)
    positive_bags = []
    for slip_index, slip_csv in enumerate(slip_csvs, start=1):
        slip_df = load_frame(slip_csv)
        run_bags = make_positive_bags(slip_df, args.slip_end_s, args.lap_period_s, args.window_step)
        for bag in run_bags:
            bag.name = f"slip{slip_index}_{bag.name}"
        positive_bags.extend(run_bags)
    normal_bags = make_normal_bags(normal_df, args.lap_period_s, args.window_step, args.normal_max_bags)

    rng.shuffle(positive_bags)
    rng.shuffle(normal_bags)
    pos_split = max(1, int(len(positive_bags) * args.normal_train_fraction))
    neg_split = max(1, int(len(normal_bags) * args.normal_train_fraction))
    train_bags = positive_bags[:pos_split] + normal_bags[:neg_split]
    val_bags = positive_bags[pos_split:] + normal_bags[neg_split:]
    rng.shuffle(train_bags)
    rng.shuffle(val_bags)

    normalization = fit_normalization(train_bags)
    train_bags = [normalize_bag(bag, normalization) for bag in train_bags]
    val_bags = [normalize_bag(bag, normalization) for bag in val_bags]

    model = MultiScaleSlipCNN(in_channels=len(FEATURE_COLUMNS), dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pos_weight = torch.tensor([max(1.0, sum(b.target == 0 for b in train_bags) / max(1, sum(b.target == 1 for b in train_bags)))], device=device)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_bags)
        losses = []
        for bag in train_bags:
            target = torch.tensor(float(bag.target), device=device)
            logit = bag_logit(model, bag, device)
            loss = F.binary_cross_entropy_with_logits(logit[None], target[None], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        val_y = np.asarray([bag.target for bag in val_bags], dtype=np.int64)
        val_scores = score_bags(model, val_bags, device)
        threshold = calibrate_threshold(val_scores, val_y, args.max_normal_fp_rate)
        epoch_metrics = metrics(val_y, val_scores, threshold)
        epoch_metrics.update({"epoch": epoch, "loss": float(np.mean(losses)), "threshold": threshold})
        history.append(epoch_metrics)

    model.eval()
    train_y = np.asarray([bag.target for bag in train_bags], dtype=np.int64)
    val_y = np.asarray([bag.target for bag in val_bags], dtype=np.int64)
    train_scores = score_bags(model, train_bags, device)
    val_scores = score_bags(model, val_bags, device)
    threshold = calibrate_threshold(val_scores, val_y, args.max_normal_fp_rate)
    summary = {
        "slip_csvs": [rel(path) for path in slip_csvs],
        "normal_csv": rel(normal_csv),
        "dataset_manifest": rel(dataset_manifest),
        "train_bags": len(train_bags),
        "val_bags": len(val_bags),
        "positive_bags_total": len(positive_bags),
        "normal_bags_total": len(normal_bags),
        "threshold": threshold,
        "train_precision": metrics(train_y, train_scores, threshold)["precision"],
        "train_recall": metrics(train_y, train_scores, threshold)["recall"],
        "train_f1": metrics(train_y, train_scores, threshold)["f1"],
        "val_precision": metrics(val_y, val_scores, threshold)["precision"],
        "val_recall": metrics(val_y, val_scores, threshold)["recall"],
        "val_f1": metrics(val_y, val_scores, threshold)["f1"],
        "val_normal_fp_rate": float(((val_scores[val_y == 0] >= threshold).sum() / max(1, (val_y == 0).sum()))),
        "val_positive_hit_rate": float(((val_scores[val_y == 1] >= threshold).sum() / max(1, (val_y == 1).sum()))),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }

    output_dir = resolve(args.output_dir)
    models_dir = resolve(args.models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / f"{args.model_name}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalization": normalization,
            "threshold": threshold,
            "feature_columns": FEATURE_COLUMNS,
            "scales": SCALES,
            "summary": summary,
            "training_mode": "multiple_instance_learning",
        },
        artifact_path,
    )
    (output_dir / f"{args.model_name}_summary.json").write_text(json.dumps({"summary": summary, "history": history}, indent=2), encoding="utf-8")
    maybe_log_wandb(
        args,
        summary,
        artifact_path,
        {
            "slip_csvs": slip_csvs,
            "normal_csv": normal_csv,
            "dataset_manifest": dataset_manifest,
        },
    )
    print(json.dumps(summary, indent=2))
    print(f"Modelo guardado en: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
