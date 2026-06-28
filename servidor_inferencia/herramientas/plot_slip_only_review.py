#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera visualizaciones de la captura de deslizamiento (slip-only) junto con sus etiquetas temporales.")
    parser.add_argument("--raw-csv", default="../datos/brutos/real_slip_only_001.csv")
    parser.add_argument("--labels-csv", default="../datos/etiquetas/real_slip_only_001_windows.csv")
    parser.add_argument("--features-csv", default="../datos/analisis/slip_only/real_slip_only_001_window_features.csv")
    parser.add_argument("--output-dir", default="../datos/figuras/slip_only")
    parser.add_argument("--chunk-s", type=float, default=120.0)
    parser.add_argument("--sample-hz", type=float, default=100.0)
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(__file__).resolve().parents[1] / candidate).resolve()


def add_label_spans(ax, labels: pd.DataFrame, start_s: float, end_s: float) -> None:
    styles = {
        "slip_confirmed_core": ("#ef4444", 0.24),
        "slip_transition": ("#f59e0b", 0.18),
        "ignore_outlier_impact": ("#7c3aed", 0.20),
        "ignore_startup": ("#64748b", 0.16),
    }
    visible = labels[(labels["relative_center_s"] >= start_s - 2.0) & (labels["relative_center_s"] <= end_s + 2.0)]
    for label, group in visible.groupby("label"):
        if label not in styles:
            continue
        color, alpha = styles[label]
        for center_s in group["relative_center_s"]:
            ax.axvspan(center_s - 0.125, center_s + 0.125, color=color, alpha=alpha, linewidth=0)


def plot_chunk(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    start_s: float,
    end_s: float,
    output_path: Path,
) -> None:
    raw_chunk = raw[(raw["t_s"] >= start_s) & (raw["t_s"] <= end_s)]
    feat_chunk = features[(features["center_s"] >= start_s) & (features["center_s"] <= end_s)]

    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=True)
    axes[0].plot(raw_chunk["t_s"], raw_chunk["acc_x_g"], label="acc_x_g", linewidth=0.8)
    axes[0].plot(raw_chunk["t_s"], raw_chunk["acc_y_g"], label="acc_y_g", linewidth=0.8)
    axes[0].plot(raw_chunk["t_s"], raw_chunk["acc_z_g"], label="acc_z_g", linewidth=0.8)
    axes[0].set_ylabel("Aceleración (g)")
    axes[0].legend(loc="upper right", ncol=3)

    axes[1].plot(feat_chunk["center_s"], feat_chunk["rms_dyn"], color="#2563eb", label="RMS dinámico")
    axes[1].set_ylabel("RMS dinámico (g)")
    axes[1].legend(loc="upper right")

    axes[2].plot(feat_chunk["center_s"], feat_chunk["x_range"], color="#16a34a", label="Rango eje X")
    axes[2].plot(feat_chunk["center_s"], feat_chunk["delta_x_p95"], color="#f97316", label="p95 |Delta X|")
    axes[2].set_ylabel("Características eje X")
    axes[2].legend(loc="upper right")

    axes[3].scatter(labels["relative_center_s"], labels["phase_offset_s"], s=8, c="#334155", alpha=0.45)
    axes[3].axhline(0, color="#ef4444", linestyle="--", linewidth=1)
    axes[3].axhline(0.30, color="#f59e0b", linestyle="--", linewidth=1)
    axes[3].axhline(-0.30, color="#f59e0b", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Desfase (s)")
    axes[3].set_xlabel("Tiempo relativo (s)")

    for ax in axes:
        add_label_spans(ax, labels, start_s, end_s)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(start_s, end_s)

    fig.suptitle(f"real_slip_only_001: revisión {start_s:.0f}-{end_s:.0f}s")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    raw_csv = resolve(args.raw_csv)
    labels_csv = resolve(args.labels_csv)
    features_csv = resolve(args.features_csv)
    output_dir = resolve(args.output_dir)

    raw = pd.read_csv(raw_csv)
    raw["t_s"] = (raw["timestamp_ms"] - raw["timestamp_ms"].iloc[0]) / 1000.0
    labels = pd.read_csv(labels_csv)
    features = pd.read_csv(features_csv)

    duration_s = float(raw["t_s"].iloc[-1])
    for start_s in np.arange(0.0, duration_s, args.chunk_s):
        end_s = min(float(start_s + args.chunk_s), duration_s)
        if end_s - start_s < 1.0:
            continue
        output_path = output_dir / f"real_slip_only_001_review_{int(start_s):04d}_{int(end_s):04d}s.png"
        plot_chunk(raw, features, labels, float(start_s), end_s, output_path)
        print(f"Gráfico guardado en: {output_path}")

    overview_path = output_dir / "real_slip_only_001_label_counts.csv"
    labels["label"].value_counts().rename_axis("label").reset_index(name="windows").to_csv(overview_path, index=False)
    print(f"Resumen de etiquetas guardado en: {overview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
