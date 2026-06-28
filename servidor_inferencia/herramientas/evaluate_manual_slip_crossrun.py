#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["x_std", "x_range", "x_slope", "dx_p95", "dx_peak", "mag_std", "dynamic_rms"]
VALID_LABELS = {"manual_slip_core": 1, "manual_normal_between": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación cruzada (cross-run) para etiquetas de deslizamiento manuales.")
    parser.add_argument("--labels-001", default="../datos/etiquetas/real_slip_manual_001_windows.csv")
    parser.add_argument("--labels-002", default="../datos/etiquetas/real_slip_manual_002_windows.csv")
    parser.add_argument("--output", default="../datos/analisis/slip_manual/manual_slip_crossrun_eval.csv")
    parser.add_argument("--summary-output", default="../datos/analisis/slip_manual/manual_slip_crossrun_eval_summary.json")
    return parser.parse_args()


def resolve_path(ruta: str) -> Path:
    candidata = Path(ruta)
    if candidata.is_absolute():
        return candidata
    return (Path(__file__).resolve().parents[1] / candidata).resolve()


def load_dataset(ruta: Path) -> tuple[np.ndarray, np.ndarray, Counter]:
    filas = []
    with ruta.open(newline="") as archivo:
        for fila in csv.DictReader(archivo):
            if fila["label"] in VALID_LABELS:
                filas.append(fila)
    x_data = np.asarray([[float(fila[nombre]) for nombre in FEATURES] for fila in filas], dtype=np.float32)
    y_data = np.asarray([VALID_LABELS[fila["label"]] for fila in filas], dtype=np.int64)
    return x_data, y_data, Counter(fila["label"] for fila in filas)


def model_factory(semilla: int = 42):
    return {
        "logreg_balanced": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=semilla),
        ),
        "histgb": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=0.05,
            random_state=semilla,
        ),
        "rf_balanced": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=semilla,
            n_jobs=-1,
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "total": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "false_positives": int(np.sum((y_true == 0) & (y_pred == 1))),
        "false_negatives": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def main() -> int:
    args = parse_args()
    conjuntos = {
        "001": load_dataset(resolve_path(args.labels_001)),
        "002": load_dataset(resolve_path(args.labels_002)),
    }
    resultados: list[dict] = []
    
    for clave_entrenamiento, clave_prueba in [("001", "002"), ("002", "001")]:
        x_train, y_train, _ = conjuntos[clave_entrenamiento]
        x_test, y_test, _ = conjuntos[clave_prueba]
        
        for nombre_modelo, modelo in model_factory().items():
            modelo.fit(x_train, y_train)
            registro = {
                "train_run": clave_entrenamiento,
                "test_run": clave_prueba,
                "model": nombre_modelo,
                "train_windows": int(len(y_train)),
                "train_positives": int(np.sum(y_train == 1)),
                "test_windows": int(len(y_test)),
                "test_positives": int(np.sum(y_test == 1)),
                **metrics(y_test, modelo.predict(x_test)),
            }
            resultados.append(registro)

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(resultados[0].keys()))
        escritor.writeheader()
        escritor.writerows(resultados)

    resumen = {
        "features": FEATURES,
        "label_counts": {clave: dict(valor[2]) for clave, valor in conjuntos.items()},
        "results": resultados,
    }
    summary_output = resolve_path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(resumen, indent=2), encoding="utf-8")

    for fila in sorted(resultados, key=lambda item: item["f1"], reverse=True):
        print(
            f"train={fila['train_run']} test={fila['test_run']} {fila['model']:16} "
            f"precisión={fila['precision']:.3f} recall={fila['recall']:.3f} "
            f"f1={fila['f1']:.3f} FP={fila['false_positives']} FN={fila['false_negatives']}"
        )
    print(f"Resultados de validación cruzada exportados a: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
