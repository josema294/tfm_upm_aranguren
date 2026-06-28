#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.vae.detector import VaeDetector
from app.windowing import quality_report


SLIP_CORE_HALF_WIDTH_S = 0.30
SLIP_TRANSITION_HALF_WIDTH_S = 0.65
STARTUP_IGNORE_S = 15.0
OUTLIER_PEAK_ABS_X_G = 8.0
OUTLIER_X_RANGE_G = 8.0


@dataclass
class Metrics:
    total: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominador = self.true_positive + self.false_positive
        return self.true_positive / denominador if denominador else 0.0

    @property
    def recall(self) -> float:
        denominador = self.true_positive + self.false_negative
        return self.true_positive / denominador if denominador else 0.0

    @property
    def f1(self) -> float:
        denominador = self.precision + self.recall
        return 2 * self.precision * self.recall / denominador if denominador else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación de modelos VAE contra etiquetas de temporización exclusivamente orientadas a deslizamientos."
    )
    parser.add_argument("--raw-csv", default="../datos/brutos/real_slip_only_001.csv")
    parser.add_argument("--labels-csv", default="../datos/etiquetas/real_slip_only_001_windows.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--output", default="../datos/analisis/slip_only/slip_only_model_eval.csv")
    parser.add_argument("--summary-output", default="../datos/analisis/slip_only/slip_only_model_eval_summary.json")
    parser.add_argument("--phase-s", type=float, default=None)
    parser.add_argument("--period-s", type=float, default=None)
    return parser.parse_args()


def resolve_path(ruta: str) -> Path:
    candidata = Path(ruta)
    if candidata.is_absolute():
        return candidata
    return (REPO_ROOT / candidata).resolve()


def read_raw(ruta: Path) -> dict[int, dict]:
    filas: dict[int, dict] = {}
    with ruta.open(newline="") as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            seq = int(fila["seq"])
            filas[seq] = {
                "seq": seq,
                "timestamp_ms": int(fila["timestamp_ms"]),
                "acc_x_g": float(fila["acc_x_g"]),
                "acc_y_g": float(fila["acc_y_g"]),
                "acc_z_g": float(fila["acc_z_g"]),
            }
    return filas


def read_raw_ordered(ruta: Path) -> list[dict]:
    with ruta.open(newline="") as archivo:
        filas = []
        for fila in csv.DictReader(archivo):
            filas.append(
                {
                    "seq": int(fila["seq"]),
                    "timestamp_ms": int(fila["timestamp_ms"]),
                    "acc_x_g": float(fila["acc_x_g"]),
                    "acc_y_g": float(fila["acc_y_g"]),
                    "acc_z_g": float(fila["acc_z_g"]),
                }
            )
    return filas


def read_labels(ruta: Path) -> list[dict]:
    with ruta.open(newline="") as archivo:
        return list(csv.DictReader(archivo))


def infer_timing(etiquetas: list[dict], phase_s: float | None, period_s: float | None) -> tuple[float, float]:
    if phase_s is not None and period_s is not None:
        return phase_s, period_s
    for fila in etiquetas:
        if fila.get("phase_s") and fila.get("period_s"):
            return float(fila["phase_s"]), float(fila["period_s"])
    raise ValueError("Se requieren los parámetros phase_s y period_s si el CSV de etiquetas omite dichos metadatos de temporización.")


def phase_offset(centro_relativo_s: float, phase_s: float, period_s: float) -> float:
    return ((centro_relativo_s - phase_s + period_s / 2) % period_s) - period_s / 2


def label_window(ventana: list[dict], timestamp_inicial_ms: int, phase_s: float, period_s: float) -> tuple[str, str, float, float, float, float]:
    centro_timestamp_ms = (int(ventana[0]["timestamp_ms"]) + int(ventana[-1]["timestamp_ms"])) / 2
    centro_relativo_s = (centro_timestamp_ms - timestamp_inicial_ms) / 1000
    desplazamiento_s = phase_offset(centro_relativo_s, phase_s, period_s)
    valores_x = [float(fila["acc_x_g"]) for fila in ventana]
    pico_abs_x = max(abs(valor) for valor in valores_x)
    rango_x = max(valores_x) - min(valores_x)

    if centro_relativo_s < STARTUP_IGNORE_S:
        etiqueta = "ignore_startup"
    elif pico_abs_x >= OUTLIER_PEAK_ABS_X_G or rango_x >= OUTLIER_X_RANGE_G:
        etiqueta = "ignore_outlier_impact"
    elif abs(desplazamiento_s) <= SLIP_CORE_HALF_WIDTH_S:
        etiqueta = "slip_confirmed_core"
    elif abs(desplazamiento_s) <= SLIP_TRANSITION_HALF_WIDTH_S:
        etiqueta = "slip_transition"
    else:
        etiqueta = "normal_cycle"

    binary_label = {
        "normal_cycle": "normal",
        "slip_confirmed_core": "anomaly",
    }.get(etiqueta, "ignore")
    return etiqueta, binary_label, centro_relativo_s, desplazamiento_s, pico_abs_x, rango_x


def class_metrics(registros: list[dict]) -> Metrics:
    tp = fp = tn = fn = 0
    for reg in registros:
        etiqueta = reg["binary_label"]
        prediccion = reg["prediction"]
        if etiqueta == "anomaly" and prediccion == "anomaly":
            tp += 1
        elif etiqueta == "normal" and prediccion == "anomaly":
            fp += 1
        elif etiqueta == "normal" and prediccion == "normal":
            tn += 1
        elif etiqueta == "anomaly" and prediccion == "normal":
            fn += 1
    return Metrics(total=len(registros), true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn)


def metrics_dict(metricas: Metrics) -> dict[str, float | int]:
    return {
        "total": metricas.total,
        "true_positive": metricas.true_positive,
        "false_positive": metricas.false_positive,
        "true_negative": metricas.true_negative,
        "false_negative": metricas.false_negative,
        "precision": metricas.precision,
        "recall": metricas.recall,
        "f1": metricas.f1,
        "accuracy": metricas.accuracy,
    }


def main() -> int:
    args = parse_args()
    raw_csv = resolve_path(args.raw_csv)
    labels_csv = resolve_path(args.labels_csv)
    models_dir = resolve_path(args.models_dir)
    output = resolve_path(args.output)
    summary_output = resolve_path(args.summary_output)

    muestras = read_raw_ordered(raw_csv)
    etiquetas = read_labels(labels_csv)
    phase_s, period_s = infer_timing(etiquetas, args.phase_s, args.period_s)
    rutas_modelos = sorted(models_dir.glob("vae_real_v*_*.pth"))
    rutas_modelos = [ruta for ruta in rutas_modelos if "window" in ruta.name]

    registros_globales: list[dict] = []
    resumenes: list[dict] = []
    
    for ruta_modelo in rutas_modelos:
        detector = VaeDetector(ruta_modelo, device="cpu")
        window_size = int(detector.config.get("window_size", 100))
        window_step = int(detector.config.get("window_step", max(1, window_size // 2)))
        timestamp_inicial_ms = int(muestras[0]["timestamp_ms"])
        registros_modelo: list[dict] = []
        ventanas_evaluadas = 0
        ventanas_ignoradas = 0

        for inicio in range(0, len(muestras) - window_size + 1, window_step):
            ventana = muestras[inicio : inicio + window_size]
            etiqueta, binary_label, centro_relativo_s, desplazamiento_s, pico_abs_x, rango_x = label_window(
                ventana, timestamp_inicial_ms, phase_s, period_s
            )
            if binary_label == "ignore":
                ventanas_ignoradas += 1
                continue

            resultado = detector.predict(ventana, quality_report(ventana, window_size))
            prediccion = "anomaly" if resultado["status"] == "anomaly" else "normal"
            registro = {
                "model": ruta_modelo.name,
                "label": etiqueta,
                "binary_label": binary_label,
                "prediction": prediccion,
                "relative_center_s": centro_relativo_s,
                "phase_offset_s": desplazamiento_s,
                "seq_start": int(ventana[0]["seq"]),
                "seq_end": int(ventana[-1]["seq"]),
                "score_normalized": float(resultado["anomaly_score"]),
                "vae_score": float(resultado["metadata"]["vae_score"]),
                "threshold": float(resultado["metadata"]["threshold"]),
                "peak_abs_x": pico_abs_x,
                "x_range": rango_x,
            }
            registros_modelo.append(registro)
            registros_globales.append(registro)
            ventanas_evaluadas += 1

        metricas = class_metrics(registros_modelo)
        resumenes.append(
            {
                "model": ruta_modelo.name,
                "window_size": window_size,
                "window_step": window_step,
                "feature_columns": detector.feature_columns,
                "phase_s": phase_s,
                "period_s": period_s,
                "evaluated_windows": ventanas_evaluadas,
                "ignored_windows": ventanas_ignoradas,
                **metrics_dict(metricas),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if registros_globales:
        with output.open("w", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(registros_globales[0].keys()))
            escritor.writeheader()
            escritor.writerows(registros_globales)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps({"models": resumenes}, indent=2), encoding="utf-8")
    
    for fila in sorted(resumenes, key=lambda item: (item["f1"], item["recall"], -item["false_positive"]), reverse=True):
        print(
            f"{fila['model']:48} total={fila['total']:4d} TP={fila['true_positive']:3d} "
            f"FP={fila['false_positive']:4d} TN={fila['true_negative']:4d} FN={fila['false_negative']:3d} "
            f"precisión={fila['precision']:.3f} recall={fila['recall']:.3f} f1={fila['f1']:.3f}"
        )
    print(f"Predicciones exportadas a: {output}")
    print(f"Resumen de evaluación exportado a: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
