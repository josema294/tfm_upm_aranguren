from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.vae.detector import VaeDetector
from app.windowing import quality_report
from entrenamiento.baseline import percentile_threshold, window_features

FEATURE_COLUMNS = ("acc_x_g", "acc_y_g", "acc_z_g")
RAW_COLUMNS = ("seq", "timestamp_ms", *FEATURE_COLUMNS)
LABEL_TO_BINARY = {
    "normal": "normal",
    "normal_quiet": "normal",
    "normal_tranquilo": "normal",
    "anomaly": "anomaly",
    "anomaly_impact": "anomaly",
    "anomalia_impacto": "anomaly",
    "anomaly_slip_candidate": "anomaly",
    "anomalia_candidato_resbalon": "anomaly",
}


def campo(fila: dict, *nombres: str, defecto: str = "") -> str:
    for nombre in nombres:
        if nombre in fila and fila[nombre] != "":
            return fila[nombre]
    return defecto


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
        description="Evaluación comparativa del modelo VAE y el sistema base RMS sobre ventanas etiquetadas."
    )
    parser.add_argument(
        "--raw-csv",
        default="../datos/brutos/real_anomaly_route_001.csv",
        help="Archivo CSV con el trayecto en bruto conteniendo las columnas seq, timestamp y aceleraciones.",
    )
    parser.add_argument(
        "--labels-csv",
        default="../datos/etiquetas/real_anomaly_route_001_windows.csv",
        help="Archivo CSV con las etiquetas por ventana generadas por el modelo de temporización.",
    )
    parser.add_argument(
        "--normal-train-csv",
        default="../datos/brutos/real_movement_004.csv",
        help="Archivo CSV de movimiento normal empleado para ajustar el umbral operativo del modelo RMS.",
    )
    parser.add_argument(
        "--vae-model",
        default="models/vae_real_v2.pth",
        help="Ruta al artefacto que contiene el modelo VAE entrenado.",
    )
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-step", type=int, default=50)
    parser.add_argument(
        "--rms-percentile",
        type=float,
        default=99.5,
        help="Percentil de energía en el conjunto de entrenamiento normal empleado como umbral anómalo RMS.",
    )
    parser.add_argument(
        "--max-abs-g",
        type=float,
        default=8.0,
        help="Omitir ventanas cuya aceleración absoluta exceda este valor límite.",
    )
    parser.add_argument(
        "--output",
        default="../datos/analisis/real_anomaly_route_001_model_eval.csv",
        help="Ruta opcional para exportar las predicciones por ventana en formato CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default="../datos/analisis/real_anomaly_route_001_model_eval_summary.json",
        help="Ruta opcional para exportar el resumen de métricas agregadas en formato JSON.",
    )
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help="Registrar métricas agregadas en Weights & Biases si se encuentra configurado."
    )
    parser.add_argument("--wandb-project", default="tfm-railway-anomaly")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="labeled-window-eval")
    return parser.parse_args()


def resolve_path(ruta: str) -> Path:
    candidata = Path(ruta)
    if candidata.is_absolute():
        return candidata
    return (REPO_ROOT / candidata).resolve()


def read_raw_samples(ruta: Path) -> list[dict]:
    muestras: list[dict] = []
    with ruta.open(newline="") as archivo:
        lector = csv.DictReader(archivo)
        faltantes = [columna for columna in RAW_COLUMNS if columna not in (lector.fieldnames or [])]
        if faltantes:
            raise ValueError(f"El archivo {ruta} no contiene las columnas requeridas: {faltantes}")
        for fila in lector:
            muestras.append(
                {
                    "seq": int(fila["seq"]),
                    "timestamp_ms": int(fila["timestamp_ms"]),
                    "acc_x_g": float(fila["acc_x_g"]),
                    "acc_y_g": float(fila["acc_y_g"]),
                    "acc_z_g": float(fila["acc_z_g"]),
                }
            )
    return muestras


def read_labels(ruta: Path) -> list[dict]:
    with ruta.open(newline="") as archivo:
        return list(csv.DictReader(archivo))


def label_seq_bounds(etiquetas: list[dict]) -> tuple[int, int] | None:
    inicios = []
    fines = []
    for fila in etiquetas:
        if campo(fila, "label", "etiqueta") in LABEL_TO_BINARY:
            inicios.append(int(campo(fila, "seq_start", "seq_inicio")))
            fines.append(int(campo(fila, "seq_end", "seq_fin")))
    if not inicios or not fines:
        return None
    return min(inicios), max(fines)


def iter_training_windows(muestras: list[dict], window_size: int, window_step: int) -> list[np.ndarray]:
    valores = np.asarray([[fila[col] for col in FEATURE_COLUMNS] for fila in muestras], dtype=np.float32)
    return [
        valores[inicio : inicio + window_size]
        for inicio in range(0, len(valores) - window_size + 1, window_step)
    ]


def fit_rms_threshold(muestras: list[dict], window_size: int, window_step: int, percentil: float) -> float:
    puntuaciones = np.asarray(
        [window_features(ventana)["energy"] for ventana in iter_training_windows(muestras, window_size, window_step)],
        dtype=np.float32,
    )
    return percentile_threshold(puntuaciones, percentil)


def sample_window_by_seq(muestras_por_seq: dict[int, dict], seq_start: int, seq_end: int) -> list[dict]:
    return [muestras_por_seq[seq] for seq in range(seq_start, seq_end + 1) if seq in muestras_por_seq]


def is_glitch(ventana: list[dict], max_abs_g: float) -> bool:
    valores = np.asarray([[fila[col] for col in FEATURE_COLUMNS] for fila in ventana], dtype=np.float32)
    return bool(np.max(np.abs(valores)) > max_abs_g)


def class_metrics(registros: list[dict], clave_prediccion: str) -> Metrics:
    tp = fp = tn = fn = 0
    for reg in registros:
        etiqueta = reg["binary_label"]
        prediccion = reg[clave_prediccion]
        if etiqueta == "anomaly" and prediccion == "anomaly":
            tp += 1
        elif etiqueta == "normal" and prediccion == "anomaly":
            fp += 1
        elif etiqueta == "normal" and prediccion == "normal":
            tn += 1
        elif etiqueta == "anomaly" and prediccion == "normal":
            fn += 1
    return Metrics(total=len(registros), true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn)


def metrics_from_score(registros: list[dict], clave_puntuacion: str, umbral: float) -> Metrics:
    tp = fp = tn = fn = 0
    for reg in registros:
        etiqueta = reg["binary_label"]
        prediccion = "anomaly" if float(reg[clave_puntuacion]) >= umbral else "normal"
        if etiqueta == "anomaly" and prediccion == "anomaly":
            tp += 1
        elif etiqueta == "normal" and prediccion == "anomaly":
            fp += 1
        elif etiqueta == "normal" and prediccion == "normal":
            tn += 1
        elif etiqueta == "anomaly" and prediccion == "normal":
            fn += 1
    return Metrics(total=len(registros), true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn)


def best_threshold(registros: list[dict], clave_puntuacion: str) -> tuple[float, Metrics]:
    puntuaciones = sorted({float(reg[clave_puntuacion]) for reg in registros})
    if not puntuaciones:
        return 0.0, Metrics(total=0, true_positive=0, false_positive=0, true_negative=0, false_negative=0)

    mejor_puntuacion = puntuaciones[0]
    mejores_metricas = metrics_from_score(registros, clave_puntuacion, mejor_puntuacion)
    for puntuacion in puntuaciones:
        metricas = metrics_from_score(registros, clave_puntuacion, puntuacion)
        if metricas.f1 > mejores_metricas.f1:
            mejor_puntuacion = puntuacion
            mejores_metricas = metricas
    return mejor_puntuacion, mejores_metricas


def print_metrics(nombre: str, metricas: Metrics) -> None:
    print(
        f"{nombre:12} total={metricas.total:4d} "
        f"TP={metricas.true_positive:4d} FP={metricas.false_positive:4d} "
        f"TN={metricas.true_negative:4d} FN={metricas.false_negative:4d} "
        f"precisión={metricas.precision:.3f} recall={metricas.recall:.3f} "
        f"f1={metricas.f1:.3f} exactitud={metricas.accuracy:.3f}"
    )


def print_label_recall(registros: list[dict], clave_prediccion: str) -> None:
    etiquetas = sorted({reg["label"] for reg in registros if reg["binary_label"] == "anomaly"})
    for etiqueta in etiquetas:
        registros_etiqueta = [reg for reg in registros if reg["label"] == etiqueta]
        detectados = sum(1 for reg in registros_etiqueta if reg[clave_prediccion] == "anomaly")
        print(f"  {clave_prediccion} {etiqueta}: {detectados}/{len(registros_etiqueta)} recall={detectados / len(registros_etiqueta):.3f}")


def label_recalls(registros: list[dict], clave_prediccion: str) -> dict[str, float]:
    recalls: dict[str, float] = {}
    etiquetas = sorted({reg["label"] for reg in registros if reg["binary_label"] == "anomaly"})
    for etiqueta in etiquetas:
        registros_etiqueta = [reg for reg in registros if reg["label"] == etiqueta]
        detectados = sum(1 for reg in registros_etiqueta if reg[clave_prediccion] == "anomaly")
        recalls[etiqueta] = detectados / len(registros_etiqueta) if registros_etiqueta else 0.0
    return recalls


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


def maybe_log_wandb(args: argparse.Namespace, resumen: dict) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        print("Advertencia: Se solicitó el uso de W&B, pero la librería no se encuentra instalada. Omitiendo registro.", file=sys.stderr)
        return
        
    metricas = resumen["metrics"]
    detalles = resumen["details"]
    wandb_metrics = {
        "eval/evaluated_windows": metricas["evaluated_windows"],
        "eval/skipped_label": metricas["skipped_label"],
        "eval/skipped_glitch": metricas["skipped_glitch"],
        "eval/skipped_size": metricas["skipped_size"],
        "eval/vae_precision": metricas["vae_precision"],
        "eval/vae_recall": metricas["vae_recall"],
        "eval/vae_f1": metricas["vae_f1"],
        "eval/vae_accuracy": metricas["vae_accuracy"],
        "eval/vae_false_positives": detalles["vae"]["false_positive"],
        "eval/vae_false_negatives": detalles["vae"]["false_negative"],
        "eval/vae_true_positives": detalles["vae"]["true_positive"],
        "eval/vae_true_negatives": detalles["vae"]["true_negative"],
        "eval/vae_best_threshold": metricas["vae_best_threshold"],
        "eval/vae_best_f1": metricas["vae_best_f1"],
        "eval/rms_precision": metricas["rms_precision"],
        "eval/rms_recall": metricas["rms_recall"],
        "eval/rms_f1": metricas["rms_f1"],
        "eval/rms_accuracy": metricas["rms_accuracy"],
        "eval/rms_false_positives": detalles["rms"]["false_positive"],
        "eval/rms_false_negatives": detalles["rms"]["false_negative"],
        "eval/rms_true_positives": detalles["rms"]["true_positive"],
        "eval/rms_true_negatives": detalles["rms"]["true_negative"],
        "eval/rms_best_threshold": metricas["rms_best_threshold"],
        "eval/rms_best_f1": metricas["rms_best_f1"],
    }
    for etiqueta, valor in detalles["vae_label_recalls"].items():
        wandb_metrics[f"label_recall/vae_{etiqueta}"] = valor
    for etiqueta, valor in detalles["rms_label_recalls"].items():
        wandb_metrics[f"label_recall/rms_{etiqueta}"] = valor

    run = wandb.init(
        project=args.wandb_project, 
        entity=args.wandb_entity, 
        name=args.wandb_run_name, 
        config=resumen["config"]
    )
    wandb.log(wandb_metrics)
    run.finish()


def main() -> int:
    args = parse_args()
    raw_csv = resolve_path(args.raw_csv)
    labels_csv = resolve_path(args.labels_csv)
    normal_train_csv = resolve_path(args.normal_train_csv)
    vae_model = resolve_path(args.vae_model)
    output = resolve_path(args.output) if args.output else None
    summary_output = resolve_path(args.summary_output) if args.summary_output else None
    
    for ruta_requerida, etiqueta_log in (
        (raw_csv, "CSV de origen"),
        (labels_csv, "CSV de etiquetas"),
        (normal_train_csv, "CSV de entrenamiento normal"),
        (vae_model, "modelo VAE"),
    ):
        if not ruta_requerida.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado ({etiqueta_log}): {ruta_requerida}. "
                "Por favor, restaure el archivo o especifique la ruta correcta."
            )

    muestras_crudas = read_raw_samples(raw_csv)
    muestras_normales = read_raw_samples(normal_train_csv)
    etiquetas = read_labels(labels_csv)
    muestras_por_seq = {fila["seq"]: fila for fila in muestras_crudas}
    raw_min_seq = min(muestras_por_seq)
    raw_max_seq = max(muestras_por_seq)
    
    limites = label_seq_bounds(etiquetas)
    if limites is not None:
        labels_min_seq, labels_max_seq = limites
        if raw_max_seq < labels_min_seq or raw_min_seq > labels_max_seq:
            raise RuntimeError(
                "Incongruencia temporal: El CSV de origen y el archivo de etiquetas no coinciden en su secuencia temporal. "
                f"origen={raw_min_seq}..{raw_max_seq}, etiquetas={labels_min_seq}..{labels_max_seq}. "
                "Verifique que ambos archivos correspondan a la misma captura de telemetría."
            )

    rms_threshold = fit_rms_threshold(
        muestras_normales,
        window_size=args.window_size,
        window_step=args.window_step,
        percentil=args.rms_percentile,
    )
    vae = VaeDetector(vae_model, device="cpu")

    registros: list[dict] = []
    skipped_label = 0
    skipped_glitch = 0
    skipped_size = 0

    for fila_etiqueta in etiquetas:
        etiqueta_actual = campo(fila_etiqueta, "label", "etiqueta")
        binary_label = LABEL_TO_BINARY.get(etiqueta_actual)
        if binary_label is None:
            skipped_label += 1
            continue

        seq_start = int(campo(fila_etiqueta, "seq_start", "seq_inicio"))
        seq_end = int(campo(fila_etiqueta, "seq_end", "seq_fin"))
        ventana = sample_window_by_seq(muestras_por_seq, seq_start, seq_end)
        
        if len(ventana) != args.window_size:
            skipped_size += 1
            continue
        if is_glitch(ventana, args.max_abs_g):
            skipped_glitch += 1
            continue

        arreglo_ventana = np.asarray([[fila[col] for col in FEATURE_COLUMNS] for fila in ventana], dtype=np.float32)
        puntuacion_rms = window_features(arreglo_ventana)["energy"]
        prediccion_rms = "anomaly" if puntuacion_rms >= rms_threshold else "normal"

        calidad = quality_report(ventana, expected_size=args.window_size)
        resultado_vae = vae.predict(ventana, calidad)
        prediccion_vae = "anomaly" if resultado_vae["status"] == "anomaly" else "normal"

        registros.append(
            {
                "window_id": campo(fila_etiqueta, "window_id", "id_ventana"),
                "label": etiqueta_actual,
                "binary_label": binary_label,
                "zone": campo(fila_etiqueta, "zone", "zona"),
                "lap_index": campo(fila_etiqueta, "lap_index", "indice_vuelta"),
                "relative_center_s": campo(fila_etiqueta, "relative_center_s", "centro_relativo_s"),
                "seq_start": seq_start,
                "seq_end": seq_end,
                "rms_score": f"{puntuacion_rms:.8f}",
                "rms_threshold": f"{rms_threshold:.8f}",
                "rms_prediction": prediccion_rms,
                "vae_score": f"{resultado_vae['metadata']['vae_score']:.8f}",
                "vae_score_normalized": f"{resultado_vae['anomaly_score']:.8f}",
                "vae_threshold": f"{resultado_vae['metadata']['threshold']:.8f}",
                "vae_prediction": prediccion_vae,
            }
        )

    print(f"Archivo crudo analizado: {raw_csv}")
    print(f"Archivo de etiquetas: {labels_csv}")
    print(f"Archivo de entrenamiento normal: {normal_train_csv}")
    print(f"Umbral RMS computado: {rms_threshold:.8f} (percentil {args.rms_percentile} de energía basal)")
    print(
        f"Ventanas evaluadas satisfactoriamente: {len(registros)} | Descartadas (etiqueta={skipped_label}, "
        f"artefactos={skipped_glitch}, tamaño={skipped_size})"
    )
    if not registros:
        raise RuntimeError(
            "Análisis fallido: Ninguna ventana etiquetada resultó válida para evaluación. "
            "Asegúrese de que --raw-csv y --labels-csv pertenezcan a la misma sesión de captura."
        )
        
    metricas_vae = class_metrics(registros, "vae_prediction")
    metricas_rms = class_metrics(registros, "rms_prediction")
    print_metrics("VAE", metricas_vae)
    print_metrics("RMS", metricas_rms)

    mejor_umbral_vae, mejores_metricas_vae = best_threshold(registros, "vae_score_normalized")
    mejor_umbral_rms, mejores_metricas_rms = best_threshold(registros, "rms_score")
    print_metrics(f"VAE óptimo@{mejor_umbral_vae:.3f}", mejores_metricas_vae)
    print_metrics(f"RMS óptimo@{mejor_umbral_rms:.3f}", mejores_metricas_rms)
    print_label_recall(registros, "vae_prediction")
    print_label_recall(registros, "rms_prediction")

    recalls_vae = label_recalls(registros, "vae_prediction")
    recalls_rms = label_recalls(registros, "rms_prediction")
    resumen = {
        "config": {
            "raw_csv": str(raw_csv),
            "labels_csv": str(labels_csv),
            "normal_train_csv": str(normal_train_csv),
            "vae_model": str(vae_model),
            "window_size": args.window_size,
            "window_step": args.window_step,
            "rms_percentile": args.rms_percentile,
            "max_abs_g": args.max_abs_g,
        },
        "metrics": {
            "evaluated_windows": len(registros),
            "skipped_label": skipped_label,
            "skipped_glitch": skipped_glitch,
            "skipped_size": skipped_size,
            "rms_threshold": rms_threshold,
            "vae_precision": metricas_vae.precision,
            "vae_recall": metricas_vae.recall,
            "vae_f1": metricas_vae.f1,
            "vae_accuracy": metricas_vae.accuracy,
            "rms_precision": metricas_rms.precision,
            "rms_recall": metricas_rms.recall,
            "rms_f1": metricas_rms.f1,
            "rms_accuracy": metricas_rms.accuracy,
            "vae_best_threshold": mejor_umbral_vae,
            "vae_best_f1": mejores_metricas_vae.f1,
            "rms_best_threshold": mejor_umbral_rms,
            "rms_best_f1": mejores_metricas_rms.f1,
            **{f"vae_recall_{etiqueta}": valor for etiqueta, valor in recalls_vae.items()},
            **{f"rms_recall_{etiqueta}": valor for etiqueta, valor in recalls_rms.items()},
        },
        "details": {
            "vae": metrics_dict(metricas_vae),
            "rms": metrics_dict(metricas_rms),
            "vae_best": metrics_dict(mejores_metricas_vae),
            "rms_best": metrics_dict(mejores_metricas_rms),
            "vae_label_recalls": recalls_vae,
            "rms_label_recalls": recalls_rms,
        },
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(registros[0].keys()) if registros else [])
            if registros:
                escritor.writeheader()
                escritor.writerows(registros)
        print(f"Predicciones por ventana exportadas a: {output}")

    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
        print(f"Resumen de evaluación exportado a: {summary_output}")

    maybe_log_wandb(args, resumen)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError) as excepcion:
        print(f"ERROR CRÍTICO: {excepcion}", file=sys.stderr)
        raise SystemExit(1)
