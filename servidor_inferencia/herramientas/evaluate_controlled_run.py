from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.vae.detector import VaeDetector
from app.windowing import quality_report
from entrenamiento.baseline import percentile_threshold, window_features

FEATURE_COLUMNS = ("acc_x_g", "acc_y_g", "acc_z_g")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación del número de eventos anómalos detectados frente a un conteo manual de referencia."
    )
    parser.add_argument("--raw-csv", default="../datos/brutos/real_anomaly_controlled_10laps_001.csv")
    parser.add_argument("--normal-train-csv", default="../datos/brutos/real_movement_004.csv")
    parser.add_argument("--vae-model", default="models/vae_real_v2.pth")
    parser.add_argument("--output", default="../datos/analisis/real_anomaly_controlled_10laps_001_events.csv")
    parser.add_argument("--summary-output", default="../datos/analisis/real_anomaly_controlled_10laps_001_summary.json")
    parser.add_argument("--manual-laps", type=int, default=10)
    parser.add_argument("--expected-anomalies-per-lap", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-step", type=int, default=50)
    parser.add_argument("--event-gap-s", type=float, default=1.0)
    parser.add_argument("--vae-thresholds", default="1.20,1.00,0.95,0.90,0.85,0.80")
    parser.add_argument("--rms-percentile", type=float, default=99.5)
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help="Registrar métricas agregadas en Weights & Biases si se encuentra configurado."
    )
    parser.add_argument("--wandb-project", default="tfm-railway-anomaly")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="controlled-10laps-eval")
    return parser.parse_args()


def resolve_path(ruta: str) -> Path:
    candidata = Path(ruta)
    if candidata.is_absolute():
        return candidata
    return (REPO_ROOT / candidata).resolve()


def read_samples(ruta: Path) -> list[dict]:
    muestras: list[dict] = []
    with ruta.open(newline="") as archivo:
        lector = csv.DictReader(archivo)
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


def iter_windows(muestras: list[dict], window_size: int, window_step: int) -> list[list[dict]]:
    return [
        muestras[inicio : inicio + window_size] 
        for inicio in range(0, len(muestras) - window_size + 1, window_step)
    ]


def fit_rms_threshold(muestras: list[dict], window_size: int, window_step: int, percentil: float) -> float:
    puntuaciones = []
    for ventana in iter_windows(muestras, window_size, window_step):
        valores = np.asarray([[fila[col] for col in FEATURE_COLUMNS] for fila in ventana], dtype=np.float32)
        puntuaciones.append(window_features(valores)["energy"])
    return percentile_threshold(np.asarray(puntuaciones, dtype=np.float32), percentil)


def build_window_records(
    muestras: list[dict], 
    detector: VaeDetector, 
    rms_threshold: float, 
    args: argparse.Namespace
) -> list[dict]:
    timestamp_base_ms = muestras[0]["timestamp_ms"]
    registros: list[dict] = []
    for indice, ventana in enumerate(iter_windows(muestras, args.window_size, args.window_step)):
        valores = np.asarray([[fila[col] for col in FEATURE_COLUMNS] for fila in ventana], dtype=np.float32)
        calidad = quality_report(ventana, args.window_size)
        resultado_vae = detector.predict(ventana, calidad)
        puntuacion_rms = window_features(valores)["energy"]
        
        centro_relativo_s = (
            (ventana[0]["timestamp_ms"] - timestamp_base_ms) + 
            (ventana[-1]["timestamp_ms"] - timestamp_base_ms)
        ) / 2000.0
        
        registros.append(
            {
                "window_id": indice,
                "relative_center_s": centro_relativo_s,
                "seq_start": ventana[0]["seq"],
                "seq_end": ventana[-1]["seq"],
                "vae_score_normalized": float(resultado_vae["anomaly_score"]),
                "vae_score": float(resultado_vae["metadata"]["vae_score"]),
                "vae_threshold": float(resultado_vae["metadata"]["threshold"]),
                "rms_score": float(puntuacion_rms),
                "rms_threshold": float(rms_threshold),
                "dynamic_rms_g": float(np.sqrt(np.mean((valores - valores.mean(axis=0, keepdims=True)) ** 2))),
                "peak_abs_g": float(np.max(np.abs(valores))),
            }
        )
    return registros


def group_events(registros: list[dict], score_key: str, umbral: float, event_gap_s: float) -> list[dict]:
    candidatos = [reg for reg in registros if float(reg[score_key]) >= umbral]
    eventos: list[dict] = []
    grupo_actual: list[dict] = []
    ultimo_t: float | None = None
    
    for reg in candidatos:
        t_actual = float(reg["relative_center_s"])
        if ultimo_t is None or t_actual - ultimo_t <= event_gap_s:
            grupo_actual.append(reg)
        else:
            eventos.append(max(grupo_actual, key=lambda item: float(item[score_key])))
            grupo_actual = [reg]
        ultimo_t = t_actual
        
    if grupo_actual:
        eventos.append(max(grupo_actual, key=lambda item: float(item[score_key])))
    return eventos


def maybe_log_wandb(args: argparse.Namespace, resumen: dict) -> None:
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        print("Advertencia: Se solicitó el uso de W&B, pero la librería no se encuentra instalada. Omitiendo registro.", file=sys.stderr)
        return
        
    metricas = resumen["metrics"]
    eventos_esperados = resumen["config"]["expected_events"]
    wandb_metrics = {
        "controlled/duration_s": metricas["duration_s"],
        "controlled/windows": metricas["windows"],
        "controlled/expected_events": eventos_esperados,
        "controlled/rms_threshold": metricas["rms_threshold"],
        "controlled/rms_events": metricas["rms_events"],
        "controlled/rms_event_error": metricas["rms_events"] - eventos_esperados,
        "controlled/rms_event_abs_error": abs(metricas["rms_events"] - eventos_esperados),
    }
    
    for item in resumen["thresholds"]:
        if item["detector"] != "vae":
            continue
        clave_umbral = f"{item['threshold']:.2f}"
        wandb_metrics[f"controlled/vae_events_at_{clave_umbral}"] = item["event_count"]
        wandb_metrics[f"controlled/vae_event_error_at_{clave_umbral}"] = item["event_count_error"]
        wandb_metrics[f"controlled/vae_event_abs_error_at_{clave_umbral}"] = abs(item["event_count_error"])

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
    normal_train_csv = resolve_path(args.normal_train_csv)
    vae_model = resolve_path(args.vae_model)
    output = resolve_path(args.output)
    summary_output = resolve_path(args.summary_output)

    muestras = read_samples(raw_csv)
    muestras_normales = read_samples(normal_train_csv)
    rms_threshold = fit_rms_threshold(muestras_normales, args.window_size, args.window_step, args.rms_percentile)
    detector = VaeDetector(vae_model, device="cpu")
    registros = build_window_records(muestras, detector, rms_threshold, args)

    eventos_esperados = args.manual_laps * args.expected_anomalies_per_lap
    umbrales_vae = [float(valor.strip()) for valor in args.vae_thresholds.split(",") if valor.strip()]
    resumenes_umbrales = []
    
    for umbral in umbrales_vae:
        eventos = group_events(registros, "vae_score_normalized", umbral, args.event_gap_s)
        resumenes_umbrales.append(
            {
                "detector": "vae",
                "threshold": umbral,
                "event_count": len(eventos),
                "expected_events": eventos_esperados,
                "event_count_error": len(eventos) - eventos_esperados,
                "event_times_s": [round(float(ev["relative_center_s"]), 3) for ev in eventos],
            }
        )

    eventos_rms = group_events(registros, "rms_score", rms_threshold, args.event_gap_s)
    resumenes_umbrales.append(
        {
            "detector": "rms",
            "threshold": rms_threshold,
            "event_count": len(eventos_rms),
            "expected_events": eventos_esperados,
            "event_count_error": len(eventos_rms) - eventos_esperados,
            "event_times_s": [round(float(ev["relative_center_s"]), 3) for ev in eventos_rms],
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as archivo:
        columnas = [
            "window_id",
            "relative_center_s",
            "seq_start",
            "seq_end",
            "vae_score_normalized",
            "vae_score",
            "vae_threshold",
            "rms_score",
            "rms_threshold",
            "dynamic_rms_g",
            "peak_abs_g",
        ]
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(registros)

    duracion_s = (muestras[-1]["timestamp_ms"] - muestras[0]["timestamp_ms"]) / 1000.0
    resumen = {
        "config": {
            "raw_csv": str(raw_csv),
            "normal_train_csv": str(normal_train_csv),
            "vae_model": str(vae_model),
            "manual_laps": args.manual_laps,
            "expected_events": eventos_esperados,
            "window_size": args.window_size,
            "window_step": args.window_step,
            "event_gap_s": args.event_gap_s,
            "rms_percentile": args.rms_percentile,
        },
        "metrics": {
            "duration_s": duracion_s,
            "windows": len(registros),
            "rms_threshold": rms_threshold,
            **{f"vae_events_at_{item['threshold']:.2f}": item["event_count"] for item in resumenes_umbrales if item["detector"] == "vae"},
            "rms_events": len(eventos_rms),
        },
        "thresholds": resumenes_umbrales,
    }
    
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    maybe_log_wandb(args, resumen)

    print(f"Archivo crudo analizado: {raw_csv}")
    print(f"Duración total: {duracion_s:.3f} s")
    print(f"Vueltas manuales contabilizadas: {args.manual_laps}")
    print(f"Eventos anómalos esperados: {eventos_esperados}")
    for item in resumenes_umbrales:
        print(
            f"[{item['detector'].upper()}] Umbral: {item['threshold']:.3f} | "
            f"Eventos detectados: {item['event_count']} | Error: {item['event_count_error']} | "
            f"Tiempos (s): {item['event_times_s']}"
        )
    print(f"Puntuaciones de ventana exportadas a: {output}")
    print(f"Resumen de evaluación exportado a: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
