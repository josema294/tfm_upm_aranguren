#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_COLUMNS = ["acc_x_g", "acc_y_g", "acc_z_g"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construir etiquetas refinadas de resbalón alrededor del núcleo dinámico de los intervalos manuales.")
    parser.add_argument("--raw-csv", required=True, help="Ruta al CSV original.")
    parser.add_argument("--events-csv", required=True, help="Ruta al CSV de eventos.")
    parser.add_argument("--window-size", type=int, default=50, help="Tamaño de la ventana (muestras).")
    parser.add_argument("--window-step", type=int, default=10, help="Paso de la ventana (muestras).")
    parser.add_argument("--core-half-s", type=float, default=0.18, help="Mitad de la duración del núcleo dinámico (s).")
    parser.add_argument("--near-ignore-s", type=float, default=0.45, help="Margen cercano a ignorar (s).")
    parser.add_argument("--context-s", type=float, default=2.0, help="Contexto adicional (s).")
    parser.add_argument("--labels-output", required=True, help="Ruta de salida para las etiquetas.")
    parser.add_argument("--summary-output", required=True, help="Ruta de salida para el resumen JSON.")
    return parser.parse_args()


def resolver(ruta: str) -> Path:
    candidata = Path(ruta)
    if candidata.is_absolute():
        return candidata
    return (REPO_ROOT / candidata).resolve()


def leer_crudo(ruta: Path) -> list[dict]:
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


def leer_intervalos(ruta: Path) -> tuple[list[dict], dict[str, int]]:
    intervalos: list[dict] = []
    activo: dict | None = None
    estadisticas = {
        "raw_events": 0,
        "completed_intervals": 0,
        "ignored_end_without_start": 0,
        "ignored_start_overwrite": 0,
        "ignored_unclosed_start": 0,
    }
    with ruta.open(newline="") as archivo:
        for fila in csv.DictReader(archivo):
            estadisticas["raw_events"] += 1
            if fila["event"] == "start":
                if activo is not None:
                    estadisticas["ignored_start_overwrite"] += 1
                activo = fila
            elif fila["event"] == "end" and activo is not None:
                intervalos.append(
                    {
                        "label": activo["label"],
                        "start_seq": int(activo["last_seq"]),
                        "end_seq": int(fila["last_seq"]),
                        "start_elapsed_s": float(activo["elapsed_s"]),
                        "end_elapsed_s": float(fila["elapsed_s"]),
                    }
                )
                estadisticas["completed_intervals"] += 1
                activo = None
            elif fila["event"] == "end":
                estadisticas["ignored_end_without_start"] += 1
    if activo is not None:
        estadisticas["ignored_unclosed_start"] += 1
    return intervalos, estadisticas


def caracteristicas_ventana(ventana: list[dict]) -> dict[str, float]:
    valores = np.asarray([[fila[columna] for columna in BASE_COLUMNS] for fila in ventana], dtype=np.float32)
    x = valores[:, 0]
    magnitud = np.sqrt(np.sum(valores**2, axis=1))
    dx = np.abs(np.diff(x, prepend=x[0]))
    centrados = valores - valores.mean(axis=0, keepdims=True)
    return {
        "x_std": float(np.std(x)),
        "x_range": float(np.ptp(x)),
        "x_slope": float(np.polyfit(np.arange(len(x)), x, 1)[0]) if len(x) > 1 else 0.0,
        "dx_p95": float(np.quantile(dx, 0.95)),
        "dx_peak": float(np.max(dx)),
        "mag_std": float(np.std(magnitud)),
        "dynamic_rms": float(np.sqrt(np.mean(np.sum(centrados**2, axis=1)))),
    }


def escalado_robusto(valores: np.ndarray) -> np.ndarray:
    mediana = np.median(valores)
    q1, q3 = np.quantile(valores, [0.25, 0.75])
    rango_intercuartilico = max(float(q3 - q1), 1e-6)
    return (valores - mediana) / rango_intercuartilico


def anotar_nucleos_intervalo(crudo: list[dict], intervalos: list[dict], muestras_mitad_nucleo: int) -> list[dict]:
    por_secuencia = {fila["seq"]: fila for fila in crudo}
    nucleos: list[dict] = []
    for indice, intervalo in enumerate(intervalos):
        inicio = min(intervalo["start_seq"], intervalo["end_seq"])
        fin = max(intervalo["start_seq"], intervalo["end_seq"])
        filas = [por_secuencia[seq] for seq in range(inicio, fin + 1) if seq in por_secuencia]
        if len(filas) < 3:
            continue

        valores = np.asarray([[fila[columna] for columna in BASE_COLUMNS] for fila in filas], dtype=np.float32)
        x = valores[:, 0]
        magnitud = np.sqrt(np.sum(valores**2, axis=1))
        dx = np.abs(np.diff(x, prepend=x[0]))
        d_magnitud = np.abs(np.diff(magnitud, prepend=magnitud[0]))
        caida_local_x = np.maximum.accumulate(x) - x
        puntaje = escalado_robusto(dx) + 0.5 * escalado_robusto(d_magnitud) + 0.5 * escalado_robusto(caida_local_x)
        indice_pico = int(np.argmax(puntaje))
        secuencia_pico = int(filas[indice_pico]["seq"])
        nucleos.append(
            {
                "interval_index": indice,
                "manual_start_seq": inicio,
                "manual_end_seq": fin,
                "core_start_seq": secuencia_pico - muestras_mitad_nucleo,
                "core_end_seq": secuencia_pico + muestras_mitad_nucleo,
                "peak_seq": secuencia_pico,
                "peak_timestamp_ms": int(filas[indice_pico]["timestamp_ms"]),
                "peak_score": float(puntaje[indice_pico]),
                "manual_duration_samples": fin - inicio + 1,
            }
        )
    return nucleos


def etiquetar_ventana(
    centro_seq: int,
    nucleos: list[dict],
    muestras_cercanas_ignorar: int,
    min_anotado: int,
    max_anotado: int,
) -> tuple[str, int | None]:
    if centro_seq < min_anotado or centro_seq > max_anotado:
        return "ignore_unannotated", None

    for nucleo in nucleos:
        if nucleo["core_start_seq"] <= centro_seq <= nucleo["core_end_seq"]:
            return "manual_slip_core", int(nucleo["interval_index"])

    for nucleo in nucleos:
        inicio = min(nucleo["manual_start_seq"], nucleo["core_start_seq"] - muestras_cercanas_ignorar)
        fin = max(nucleo["manual_end_seq"], nucleo["core_end_seq"] + muestras_cercanas_ignorar)
        if inicio <= centro_seq <= fin:
            return "ignore_near_manual_slip", int(nucleo["interval_index"])

    return "manual_normal_between", None


def main() -> int:
    args = parse_args()
    csv_crudo = resolver(args.raw_csv)
    csv_eventos = resolver(args.events_csv)
    salida_etiquetas = resolver(args.labels_output)
    salida_resumen = resolver(args.summary_output)

    crudo = leer_crudo(csv_crudo)
    intervalos, estadisticas_eventos = leer_intervalos(csv_eventos)
    if not intervalos:
        raise ValueError(f"No se encontraron intervalos completos de inicio/fin en {csv_eventos}")

    primer_timestamp_ms = crudo[0]["timestamp_ms"]
    muestras_mitad_nucleo = int(round(args.core_half_s * 100))
    muestras_cercanas_ignorar = int(round(args.near_ignore_s * 100))
    muestras_contexto = int(round(args.context_s * 100))
    nucleos = anotar_nucleos_intervalo(crudo, intervalos, muestras_mitad_nucleo)
    min_anotado = min(nucleo["manual_start_seq"] for nucleo in nucleos) - muestras_contexto
    max_anotado = max(nucleo["manual_end_seq"] for nucleo in nucleos) + muestras_contexto

    etiquetas: list[dict] = []
    for inicio in range(0, len(crudo) - args.window_size + 1, args.window_step):
        ventana = crudo[inicio : inicio + args.window_size]
        centro = ventana[len(ventana) // 2]
        etiqueta, indice_intervalo = etiquetar_ventana(centro["seq"], nucleos, muestras_cercanas_ignorar, min_anotado, max_anotado)
        nucleo = nucleos[indice_intervalo] if indice_intervalo is not None else None
        etiquetas.append(
            {
                "window_id": len(etiquetas),
                "relative_center_s": (centro["timestamp_ms"] - primer_timestamp_ms) / 1000,
                "seq_start": ventana[0]["seq"],
                "seq_end": ventana[-1]["seq"],
                "seq_center": centro["seq"],
                "label": etiqueta,
                "interval_index": "" if indice_intervalo is None else indice_intervalo,
                "core_peak_seq": "" if nucleo is None else nucleo["peak_seq"],
                "core_start_seq": "" if nucleo is None else nucleo["core_start_seq"],
                "core_end_seq": "" if nucleo is None else nucleo["core_end_seq"],
                **caracteristicas_ventana(ventana),
            }
        )

    salida_etiquetas.parent.mkdir(parents=True, exist_ok=True)
    with salida_etiquetas.open("w", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(etiquetas[0].keys()))
        escritor.writeheader()
        escritor.writerows(etiquetas)

    conteos = Counter(fila["label"] for fila in etiquetas)
    resumen = {
        "raw_csv": str(csv_crudo),
        "events_csv": str(csv_eventos),
        "labels_output": str(salida_etiquetas),
        "rows": len(crudo),
        "duration_s": (crudo[-1]["timestamp_ms"] - crudo[0]["timestamp_ms"]) / 1000,
        "intervals": len(intervalos),
        "cores": len(nucleos),
        "event_stats": estadisticas_eventos,
        "window_size": args.window_size,
        "window_step": args.window_step,
        "core_half_s": args.core_half_s,
        "near_ignore_s": args.near_ignore_s,
        "context_s": args.context_s,
        "label_counts": dict(conteos),
        "core_examples": nucleos[:5],
    }
    salida_resumen.parent.mkdir(parents=True, exist_ok=True)
    salida_resumen.write_text(json.dumps(resumen, indent=2), encoding="utf-8")

    print(json.dumps(resumen, indent=2))
    print(f"Etiquetas guardadas en: {salida_etiquetas}")
    print(f"Resumen guardado en: {salida_resumen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
