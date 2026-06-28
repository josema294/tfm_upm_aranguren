#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construir etiquetas de ventana a partir de anotaciones manuales de eventos de teclado.")
    parser.add_argument("--raw-csv", default="../datos/brutos/real_slip_manual_001.csv", help="Ruta al CSV original.")
    parser.add_argument("--events-csv", default="../datos/brutos/real_slip_manual_001_events.csv", help="Ruta al CSV de eventos.")
    parser.add_argument("--window-size", type=int, default=50, help="Tamaño de la ventana (muestras).")
    parser.add_argument("--window-step", type=int, default=10, help="Paso de la ventana (muestras).")
    parser.add_argument("--edge-ignore-s", type=float, default=0.12, help="Margen de borde a ignorar (s).")
    parser.add_argument("--near-ignore-s", type=float, default=0.40, help="Margen cercano a ignorar (s).")
    parser.add_argument("--context-s", type=float, default=2.0, help="Contexto adicional (s).")
    parser.add_argument("--labels-output", default="../datos/etiquetas/real_slip_manual_001_windows.csv", help="Ruta de salida para las etiquetas de ventana.")
    parser.add_argument("--summary-output", default="../datos/analisis/slip_manual/real_slip_manual_001_summary.json", help="Ruta de salida para el resumen JSON.")
    parser.add_argument("--figures-dir", default="../datos/figuras/slip_manual", help="Directorio para las figuras.")
    return parser.parse_args()


def resolver(ruta: str) -> Path:
    candidata = Path(ruta)
    if candidata.is_absolute():
        return candidata
    return (REPO_ROOT / candidata).resolve()


def leer_crudo(ruta: Path) -> list[dict]:
    with ruta.open(newline="") as archivo:
        return [
            {
                "pc_timestamp_ns": int(fila["pc_timestamp_ns"]),
                "seq": int(fila["seq"]),
                "timestamp_ms": int(fila["timestamp_ms"]),
                "acc_x_g": float(fila["acc_x_g"]),
                "acc_y_g": float(fila["acc_y_g"]),
                "acc_z_g": float(fila["acc_z_g"]),
            }
            for fila in csv.DictReader(archivo)
        ]


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
                        "start_elapsed_s": float(activo["elapsed_s"]),
                        "end_elapsed_s": float(fila["elapsed_s"]),
                        "start_seq": int(activo["last_seq"]),
                        "end_seq": int(fila["last_seq"]),
                        "start_timestamp_ms": int(activo["last_timestamp_ms"]),
                        "end_timestamp_ms": int(fila["last_timestamp_ms"]),
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
    valores = np.asarray([[fila["acc_x_g"], fila["acc_y_g"], fila["acc_z_g"]] for fila in ventana], dtype=np.float32)
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


def etiquetar_ventana(centro_seq: int, intervalos: list[dict], muestras_borde_ignorar: int, muestras_cercanas_ignorar: int, min_anotado: int, max_anotado: int) -> tuple[str, int | None]:
    if centro_seq < min_anotado or centro_seq > max_anotado:
        return "ignore_unannotated", None

    for indice, intervalo in enumerate(intervalos):
        inicio = min(intervalo["start_seq"], intervalo["end_seq"])
        fin = max(intervalo["start_seq"], intervalo["end_seq"])
        if inicio <= centro_seq <= fin:
            if centro_seq - inicio < muestras_borde_ignorar or fin - centro_seq < muestras_borde_ignorar:
                return "ignore_manual_edge", indice
            return "manual_slip_core", indice

    for indice, intervalo in enumerate(intervalos):
        inicio = min(intervalo["start_seq"], intervalo["end_seq"])
        fin = max(intervalo["start_seq"], intervalo["end_seq"])
        if inicio - muestras_cercanas_ignorar <= centro_seq <= fin + muestras_cercanas_ignorar:
            return "ignore_near_manual_slip", indice

    return "manual_normal_between", None


def guardar_graficos(crudo: list[dict], intervalos: list[dict], etiquetas_generadas: list[dict], directorio_figuras: Path, raiz_nombre: str) -> None:
    directorio_figuras.mkdir(parents=True, exist_ok=True)
    t = np.asarray([(fila["timestamp_ms"] - crudo[0]["timestamp_ms"]) / 1000 for fila in crudo], dtype=np.float32)
    x = np.asarray([fila["acc_x_g"] for fila in crudo], dtype=np.float32)
    y = np.asarray([fila["acc_y_g"] for fila in crudo], dtype=np.float32)
    z = np.asarray([fila["acc_z_g"] for fila in crudo], dtype=np.float32)
    magnitud = np.sqrt(x**2 + y**2 + z**2)

    t_etiqueta = np.asarray([float(fila["relative_center_s"]) for fila in etiquetas_generadas], dtype=np.float32)
    puntaje_etiqueta = np.asarray([float(fila["dx_p95"]) for fila in etiquetas_generadas], dtype=np.float32)
    color_etiqueta = [
        "tab:red" if fila["label"] == "manual_slip_core" else "tab:blue" if fila["label"] == "manual_normal_between" else "0.7"
        for fila in etiquetas_generadas
    ]

    inicio = max(0.0, intervalos[0]["start_elapsed_s"] - 5.0)
    fin = min(float(t[-1]), intervalos[-1]["end_elapsed_s"] + 5.0)
    tamano_bloque = 60.0
    actual = inicio
    while actual < fin:
        detener = min(actual + tamano_bloque, fin)
        mascara = (t >= actual) & (t <= detener)
        fig, ejes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
        ejes[0].plot(t[mascara], x[mascara], label="x", linewidth=0.9)
        ejes[0].plot(t[mascara], y[mascara], label="y", linewidth=0.8, alpha=0.8)
        ejes[0].plot(t[mascara], z[mascara], label="z", linewidth=0.8, alpha=0.8)
        ejes[0].plot(t[mascara], magnitud[mascara], label="magnitud", linewidth=0.8, alpha=0.6)
        for intervalo in intervalos:
            a = intervalo["start_elapsed_s"]
            b = intervalo["end_elapsed_s"]
            if b >= actual and a <= detener:
                ejes[0].axvspan(a, b, color="tab:red", alpha=0.18)
                ejes[1].axvspan(a, b, color="tab:red", alpha=0.18)
        ejes[0].set_ylabel("aceleración (g)")
        ejes[0].legend(loc="upper right", ncol=4)
        mascara_L = (t_etiqueta >= actual) & (t_etiqueta <= detener)
        ejes[1].scatter(t_etiqueta[mascara_L], puntaje_etiqueta[mascara_L], c=np.asarray(color_etiqueta, dtype=object)[mascara_L], s=12)
        ejes[1].set_ylabel("dx p95")
        ejes[1].set_xlabel("tiempo desde el inicio de la captura (s)")
        ejes[1].grid(True, alpha=0.2)
        fig.suptitle(f"{raiz_nombre}: {actual:.0f}-{detener:.0f}s")
        fig.tight_layout()
        fig.savefig(directorio_figuras / f"{raiz_nombre}_review_{int(actual):04d}_{int(detener):04d}s.png", dpi=140)
        plt.close(fig)
        actual = detener


def main() -> int:
    args = parse_args()
    csv_crudo = resolver(args.raw_csv)
    csv_eventos = resolver(args.events_csv)
    salida_etiquetas = resolver(args.labels_output)
    salida_resumen = resolver(args.summary_output)
    directorio_figuras = resolver(args.figures_dir)

    crudo = leer_crudo(csv_crudo)
    intervalos, estadisticas_eventos = leer_intervalos(csv_eventos)
    if not intervalos:
        raise ValueError(f"No se encontraron intervalos completos de inicio/fin en {csv_eventos}")

    primer_timestamp_ms = crudo[0]["timestamp_ms"]
    muestras_borde_ignorar = int(round(args.edge_ignore_s * 100))
    muestras_cercanas_ignorar = int(round(args.near_ignore_s * 100))
    muestras_contexto = int(round(args.context_s * 100))
    min_anotado = min(intervalo["start_seq"] for intervalo in intervalos) - muestras_contexto
    max_anotado = max(intervalo["end_seq"] for intervalo in intervalos) + muestras_contexto

    etiquetas: list[dict] = []
    for inicio in range(0, len(crudo) - args.window_size + 1, args.window_step):
        ventana = crudo[inicio : inicio + args.window_size]
        centro = ventana[len(ventana) // 2]
        etiqueta, indice_intervalo = etiquetar_ventana(
            centro["seq"],
            intervalos,
            muestras_borde_ignorar,
            muestras_cercanas_ignorar,
            min_anotado,
            max_anotado,
        )
        caracteristicas = caracteristicas_ventana(ventana)
        etiquetas.append(
            {
                "window_id": len(etiquetas),
                "relative_center_s": (centro["timestamp_ms"] - primer_timestamp_ms) / 1000,
                "seq_start": ventana[0]["seq"],
                "seq_end": ventana[-1]["seq"],
                "seq_center": centro["seq"],
                "label": etiqueta,
                "interval_index": "" if indice_intervalo is None else indice_intervalo,
                **caracteristicas,
            }
        )

    salida_etiquetas.parent.mkdir(parents=True, exist_ok=True)
    with salida_etiquetas.open("w", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(etiquetas[0].keys()))
        escritor.writeheader()
        escritor.writerows(etiquetas)

    conteos = Counter(fila["label"] for fila in etiquetas)
    duraciones = [intervalo["end_elapsed_s"] - intervalo["start_elapsed_s"] for intervalo in intervalos]
    brechas = [
        intervalos[indice + 1]["start_elapsed_s"] - intervalos[indice]["start_elapsed_s"]
        for indice in range(len(intervalos) - 1)
    ]
    resumen = {
        "raw_csv": str(csv_crudo),
        "events_csv": str(csv_eventos),
        "labels_output": str(salida_etiquetas),
        "rows": len(crudo),
        "duration_s": (crudo[-1]["timestamp_ms"] - crudo[0]["timestamp_ms"]) / 1000,
        "intervals": len(intervalos),
        "event_stats": estadisticas_eventos,
        "first_event_s": intervalos[0]["start_elapsed_s"],
        "last_event_s": intervalos[-1]["end_elapsed_s"],
        "event_duration_s": {
            "min": min(duraciones),
            "median": float(np.median(duraciones)),
            "mean": float(np.mean(duraciones)),
            "max": max(duraciones),
        },
        "start_to_start_gap_s": {
            "min": min(brechas) if brechas else None,
            "median": float(np.median(brechas)) if brechas else None,
            "mean": float(np.mean(brechas)) if brechas else None,
            "max": max(brechas) if brechas else None,
        },
        "window_size": args.window_size,
        "window_step": args.window_step,
        "edge_ignore_s": args.edge_ignore_s,
        "near_ignore_s": args.near_ignore_s,
        "context_s": args.context_s,
        "label_counts": dict(conteos),
    }
    salida_resumen.parent.mkdir(parents=True, exist_ok=True)
    salida_resumen.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    guardar_graficos(crudo, intervalos, etiquetas, directorio_figuras, csv_crudo.stem)

    print(json.dumps(resumen, indent=2))
    print(f"Etiquetas guardadas en: {salida_etiquetas}")
    print(f"Resumen guardado en: {salida_resumen}")
    print(f"Figuras guardadas en: {directorio_figuras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
