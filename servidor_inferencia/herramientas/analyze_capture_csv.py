#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EstadisticasCaptura:
    filas: int
    primer_seq: int | None
    ultimo_seq: int | None
    min_seq: int | None
    max_seq: int | None
    reinicios_probables: int
    eventos_desordenados: int
    secuencias_duplicadas: int
    paquetes_perdidos: int
    eventos_caida: int
    duracion_esp_s: float | None
    duracion_pc_s: float | None
    tasa_esp_hz: float | None
    tasa_pc_hz: float | None
    max_brecha_timestamp_ms: int | None
    filas_malformadas: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analizar la calidad del CSV de captura UDP del ESP32.")
    parser.add_argument("csv_path", help="Ruta al archivo CSV de captura para su análisis.")
    parser.add_argument("--expected-hz", type=float, default=100.0, help="Frecuencia de muestreo esperada (Hz).")
    parser.add_argument("--max-loss-ratio", type=float, default=0.01, help="Proporción máxima de pérdida de paquetes aceptable.")
    parser.add_argument("--max-rate-error", type=float, default=0.05, help="Error relativo máximo de la frecuencia de muestreo.")
    return parser.parse_args()


def leer_filas(ruta: Path) -> tuple[list[tuple[int, int, int | None]], int]:
    filas: list[tuple[int, int, int | None]] = []
    malformadas = 0
    with ruta.open(newline="") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            try:
                seq = int(fila["seq"])
                timestamp_ms = int(fila["timestamp_ms"])
                pc_raw = fila.get("pc_timestamp_ns")
                pc_timestamp_ns = int(pc_raw) if pc_raw else None
            except (KeyError, TypeError, ValueError):
                malformadas += 1
                continue
            filas.append((seq, timestamp_ms, pc_timestamp_ns))
    return filas, malformadas


def analizar(ruta: Path) -> EstadisticasCaptura:
    filas, malformadas = leer_filas(ruta)
    if not filas:
        return EstadisticasCaptura(
            filas=0,
            primer_seq=None,
            ultimo_seq=None,
            min_seq=None,
            max_seq=None,
            reinicios_probables=0,
            eventos_desordenados=0,
            secuencias_duplicadas=0,
            paquetes_perdidos=0,
            eventos_caida=0,
            duracion_esp_s=None,
            duracion_pc_s=None,
            tasa_esp_hz=None,
            tasa_pc_hz=None,
            max_brecha_timestamp_ms=None,
            filas_malformadas=malformadas,
        )

    secuencias = [r[0] for r in filas]
    timestamps_pc = [r[2] for r in filas if r[2] is not None]

    reinicios_probables = 0
    eventos_desordenados = 0
    secuencias_duplicadas_crudas = 0

    for idx in range(1, len(filas)):
        seq_prev, ts_prev, _ = filas[idx - 1]
        seq, ts, _ = filas[idx]
        delta_seq = seq - seq_prev
        delta_ts = ts - ts_prev
        if delta_seq < 0:
            # UDP puede llegar ligeramente desordenado. Las inversiones de una sola muestra
            # se tratan como desorden de transporte, no como reinicios del ESP32.
            if delta_seq <= -100 or delta_ts <= -1000:
                reinicios_probables += 1
            else:
                eventos_desordenados += 1
        elif delta_seq == 0:
            secuencias_duplicadas_crudas += 1

    # La pérdida se calcula tras ordenar por seq; de lo contrario, el desorden UDP genera
    # eventos de pérdida falsos entre paquetes adyacentes intercambiados.
    filas_ordenadas = sorted(filas, key=lambda f: f[0])
    filas_sin_duplicados: list[tuple[int, int, int | None]] = []
    secuencias_duplicadas = 0
    seq_anterior: int | None = None
    for fila in filas_ordenadas:
        if seq_anterior is not None and fila[0] == seq_anterior:
            secuencias_duplicadas += 1
            continue
        filas_sin_duplicados.append(fila)
        seq_anterior = fila[0]

    secuencias_ordenadas = [r[0] for r in filas_sin_duplicados]
    timestamps_ordenados = [r[1] for r in filas_sin_duplicados]
    paquetes_perdidos = 0
    eventos_caida = 0
    max_brecha_timestamp_ms = 0

    for idx in range(1, len(filas_sin_duplicados)):
        seq_prev, ts_prev, _ = filas_sin_duplicados[idx - 1]
        seq, ts, _ = filas_sin_duplicados[idx]
        delta_seq = seq - seq_prev
        delta_ts = ts - ts_prev
        if delta_seq > 1:
            eventos_caida += 1
            paquetes_perdidos += delta_seq - 1
        if delta_ts > max_brecha_timestamp_ms:
            max_brecha_timestamp_ms = delta_ts

    duracion_esp_s = None
    tasa_esp_hz = None
    if timestamps_ordenados[-1] > timestamps_ordenados[0]:
        duracion_esp_s = (timestamps_ordenados[-1] - timestamps_ordenados[0]) / 1000.0
        tasa_esp_hz = len(filas_sin_duplicados) / duracion_esp_s if duracion_esp_s > 0 else None

    duracion_pc_s = None
    tasa_pc_hz = None
    if len(timestamps_pc) >= 2 and timestamps_pc[-1] > timestamps_pc[0]:
        duracion_pc_s = (timestamps_pc[-1] - timestamps_pc[0]) / 1_000_000_000.0
        tasa_pc_hz = len(filas) / duracion_pc_s if duracion_pc_s > 0 else None

    return EstadisticasCaptura(
        filas=len(filas),
        primer_seq=secuencias[0],
        ultimo_seq=secuencias[-1],
        min_seq=min(secuencias),
        max_seq=max(secuencias),
        reinicios_probables=reinicios_probables,
        eventos_desordenados=eventos_desordenados,
        secuencias_duplicadas=secuencias_duplicadas or secuencias_duplicadas_crudas,
        paquetes_perdidos=paquetes_perdidos,
        eventos_caida=eventos_caida,
        duracion_esp_s=duracion_esp_s,
        duracion_pc_s=duracion_pc_s,
        tasa_esp_hz=tasa_esp_hz,
        tasa_pc_hz=tasa_pc_hz,
        max_brecha_timestamp_ms=max_brecha_timestamp_ms,
        filas_malformadas=malformadas,
    )


def fmt(valor: float | int | None, decimales: int = 3) -> str:
    if valor is None:
        return "n/d"
    if isinstance(valor, float):
        return f"{valor:.{decimales}f}"
    return str(valor)


def imprimir_informe(ruta: Path, stats: EstadisticasCaptura, expected_hz: float, max_loss_ratio: float, max_rate_error: float) -> int:
    paquetes_esperados = None
    proporcion_perdida = None
    if stats.min_seq is not None and stats.max_seq is not None and stats.reinicios_probables == 0:
        paquetes_esperados = stats.max_seq - stats.min_seq + 1
        if paquetes_esperados > 0:
            proporcion_perdida = stats.paquetes_perdidos / paquetes_esperados

    frecuencia_detectada = stats.tasa_pc_hz or stats.tasa_esp_hz
    error_frecuencia = None
    if frecuencia_detectada is not None:
        error_frecuencia = abs(frecuencia_detectada - expected_hz) / expected_hz

    aceptado = (
        stats.filas > 0
        and stats.reinicios_probables == 0
        and (proporcion_perdida is None or proporcion_perdida <= max_loss_ratio)
        and (error_frecuencia is None or error_frecuencia <= max_rate_error)
    )

    print(f"Archivo: {ruta}")
    print(f"Filas: {stats.filas}")
    print(f"Secuencia: primera={stats.primer_seq} última={stats.ultimo_seq} min={stats.min_seq} max={stats.max_seq}")
    print(f"Reinicios probables: {stats.reinicios_probables}")
    print(f"Eventos UDP desordenados: {stats.eventos_desordenados}")
    print(f"Secuencias duplicadas: {stats.secuencias_duplicadas}")
    print(f"Eventos de caída: {stats.eventos_caida}")
    print(f"Paquetes perdidos: {stats.paquetes_perdidos}")
    print(f"Proporción de pérdida: {fmt(proporcion_perdida, 4)}")
    print(f"Duración ESP (s): {fmt(stats.duracion_esp_s)}")
    print(f"Duración PC (s): {fmt(stats.duracion_pc_s)}")
    print(f"Frecuencia ESP (Hz): {fmt(stats.tasa_esp_hz)}")
    print(f"Frecuencia PC (Hz): {fmt(stats.tasa_pc_hz)}")
    print(f"Brecha máxima de timestamp (ms): {stats.max_brecha_timestamp_ms}")
    print(f"Filas malformadas: {stats.filas_malformadas}")
    print(f"Veredicto: {'ACEPTABLE' if aceptado else 'REVISAR'}")

    if stats.reinicios_probables:
        print("Motivo: decremento pronunciado de secuencia/timestamp; probable reinicio del ESP32 o firmware.")
    elif proporcion_perdida is not None and proporcion_perdida > max_loss_ratio:
        print("Motivo: la pérdida de paquetes excede el umbral configurado.")
    elif error_frecuencia is not None and error_frecuencia > max_rate_error:
        print("Motivo: la frecuencia de captura difiere de la frecuencia de muestreo esperada.")
    elif stats.filas == 0:
        print("Motivo: captura vacía.")

    return 0 if aceptado else 2


def main() -> int:
    args = parse_args()
    ruta = Path(args.csv_path)
    stats = analizar(ruta)
    return imprimir_informe(ruta, stats, args.expected_hz, args.max_loss_ratio, args.max_rate_error)


if __name__ == "__main__":
    raise SystemExit(main())
