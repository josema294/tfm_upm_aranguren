#!/usr/bin/env python3
import argparse
import csv
import pathlib
import sys
import time

import serial


def parse_args():
    parser = argparse.ArgumentParser(
        description="Captura de datos CSV por puerto serie desde el firmware del ADXL345 ESP32."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Ruta del puerto serie (ej: /dev/ttyUSB0).")
    parser.add_argument("--baud", type=int, default=115200, help="Tasa de baudios del puerto serie.")
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida para el archivo CSV (ej: ../datos/brutos/normal_001.csv).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Duración de la captura en segundos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ruta_salida = pathlib.Path(args.output)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    tiempo_limite = time.monotonic() + args.seconds
    filas_escritas = 0

    with serial.Serial(args.port, args.baud, timeout=1) as conexion_serie:
        # Esperamos a que el ESP32 se reinicie y empiece a imprimir la cabecera CSV
        time.sleep(2)
        conexion_serie.reset_input_buffer()

        with ruta_salida.open("w", newline="") as f:
            escritor_csv = csv.writer(f)
            escritor_csv.writerow(["pc_timestamp_ns", "timestamp_ms", "acc_x_g", "acc_y_g", "acc_z_g"])

            while time.monotonic() < tiempo_limite:
                linea_cruda = conexion_serie.readline()
                if not linea_cruda:
                    continue

                linea = linea_cruda.decode("utf-8", errors="replace").strip()
                if not linea or linea.startswith("timestamp_ms") or linea.startswith("ERROR"):
                    continue

                partes = linea.split(",")
                if len(partes) != 4:
                    continue

                try:
                    timestamp_ms = int(partes[0])
                    acc_x = float(partes[1])
                    acc_y = float(partes[2])
                    acc_z = float(partes[3])
                except ValueError:
                    continue

                escritor_csv.writerow([time.time_ns(), timestamp_ms, acc_x, acc_y, acc_z])
                filas_escritas += 1

    print(f"Captura completada. Se han escrito {filas_escritas} filas en {ruta_salida}")
    if filas_escritas == 0:
        print("No se ha capturado ninguna fila válida. Verifique el puerto, la tasa de baudios y asegúrese de que el Monitor Serie del IDE esté cerrado.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
