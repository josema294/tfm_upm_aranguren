from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import load_settings


COUNTERS = (
    "incoming_rows_total",
    "accepted_rows_total",
    "discarded_rows_total",
    "samples_served_total",
    "results_total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor de telemetría y flujo de datos en tiempo real proveniente del servidor (VPS)."
    )
    parser.add_argument("--session-id", help="Identificador de la sesión a monitorizar. Por defecto asume la sesión activa del VPS.")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalo de sondeo en segundos.")
    parser.add_argument("--duration", type=float, default=0.0, help="Tiempo de ejecución en segundos. 0 indica ejecución indefinida.")
    return parser.parse_args()


def get_json(client: httpx.Client, ruta: str) -> dict:
    respuesta = client.get(ruta)
    respuesta.raise_for_status()
    return respuesta.json()


def resolve_session_id(client: httpx.Client, session_id_solicitada: str | None) -> str | None:
    if session_id_solicitada:
        return session_id_solicitada
    control = get_json(client, "/api/v1/live/control")
    return control.get("active_session_id")


def read_snapshot(client: httpx.Client, session_id: str | None) -> dict | None:
    session_id = resolve_session_id(client, session_id)
    if not session_id:
        return None
    flujo = get_json(client, f"/api/v1/live/{session_id}/control")
    resumen = get_json(client, f"/api/v1/sessions/{session_id}/summary")
    return {
        "time": time.monotonic(),
        "session_id": session_id,
        "capture_enabled": bool(flujo.get("capture_enabled")),
        "window_count": int(resumen.get("window_count", 0)),
        **{clave: int(flujo.get(clave, 0)) for clave in COUNTERS},
    }


def compute_rate(actual: dict, previo: dict, clave: str, tiempo_transcurrido: float) -> float:
    if tiempo_transcurrido <= 0:
        return 0.0
    return max(0, actual[clave] - previo[clave]) / tiempo_transcurrido


def print_header() -> None:
    print(
        "hora      sesión               cap  esp_in/s  aceptado/s worker/s  result/s   descart/s  "
        "aceptado_total worker_total result_total ventanas"
    )


def print_row(actual: dict, previo: dict | None) -> None:
    hora_actual = datetime.now().strftime("%H:%M:%S")
    if previo is None:
        tiempo_transcurrido = 0.0
    else:
        tiempo_transcurrido = actual["time"] - previo["time"]

    tasas = {
        clave: 0.0 if previo is None else compute_rate(actual, previo, clave, tiempo_transcurrido)
        for clave in COUNTERS
    }
    
    estado_captura = "on " if actual["capture_enabled"] else "off"
    
    print(
        f"{hora_actual}  "
        f"{actual['session_id'][:19]:19} "
        f"{estado_captura} "
        f"{tasas['incoming_rows_total']:8.1f} "
        f"{tasas['accepted_rows_total']:11.1f} "
        f"{tasas['samples_served_total']:8.1f} "
        f"{tasas['results_total']:9.1f} "
        f"{tasas['discarded_rows_total']:9.1f} "
        f"{actual['accepted_rows_total']:14} "
        f"{actual['samples_served_total']:12} "
        f"{actual['results_total']:12} "
        f"{actual['window_count']:7}"
    )


def main() -> int:
    args = parse_args()
    configuracion = load_settings()
    if not configuracion.vps_api_key:
        raise RuntimeError("La variable de entorno VPS_API_KEY resulta indispensable en el archivo .env para el monitoreo remoto.")

    cabeceras = {"X-API-Key": configuracion.vps_api_key}
    tiempo_espera_segundos = float(os.getenv("HTTP_TIMEOUT_SECONDS", "10"))
    limite_tiempo = httpx.Timeout(tiempo_espera_segundos)
    inicio = time.monotonic()
    previo: dict | None = None
    print_header()

    with httpx.Client(base_url=configuracion.vps_base_url, headers=cabeceras, timeout=limite_tiempo) as cliente:
        while True:
            estado = read_snapshot(cliente, args.session_id)
            if estado is None:
                print(f"{datetime.now().strftime('%H:%M:%S')}  No se detecta ninguna sesión activa en el servidor remoto.")
            else:
                print_row(estado, previo)
                previo = estado

            if args.duration > 0 and time.monotonic() - inicio >= args.duration:
                break
            time.sleep(max(0.2, args.interval))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as excepcion:
        print(f"Error crítico en monitorización: {excepcion}", file=sys.stderr)
        raise SystemExit(1)
