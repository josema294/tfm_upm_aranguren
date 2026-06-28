#!/usr/bin/env python3
import argparse
import csv
import select
import pathlib
import socket
import sys
import time
import termios
import tty
from contextlib import contextmanager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recepción de paquetes UDP del ADXL345 (ESP32) para su almacenamiento en formato CSV."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host o interfaz de red a escuchar.")
    parser.add_argument("--port", type=int, default=5005, help="Puerto UDP de escucha.")
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida del archivo CSV (ej: ../datos/brutos/wifi_test_001.csv).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir un archivo CSV existente. Por defecto, los archivos están protegidos.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Duración de la captura en segundos. Ignorado si se utiliza --forever.",
    )
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Capturar indefinidamente hasta pulsar Ctrl+C.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="Volcar el buffer al archivo CSV cada N filas para observar el progreso en disco.",
    )
    parser.add_argument(
        "--progress-every",
        type=float,
        default=5.0,
        help="Mostrar el progreso de captura en consola cada N segundos.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help=(
            "Habilitar anotaciones por teclado de forma asíncrona. Teclas: "
            "e=iniciar anomalía, s=finalizar anomalía, i=impacto, p=deslizamiento, q=detener captura."
        ),
    )
    parser.add_argument(
        "--events-output",
        default=None,
        help="Ruta de salida del CSV para las anotaciones por teclado. Si no se especifica, usa <output_name>_events.csv.",
    )
    parser.add_argument(
        "--anomaly-label",
        default="manual_anomaly",
        help="Etiqueta genérica utilizada para las anomalías anotadas manualmente (teclas e/s).",
    )
    return parser.parse_args()


@contextmanager
def raw_terminal(enabled):
    if not enabled:
        yield
        return
    if not sys.stdin.isatty():
        raise RuntimeError("--annotate requires an interactive terminal")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key():
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    return sys.stdin.read(1)


def default_events_path(output_path):
    return output_path.with_name(f"{output_path.stem}_events.csv")


def main():
    args = parse_args()
    ruta_salida = pathlib.Path(args.output)
    ruta_eventos = pathlib.Path(args.events_output) if args.events_output else default_events_path(ruta_salida)
    if ruta_salida.exists() and not args.overwrite:
        print(
            f"Denegado el sobrescribir un archivo existente: {ruta_salida}. "
            "Utilice el flag --overwrite únicamente si es su intención.",
            file=sys.stderr,
        )
        return 2
    if args.annotate and ruta_eventos.exists() and not args.overwrite:
        print(
            f"Denegado el sobrescribir un archivo de eventos existente: {ruta_eventos}. "
            "Utilice el flag --overwrite únicamente si es su intención.",
            file=sys.stderr,
        )
        return 2
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    if args.annotate:
        ruta_eventos.parent.mkdir(parents=True, exist_ok=True)

    socket_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket_udp.bind((args.host, args.port))
    socket_udp.settimeout(1.0)

    tiempo_inicio = time.monotonic()
    tiempo_limite = None if args.forever else tiempo_inicio + args.seconds
    siguiente_progreso = tiempo_inicio + args.progress_every
    filas_escritas = 0
    primera_secuencia = None
    ultima_secuencia = None
    ultimo_timestamp_ms = None
    etiqueta_activa = None

    print(f"Escuchando paquetes UDP en {args.host}:{args.port}")
    if args.forever:
        print("Modo de captura: continuo. Pulse Ctrl+C para detener y guardar el resumen.")
    else:
        print(f"Modo de captura: {args.seconds:g} segundos.")
    if args.annotate:
        print(f"Modo de anotación activo: almacenando eventos de teclado en {ruta_eventos}")
        print("Atajos: e=inicio anomalía genérica, s=fin anomalía actual, i=impacto, p=deslizamiento, q=detener")

    archivo_eventos = ruta_eventos.open("w", newline="") if args.annotate else None
    try:
        escritor_eventos = csv.writer(archivo_eventos) if archivo_eventos else None
        if escritor_eventos:
            escritor_eventos.writerow(
                [
                    "pc_timestamp_ns",
                    "elapsed_s",
                    "event",
                    "label",
                    "last_seq",
                    "last_timestamp_ms",
                    "rows_written",
                ]
            )
            archivo_eventos.flush()

        with raw_terminal(args.annotate), ruta_salida.open("w", newline="") as f:
            escritor_csv = csv.writer(f)
            escritor_csv.writerow(
                [
                    "pc_timestamp_ns",
                    "sender_ip",
                    "seq",
                    "timestamp_ms",
                    "acc_x_g",
                    "acc_y_g",
                    "acc_z_g",
                ]
            )
            f.flush()

            try:
                while tiempo_limite is None or time.monotonic() < tiempo_limite:
                    ahora = time.monotonic()
                    if args.annotate:
                        tecla = read_key()
                        if tecla:
                            tiempo_evento_ns = time.time_ns()
                            tiempo_transcurrido = ahora - tiempo_inicio
                            evento = None
                            etiqueta = etiqueta_activa
                            if tecla == "e":
                                etiqueta_activa = args.anomaly_label
                                evento = "start"
                                etiqueta = etiqueta_activa
                            elif tecla == "i":
                                etiqueta_activa = "manual_impact"
                                evento = "start"
                                etiqueta = etiqueta_activa
                            elif tecla == "p":
                                etiqueta_activa = "manual_slip"
                                evento = "start"
                                etiqueta = etiqueta_activa
                            elif tecla == "s":
                                evento = "end"
                                etiqueta = etiqueta_activa or args.anomaly_label
                                etiqueta_activa = None
                            elif tecla == "q":
                                evento = "stop_capture"
                                etiqueta = etiqueta_activa
                            else:
                                evento = f"key_{tecla!r}"

                            if escritor_eventos:
                                escritor_eventos.writerow(
                                    [
                                        tiempo_evento_ns,
                                        f"{tiempo_transcurrido:.6f}",
                                        evento,
                                        etiqueta,
                                        ultima_secuencia,
                                        ultimo_timestamp_ms,
                                        filas_escritas,
                                    ]
                                )
                                archivo_eventos.flush()
                            print(
                                f"\nEvento registrado: {evento} etiqueta={etiqueta} transcurrido={tiempo_transcurrido:.3f}s "
                                f"ultima_seq={ultima_secuencia}",
                                flush=True,
                            )
                            if tecla == "q":
                                break

                    if ahora >= siguiente_progreso:
                        tiempo_transcurrido = ahora - tiempo_inicio
                        tasa_captura = filas_escritas / tiempo_transcurrido if tiempo_transcurrido > 0 else 0.0
                        print(
                            f"Progreso actual: filas={filas_escritas} tiempo={tiempo_transcurrido:.1f}s "
                            f"tasa={tasa_captura:.1f} filas/s ultima_seq={ultima_secuencia}"
                        )
                        siguiente_progreso = ahora + args.progress_every

                    try:
                        paquete, direccion = socket_udp.recvfrom(1024)
                    except socket.timeout:
                        continue

                    linea = paquete.decode("utf-8", errors="replace").strip()
                    partes = linea.split(",")
                    if len(partes) != 5:
                        continue

                    try:
                        secuencia = int(partes[0])
                        timestamp_ms = int(partes[1])
                        acc_x = float(partes[2])
                        acc_y = float(partes[3])
                        acc_z = float(partes[4])
                    except ValueError:
                        continue

                    if primera_secuencia is None:
                        primera_secuencia = secuencia
                    ultima_secuencia = secuencia
                    ultimo_timestamp_ms = timestamp_ms

                    escritor_csv.writerow(
                        [time.time_ns(), direccion[0], secuencia, timestamp_ms, acc_x, acc_y, acc_z]
                    )
                    filas_escritas += 1
                    if args.flush_every > 0 and filas_escritas % args.flush_every == 0:
                        f.flush()
            except KeyboardInterrupt:
                print("\nCaptura interrumpida por el usuario.")
            finally:
                f.flush()
                if archivo_eventos:
                    archivo_eventos.flush()
    finally:
        if archivo_eventos:
            archivo_eventos.close()

    socket_udp.close()
    print(f"Captura finalizada: se han escrito {filas_escritas} filas en {ruta_salida}")
    if args.annotate:
        print(f"Los eventos de anotación manual se han guardado en {ruta_eventos}")
    if primera_secuencia is not None and ultima_secuencia is not None:
        paquetes_esperados = ultima_secuencia - primera_secuencia + 1
        paquetes_perdidos = paquetes_esperados - filas_escritas
        print(f"Rango de secuencias: {primera_secuencia}..{ultima_secuencia}; estimación de paquetes perdidos: {paquetes_perdidos}")

    if filas_escritas == 0:
        print("No se ha capturado ningún paquete UDP. Compruebe la configuración WiFi del ESP32, la IP del ordenador y el cortafuegos.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
