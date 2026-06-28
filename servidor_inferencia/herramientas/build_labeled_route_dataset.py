from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_COLUMNS = ("acc_x_g", "acc_y_g", "acc_z_g")
RAW_COLUMNS = ("pc_timestamp_ns", "sender_ip", "seq", "timestamp_ms", *FEATURE_COLUMNS)


@dataclass
class Ventana:
    id_ventana: int
    filas: list[dict]
    inicio_relativo_s: float
    fin_relativo_s: float
    centro_relativo_s: float
    rms_g: float
    rms_dinamico_g: float
    pico_abs_g: float
    muestras_perdidas: int
    evidencia: float = 0.0
    etiqueta: str = "ignorar"
    zona: str = "no_asignada"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construir etiquetas multiclase para la captura de la ruta de anomalías."
    )
    parser.add_argument("--raw-csv", default="datos/brutos/real_anomaly_route_001.csv", help="Ruta al CSV original.")
    parser.add_argument("--windows-csv", default="datos/etiquetas/real_anomaly_route_001_windows.csv", help="Ruta al CSV de ventanas generado.")
    parser.add_argument("--anomaly-segments-csv", default="datos/procesados/real_anomaly_route_001_anomaly_segments.csv", help="CSV de segmentos de anomalía.")
    parser.add_argument("--normal-segments-csv", default="datos/procesados/real_anomaly_route_001_normal_segments.csv", help="CSV de segmentos normales.")
    parser.add_argument("--ignore-intervals-csv", default="datos/etiquetas/real_anomaly_route_001_ignore_intervals.csv", help="CSV de intervalos a ignorar.")
    parser.add_argument("--window-size", type=int, default=100, help="Tamaño de la ventana (número de muestras).")
    parser.add_argument("--window-step", type=int, default=50, help="Paso de desplazamiento de la ventana.")
    parser.add_argument("--glitch-threshold-g", type=float, default=8.0, help="Umbral para descartar fallos del sensor (g).")
    parser.add_argument("--impact-percentile", type=float, default=85.0, help="Percentil de evidencia para considerar un impacto.")
    parser.add_argument("--impact-min-distance-s", type=float, default=2.0, help="Distancia mínima entre impactos (s).")
    parser.add_argument("--impact-half-width-s", type=float, default=0.45, help="Ancho medio de la zona de impacto (s).")
    parser.add_argument("--slip-offset-before-s", type=float, default=-4.5, help="Desplazamiento temporal del resbalón previo (s).")
    parser.add_argument("--slip-offset-after-s", type=float, default=6.0, help="Desplazamiento temporal del resbalón posterior (s).")
    parser.add_argument("--slip-half-width-s", type=float, default=0.55, help="Ancho medio de la zona de resbalón (s).")
    parser.add_argument("--slip-dedup-distance-s", type=float, default=1.0, help="Distancia para desduplicar resbalones (s).")
    parser.add_argument("--quiet-percentile", type=float, default=60.0, help="Percentil máximo para considerar zona tranquila.")
    parser.add_argument("--quiet-min-distance-s", type=float, default=1.8, help="Distancia mínima a un impacto/resbalón para ser zona tranquila (s).")
    parser.add_argument(
        "--manual-ignore-interval",
        action="append",
        default=[],
        help="Intervalo manual a ignorar en segundos, formato: inicio:fin:motivo.",
    )
    parser.add_argument("--auto-ignore-low-rms-threshold", type=float, default=0.05, help="Umbral inferior RMS para ignorar automáticamente.")
    parser.add_argument("--auto-ignore-padding-s", type=float, default=0.75, help="Margen de padding para el ignorado automático (s).")
    parser.add_argument("--auto-ignore-min-duration-s", type=float, default=2.0, help="Duración mínima para aplicar el ignorado automático (s).")
    return parser.parse_args()


def leer_crudo(ruta: Path) -> list[dict]:
    filas: list[dict] = []
    with ruta.open(newline="") as archivo:
        lector = csv.DictReader(archivo)
        faltantes = [columna for columna in RAW_COLUMNS if columna not in (lector.fieldnames or [])]
        if faltantes:
            raise ValueError(f"Faltan las siguientes columnas en {ruta}: {faltantes}")
        for fila in lector:
            filas.append(
                {
                    "pc_timestamp_ns": fila["pc_timestamp_ns"],
                    "sender_ip": fila["sender_ip"],
                    "seq": int(fila["seq"]),
                    "timestamp_ms": int(fila["timestamp_ms"]),
                    "acc_x_g": float(fila["acc_x_g"]),
                    "acc_y_g": float(fila["acc_y_g"]),
                    "acc_z_g": float(fila["acc_z_g"]),
                }
            )
    return filas


def calcular_z_robusto(valores: list[float]) -> np.ndarray:
    arreglo = np.asarray(valores, dtype=np.float64)
    mediana = np.median(arreglo)
    rango_intercuartilico = np.percentile(arreglo, 75) - np.percentile(arreglo, 25)
    if rango_intercuartilico < 1e-9:
        rango_intercuartilico = 1.0
    return (arreglo - mediana) / rango_intercuartilico


def metricas_ventana(filas: list[dict]) -> tuple[float, float, float, int]:
    valores = np.asarray([[fila[col] for col in FEATURE_COLUMNS] for fila in filas], dtype=np.float32)
    rms_g = float(np.sqrt(np.mean(valores**2)))
    centrados = valores - valores.mean(axis=0, keepdims=True)
    rms_dinamico_g = float(np.sqrt(np.mean(centrados**2)))
    pico_abs_g = float(np.max(np.abs(valores)))
    secuencias = [int(fila["seq"]) for fila in filas]
    muestras_perdidas = sum(max(0, b - a - 1) for a, b in zip(secuencias, secuencias[1:]))
    return rms_g, rms_dinamico_g, pico_abs_g, muestras_perdidas


def construir_ventanas(filas: list[dict], tamano_ventana: int, paso_ventana: int) -> list[Ventana]:
    timestamp_base_ms = filas[0]["timestamp_ms"]
    ventanas: list[Ventana] = []
    for inicio in range(0, len(filas) - tamano_ventana + 1, paso_ventana):
        filas_ventana = filas[inicio : inicio + tamano_ventana]
        inicio_relativo_s = (filas_ventana[0]["timestamp_ms"] - timestamp_base_ms) / 1000.0
        fin_relativo_s = (filas_ventana[-1]["timestamp_ms"] - timestamp_base_ms) / 1000.0
        rms_g, rms_dinamico_g, pico_abs_g, muestras_perdidas = metricas_ventana(filas_ventana)
        ventanas.append(
            Ventana(
                id_ventana=len(ventanas),
                filas=filas_ventana,
                inicio_relativo_s=inicio_relativo_s,
                fin_relativo_s=fin_relativo_s,
                centro_relativo_s=(inicio_relativo_s + fin_relativo_s) / 2.0,
                rms_g=rms_g,
                rms_dinamico_g=rms_dinamico_g,
                pico_abs_g=pico_abs_g,
                muestras_perdidas=muestras_perdidas,
            )
        )
    return ventanas


def asignar_evidencia(ventanas: list[Ventana], umbral_glitch_g: float) -> tuple[float, float]:
    validas = [v for v in ventanas if v.pico_abs_g <= umbral_glitch_g]
    z_dinamico = calcular_z_robusto([v.rms_dinamico_g for v in validas])
    z_pico = calcular_z_robusto([v.pico_abs_g for v in validas])
    for ventana, puntaje_dinamico, puntaje_pico in zip(validas, z_dinamico, z_pico):
        ventana.evidencia = float(puntaje_dinamico + 0.35 * puntaje_pico)
    for ventana in ventanas:
        if ventana.pico_abs_g > umbral_glitch_g:
            ventana.evidencia = float("inf")
    evidencias_validas = [v.evidencia for v in validas]
    return float(np.percentile(evidencias_validas, 0)), float(np.percentile(evidencias_validas, 100))


def detectar_eventos_impacto(
    ventanas: list[Ventana],
    umbral_glitch_g: float,
    percentil_impacto: float,
    distancia_minima_s: float,
) -> list[Ventana]:
    evidencias_validas = [v.evidencia for v in ventanas if v.pico_abs_g <= umbral_glitch_g]
    umbral = float(np.percentile(evidencias_validas, percentil_impacto))
    candidatos: list[Ventana] = []
    for indice, ventana in enumerate(ventanas):
        if ventana.pico_abs_g > umbral_glitch_g or ventana.evidencia < umbral:
            continue
        vecinas = ventanas[max(0, indice - 2) : indice] + ventanas[indice + 1 : indice + 3]
        if all(ventana.evidencia >= vecina.evidencia for vecina in vecinas):
            candidatos.append(ventana)

    seleccionadas: list[Ventana] = []
    for ventana in sorted(candidatos, key=lambda v: v.evidencia, reverse=True):
        if all(abs(ventana.centro_relativo_s - evento.centro_relativo_s) >= distancia_minima_s for evento in seleccionadas):
            seleccionadas.append(ventana)
    return sorted(seleccionadas, key=lambda v: v.centro_relativo_s)


def construir_centros_resbalon(
    eventos_impacto: list[Ventana],
    duracion_total_s: float,
    desplazamientos: tuple[float, float],
    distancia_desduplicacion_s: float,
) -> list[float]:
    centros_crudos: list[float] = []
    for evento in eventos_impacto:
        for desp in desplazamientos:
            centro = evento.centro_relativo_s + desp
            if 0.0 <= centro <= duracion_total_s:
                centros_crudos.append(centro)

    centros: list[float] = []
    for centro in sorted(centros_crudos):
        if not centros or abs(centro - centros[-1]) >= distancia_desduplicacion_s:
            centros.append(centro)
    return centros


def distancia_mas_cercana(valor: float, centros: list[float]) -> float:
    if not centros:
        return float("inf")
    return float(min(abs(valor - centro) for centro in centros))


def etiquetar_ventanas(
    ventanas: list[Ventana],
    eventos_impacto: list[Ventana],
    centros_resbalon: list[float],
    args: argparse.Namespace,
    intervalos_ignorados: list[dict],
) -> None:
    centros_impacto = [evento.centro_relativo_s for evento in eventos_impacto]
    evidencias_validas = [v.evidencia for v in ventanas if v.pico_abs_g <= args.glitch_threshold_g]
    umbral_tranquilo = float(np.percentile(evidencias_validas, args.quiet_percentile))

    for ventana in ventanas:
        motivo_ignorado = obtener_motivo_intervalo(ventana.centro_relativo_s, intervalos_ignorados)
        if motivo_ignorado is not None:
            ventana.etiqueta = "ignorar"
            ventana.zona = motivo_ignorado
            continue

        distancia_impacto = distancia_mas_cercana(ventana.centro_relativo_s, centros_impacto)
        distancia_resbalon = distancia_mas_cercana(ventana.centro_relativo_s, centros_resbalon)

        if ventana.pico_abs_g > args.glitch_threshold_g:
            ventana.etiqueta = "ignorar"
            ventana.zona = "fallo_sensor"
        elif distancia_impacto <= args.impact_half_width_s:
            ventana.etiqueta = "anomalia_impacto"
            ventana.zona = "pico_impacto"
        elif distancia_resbalon <= args.slip_half_width_s:
            ventana.etiqueta = "anomalia_candidato_resbalon"
            ventana.zona = "candidato_resbalon"
        elif (
            distancia_impacto >= args.quiet_min_distance_s
            and distancia_resbalon >= args.quiet_min_distance_s
            and ventana.evidencia <= umbral_tranquilo
        ):
            ventana.etiqueta = "normal_tranquilo"
            ventana.zona = "tranquilo"
        else:
            ventana.etiqueta = "ignorar"
            ventana.zona = "transicion_o_incertidumbre"


def etiquetar_fila(
    tiempo_relativo_s: float,
    centros_impacto: list[float],
    centros_resbalon: list[float],
    args: argparse.Namespace,
    intervalos_ignorados: list[dict],
) -> tuple[str, str]:
    motivo_ignorado = obtener_motivo_intervalo(tiempo_relativo_s, intervalos_ignorados)
    if motivo_ignorado is not None:
        return "ignorar", motivo_ignorado

    distancia_impacto = distancia_mas_cercana(tiempo_relativo_s, centros_impacto)
    distancia_resbalon = distancia_mas_cercana(tiempo_relativo_s, centros_resbalon)
    if distancia_impacto <= args.impact_half_width_s:
        return "anomalia_impacto", "pico_impacto"
    if distancia_resbalon <= args.slip_half_width_s:
        return "anomalia_candidato_resbalon", "candidato_resbalon"
    if distancia_impacto >= args.quiet_min_distance_s and distancia_resbalon >= args.quiet_min_distance_s:
        return "normal_tranquilo", "candidato_tranquilo"
    return "ignorar", "transicion_o_incertidumbre"


def guardar_ventanas(ruta: Path, ventanas: list[Ventana]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nombres_campos = [
        "id_ventana",
        "etiqueta",
        "zona",
        "indice_vuelta",
        "fase_s",
        "inicio_relativo_s",
        "fin_relativo_s",
        "centro_relativo_s",
        "seq_inicio",
        "seq_fin",
        "timestamp_inicio_ms",
        "timestamp_fin_ms",
        "muestras_recibidas",
        "muestras_perdidas",
        "rms_g",
        "rms_dinamico_g",
        "pico_abs_g",
        "puntaje_evidencia",
    ]
    with ruta.open("w", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=nombres_campos)
        escritor.writeheader()
        for ventana in ventanas:
            escritor.writerow(
                {
                    "id_ventana": ventana.id_ventana,
                    "etiqueta": ventana.etiqueta,
                    "zona": ventana.zona,
                    "indice_vuelta": "",
                    "fase_s": "",
                    "inicio_relativo_s": round(ventana.inicio_relativo_s, 3),
                    "fin_relativo_s": round(ventana.fin_relativo_s, 3),
                    "centro_relativo_s": round(ventana.centro_relativo_s, 3),
                    "seq_inicio": ventana.filas[0]["seq"],
                    "seq_fin": ventana.filas[-1]["seq"],
                    "timestamp_inicio_ms": ventana.filas[0]["timestamp_ms"],
                    "timestamp_fin_ms": ventana.filas[-1]["timestamp_ms"],
                    "muestras_recibidas": len(ventana.filas),
                    "muestras_perdidas": ventana.muestras_perdidas,
                    "rms_g": round(ventana.rms_g, 8),
                    "rms_dinamico_g": round(ventana.rms_dinamico_g, 8),
                    "pico_abs_g": round(ventana.pico_abs_g, 8),
                    "puntaje_evidencia": round(ventana.evidencia, 8) if np.isfinite(ventana.evidencia) else "",
                }
            )


def guardar_segmentos(
    ruta_anomalia: Path,
    ruta_normal: Path,
    filas: list[dict],
    centros_impacto: list[float],
    centros_resbalon: list[float],
    args: argparse.Namespace,
    intervalos_ignorados: list[dict],
) -> tuple[dict[str, int], dict[str, int]]:
    ruta_anomalia.parent.mkdir(parents=True, exist_ok=True)
    ruta_normal.parent.mkdir(parents=True, exist_ok=True)
    timestamp_base_ms = filas[0]["timestamp_ms"]
    nombres_campos = [
        "pc_timestamp_ns",
        "sender_ip",
        "seq",
        "timestamp_ms",
        "acc_x_g",
        "acc_y_g",
        "acc_z_g",
        "etiqueta",
        "zona",
        "indice_vuelta",
        "fase_s",
        "tiempo_relativo_s",
    ]
    conteos_anomalia: dict[str, int] = {}
    conteos_normales: dict[str, int] = {}
    with ruta_anomalia.open("w", newline="") as archivo_anomalia, ruta_normal.open("w", newline="") as archivo_normal:
        escritor_anomalia = csv.DictWriter(archivo_anomalia, fieldnames=nombres_campos)
        escritor_normal = csv.DictWriter(archivo_normal, fieldnames=nombres_campos)
        escritor_anomalia.writeheader()
        escritor_normal.writeheader()
        for fila in filas:
            if max(abs(fila[col]) for col in FEATURE_COLUMNS) > args.glitch_threshold_g:
                continue
            tiempo_relativo_s = (fila["timestamp_ms"] - timestamp_base_ms) / 1000.0
            etiqueta, zona = etiquetar_fila(tiempo_relativo_s, centros_impacto, centros_resbalon, args, intervalos_ignorados)
            if etiqueta == "ignorar":
                continue
            salida = {
                "pc_timestamp_ns": fila["pc_timestamp_ns"],
                "sender_ip": fila["sender_ip"],
                "seq": fila["seq"],
                "timestamp_ms": fila["timestamp_ms"],
                "acc_x_g": fila["acc_x_g"],
                "acc_y_g": fila["acc_y_g"],
                "acc_z_g": fila["acc_z_g"],
                "etiqueta": etiqueta,
                "zona": zona,
                "indice_vuelta": "",
                "fase_s": "",
                "tiempo_relativo_s": round(tiempo_relativo_s, 3),
            }
            if etiqueta.startswith("anomalia_"):
                escritor_anomalia.writerow(salida)
                conteos_anomalia[etiqueta] = conteos_anomalia.get(etiqueta, 0) + 1
            elif etiqueta == "normal_tranquilo":
                escritor_normal.writerow(salida)
                conteos_normales[etiqueta] = conteos_normales.get(etiqueta, 0) + 1
    return conteos_anomalia, conteos_normales


def procesar_intervalos_ignorados_manuales(valores: list[str]) -> list[dict]:
    intervalos: list[dict] = []
    for valor in valores:
        partes = valor.split(":", 2)
        if len(partes) != 3:
            raise ValueError("--manual-ignore-interval debe utilizar el formato inicio:fin:motivo")
        inicio_s, fin_s, motivo = partes
        intervalos.append(
            {
                "inicio_s": float(inicio_s),
                "fin_s": float(fin_s),
                "motivo": motivo or "ignorado_manual",
                "origen": "manual",
            }
        )
    return intervalos


def obtener_motivo_intervalo(tiempo_relativo_s: float, intervalos: list[dict]) -> str | None:
    for intervalo in intervalos:
        if intervalo["inicio_s"] <= tiempo_relativo_s <= intervalo["fin_s"]:
            return str(intervalo["motivo"])
    return None


def fusionar_intervalos(intervalos: list[dict], brecha_max_s: float = 0.75) -> list[dict]:
    if not intervalos:
        return []
    ordenados = sorted(intervalos, key=lambda item: (item["inicio_s"], item["fin_s"]))
    fusionados = [ordenados[0].copy()]
    for intervalo in ordenados[1:]:
        ultimo = fusionados[-1]
        mismo_motivo = intervalo["motivo"] == ultimo["motivo"] and intervalo["origen"] == ultimo["origen"]
        if mismo_motivo and intervalo["inicio_s"] <= ultimo["fin_s"] + brecha_max_s:
            ultimo["fin_s"] = max(ultimo["fin_s"], intervalo["fin_s"])
        else:
            fusionados.append(intervalo.copy())
    return fusionados


def detectar_intervalos_ignorados_automaticos(ventanas: list[Ventana], args: argparse.Namespace) -> list[dict]:
    """Detecta intervalos planos o de reposicionamiento no representativos.

    Estas reglas apuntan intencionalmente solo a ventanas planas muy inusuales cerca de eventos de alta
    evidencia. No es un detector general de anomalías; previenen que segmentos de reposicionamiento/parada
    se cuenten como tramos normales tranquilos.
    """
    candidatos_planos: list[dict] = []
    for ventana in ventanas:
        es_plano = ventana.rms_dinamico_g <= args.auto_ignore_low_rms_threshold
        if es_plano and ventana.pico_abs_g <= args.glitch_threshold_g:
            candidatos_planos.append(
                {
                    "inicio_s": max(0.0, ventana.inicio_relativo_s - args.auto_ignore_padding_s),
                    "fin_s": ventana.fin_relativo_s + args.auto_ignore_padding_s,
                    "motivo": "ignorado_automatico_plano_o_reposicionamiento",
                    "origen": "automatico",
                }
            )
    fusionados = fusionar_intervalos(candidatos_planos)
    return [
        intervalo
        for intervalo in fusionados
        if intervalo["fin_s"] - intervalo["inicio_s"] >= args.auto_ignore_min_duration_s
    ]


def guardar_intervalos_ignorados(ruta: Path, intervalos: list[dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nombres_campos = ["inicio_s", "fin_s", "duracion_s", "motivo", "origen"]
    with ruta.open("w", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=nombres_campos)
        escritor.writeheader()
        for intervalo in intervalos:
            inicio_s = float(intervalo["inicio_s"])
            fin_s = float(intervalo["fin_s"])
            escritor.writerow(
                {
                    "inicio_s": round(inicio_s, 3),
                    "fin_s": round(fin_s, 3),
                    "duracion_s": round(fin_s - inicio_s, 3),
                    "motivo": intervalo["motivo"],
                    "origen": intervalo["origen"],
                }
            )


def contar_etiquetas(ventanas: list[Ventana]) -> dict[str, int]:
    conteos: dict[str, int] = {}
    for ventana in ventanas:
        conteos[ventana.etiqueta] = conteos.get(ventana.etiqueta, 0) + 1
    return conteos


def main() -> int:
    args = parse_args()
    ruta_csv_crudo = Path(args.raw_csv)
    if not ruta_csv_crudo.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo CSV crudo: {ruta_csv_crudo}. "
            "Restaure el CSV original de la ruta o especifique --raw-csv datos/brutos/real_anomaly_route_002.csv."
        )
    filas = leer_crudo(ruta_csv_crudo)
    ventanas = construir_ventanas(filas, args.window_size, args.window_step)
    asignar_evidencia(ventanas, args.glitch_threshold_g)

    eventos_impacto = detectar_eventos_impacto(
        ventanas,
        umbral_glitch_g=args.glitch_threshold_g,
        percentil_impacto=args.impact_percentile,
        distancia_minima_s=args.impact_min_distance_s,
    )
    duracion_total_s = (filas[-1]["timestamp_ms"] - filas[0]["timestamp_ms"]) / 1000.0
    centros_resbalon = construir_centros_resbalon(
        eventos_impacto,
        duracion_total_s=duracion_total_s,
        desplazamientos=(args.slip_offset_before_s, args.slip_offset_after_s),
        distancia_desduplicacion_s=args.slip_dedup_distance_s,
    )
    intervalos_ignorados = fusionar_intervalos(
        [
            *procesar_intervalos_ignorados_manuales(args.manual_ignore_interval),
            *detectar_intervalos_ignorados_automaticos(ventanas, args),
        ]
    )
    etiquetar_ventanas(ventanas, eventos_impacto, centros_resbalon, args, intervalos_ignorados)

    guardar_ventanas(Path(args.windows_csv), ventanas)
    guardar_intervalos_ignorados(Path(args.ignore_intervals_csv), intervalos_ignorados)
    centros_impacto = [evento.centro_relativo_s for evento in eventos_impacto]
    conteos_anomalia, conteos_normales = guardar_segmentos(
        Path(args.anomaly_segments_csv),
        Path(args.normal_segments_csv),
        filas,
        centros_impacto,
        centros_resbalon,
        args,
        intervalos_ignorados,
    )

    print(f"Archivo CSV crudo: {ruta_csv_crudo}")
    print(f"Ventanas procesadas: {len(ventanas)}")
    print(f"Eventos de impacto: {len(eventos_impacto)}")
    print(f"Centros candidatos a resbalón: {len(centros_resbalon)}")
    print(f"Intervalos ignorados: {len(intervalos_ignorados)}")
    print(f"Etiquetas de ventana: {contar_etiquetas(ventanas)}")
    print(f"Etiquetas de segmentos anómalos: {conteos_anomalia}")
    print(f"Etiquetas de segmentos normales: {conteos_normales}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
