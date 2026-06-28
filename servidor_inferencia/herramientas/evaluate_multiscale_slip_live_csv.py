#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from herramientas.train_multiscale_slip_cnn import BASE_COLUMNS, FEATURE_COLUMNS, MultiScaleSlipCNN, SCALES, derived_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa una CNN de deslizamiento multiescala sobre un flujo de datos CSV continuo, separando por vueltas estimadas."
    )
    parser.add_argument("csv_path")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-step", type=int, default=25)
    parser.add_argument("--lap-period-s", type=float, default=10.0)
    parser.add_argument("--split-s", type=float, default=600.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold-override", type=float, default=None)
    parser.add_argument("--output-csv")
    return parser.parse_args()


def centered_features(registros: list[dict], indice_central: int, escala: int) -> np.ndarray | None:
    mitad_escala = escala // 2
    inicio = indice_central - mitad_escala
    fin = inicio + escala
    if inicio < 0 or fin > len(registros):
        return None
    valores = np.asarray(
        [[registros[indice][columna] for columna in BASE_COLUMNS] for indice in range(inicio, fin)],
        dtype=np.float32,
    )
    return derived_features(valores)


def load_model(ruta: Path, dispositivo: torch.device) -> tuple[MultiScaleSlipCNN, dict, float]:
    artefacto = torch.load(ruta, map_location=dispositivo)
    modelo = MultiScaleSlipCNN(in_channels=len(artefacto.get("feature_columns", FEATURE_COLUMNS))).to(dispositivo)
    modelo.load_state_dict(artefacto["model_state_dict"])
    modelo.eval()
    return modelo, artefacto["normalization"], float(artefacto.get("threshold", 0.5))


def normalize(caracteristicas: np.ndarray, normalizacion: dict, escala: int) -> np.ndarray:
    parametros = normalizacion[str(escala)]
    media = np.asarray(parametros["mean"], dtype=np.float32)[:, None]
    desviacion = np.asarray(parametros["std"], dtype=np.float32)[:, None]
    desviacion = np.where(desviacion < 1e-6, 1.0, desviacion)
    return ((caracteristicas - media) / desviacion).astype(np.float32)


def print_segment(nombre: str, marco: pd.DataFrame, umbral: float) -> None:
    if marco.empty:
        print(f"Segmento {nombre}: Vacío")
        return
    por_vuelta = marco.groupby("lap").probability.max()
    print(f"\n{nombre}: ventanas_totales={len(marco)} vueltas_identificadas={len(por_vuelta)}")
    umbrales = []
    for umbral_candidato in [umbral, 0.5, 0.3, 0.2, 0.1, 0.05]:
        if not any(abs(umbral_candidato - existente) < 1e-9 for existente in umbrales):
            umbrales.append(umbral_candidato)
    for umbral_candidato in umbrales:
        aciertos = int((por_vuelta >= umbral_candidato).sum())
        ratio = aciertos / len(por_vuelta)
        print(f"  umbral={umbral_candidato:.3f} detecciones_por_vuelta={aciertos}/{len(por_vuelta)} {ratio:.1%}")


def main() -> int:
    args = parse_args()
    dispositivo = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() 
        else "cpu" if args.device == "auto" 
        else args.device
    )
    modelo, normalizacion, umbral_artefacto = load_model(Path(args.model_path), dispositivo)
    umbral_operativo = float(args.threshold_override) if args.threshold_override is not None else umbral_artefacto

    df = pd.read_csv(args.csv_path)
    mascara_invalidos = (df[BASE_COLUMNS].abs() > 8.5).any(axis=1)
    for columna in BASE_COLUMNS:
        df.loc[mascara_invalidos, columna] = np.nan
        df[columna] = df[columna].interpolate().ffill().bfill()
        
    registros = df.to_dict("records")
    primer_timestamp_ms = int(registros[0]["timestamp_ms"])

    filas_resultados = []
    with torch.no_grad():
        for inicio in range(0, len(registros) - args.window_size + 1, args.window_step):
            indice_central = inicio + args.window_size // 2
            caracteristicas_por_escala = {}
            for escala in SCALES:
                caracteristicas = centered_features(registros, indice_central, escala)
                if caracteristicas is None:
                    break
                caracteristicas_por_escala[escala] = normalize(caracteristicas, normalizacion, escala)
                
            if len(caracteristicas_por_escala) != len(SCALES):
                continue
                
            tensores = [
                torch.as_tensor(caracteristicas_por_escala[escala][None, :, :], dtype=torch.float32, device=dispositivo)
                for escala in SCALES
            ]
            probabilidad = float(torch.sigmoid(modelo(*tensores)).detach().cpu().numpy()[0])
            t_s = (int(registros[inicio]["timestamp_ms"]) - primer_timestamp_ms) / 1000
            
            filas_resultados.append(
                {
                    "t_s": t_s,
                    "lap": int(t_s // args.lap_period_s),
                    "seq_start": int(registros[inicio]["seq"]),
                    "seq_end": int(registros[inicio + args.window_size - 1]["seq"]),
                    "probability": probabilidad,
                    "status": "anomaly" if probabilidad >= umbral_operativo else "normal",
                }
            )

    resultado_final = pd.DataFrame(filas_resultados)
    print(f"Modelo empleado: {args.model_path}")
    print(f"Umbral del artefacto: {umbral_artefacto:.6f}")
    print(f"Umbral operativo configurado: {umbral_operativo:.6f}")
    print(
        f"Muestras procesadas: {len(df)} | "
        f"Valores atípicos interpolados: {int(mascara_invalidos.sum())} | "
        f"Ventanas de inferencia generadas: {len(resultado_final)}"
    )
    
    print_segment("seccion_inicial_deslizamiento", resultado_final[resultado_final.t_s < args.split_s], umbral_operativo)
    print_segment("seccion_mixta_final", resultado_final[resultado_final.t_s >= args.split_s], umbral_operativo)
    print_segment("evaluacion_global", resultado_final, umbral_operativo)
    
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        resultado_final.to_csv(args.output_csv, index=False)
        print(f"Resultados de inferencia exportados a: {args.output_csv}")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
