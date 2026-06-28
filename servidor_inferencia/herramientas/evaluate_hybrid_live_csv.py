from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.hybrid.detector import HybridVaeSlipDetector
from app.slip.detector import SlipDetector
from app.vae.detector import VaeDetector
from app.windowing import quality_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua el detector hibrido sobre un CSV continuo agrupando por vueltas.")
    parser.add_argument("csv_path")
    parser.add_argument("--vae-model", required=True)
    parser.add_argument("--slip-model", required=True)
    parser.add_argument("--slip-threshold", type=float, default=0.30)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-step", type=int, default=25)
    parser.add_argument("--lap-period-s", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--positive", action="store_true", help="Indica que todas las vueltas del CSV contienen patinaje.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def read_samples(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            samples.append(
                {
                    "seq": int(row["seq"]),
                    "timestamp_ms": int(row["timestamp_ms"]),
                    "acc_x_g": float(row["acc_x_g"]),
                    "acc_y_g": float(row["acc_y_g"]),
                    "acc_z_g": float(row["acc_z_g"]),
                }
            )
    return samples


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    samples = read_samples(csv_path)
    first_timestamp_ms = samples[0]["timestamp_ms"]

    vae = VaeDetector(args.vae_model, device=args.device)
    slip = SlipDetector(args.slip_model, device=args.device, threshold_override=args.slip_threshold)
    hybrid = HybridVaeSlipDetector(vae, slip)

    rows = []
    lap_summary: dict[int, dict] = {}
    counts = {"normal": 0, "anomaly": 0, "unreliable": 0, "impact_or_general": 0, "slip": 0}
    for start in range(0, len(samples) - args.window_size + 1, args.window_step):
        window = samples[start : start + args.window_size]
        quality = quality_report(window, expected_size=args.window_size)
        prediction = hybrid.predict(window, quality)
        t_s = (window[0]["timestamp_ms"] - first_timestamp_ms) / 1000.0
        lap = int(t_s // args.lap_period_s)
        status = prediction["status"]
        anomaly_type = prediction.get("metadata", {}).get("anomaly_type", "none")

        counts[status] = counts.get(status, 0) + 1
        if status == "anomaly":
            counts[anomaly_type] = counts.get(anomaly_type, 0) + 1

        lap_info = lap_summary.setdefault(
            lap,
            {
                "lap": lap,
                "windows": 0,
                "detected_hybrid": False,
                "detected_slip": False,
                "detected_vae_general": False,
            },
        )
        lap_info["windows"] += 1
        lap_info["detected_hybrid"] = lap_info["detected_hybrid"] or status == "anomaly"
        lap_info["detected_slip"] = lap_info["detected_slip"] or anomaly_type == "slip"
        lap_info["detected_vae_general"] = lap_info["detected_vae_general"] or anomaly_type == "impact_or_general"

        rows.append(
            {
                "t_s": round(t_s, 3),
                "lap": lap,
                "seq_start": window[0]["seq"],
                "seq_end": window[-1]["seq"],
                "status": status,
                "anomaly_type": anomaly_type,
                "hybrid_score": prediction["anomaly_score"],
                "vae_status": prediction["metadata"]["vae"]["status"],
                "slip_status": prediction["metadata"]["slip"]["status"],
                "slip_probability": prediction["metadata"]["slip"].get("slip_probability", ""),
            }
        )

    laps = list(lap_summary.values())
    detected_hybrid = sum(1 for lap in laps if lap["detected_hybrid"])
    detected_slip = sum(1 for lap in laps if lap["detected_slip"])
    detected_vae_general = sum(1 for lap in laps if lap["detected_vae_general"])
    total_laps = len(laps)

    summary = {
        "config": {
            "csv_path": str(csv_path),
            "vae_model": str(args.vae_model),
            "slip_model": str(args.slip_model),
            "slip_threshold": args.slip_threshold,
            "window_size": args.window_size,
            "window_step": args.window_step,
            "lap_period_s": args.lap_period_s,
            "positive": args.positive,
        },
        "windows": {
            "total": len(rows),
            **counts,
        },
        "laps": {
            "total": total_laps,
            "detected_hybrid": detected_hybrid,
            "detected_slip": detected_slip,
            "detected_vae_general": detected_vae_general,
            "ratio_hybrid": detected_hybrid / total_laps if total_laps else 0.0,
            "ratio_slip": detected_slip / total_laps if total_laps else 0.0,
            "ratio_vae_general": detected_vae_general / total_laps if total_laps else 0.0,
        },
    }

    if args.positive:
        summary["classification"] = {
            "true_positive": detected_hybrid,
            "false_negative": total_laps - detected_hybrid,
        }
    else:
        summary["classification"] = {
            "false_positive": detected_hybrid,
            "true_negative": total_laps - detected_hybrid,
        }

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Predicciones exportadas a: {output_csv}")
    print(f"Resumen exportado a: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
