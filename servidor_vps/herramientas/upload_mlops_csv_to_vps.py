#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import os
import urllib.parse
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a MLOps synthetic vibration CSV and upload it to the VPS API."
    )
    parser.add_argument("--input", required=True, help="Input CSV with timestamp,accel_x,accel_y,accel_z.")
    parser.add_argument("--session-id", required=True, help="Destination session id.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8055", help="VPS API base URL.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("VPS_API_KEY") or os.getenv("TFM_API_KEY"),
        help="API key for protected VPS API. Defaults to VPS_API_KEY or TFM_API_KEY.",
    )
    return parser.parse_args()


def convert_csv(input_path: str) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["seq", "timestamp_ms", "acc_x_g", "acc_y_g", "acc_z_g"])

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "accel_x", "accel_y", "accel_z"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"Input CSV must include columns: {', '.join(sorted(required))}")

        for seq, row in enumerate(reader):
            timestamp_ms = int(round(float(row["timestamp"]) * 1000.0))
            writer.writerow(
                [
                    seq,
                    timestamp_ms,
                    float(row["accel_x"]),
                    float(row["accel_y"]),
                    float(row["accel_z"]),
                ]
            )

    return output.getvalue().encode("utf-8")


def upload_csv(base_url: str, session_id: str, payload: bytes, api_key: str | None = None) -> bytes:
    boundary = "----tfm-boundary"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="converted.csv"\r\n',
            b"Content-Type: text/csv\r\n\r\n",
            payload,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    quoted_session = urllib.parse.quote(session_id, safe="")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["X-API-Key"] = api_key

    request = urllib.request.Request(
        f"{base_url.rstrip()}/api/v1/sessions/{quoted_session}/csv",
        data=body,
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main() -> int:
    args = parse_args()
    payload = convert_csv(args.input)
    response = upload_csv(args.base_url, args.session_id, payload, args.api_key)
    print(response.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
