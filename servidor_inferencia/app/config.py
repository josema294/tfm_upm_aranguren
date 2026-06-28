from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vps_base_url: str
    vps_api_key: str
    session_id: str
    poll_interval_seconds: float
    samples_limit: int
    window_size: int
    window_step: int
    detector_type: str
    model_path: str
    slip_model_path: str
    slip_threshold: float | None
    device: str
    http_timeout_seconds: float
    job_check_interval_seconds: float
    job_poll_interval_seconds: float
    heartbeat_interval_seconds: float
    live_discovery_interval_seconds: float
    live_session_max_age_seconds: float
    live_max_sessions: int
    http_retries: int


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_settings() -> Settings:
    load_dotenv()
    slip_threshold_raw = os.getenv("SLIP_THRESHOLD")
    return Settings(
        vps_base_url=os.getenv("VPS_BASE_URL", "http://127.0.0.1:8055").rstrip("/"),
        vps_api_key=os.getenv("VPS_API_KEY", ""),
        session_id=os.getenv("SESSION_ID", "live_test_001"),
        poll_interval_seconds=float(os.getenv("POLL_INTERVAL_SECONDS", "0.5")),
        samples_limit=int(os.getenv("SAMPLES_LIMIT", "500")),
        window_size=int(os.getenv("WINDOW_SIZE", "100")),
        window_step=int(os.getenv("WINDOW_STEP", "50")),
        detector_type=os.getenv("DETECTOR_TYPE", "placeholder"),
        model_path=os.getenv("MODEL_PATH", "models/autoencoder_v1.pth"),
        slip_model_path=os.getenv("SLIP_MODEL_PATH", "models/slip_mil_w30_50_100_testperf_plus_validation_v2.pth"),
        slip_threshold=float(slip_threshold_raw) if slip_threshold_raw else None,
        device=os.getenv("TORCH_DEVICE", "auto"),
        http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "30")),
        job_check_interval_seconds=float(os.getenv("JOB_CHECK_INTERVAL_SECONDS", "10")),
        job_poll_interval_seconds=float(os.getenv("JOB_POLL_INTERVAL_SECONDS", "3")),
        heartbeat_interval_seconds=float(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "15")),
        live_discovery_interval_seconds=float(os.getenv("LIVE_DISCOVERY_INTERVAL_SECONDS", "1")),
        live_session_max_age_seconds=float(os.getenv("LIVE_SESSION_MAX_AGE_SECONDS", "30")),
        live_max_sessions=int(os.getenv("LIVE_MAX_SESSIONS", "1")),
        http_retries=int(os.getenv("HTTP_RETRIES", "3")),
    )
