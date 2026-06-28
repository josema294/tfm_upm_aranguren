from __future__ import annotations

import argparse
import socket
import time
from datetime import UTC, datetime
from typing import Protocol

import httpx

from .client import VpsClient
from .config import load_settings
from .detector import PlaceholderDetector
from .windowing import WindowBuffer, quality_report, window_to_array


class Detector(Protocol):
    def predict(self, samples, quality: dict) -> dict: ...


class JobCancelled(RuntimeError):
    pass


class LiveSessionState:
    def __init__(self, after_seq: int = -1):
        self.buffer = WindowBuffer(0, 0)
        self.last_seq = after_seq
        self.windows_sent = 0
        self.last_idle_log = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference worker that pulls samples from the VPS.")
    parser.add_argument("--session-id", help="Session to consume. Defaults to SESSION_ID env.")
    parser.add_argument("--after-seq", type=int, default=-1, help="Initial sequence cursor.")
    parser.add_argument("--once", action="store_true", help="Run one polling iteration and exit.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Continuously process a live session as new ESP32 samples arrive.",
    )
    parser.add_argument(
        "--from-end",
        action="store_true",
        help="In live mode, skip already stored samples and start from the current end of the session.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Register as an inference worker and process queued jobs plus recent live sessions.",
    )
    parser.add_argument(
        "--worker-id",
        help="Worker id for daemon mode. Defaults to hostname.",
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        help="Process available samples until the VPS returns no new data, then exit.",
    )
    return parser.parse_args()


def build_detector(settings) -> tuple[Detector, str]:
    if settings.detector_type == "hybrid":
        from .hybrid.detector import HybridVaeSlipDetector
        from .slip.detector import SlipDetector
        from .vae.detector import VaeDetector

        vae_detector = VaeDetector(settings.model_path, device=settings.device)
        slip_detector = SlipDetector(
            settings.slip_model_path,
            device=settings.device,
            threshold_override=settings.slip_threshold,
        )
        return HybridVaeSlipDetector(vae_detector, slip_detector), "hybrid_vae_slip_cnn"
    if settings.detector_type == "vae":
        from .vae.detector import VaeDetector

        return VaeDetector(settings.model_path, device=settings.device), "vae"
    if settings.detector_type == "slip":
        from .slip.detector import SlipDetector

        return (
            SlipDetector(
                settings.slip_model_path,
                device=settings.device,
                threshold_override=settings.slip_threshold,
            ),
            "slip_cnn",
        )
    if settings.detector_type == "autoencoder":
        from .autoencoder.detector import AutoencoderDetector

        return AutoencoderDetector(settings.model_path, device=settings.device), "autoencoder"
    if settings.detector_type == "placeholder":
        return PlaceholderDetector(), "placeholder_energy_v0"
    raise ValueError(f"Unknown DETECTOR_TYPE: {settings.detector_type}")


def predict_window(detector: Detector, detector_name: str, window: list[dict], quality: dict) -> dict:
    if detector_name in {"autoencoder", "vae", "slip_cnn", "hybrid_vae_slip_cnn"}:
        return detector.predict(window, quality)
    return detector.predict(window_to_array(window), quality)


def post_ready_windows(
    client: VpsClient,
    session_id: str,
    settings,
    detector: Detector,
    detector_name: str,
    buffer: WindowBuffer,
    windows_sent: int,
    job_check,
) -> int:
    for window in buffer.pop_ready_windows():
        job_check()
        quality = quality_report(window, settings.window_size)
        predict_started = time.perf_counter()
        prediction = predict_window(detector, detector_name, window, quality)
        predict_ms = (time.perf_counter() - predict_started) * 1000.0
        payload = {
            "session_id": session_id,
            "source": "pc_inference",
            "window_start_ms": quality["timestamp_start_ms"],
            "window_end_ms": quality["timestamp_end_ms"],
            "status": prediction["status"],
            "anomaly_score": prediction["anomaly_score"],
            "reconstruction_error": prediction["reconstruction_error"],
            "quality": quality,
            "metadata": {"model": detector_name, **prediction.get("metadata", {})},
        }
        post_started = time.perf_counter()
        client.post_result(payload)
        post_ms = (time.perf_counter() - post_started) * 1000.0
        windows_sent += 1
        print(
            "result",
            f"window={windows_sent}",
            f"status={payload['status']}",
            f"score={payload['anomaly_score']:.6f}",
            f"seq={quality['seq_start']}..{quality['seq_end']}",
            f"predict_ms={predict_ms:.1f}",
            f"post_ms={post_ms:.1f}",
        )
        if predict_ms >= 100.0 or post_ms >= 250.0:
            print(
                "inference_step_slow",
                f"session={session_id}",
                f"window={windows_sent}",
                f"predict_ms={predict_ms:.1f}",
                f"post_ms={post_ms:.1f}",
            )
    return windows_sent


def parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def session_age_seconds(session: dict) -> float | None:
    updated_at = parse_timestamp(str(session.get("updated_at", "")))
    if updated_at is None:
        return None
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return (datetime.now(UTC) - updated_at).total_seconds()


def recent_live_sessions(client: VpsClient, max_age_seconds: float) -> list[dict]:
    sessions = []
    for session in client.list_sessions():
        if session.get("mode") != "live":
            continue
        age = session_age_seconds(session)
        if age is None or age > max_age_seconds:
            continue
        sessions.append(session)
    sessions.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return sessions


def process_session(
    client: VpsClient,
    session_id: str,
    settings,
    detector: Detector,
    detector_name: str,
    after_seq: int = -1,
    once: bool = False,
    drain: bool = True,
    job_id: str | None = None,
) -> int:
    buffer = WindowBuffer(settings.window_size, settings.window_step)
    last_seq = after_seq
    windows_sent = 0
    last_job_check = 0.0

    def ensure_job_active(force: bool = False) -> None:
        nonlocal last_job_check
        if job_id is None:
            return
        now = time.monotonic()
        if not force and now - last_job_check < settings.job_check_interval_seconds:
            return
        last_job_check = now
        job = client.get_job(job_id)
        if job is None or job.get("status") != "running":
            raise JobCancelled(f"job {job_id} is no longer active")

    while True:
        ensure_job_active(force=True)
        samples = client.get_samples(session_id, last_seq, settings.samples_limit)
        if samples:
            last_seq = int(samples[-1]["seq"])
            buffer.extend(samples)
            print(f"received={len(samples)} last_seq={last_seq} buffered={len(buffer.samples)}")

        windows_sent = post_ready_windows(
            client,
            session_id,
            settings,
            detector,
            detector_name,
            buffer,
            windows_sent,
            ensure_job_active,
        )

        if once:
            break
        if drain and not samples:
            break
        time.sleep(settings.poll_interval_seconds)

    return windows_sent


def find_latest_seq(client: VpsClient, session_id: str, settings, after_seq: int) -> int:
    last_seq = after_seq
    skipped = 0
    while True:
        samples = client.get_samples(session_id, last_seq, settings.samples_limit)
        if not samples:
            break
        skipped += len(samples)
        last_seq = int(samples[-1]["seq"])
    print(f"live cursor positioned at seq={last_seq}; skipped historical samples={skipped}")
    return last_seq


def run_live(
    client: VpsClient,
    session_id: str,
    settings,
    detector: Detector,
    detector_name: str,
    worker_id: str,
    after_seq: int,
    from_end: bool,
) -> None:
    capabilities = [detector_name, "live_session_inference"]
    buffer = WindowBuffer(settings.window_size, settings.window_step)
    last_seq = after_seq
    windows_sent = 0
    last_heartbeat = 0.0
    last_idle_log = 0.0

    if from_end:
        last_seq = find_latest_seq(client, session_id, settings, after_seq)

    print(f"Worker: {worker_id}")
    print("Mode: live")
    print(f"Session: {session_id}")
    print(f"Initial cursor: after_seq={last_seq}")

    while True:
        try:
            now = time.monotonic()
            if now - last_heartbeat >= settings.heartbeat_interval_seconds:
                client.heartbeat(worker_id, capabilities, current_job_id=f"live:{session_id}")
                last_heartbeat = now

            samples = client.get_samples(session_id, last_seq, settings.samples_limit)
            if samples:
                last_seq = int(samples[-1]["seq"])
                buffer.extend(samples)
                print(f"live received={len(samples)} last_seq={last_seq} buffered={len(buffer.samples)}")
                windows_sent = post_ready_windows(
                    client,
                    session_id,
                    settings,
                    detector,
                    detector_name,
                    buffer,
                    windows_sent,
                    lambda: None,
                )
            elif now - last_idle_log >= 10.0:
                print(f"live waiting session={session_id} after_seq={last_seq} windows={windows_sent}")
                last_idle_log = now
        except httpx.HTTPError as exc:
            print(f"VPS communication error in live mode: {exc}")

        time.sleep(settings.poll_interval_seconds)


def process_live_once(
    client: VpsClient,
    session_id: str,
    settings,
    detector: Detector,
    detector_name: str,
    state: LiveSessionState,
) -> int:
    if state.buffer.window_size == 0:
        state.buffer = WindowBuffer(settings.window_size, settings.window_step)

    started = time.perf_counter()
    samples = client.get_samples(session_id, state.last_seq, settings.samples_limit)
    get_ms = (time.perf_counter() - started) * 1000.0
    if samples:
        state.last_seq = int(samples[-1]["seq"])
        state.buffer.extend(samples)
        print(
            f"live session={session_id}",
            f"received={len(samples)}",
            f"last_seq={state.last_seq}",
            f"buffered={len(state.buffer.samples)}",
            f"get_ms={get_ms:.1f}",
        )
        if get_ms >= 250.0:
            print(
                "live_fetch_slow",
                f"session={session_id}",
                f"received={len(samples)}",
                f"after_seq={state.last_seq}",
                f"get_ms={get_ms:.1f}",
            )
        before = state.windows_sent
        state.windows_sent = post_ready_windows(
            client,
            session_id,
            settings,
            detector,
            detector_name,
            state.buffer,
            state.windows_sent,
            lambda: None,
        )
        return state.windows_sent - before

    now = time.monotonic()
    if now - state.last_idle_log >= 10.0:
        print(f"live waiting session={session_id} after_seq={state.last_seq} windows={state.windows_sent}")
        state.last_idle_log = now
    return 0


def run_daemon(client: VpsClient, settings, detector: Detector, detector_name: str, worker_id: str) -> None:
    capabilities = [detector_name, "full_session_inference", "live_session_inference"]
    last_heartbeat = 0.0
    last_live_discovery = 0.0
    live_states: dict[str, LiveSessionState] = {}
    active_live_session_ids: list[str] = []
    print(f"Worker: {worker_id}")
    print("Mode: daemon")

    while True:
        current_live_job = f"live:{active_live_session_ids[0]}" if active_live_session_ids else None
        try:
            now = time.monotonic()
            if now - last_heartbeat >= settings.heartbeat_interval_seconds:
                client.heartbeat(worker_id, capabilities, current_job_id=current_live_job)
                last_heartbeat = now
            job = client.claim_next_job(worker_id)
        except httpx.HTTPError as exc:
            print(f"VPS communication error while polling daemon work: {exc}")
            time.sleep(settings.job_poll_interval_seconds)
            continue

        if not job:
            try:
                now = time.monotonic()
                if now - last_live_discovery >= settings.live_discovery_interval_seconds:
                    live_sessions = recent_live_sessions(client, settings.live_session_max_age_seconds)
                    active_live_session_ids = [
                        session["session_id"] for session in live_sessions[: max(1, settings.live_max_sessions)]
                    ]
                    for session_id in active_live_session_ids:
                        live_states.setdefault(session_id, LiveSessionState())
                    last_live_discovery = now

                windows_sent = 0
                for session_id in active_live_session_ids:
                    windows_sent += process_live_once(
                        client,
                        session_id,
                        settings,
                        detector,
                        detector_name,
                        live_states[session_id],
                    )
                if not active_live_session_ids:
                    time.sleep(settings.job_poll_interval_seconds)
                elif windows_sent == 0:
                    time.sleep(settings.poll_interval_seconds)
            except httpx.HTTPError as exc:
                print(f"VPS communication error while processing live sessions: {exc}")
                time.sleep(settings.job_poll_interval_seconds)
            continue

        job_id = job["job_id"]
        session_id = job["session_id"]
        print(f"claimed job={job_id} session={session_id}")
        client.heartbeat(worker_id, capabilities, current_job_id=job_id)
        last_heartbeat = time.monotonic()
        try:
            windows_sent = process_session(
                client,
                session_id,
                settings,
                detector,
                detector_name,
                after_seq=-1,
                drain=True,
                job_id=job_id,
            )
        except JobCancelled as exc:
            client.heartbeat(worker_id, capabilities)
            last_heartbeat = time.monotonic()
            print(f"cancelled job={job_id}: {exc}")
        except Exception as exc:
            client.fail_job(job_id, worker_id, str(exc))
            client.heartbeat(worker_id, capabilities)
            last_heartbeat = time.monotonic()
            print(f"failed job={job_id}: {exc}")
        else:
            client.complete_job(job_id, worker_id)
            client.heartbeat(worker_id, capabilities)
            last_heartbeat = time.monotonic()
            print(f"completed job={job_id} windows={windows_sent}")


def main() -> int:
    args = parse_args()
    settings = load_settings()
    session_id = args.session_id or settings.session_id

    client = VpsClient(
        settings.vps_base_url,
        api_key=settings.vps_api_key,
        timeout=settings.http_timeout_seconds,
        retries=settings.http_retries,
    )
    detector, detector_name = build_detector(settings)

    print(f"VPS: {settings.vps_base_url}")
    print(f"Detector: {detector_name}")

    try:
        if args.daemon:
            worker_id = args.worker_id or socket.gethostname()
            run_daemon(client, settings, detector, detector_name, worker_id)
        elif args.live:
            worker_id = args.worker_id or socket.gethostname()
            run_live(
                client,
                session_id,
                settings,
                detector,
                detector_name,
                worker_id,
                args.after_seq,
                args.from_end,
            )
        else:
            print(f"Session: {session_id}")
            process_session(
                client,
                session_id,
                settings,
                detector,
                detector_name,
                after_seq=args.after_seq,
                once=args.once,
                drain=args.drain,
            )
    finally:
        client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
