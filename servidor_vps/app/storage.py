from __future__ import annotations

import csv
import json
import os
import shutil
import threading
import time
import uuid
from collections import deque
from json import JSONDecodeError
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Deque

from .models import InferenceJob, InferenceResult, SampleBatch, SessionSummary, WorkerHeartbeat, WorkerInfo


CSV_HEADER = ["seq", "timestamp_ms", "acc_x_g", "acc_y_g", "acc_z_g"]
SAMPLE_CACHE_MAX_ROWS = 20000
SAMPLE_FLUSH_ROWS = int(os.getenv("TFM_SAMPLE_FLUSH_ROWS", "100"))
SAMPLE_FLUSH_SECONDS = float(os.getenv("TFM_SAMPLE_FLUSH_SECONDS", "1.0"))
RESULT_FLUSH_ROWS = int(os.getenv("TFM_RESULT_FLUSH_ROWS", "10"))
RESULT_FLUSH_SECONDS = float(os.getenv("TFM_RESULT_FLUSH_SECONDS", "1.0"))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def read_json_file(path: Path) -> dict | None:
    try:
        text = path.read_text()
        if not text.strip():
            return None
        return json.loads(text)
    except (FileNotFoundError, JSONDecodeError):
        return None


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=path.parent) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


class FileStorage:
    def __init__(self, root: Path):
        self.root = root
        self.sessions_dir = root / "sessions"
        self.results_dir = root / "results"
        self.jobs_dir = root / "jobs"
        self.workers_dir = root / "workers"
        self.live_control_path = root / "live_control.json"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.workers_dir.mkdir(parents=True, exist_ok=True)
        self.sample_cache: dict[str, Deque[dict]] = {}
        self.pending_sample_rows: dict[str, list[list]] = {}
        self.pending_results: dict[str, list[dict]] = {}
        self.metadata_cache: dict[str, dict] = {}
        self.dirty_metadata: set[str] = set()
        self.last_sample_flush: dict[str, float] = {}
        self.last_result_flush: dict[str, float] = {}
        self.lock = threading.RLock()

    def _session_dir(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        path = self.sessions_dir / safe_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _data_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "samples.csv"

    def _meta_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "metadata.json"

    def _results_path(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return self.results_dir / f"{safe_id}.jsonl"

    def _job_path(self, job_id: str) -> Path:
        safe_id = job_id.replace("/", "_").replace("\\", "_")
        return self.jobs_dir / f"{safe_id}.json"

    def _worker_path(self, worker_id: str) -> Path:
        safe_id = worker_id.replace("/", "_").replace("\\", "_")
        return self.workers_dir / f"{safe_id}.json"

    def _live_control_locked(self) -> dict:
        data = read_json_file(self.live_control_path) or {}
        return {
            "active_session_id": data.get("active_session_id"),
            "capture_enabled": bool(data.get("capture_enabled", False)),
            "updated_at": data.get("updated_at"),
        }

    def live_capture_control(self) -> dict:
        with self.lock:
            return self._live_control_locked()

    def delete_session(self, session_id: str) -> bool:
        with self.lock:
            data_path = self._data_path(session_id)
            session_dir = data_path.parent
            existed = session_dir.exists() or self._results_path(session_id).exists()

            if session_dir.exists():
                shutil.rmtree(session_dir)

            results_path = self._results_path(session_id)
            if results_path.exists():
                results_path.unlink()

            for job_path in self.jobs_dir.glob("*.json"):
                data = read_json_file(job_path)
                if data and data.get("session_id") == session_id:
                    job_path.unlink()

            self.sample_cache.pop(session_id, None)
            self.pending_sample_rows.pop(session_id, None)
            self.pending_results.pop(session_id, None)
            self.metadata_cache.pop(session_id, None)
            self.dirty_metadata.discard(session_id)
            self.last_sample_flush.pop(session_id, None)
            self.last_result_flush.pop(session_id, None)
            control = self._live_control_locked()
            if control.get("active_session_id") == session_id:
                write_json_atomic(
                    self.live_control_path,
                    {
                        "active_session_id": None,
                        "capture_enabled": False,
                        "updated_at": utc_now_iso(),
                    },
                )
            return existed

    def _cache_samples(self, session_id: str, samples: list[dict]) -> None:
        if not samples:
            return
        cache = self.sample_cache.setdefault(session_id, deque(maxlen=SAMPLE_CACHE_MAX_ROWS))
        cache.extend(samples)

    def create_or_update_metadata(self, session_id: str, mode: str, rows_delta: int) -> SessionSummary:
        with self.lock:
            data = self._update_metadata_locked(session_id, mode, rows_delta)
            self._flush_metadata_locked(session_id)
            return SessionSummary(**data)

    def _metadata_locked(self, session_id: str) -> dict:
        cached = self.metadata_cache.get(session_id)
        if cached is not None:
            return cached
        data = read_json_file(self._meta_path(session_id)) or {}
        if data:
            self.metadata_cache[session_id] = data
        return data

    def _update_metadata_locked(self, session_id: str, mode: str, rows_delta: int) -> dict:
        now = utc_now_iso()
        data = dict(self._metadata_locked(session_id))
        if data:
            data["rows"] = int(data.get("rows", 0)) + rows_delta
            data["updated_at"] = now
        else:
            data = {
                "session_id": session_id,
                "mode": mode,
                "rows": rows_delta,
                "created_at": now,
                "updated_at": now,
            }
        data.setdefault("session_id", session_id)
        data.setdefault("mode", mode)
        data.setdefault("created_at", now)
        data.setdefault("capture_enabled", False)
        data.setdefault("incoming_batches_total", 0)
        data.setdefault("incoming_rows_total", 0)
        data.setdefault("accepted_rows_total", 0)
        data.setdefault("discarded_rows_total", 0)
        data.setdefault("samples_served_total", 0)
        data.setdefault("results_total", 0)
        self.metadata_cache[session_id] = data
        self.dirty_metadata.add(session_id)
        return data

    def _record_batch_metrics_locked(
        self,
        session_id: str,
        incoming_rows: int,
        accepted_rows: int,
        source_session_id: str,
        device_id: str,
    ) -> dict:
        data = self._update_metadata_locked(session_id, "live", accepted_rows)
        now = utc_now_iso()
        data["last_batch_at"] = now
        data["last_source_session_id"] = source_session_id
        data["last_device_id"] = device_id
        data["incoming_batches_total"] = int(data.get("incoming_batches_total", 0)) + 1
        data["incoming_rows_total"] = int(data.get("incoming_rows_total", 0)) + incoming_rows
        data["accepted_rows_total"] = int(data.get("accepted_rows_total", 0)) + accepted_rows
        data["discarded_rows_total"] = int(data.get("discarded_rows_total", 0)) + max(0, incoming_rows - accepted_rows)
        self.metadata_cache[session_id] = data
        self.dirty_metadata.add(session_id)
        return data

    def set_capture_enabled(self, session_id: str, enabled: bool) -> dict:
        return self.set_live_capture_control(session_id, enabled)

    def set_live_capture_control(self, session_id: str, enabled: bool) -> dict:
        with self.lock:
            now = utc_now_iso()
            write_json_atomic(
                self.live_control_path,
                {
                    "active_session_id": session_id,
                    "capture_enabled": enabled,
                    "updated_at": now,
                },
            )
            data = self._update_metadata_locked(session_id, "live", 0)
            data["capture_enabled"] = enabled
            data["capture_control_updated_at"] = now
            self.metadata_cache[session_id] = data
            self.dirty_metadata.add(session_id)
            self._flush_metadata_locked(session_id)
            return self.live_flow_status_locked(session_id)

    def finish_live_capture(self, session_id: str) -> dict:
        with self.lock:
            now = utc_now_iso()
            self._flush_samples_locked(session_id)
            self._flush_results_locked(session_id)
            data = self._update_metadata_locked(session_id, "live", 0)
            data["capture_enabled"] = False
            data["finished_at"] = now
            data["capture_control_updated_at"] = now
            self.metadata_cache[session_id] = data
            self.dirty_metadata.add(session_id)
            self._flush_metadata_locked(session_id)
            control = self._live_control_locked()
            if control.get("active_session_id") == session_id:
                write_json_atomic(
                    self.live_control_path,
                    {
                        "active_session_id": None,
                        "capture_enabled": False,
                        "updated_at": now,
                    },
                )
            return self.live_flow_status_locked(session_id)

    def live_flow_status_locked(self, session_id: str) -> dict:
        data = dict(self._metadata_locked(session_id))
        if not data:
            data = dict(self._update_metadata_locked(session_id, "live", 0))
        control = self._live_control_locked()
        is_active_session = control.get("active_session_id") == session_id
        return {
            "session_id": session_id,
            "capture_enabled": bool(control.get("capture_enabled")) if is_active_session else False,
            "active_session_id": control.get("active_session_id"),
            "incoming_batches_total": int(data.get("incoming_batches_total", 0)),
            "incoming_rows_total": int(data.get("incoming_rows_total", 0)),
            "accepted_rows_total": int(data.get("accepted_rows_total", data.get("rows", 0))),
            "discarded_rows_total": int(data.get("discarded_rows_total", 0)),
            "samples_served_total": int(data.get("samples_served_total", 0)),
            "results_total": int(data.get("results_total", 0)),
            "rows": int(data.get("rows", 0)),
            "last_batch_at": data.get("last_batch_at"),
            "last_source_session_id": data.get("last_source_session_id"),
            "last_device_id": data.get("last_device_id"),
            "updated_at": data.get("updated_at"),
            "capture_control_updated_at": data.get("capture_control_updated_at"),
        }

    def live_flow_status(self, session_id: str) -> dict:
        with self.lock:
            return self.live_flow_status_locked(session_id)

    def _flush_metadata_locked(self, session_id: str) -> None:
        if session_id not in self.dirty_metadata:
            return
        data = self.metadata_cache.get(session_id)
        if not data:
            return
        write_json_atomic(self._meta_path(session_id), data)
        self.dirty_metadata.discard(session_id)

    def _flush_samples_locked(self, session_id: str) -> int:
        rows = self.pending_sample_rows.get(session_id, [])
        if not rows:
            return 0
        data_path = self._data_path(session_id)
        write_header = not data_path.exists()
        with data_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        flushed = len(rows)
        self.pending_sample_rows[session_id] = []
        self.last_sample_flush[session_id] = time.monotonic()
        self._flush_metadata_locked(session_id)
        return flushed

    def _flush_results_locked(self, session_id: str) -> int:
        rows = self.pending_results.get(session_id, [])
        if not rows:
            return 0
        path = self._results_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for payload in rows:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
        flushed = len(rows)
        self.pending_results[session_id] = []
        self.last_result_flush[session_id] = time.monotonic()
        return flushed

    def _maybe_flush_samples_locked(self, session_id: str) -> int:
        rows = self.pending_sample_rows.get(session_id, [])
        if not rows:
            return 0
        now = time.monotonic()
        if session_id not in self.last_sample_flush:
            self.last_sample_flush[session_id] = now
        elapsed = now - self.last_sample_flush[session_id]
        if len(rows) >= SAMPLE_FLUSH_ROWS or elapsed >= SAMPLE_FLUSH_SECONDS:
            return self._flush_samples_locked(session_id)
        return 0

    def _maybe_flush_results_locked(self, session_id: str) -> int:
        rows = self.pending_results.get(session_id, [])
        if not rows:
            return 0
        now = time.monotonic()
        if session_id not in self.last_result_flush:
            self.last_result_flush[session_id] = now
        elapsed = now - self.last_result_flush[session_id]
        if len(rows) >= RESULT_FLUSH_ROWS or elapsed >= RESULT_FLUSH_SECONDS:
            return self._flush_results_locked(session_id)
        return 0

    def flush_session(self, session_id: str) -> None:
        with self.lock:
            self._flush_samples_locked(session_id)
            self._flush_results_locked(session_id)
            self._flush_metadata_locked(session_id)

    def append_batch(self, batch: SampleBatch) -> tuple[int, str]:
        cached_rows = []
        csv_rows = []
        rows = len(batch.samples)
        for offset, sample in enumerate(batch.samples):
            seq = batch.seq_start + offset
            csv_rows.append([seq, sample.timestamp_ms, sample.acc_x_g, sample.acc_y_g, sample.acc_z_g])
            cached_rows.append(
                {
                    "seq": seq,
                    "timestamp_ms": sample.timestamp_ms,
                    "acc_x_g": sample.acc_x_g,
                    "acc_y_g": sample.acc_y_g,
                    "acc_z_g": sample.acc_z_g,
                }
            )
        with self.lock:
            control = self._live_control_locked()
            if not control["active_session_id"]:
                return 0, ""
            target_session_id = str(control["active_session_id"])
            metadata = self._metadata_locked(target_session_id)
            capture_enabled = bool(control["capture_enabled"])
            if not capture_enabled:
                self._record_batch_metrics_locked(target_session_id, rows, 0, batch.session_id, batch.device_id)
                self._flush_metadata_locked(target_session_id)
                return 0, target_session_id
            self.pending_sample_rows.setdefault(target_session_id, []).extend(csv_rows)
            self._cache_samples(target_session_id, cached_rows)
            self._record_batch_metrics_locked(target_session_id, rows, rows, batch.session_id, batch.device_id)
            self._maybe_flush_samples_locked(target_session_id)
        return rows, target_session_id

    def replace_csv(self, session_id: str, csv_bytes: bytes, max_rows: int | None = None) -> int:
        text = csv_bytes.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            raise ValueError("CSV must include a header row")

        fieldnames = set(reader.fieldnames)
        internal_required = set(CSV_HEADER)
        mlops_required = {"timestamp", "accel_x", "accel_y", "accel_z"}
        is_internal_format = internal_required.issubset(fieldnames)
        is_mlops_format = mlops_required.issubset(fieldnames)

        if not is_internal_format and not is_mlops_format:
            raise ValueError(
                "CSV must include either columns "
                f"{', '.join(CSV_HEADER)} or timestamp, accel_x, accel_y, accel_z"
            )

        rows = []
        cached_rows = []
        for seq, row in enumerate(reader):
            if max_rows is not None and seq >= max_rows:
                raise ValueError(f"CSV exceeds maximum row limit: {max_rows}")
            if is_internal_format:
                row_seq = int(row["seq"])
                timestamp_ms = int(row["timestamp_ms"])
                acc_x_g = float(row["acc_x_g"])
                acc_y_g = float(row["acc_y_g"])
                acc_z_g = float(row["acc_z_g"])
            else:
                row_seq = seq
                timestamp_ms = int(round(float(row["timestamp"]) * 1000.0))
                acc_x_g = float(row["accel_x"])
                acc_y_g = float(row["accel_y"])
                acc_z_g = float(row["accel_z"])
            rows.append([row_seq, timestamp_ms, acc_x_g, acc_y_g, acc_z_g])
            cached_rows.append(
                {
                    "seq": row_seq,
                    "timestamp_ms": timestamp_ms,
                    "acc_x_g": acc_x_g,
                    "acc_y_g": acc_y_g,
                    "acc_z_g": acc_z_g,
                }
            )

        data_path = self._data_path(session_id)
        with self.lock:
            self.pending_sample_rows.pop(session_id, None)
            self.pending_results.pop(session_id, None)
            self.metadata_cache.pop(session_id, None)
            self.dirty_metadata.discard(session_id)
            self.last_sample_flush.pop(session_id, None)
            self.last_result_flush.pop(session_id, None)
            with NamedTemporaryFile("w", newline="", delete=False, dir=data_path.parent) as tmp:
                writer = csv.writer(tmp)
                writer.writerow(CSV_HEADER)
                writer.writerows(rows)
                tmp_path = Path(tmp.name)
            tmp_path.replace(data_path)

            meta_path = self._meta_path(session_id)
            if meta_path.exists():
                meta_path.unlink()
            self.clear_results(session_id)
            self.sample_cache.pop(session_id, None)
            self._cache_samples(session_id, cached_rows[-SAMPLE_CACHE_MAX_ROWS:])
            self.create_or_update_metadata(session_id, "csv_upload", len(rows))
        return len(rows)

    def clear_results(self, session_id: str) -> None:
        with self.lock:
            self.pending_results.pop(session_id, None)
            self.last_result_flush.pop(session_id, None)
            path = self._results_path(session_id)
            if path.exists():
                path.unlink()

    def clear_jobs(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            path.unlink()

    def enqueue_inference_job(self, session_id: str) -> InferenceJob:
        self.clear_jobs()
        now = utc_now_iso()
        job = InferenceJob(
            job_id=str(uuid.uuid4()),
            session_id=session_id,
            created_at=now,
            updated_at=now,
        )
        write_json_atomic(self._job_path(job.job_id), job.model_dump())
        return job

    def get_job(self, job_id: str) -> InferenceJob | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        data = read_json_file(path)
        return InferenceJob(**data) if data is not None else None

    def list_jobs(self, status: str | None = None) -> list[InferenceJob]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            data = read_json_file(path)
            if data is None:
                continue
            job = InferenceJob(**data)
            if status is None or job.status == status:
                jobs.append(job)
        return jobs

    def _write_job(self, job: InferenceJob) -> InferenceJob:
        write_json_atomic(self._job_path(job.job_id), job.model_dump())
        return job

    def claim_next_job(self, worker_id: str) -> InferenceJob | None:
        pending_jobs = self.list_jobs(status="pending")
        if not pending_jobs:
            return None

        job = pending_jobs[0]
        now = utc_now_iso()
        updated = job.model_copy(
            update={
                "status": "running",
                "claimed_by": worker_id,
                "claimed_at": now,
                "updated_at": now,
                "error": None,
            }
        )
        return self._write_job(updated)

    def complete_job(self, job_id: str, worker_id: str) -> InferenceJob | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        data = read_json_file(path)
        if data is None:
            return None
        job = InferenceJob(**data)
        if job.claimed_by != worker_id:
            raise ValueError("Job is not claimed by this worker")
        now = utc_now_iso()
        updated = job.model_copy(
            update={
                "status": "completed",
                "completed_at": now,
                "updated_at": now,
                "error": None,
            }
        )
        return self._write_job(updated)

    def fail_job(self, job_id: str, worker_id: str, error: str | None) -> InferenceJob | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        data = read_json_file(path)
        if data is None:
            return None
        job = InferenceJob(**data)
        if job.claimed_by != worker_id:
            raise ValueError("Job is not claimed by this worker")
        now = utc_now_iso()
        updated = job.model_copy(
            update={
                "status": "failed",
                "updated_at": now,
                "error": error or "Unknown inference worker error",
            }
        )
        return self._write_job(updated)

    def register_worker(self, heartbeat: WorkerHeartbeat) -> WorkerInfo:
        path = self._worker_path(heartbeat.worker_id)
        now = utc_now_iso()
        if path.exists():
            data = read_json_file(path)
            first_seen_at = data["first_seen_at"] if data and data.get("first_seen_at") else now
        else:
            first_seen_at = now
        worker = WorkerInfo(
            worker_id=heartbeat.worker_id,
            capabilities=heartbeat.capabilities,
            current_job_id=heartbeat.current_job_id,
            first_seen_at=first_seen_at,
            last_seen_at=now,
        )
        write_json_atomic(path, worker.model_dump())
        return worker

    def list_workers(self) -> list[WorkerInfo]:
        workers = []
        for path in sorted(self.workers_dir.glob("*.json")):
            data = read_json_file(path)
            if data is None:
                continue
            workers.append(WorkerInfo(**data))
        return workers

    def list_sessions(self) -> list[SessionSummary]:
        with self.lock:
            summaries_by_id = {}
            for meta_path in sorted(self.sessions_dir.glob("*/metadata.json")):
                data = read_json_file(meta_path)
                if data is None:
                    continue
                summaries_by_id[data["session_id"]] = SessionSummary(**data)
            for session_id, data in self.metadata_cache.items():
                if data:
                    summaries_by_id[session_id] = SessionSummary(**data)
            return list(summaries_by_id.values())

    def session_csv_path(self, session_id: str) -> Path:
        self.flush_session(session_id)
        return self._data_path(session_id)

    def read_samples_after(self, session_id: str, after_seq: int, limit: int) -> list[dict]:
        with self.lock:
            cached = self.sample_cache.get(session_id)
            if cached:
                first_cached_seq = int(cached[0]["seq"])
                if after_seq >= first_cached_seq - 1:
                    samples = []
                    for sample in list(cached):
                        if int(sample["seq"]) > after_seq:
                            samples.append(sample)
                            if len(samples) >= limit:
                                break
                    return samples

        path = self._data_path(session_id)
        if not path.exists():
            return []
        samples = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq = int(row["seq"])
                if seq <= after_seq:
                    continue
                samples.append(
                    {
                        "seq": seq,
                        "timestamp_ms": int(row["timestamp_ms"]),
                        "acc_x_g": float(row["acc_x_g"]),
                        "acc_y_g": float(row["acc_y_g"]),
                        "acc_z_g": float(row["acc_z_g"]),
                    }
                )
                if len(samples) >= limit:
                    break
        return samples

    def record_samples_served(self, session_id: str, rows: int) -> None:
        if rows <= 0:
            return
        with self.lock:
            data = self._update_metadata_locked(session_id, "live", 0)
            data["samples_served_total"] = int(data.get("samples_served_total", 0)) + rows
            self.metadata_cache[session_id] = data
            self.dirty_metadata.add(session_id)
            self._maybe_flush_samples_locked(session_id)

    def append_result(self, result: InferenceResult) -> None:
        payload = result.model_dump()
        payload["created_at"] = utc_now_iso()
        with self.lock:
            self.pending_results.setdefault(result.session_id, []).append(payload)
            data = self._update_metadata_locked(result.session_id, "live", 0)
            data["results_total"] = int(data.get("results_total", 0)) + 1
            self.metadata_cache[result.session_id] = data
            self.dirty_metadata.add(result.session_id)
            self._maybe_flush_results_locked(result.session_id)

    def read_results(self, session_id: str, limit: int) -> list[dict]:
        with self.lock:
            self._maybe_flush_results_locked(session_id)
            path = self._results_path(session_id)
            results = []
            if path.exists():
                lines = path.read_text().splitlines()
                results.extend(json.loads(line) for line in lines if line.strip())
            results.extend(self.pending_results.get(session_id, []))
            return results[-limit:]

    def summarize_results(self, session_id: str) -> dict:
        with self.lock:
            self._maybe_flush_results_locked(session_id)
            path = self._results_path(session_id)
            results = []
            if path.exists():
                results.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
            results.extend(self.pending_results.get(session_id, []))

        if not results:
            return {
                "session_id": session_id,
                "window_count": 0,
                "session_status": "pending",
                "status_counts": {"normal": 0, "anomaly": 0, "unreliable": 0},
                "anomaly_ratio": 0.0,
                "unreliable_ratio": 0.0,
                "max_anomaly_score": None,
                "max_reconstruction_error": None,
                "latest_status": None,
                "latest_result": None,
            }
        status_counts = {"normal": 0, "anomaly": 0, "unreliable": 0}
        max_anomaly_score = None
        max_reconstruction_error = None
        max_score_result = None

        for result in results:
            status = result.get("status")
            if status in status_counts:
                status_counts[status] += 1

            score = result.get("anomaly_score")
            if score is not None and (max_anomaly_score is None or score > max_anomaly_score):
                max_anomaly_score = score
                max_score_result = result

            reconstruction_error = result.get("reconstruction_error")
            if reconstruction_error is not None and (
                max_reconstruction_error is None or reconstruction_error > max_reconstruction_error
            ):
                max_reconstruction_error = reconstruction_error

        window_count = len(results)
        anomaly_ratio = status_counts["anomaly"] / window_count if window_count else 0.0
        unreliable_ratio = status_counts["unreliable"] / window_count if window_count else 0.0
        latest_result = results[-1] if results else None

        session_status = "normal"
        if not window_count:
            session_status = "pending"
        elif unreliable_ratio >= 0.2:
            session_status = "bad_data"
        elif anomaly_ratio >= 0.05:
            session_status = "anomaly"
        elif anomaly_ratio >= 0.01:
            session_status = "review"

        return {
            "session_id": session_id,
            "window_count": window_count,
            "session_status": session_status,
            "status_counts": status_counts,
            "anomaly_ratio": anomaly_ratio,
            "unreliable_ratio": unreliable_ratio,
            "max_anomaly_score": max_anomaly_score,
            "max_reconstruction_error": max_reconstruction_error,
            "latest_status": latest_result.get("status") if latest_result else None,
            "latest_result": latest_result,
            "max_score_result": max_score_result,
        }
