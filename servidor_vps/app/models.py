from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Sample(BaseModel):
    timestamp_ms: int = Field(ge=0)
    acc_x_g: float
    acc_y_g: float
    acc_z_g: float


class SampleBatch(BaseModel):
    device_id: str = Field(min_length=1, max_length=80)
    session_id: str = Field(min_length=1, max_length=120)
    seq_start: int = Field(ge=0)
    sample_rate_hz: float = Field(gt=0)
    samples: list[Sample] = Field(min_length=1, max_length=500)


class InferenceResult(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    source: Literal["pc_inference", "manual", "test"] = "pc_inference"
    window_start_ms: int = Field(ge=0)
    window_end_ms: int = Field(ge=0)
    status: Literal["normal", "anomaly", "unreliable"]
    anomaly_score: float | None = None
    reconstruction_error: float | None = None
    quality: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class SessionSummary(BaseModel):
    session_id: str
    mode: Literal["csv_upload", "live"]
    rows: int
    created_at: str
    updated_at: str


class AcceptedResponse(BaseModel):
    ok: bool = True
    session_id: str
    rows_received: int


class LiveCaptureControl(BaseModel):
    capture_enabled: bool
    session_id: str | None = Field(default=None, min_length=1, max_length=120)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=512)


class WorkerHeartbeat(BaseModel):
    worker_id: str = Field(min_length=1, max_length=120)
    capabilities: list[str] = Field(default_factory=list)
    current_job_id: str | None = None


class WorkerInfo(BaseModel):
    worker_id: str
    capabilities: list[str] = Field(default_factory=list)
    current_job_id: str | None = None
    first_seen_at: str
    last_seen_at: str


class JobCreateResponse(BaseModel):
    ok: bool = True
    job_id: str
    session_id: str


class JobClaimRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=120)


class JobStatusUpdate(BaseModel):
    worker_id: str = Field(min_length=1, max_length=120)
    error: str | None = None


class InferenceJob(BaseModel):
    job_id: str
    session_id: str
    job_type: Literal["full_session_inference"] = "full_session_inference"
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    created_at: str
    updated_at: str
    claimed_by: str | None = None
    claimed_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
