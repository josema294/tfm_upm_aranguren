from __future__ import annotations

import asyncio
import html
import json
import os
import time
import mimetypes
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .auth import (
    clear_session_cookie,
    create_session_token,
    load_auth_settings,
    request_has_valid_api_key,
    request_has_valid_session,
    require_machine_or_web_auth,
    set_session_cookie,
    verify_password,
)
from .models import (
    AcceptedResponse,
    InferenceJob,
    InferenceResult,
    JobClaimRequest,
    JobCreateResponse,
    JobStatusUpdate,
    LiveCaptureControl,
    LoginRequest,
    SampleBatch,
    SessionSummary,
    WorkerHeartbeat,
    WorkerInfo,
)
from .storage import FileStorage


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DOCS_DIR = APP_DIR.parent / "docs"
MEMORIA_PDF_PATH = DOCS_DIR / "memoria_tfm.pdf"
DATA_ROOT = Path(os.getenv("TFM_SERVER_DATA", "server_data"))
HOME_VIDEO_CANDIDATES = [
    Path(os.getenv("TFM_HOME_VIDEO", "")) if os.getenv("TFM_HOME_VIDEO") else None,
    Path("/data/videohome.mp4"),
    Path("/data/videohome/tfm-vtrain.webm"),
    DATA_ROOT.parent / "videohome.mp4",
    DATA_ROOT.parent / "videohome" / "tfm-vtrain.webm",
]
storage = FileStorage(DATA_ROOT)
auth_settings = load_auth_settings()
login_failures: dict[str, list[float]] = {}
LOGIN_MAX_ATTEMPTS = int(os.getenv("TFM_LOGIN_MAX_ATTEMPTS", "10"))
LOGIN_WINDOW_SECONDS = int(os.getenv("TFM_LOGIN_WINDOW_SECONDS", "300"))

app = FastAPI(
    title="Servidor de deteccion de anomalias ferroviarias",
    version="0.1.0",
    description="API puente para lotes de vibracion del ESP32, sesiones CSV y resultados de inferencia.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


LOGIN_HTML = """<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Login · Monitorización Automatizada del Estado de Sistemas Ferroviarios mediante Deep Learning
</title>
    <link rel="icon" type="image/png" href="/ui/favicon.png" />
    <style>
      body { align-items: center; background: #f5f7fa; color: #172033; display: flex; font-family: system-ui, sans-serif; justify-content: center; margin: 0; min-height: 100vh; }
      main { background: #fff; border: 1px solid #dfe5ef; border-radius: 8px; box-sizing: border-box; max-width: 390px; padding: 24px; width: 92vw; }
      h1 { font-size: 22px; margin: 0 0 6px; }
      p { color: #61708a; margin: 0 0 18px; }
      label { color: #34425a; display: grid; gap: 6px; margin: 12px 0; }
      input { border: 1px solid #c9d3e2; border-radius: 6px; font: inherit; padding: 10px; }
      button { background: #1d4ed8; border: 0; border-radius: 6px; color: #fff; cursor: pointer; font: inherit; font-weight: 700; margin-top: 10px; padding: 10px 12px; width: 100%; }
      .error { background: #fdecec; border-radius: 6px; color: #a81818; margin-top: 12px; padding: 10px; }
    </style>
  </head>
  <body>
    <main>
      <h1>Monitorización Automatizada del Estado de Sistemas Ferroviarios mediante Deep Learning
</h1>
      <p>Acceso protegido al panel de anomalias ferroviarias.</p>
      <form method="post" action="/login">
        <label>Usuario <input name="username" autocomplete="username" required /></label>
        <label>Contrasena <input name="password" type="password" autocomplete="current-password" required /></label>
        <button type="submit">Entrar</button>
      </form>
    </main>
  </body>
</html>"""


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    public_paths = {"/login", "/api/v1/auth/login"}
    if path in public_paths:
        return await call_next(request)

    authenticated = request_has_valid_session(request, auth_settings) or request_has_valid_api_key(request, auth_settings)
    if authenticated:
        return await call_next(request)

    accepts_html = "text/html" in request.headers.get("accept", "")
    if accepts_html and not path.startswith("/api/"):
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    return JSONResponse({"detail": "Authentication required"}, status_code=status.HTTP_401_UNAUTHORIZED)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "media-src 'self'; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'",
    )
    response.headers.setdefault("Cache-Control", "no-store")
    if auth_settings.secure_cookie:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response


def _client_ip(request: Request) -> str:
    return (
        request.headers.get("cf-connecting-ip")
        or request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        or (request.client.host if request.client else "unknown")
    )


def _login_allowed(client_ip: str) -> bool:
    now = time.time()
    cutoff = now - LOGIN_WINDOW_SECONDS
    failures = [failure_time for failure_time in login_failures.get(client_ip, []) if failure_time >= cutoff]
    login_failures[client_ip] = failures
    return len(failures) < LOGIN_MAX_ATTEMPTS


def _record_login_failure(client_ip: str) -> None:
    login_failures.setdefault(client_ip, []).append(time.time())


def _clear_login_failures(client_ip: str) -> None:
    login_failures.pop(client_ip, None)


def _render_markdown_document(markdown: str) -> str:
    html_parts: list[str] = []
    paragraph: list[str] = []
    code_block: list[str] = []
    in_code = False
    in_list = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            html_parts.append(f"<p>{html.escape(' '.join(paragraph))}</p>")
            paragraph = []

    def flush_list() -> None:
        nonlocal in_list
        if in_list:
            html_parts.append("</ul>")
            in_list = False

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                html_parts.append(f"<pre><code>{html.escape(chr(10).join(code_block))}</code></pre>")
                code_block = []
                in_code = False
            else:
                flush_paragraph()
                flush_list()
                in_code = True
            continue
        if in_code:
            code_block.append(line)
            continue
        if not stripped:
            flush_paragraph()
            flush_list()
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            flush_list()
            html_parts.append(f"<h1>{html.escape(stripped[2:].strip())}</h1>")
        elif stripped.startswith("## "):
            flush_paragraph()
            flush_list()
            html_parts.append(f"<h2>{html.escape(stripped[3:].strip())}</h2>")
        elif stripped.startswith("### "):
            flush_paragraph()
            flush_list()
            html_parts.append(f"<h3>{html.escape(stripped[4:].strip())}</h3>")
        elif stripped.startswith("- "):
            flush_paragraph()
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{html.escape(stripped[2:].strip())}</li>")
        else:
            paragraph.append(stripped)

    if in_code:
        html_parts.append(f"<pre><code>{html.escape(chr(10).join(code_block))}</code></pre>")
    flush_paragraph()
    flush_list()
    return "\n".join(html_parts)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.png", media_type="image/png")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request_has_valid_session(request, auth_settings):
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return HTMLResponse(LOGIN_HTML)


@app.post("/login")
async def login_form(request: Request) -> RedirectResponse:
    client_ip = _client_ip(request)
    if not _login_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many login attempts")
    form = await request.form()
    username = str(form.get("username", ""))
    password = str(form.get("password", ""))
    if username != auth_settings.username or not verify_password(password, auth_settings.password_hash):
        _record_login_failure(client_ip)
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    _clear_login_failures(client_ip)
    response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    set_session_cookie(response, create_session_token(username, auth_settings), auth_settings)
    return response


@app.post("/api/v1/auth/login")
def login_api(payload: LoginRequest, request: Request) -> Response:
    client_ip = _client_ip(request)
    if not _login_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many login attempts")
    if payload.username != auth_settings.username or not verify_password(payload.password, auth_settings.password_hash):
        _record_login_failure(client_ip)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    _clear_login_failures(client_ip)
    response = JSONResponse({"ok": True})
    set_session_cookie(response, create_session_token(payload.username, auth_settings), auth_settings)
    return response


@app.post("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)
    clear_session_cookie(response, auth_settings)
    return response


@app.get("/docs", include_in_schema=False)
def docs(request: Request):
    require_machine_or_web_auth(request, auth_settings)
    docs_path = DOCS_DIR / "server_api.md"
    if not docs_path.exists():
        raise HTTPException(status_code=404, detail="API documentation not found")
    body = _render_markdown_document(docs_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        f"""<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Documentacion API</title>
    <style>
      body {{ background: #f5f7fa; color: #172033; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; }}
      main {{ background: #fff; border: 1px solid #dfe5ef; border-radius: 8px; box-sizing: border-box; margin: 24px auto; max-width: 980px; padding: 24px; width: calc(100% - 32px); }}
      h1, h2, h3 {{ color: #172033; line-height: 1.2; }}
      h1 {{ margin-top: 0; }}
      p, li {{ color: #52627a; line-height: 1.55; }}
      a {{ color: #1d4ed8; font-weight: 650; }}
      pre {{ background: #0f172a; border-radius: 8px; color: #e5e7eb; overflow-x: auto; padding: 14px; }}
      code {{ font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }}
      .actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }}
      .button {{ background: #e8eef8; border-radius: 6px; color: #1f3356; display: inline-block; padding: 9px 12px; text-decoration: none; }}
    </style>
  </head>
  <body>
    <main>
      <div class="actions">
        <a class="button" href="/">Volver al panel</a>
        <a class="button" href="/openapi.json">Descargar OpenAPI JSON</a>
      </div>
      {body}
    </main>
  </body>
</html>"""
    )


@app.get("/openapi.json", include_in_schema=False)
def openapi_json(request: Request):
    require_machine_or_web_auth(request, auth_settings)
    return app.openapi()


@app.get("/docs/memoria_tfm.pdf", include_in_schema=False)
def memoria_tfm_pdf(request: Request) -> FileResponse:
    require_machine_or_web_auth(request, auth_settings)
    if not MEMORIA_PDF_PATH.exists():
        raise HTTPException(status_code=404, detail="TFM report PDF not found")
    return FileResponse(
        MEMORIA_PDF_PATH,
        media_type="application/pdf",
        filename="memoria_tfm_monitorizacion_ferroviaria.pdf",
    )


@app.get("/health")
def health(request: Request) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    return {"ok": True}


@app.get("/media/videohome")
def home_video(request: Request) -> FileResponse:
    require_machine_or_web_auth(request, auth_settings)
    video_path = next((path for path in HOME_VIDEO_CANDIDATES if path and path.exists()), None)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Home video not found")
    media_type = mimetypes.guess_type(video_path.name)[0] or "application/octet-stream"
    return FileResponse(video_path, media_type=media_type, filename=video_path.name)


@app.post("/api/v1/samples/batch", response_model=AcceptedResponse)
def receive_sample_batch(request: Request, batch: SampleBatch) -> AcceptedResponse:
    require_machine_or_web_auth(request, auth_settings)
    started = time.perf_counter()
    rows, stored_session_id = storage.append_batch(batch)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        "sample_batch",
        f"source_session={batch.session_id}",
        f"stored_session={stored_session_id or 'none'}",
        f"device={batch.device_id}",
        f"seq={batch.seq_start}..{batch.seq_start + len(batch.samples) - 1}",
        f"incoming_rows={len(batch.samples)}",
        f"accepted_rows={rows}",
        f"write_ms={elapsed_ms:.1f}",
    )
    if elapsed_ms >= 100.0:
        print(
            "sample_batch_slow",
            f"session={batch.session_id}",
            f"rows={rows}",
            f"write_ms={elapsed_ms:.1f}",
        )
    return AcceptedResponse(session_id=stored_session_id, rows_received=rows)


@app.post("/api/v1/sessions/{session_id}/csv", response_model=AcceptedResponse)
async def upload_session_csv(request: Request, session_id: str, file: UploadFile = File(...)) -> AcceptedResponse:
    require_machine_or_web_auth(request, auth_settings)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > auth_settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"CSV upload exceeds {auth_settings.max_upload_bytes} bytes")
    csv_bytes = await file.read(auth_settings.max_upload_bytes + 1)
    if len(csv_bytes) > auth_settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"CSV upload exceeds {auth_settings.max_upload_bytes} bytes")
    try:
        rows = storage.replace_csv(session_id, csv_bytes, max_rows=auth_settings.max_csv_rows)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    storage.clear_results(session_id)
    storage.enqueue_inference_job(session_id)
    return AcceptedResponse(session_id=session_id, rows_received=rows)


@app.post("/api/v1/sessions/{session_id}/jobs", response_model=JobCreateResponse)
def enqueue_session_job(request: Request, session_id: str, clear_results: bool = True) -> JobCreateResponse:
    require_machine_or_web_auth(request, auth_settings)
    if not storage.session_csv_path(session_id).exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if clear_results:
        storage.clear_results(session_id)
    job = storage.enqueue_inference_job(session_id)
    return JobCreateResponse(job_id=job.job_id, session_id=job.session_id)


@app.get("/api/v1/sessions", response_model=list[SessionSummary])
def list_sessions(request: Request) -> list[SessionSummary]:
    require_machine_or_web_auth(request, auth_settings)
    return storage.list_sessions()


@app.get("/api/v1/sessions/{session_id}/csv")
def download_session_csv(request: Request, session_id: str) -> FileResponse:
    require_machine_or_web_auth(request, auth_settings)
    path = storage.session_csv_path(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return FileResponse(path, media_type="text/csv", filename=f"{session_id}.csv")


@app.delete("/api/v1/sessions/{session_id}")
def delete_session(request: Request, session_id: str) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    deleted = storage.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "session_id": session_id}


@app.get("/api/v1/sessions/{session_id}/samples")
def get_session_samples(request: Request, session_id: str, after_seq: int = -1, limit: int = 500) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    limit = max(1, min(limit, 5000))
    started = time.perf_counter()
    samples = storage.read_samples_after(session_id, after_seq, limit)
    if request_has_valid_api_key(request, auth_settings):
        storage.record_samples_served(session_id, len(samples))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if elapsed_ms >= 100.0:
        last_seq = samples[-1]["seq"] if samples else None
        print(
            "sample_read_slow",
            f"session={session_id}",
            f"after_seq={after_seq}",
            f"limit={limit}",
            f"rows={len(samples)}",
            f"last_seq={last_seq}",
            f"read_ms={elapsed_ms:.1f}",
        )
    return {"session_id": session_id, "after_seq": after_seq, "samples": samples}


@app.post("/api/v1/inference/results")
def post_inference_result(request: Request, result: InferenceResult) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    if result.window_end_ms <= result.window_start_ms:
        raise HTTPException(status_code=400, detail="window_end_ms must be greater than window_start_ms")
    storage.append_result(result)
    return {"ok": True}


@app.get("/api/v1/sessions/{session_id}/results")
def get_inference_results(request: Request, session_id: str, limit: int = 100) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    limit = max(1, min(limit, 5000))
    return {"session_id": session_id, "results": storage.read_results(session_id, limit)}


@app.get("/api/v1/sessions/{session_id}/summary")
def get_inference_summary(request: Request, session_id: str) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    return storage.summarize_results(session_id)


@app.get("/api/v1/live/{session_id}/control")
def get_live_capture_control(request: Request, session_id: str) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    return storage.live_flow_status(session_id)


@app.post("/api/v1/live/{session_id}/control")
def set_live_capture_control(request: Request, session_id: str, payload: LiveCaptureControl) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    return storage.set_capture_enabled(session_id, payload.capture_enabled)


@app.get("/api/v1/live/control")
def get_global_live_capture_control(request: Request) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    return storage.live_capture_control()


@app.post("/api/v1/live/control")
def set_global_live_capture_control(request: Request, payload: LiveCaptureControl) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    session_id = (payload.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return storage.set_live_capture_control(session_id, payload.capture_enabled)


@app.post("/api/v1/live/control/finish")
def finish_global_live_capture(request: Request, payload: LiveCaptureControl) -> dict:
    require_machine_or_web_auth(request, auth_settings)
    session_id = (payload.session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return storage.finish_live_capture(session_id)


def _sse_message(event: str, payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return f"event: {event}\ndata: {data}\n\n"


@app.get("/api/v1/live/{session_id}/events")
async def live_session_events(request: Request, session_id: str, limit: int = 120, interval: float = 1.0) -> StreamingResponse:
    require_machine_or_web_auth(request, auth_settings)
    limit = max(1, min(limit, 300))
    interval = max(0.25, min(interval, 5.0))

    async def event_stream():
        last_payload = ""
        while not await request.is_disconnected():
            session = next((item.model_dump() for item in storage.list_sessions() if item.session_id == session_id), None)
            if session is None:
                yield _sse_message("missing", {"session_id": session_id})
                await asyncio.sleep(interval)
                continue

            payload = {
                "session": session,
                "summary": storage.summarize_results(session_id),
                "results": storage.read_results(session_id, limit),
                "flow": storage.live_flow_status(session_id),
                "server_time": time.time(),
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if encoded != last_payload:
                yield f"event: snapshot\ndata: {encoded}\n\n"
                last_payload = encoded
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(interval)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/workers/heartbeat", response_model=WorkerInfo)
def worker_heartbeat(request: Request, heartbeat: WorkerHeartbeat) -> WorkerInfo:
    require_machine_or_web_auth(request, auth_settings)
    return storage.register_worker(heartbeat)


@app.get("/api/v1/workers", response_model=list[WorkerInfo])
def list_workers(request: Request) -> list[WorkerInfo]:
    require_machine_or_web_auth(request, auth_settings)
    return storage.list_workers()


@app.get("/api/v1/inference/jobs", response_model=list[InferenceJob])
def list_inference_jobs(request: Request, status: str | None = None) -> list[InferenceJob]:
    require_machine_or_web_auth(request, auth_settings)
    return storage.list_jobs(status=status)


@app.get("/api/v1/inference/jobs/{job_id}", response_model=InferenceJob)
def get_inference_job(request: Request, job_id: str) -> InferenceJob:
    require_machine_or_web_auth(request, auth_settings)
    job = storage.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/v1/inference/jobs/next", response_model=InferenceJob | None)
def claim_next_inference_job(http_request: Request, request: JobClaimRequest) -> InferenceJob | None:
    require_machine_or_web_auth(http_request, auth_settings)
    return storage.claim_next_job(request.worker_id)


@app.post("/api/v1/inference/jobs/{job_id}/complete", response_model=InferenceJob)
def complete_inference_job(request: Request, job_id: str, update: JobStatusUpdate) -> InferenceJob:
    require_machine_or_web_auth(request, auth_settings)
    try:
        job = storage.complete_job(job_id, update.worker_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/v1/inference/jobs/{job_id}/fail", response_model=InferenceJob)
def fail_inference_job(request: Request, job_id: str, update: JobStatusUpdate) -> InferenceJob:
    require_machine_or_web_auth(request, auth_settings)
    try:
        job = storage.fail_job(job_id, update.worker_id, update.error)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    storage.clear_results(job.session_id)
    return job
