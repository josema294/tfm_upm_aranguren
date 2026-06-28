const sessionsBody = document.querySelector("#sessionsBody");
const detailEl = document.querySelector("#detail");
const liveDetailEl = document.querySelector("#liveDetail");
const jobsDetailEl = document.querySelector("#jobsDetail");
const architectureDetailEl = document.querySelector("#architectureDetail");
const toastEl = document.querySelector("#toast");
const sortHeaders = document.querySelectorAll(".sort-header");
const tabButtons = document.querySelectorAll(".tab");
const tabPanels = document.querySelectorAll(".tab-panel");
const workerSelect = document.querySelector("#workerSelect");
const syncStatusEl = document.querySelector("#syncStatus");
const liveSessionLabel = document.querySelector("#liveSessionLabel");
const liveAlert = document.querySelector("#liveAlert");
const liveMetrics = document.querySelector("#liveMetrics");
const liveScoreCanvas = document.querySelector("#liveScoreCanvas");
const liveScoreLabel = document.querySelector("#liveScoreLabel");
const liveWindowCanvas = document.querySelector("#liveWindowCanvas");
const liveWindowLabel = document.querySelector("#liveWindowLabel");
const liveWindowHint = document.querySelector("#liveWindowHint");
const liveRefreshBtn = document.querySelector("#liveRefreshBtn");
const liveControlForm = document.querySelector("#liveControlForm");
const liveControlSessionId = document.querySelector("#liveControlSessionId");
const liveControlStartBtn = document.querySelector("#liveControlStartBtn");
const liveControlPauseBtn = document.querySelector("#liveControlPauseBtn");
const liveControlFinishBtn = document.querySelector("#liveControlFinishBtn");
const liveAutoFollow = document.querySelector("#liveAutoFollow");
const liveWindowSlider = document.querySelector("#liveWindowSlider");
const liveYZoomLabel = document.querySelector("#liveYZoomLabel");
const liveSessionsBody = document.querySelector("#liveSessionsBody");
const liveSessionsRefreshBtn = document.querySelector("#liveSessionsRefreshBtn");
const vpsNode = document.querySelector("#vpsNode");
const workerNode = document.querySelector("#workerNode");
const espNode = document.querySelector("#espNode");
const prevSessionsPageBtn = document.querySelector("#prevSessionsPage");
const nextSessionsPageBtn = document.querySelector("#nextSessionsPage");
const sessionsPageInfo = document.querySelector("#sessionsPageInfo");
const sessionsBtn = document.querySelector("#sessionsBtn");

document.querySelectorAll(".institution-card img").forEach((image) => {
  image.addEventListener("error", () => {
    image.hidden = true;
    image.closest(".institution-card")?.classList.add("no-logo");
  });
});

let sessionRows = [];
let sessionSort = { key: "updated_at", direction: "desc" };
let sessionsPage = 1;
const sessionsPageSize = 5;
const workerGreenMs = 20000;
const workerYellowMs = 45000;
const liveGreenMs = 15000;
const liveYellowMs = 60000;
const autoRefreshMs = 5000;
const liveRefreshMs = 5000;
let latestWorkers = [];
let latestJobs = [];
let detailWindows = [];
let detailWindowIndex = 0;
let sessionsLoadSeq = 0;
let sessionsLoading = false;
let liveRefreshSeq = 0;
let liveRenderedWindowKey = "";
let liveRateSnapshot = null;
let liveRateHistory = [];
let liveWindows = [];
let liveWindowIndex = 0;
let liveCurrentSessionId = "";
let liveActiveSessionId = "";
let liveEventSource = null;
let liveEventSessionId = "";
let liveSseConnected = false;
let liveCaptureEnabled = false;
let liveCaptureStartPending = false;
const windowSeriesOptions = {
  x: true,
  y: true,
  z: true,
  rx: false,
  ry: false,
  rz: false,
};
let windowYZoom = 1;

const sessionStatusLabels = {
  pending: "pendiente",
  normal: "normal",
  review: "revisar",
  anomaly: "anomalia",
  bad_data: "datos malos",
  running: "en ejecucion",
  completed: "completado",
  failed: "fallido",
  cancelled: "cancelado",
};

function showToast(message) {
  toastEl.textContent = message;
  toastEl.hidden = false;
  setTimeout(() => {
    toastEl.hidden = true;
  }, 2600);
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined) return "n/a";
  return Number(value).toLocaleString("es-ES", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDateTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "n/a";
  const pad = (number) => String(number).padStart(2, "0");
  const day = pad(date.getDate());
  const month = pad(date.getMonth() + 1);
  const year = pad(date.getFullYear() % 100);
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  return `${day}/${month}/${year} - ${hours}:${minutes}:${seconds}`;
}

function sessionStatus(summary) {
  return summary?.session_status || "pending";
}

function sessionStatusLabel(status) {
  return sessionStatusLabels[status] || status;
}

async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (response.status === 401) {
    window.location.href = "/login";
    throw new Error("Authentication required");
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

async function refreshHealth() {
  try {
    await api("/health");
    setConnectionState(vpsNode, "ok");
  } catch (error) {
    setConnectionState(vpsNode, "error");
  }
}

function setActiveTab(tabId) {
  for (const button of tabButtons) {
    button.classList.toggle("active", button.dataset.tab === tabId);
  }
  for (const panel of tabPanels) {
    panel.classList.toggle("active", panel.id === `tab-${tabId}`);
  }
  if (tabId === "async" || tabId === "sync") {
    loadSessions(false).catch((error) => showToast(error.message));
  }
}

function setConnectionState(element, state) {
  if (!element) return;
  element.classList.remove("ok", "pending", "error");
  element.classList.add(state);
}

function ageMs(isoDate) {
  const parsed = Date.parse(isoDate);
  if (!parsed) return Number.POSITIVE_INFINITY;
  return Date.now() - parsed;
}

function stateFromAge(age, greenMs, yellowMs) {
  if (age <= greenMs) return "ok";
  if (age <= yellowMs) return "pending";
  return "error";
}

function formatAge(age) {
  if (!Number.isFinite(age)) return "sin heartbeat";
  if (age < 1000) return "ahora";
  return `${Math.round(age / 1000)}s`;
}

function nodeText(node, text) {
  const small = node?.querySelector("small");
  if (small) small.textContent = text;
}

function latestLiveSession() {
  return sessionRows
    .map((row) => row.session)
    .filter((session) => session.mode === "live")
    .sort((left, right) => (Date.parse(right.updated_at) || 0) - (Date.parse(left.updated_at) || 0))[0];
}

function liveRows() {
  return sessionRows
    .filter((row) => row.session.mode === "live")
    .sort((left, right) => (Date.parse(right.session.updated_at) || 0) - (Date.parse(left.session.updated_at) || 0));
}

function renderLiveSessionsTable() {
  if (!liveSessionsBody) return;
  const rows = liveRows();
  liveSessionsBody.innerHTML = "";
  if (!rows.length) {
    liveSessionsBody.innerHTML = '<tr><td colspan="6">No hay sesiones live guardadas.</td></tr>';
    return;
  }

  for (const { session, summary } of rows) {
    const status = sessionStatus(summary);
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${session.session_id}</td>
      <td>${session.rows}</td>
      <td>${formatDateTime(session.updated_at)}</td>
      <td><span class="badge ${status}">${sessionStatusLabel(status)}</span></td>
      <td>${summary ? `${(summary.anomaly_ratio * 100).toFixed(2)}%` : "n/a"}</td>
      <td class="actions">
        <button class="secondary" data-live-action="view" data-id="${session.session_id}">Ver</button>
        <a class="download" href="/api/v1/sessions/${session.session_id}/csv">CSV</a>
        <button class="danger-button" data-live-action="delete" data-id="${session.session_id}">Borrar</button>
      </td>
    `;
    liveSessionsBody.appendChild(row);
  }
}

function updateSyncStatus(live = null) {
  if (!live) {
    setConnectionState(syncStatusEl, "error");
    setConnectionState(espNode, "error");
    syncStatusEl.querySelector("strong").textContent = "Registro live en espera";
    syncStatusEl.querySelector("p").textContent = "Define una sesion y pulsa Iniciar registro para aceptar datos.";
    nodeText(espNode, "VPS en espera");
    return;
  }

  const liveAge = ageMs(live.updated_at);
  const state = stateFromAge(liveAge, liveGreenMs, liveYellowMs);
  setConnectionState(syncStatusEl, state);
  setConnectionState(espNode, state);
  syncStatusEl.querySelector("strong").textContent =
    state === "ok" ? "ESP32 enviando datos" : state === "pending" ? "ESP32 con datos antiguos" : "ESP32 desconectado";
  syncStatusEl.querySelector("p").textContent = `${live.session_id} · ${live.rows} filas · actualizada ${live.updated_at}`;
  nodeText(
    espNode,
    state === "ok" ? `Sesion ${live.session_id}` : state === "pending" ? "Datos antiguos" : "Sin datos recientes",
  );
}

function latestResult(results) {
  return results.length ? results[results.length - 1] : null;
}

function formatOptional(value, digits = 3) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? "n/a"
    : formatNumber(value, digits);
}

function resultBreakdown(result) {
  const metadata = result?.metadata || {};
  const vae = metadata.vae || {};
  const slip = metadata.slip || {};
  const anomalyType = metadata.anomaly_type || "none";
  return {
    anomalyType,
    anomalyLabel: anomalyTypeLabel(anomalyType),
    vaeStatus: vae.status || "n/a",
    vaeScore: vae.anomaly_score,
    vaeRawScore: vae.vae_score,
    vaeThreshold: vae.threshold,
    slipStatus: slip.status || "n/a",
    slipScore: slip.anomaly_score,
    slipProbability: slip.slip_probability,
    slipThreshold: slip.slip_threshold,
    slipWindowSize: slip.slip_window_size,
    slipArchitecture: slip.architecture,
  };
}

function anomalyTypeLabel(type) {
  const labels = {
    none: "sin anomalia",
    slip: "slip / patinaje",
    impact_or_general: "impacto / anomalia general",
    unreliable_window: "ventana no fiable",
  };
  return labels[type] || type || "n/a";
}

function resultDetailText(result) {
  if (!result) return "Sin resultado";
  const breakdown = resultBreakdown(result);
  return [
    `tipo ${breakdown.anomalyLabel}`,
    `VAE ${breakdown.vaeStatus} score ${formatOptional(breakdown.vaeScore)}`,
    `CNN slip ${breakdown.slipStatus} prob ${formatOptional(breakdown.slipProbability)}`,
  ].join(" · ");
}

function resultModelChips(result) {
  const breakdown = resultBreakdown(result);
  return `
    <div class="model-breakdown">
      <span class="model-chip type">Tipo: ${breakdown.anomalyLabel}</span>
      <span class="model-chip vae">VAE: ${breakdown.vaeStatus} · score ${formatOptional(breakdown.vaeScore)}</span>
      <span class="model-chip slip">CNN slip: ${breakdown.slipStatus} · prob ${formatOptional(breakdown.slipProbability)} / umbral ${formatOptional(breakdown.slipThreshold)}</span>
    </div>
  `;
}

function resultPointColor(result) {
  if (result.status === "unreliable") return "#f59e0b";
  if (result.status !== "anomaly") return "#1d4ed8";
  const type = resultBreakdown(result).anomalyType;
  if (type === "slip") return "#f97316";
  if (type === "impact_or_general") return "#dc2626";
  return "#ef4444";
}

function setLiveAlert(state, title, message) {
  liveAlert.classList.remove("idle", "normal", "warning", "danger");
  liveAlert.classList.add(state);
  liveAlert.querySelector("strong").textContent = title;
  liveAlert.querySelector("span").textContent = message;
}

function renderLiveEmpty(message = "No hay una sesion live reciente.", flow = null) {
  liveWindows = [];
  liveWindowIndex = 0;
  liveCurrentSessionId = "";
  liveRenderedWindowKey = "";
  liveRateSnapshot = null;
  liveRateHistory = [];
  liveSessionLabel.textContent = "Sin sesion live activa.";
  liveScoreLabel.textContent = "Sin ventanas";
  liveWindowLabel.textContent = "Sin ventana";
  liveWindowHint.textContent = message;
  liveMetrics.innerHTML = `
    <div class="metric"><span>Sesion</span><strong>n/a</strong></div>
    <div class="metric"><span>Filas recibidas</span><strong>0</strong></div>
    <div class="metric"><span>Tasa entrada · media 10s</span><strong>n/a</strong><small>inst n/a</small></div>
    <div class="metric"><span>Ventanas inferidas</span><strong>0</strong></div>
    <div class="metric"><span>Flujo worker</span><strong>0</strong><small>muestras servidas</small></div>
    <div class="metric"><span>Descartadas</span><strong>0</strong><small>captura pausada</small></div>
    <div class="metric"><span>Ultimo score</span><strong>n/a</strong></div>
    <div class="metric"><span>Pico anomalia</span><strong>n/a</strong></div>
    <div class="metric"><span>Ratio anomalia</span><strong>0.00%</strong></div>
  `;
  updateLiveCaptureControls(flow);
  setLiveAlert("idle", "Esperando datos", message);
  clearCanvas(liveScoreCanvas, "Esperando resultados live");
  clearCanvas(liveWindowCanvas, "Esperando ventana inferida");
}

function formatInteger(value) {
  return Number.isFinite(Number(value)) ? Number(value).toLocaleString("es-ES") : "n/a";
}

function updateLiveCaptureControls(flow) {
  const activeSessionId =
    flow && Object.prototype.hasOwnProperty.call(flow, "active_session_id")
      ? flow.active_session_id
      : flow?.session_id || liveCurrentSessionId;
  const hasSession = Boolean(activeSessionId);
  const enabled = Boolean(flow?.capture_enabled);
  liveActiveSessionId = activeSessionId || "";
  liveCaptureEnabled = enabled;
  if (activeSessionId && liveControlSessionId && liveControlSessionId !== document.activeElement) {
    liveControlSessionId.value = activeSessionId;
  }
  if (liveControlStartBtn) {
    liveControlStartBtn.disabled = hasSession;
  }
  if (liveControlPauseBtn) {
    liveControlPauseBtn.disabled = !hasSession;
    liveControlPauseBtn.textContent = enabled ? "Pausar registro" : "Reanudar registro";
    liveControlPauseBtn.classList.toggle("danger-button", enabled);
    liveControlPauseBtn.classList.toggle("secondary", !enabled);
  }
  if (liveControlFinishBtn) {
    liveControlFinishBtn.disabled = !hasSession;
  }
}

function updateLiveRate(sessionId, rows) {
  const now = Date.now();
  let instantHz = null;
  if (liveRateSnapshot?.sessionId === sessionId && now > liveRateSnapshot.timeMs) {
    const deltaRows = rows - liveRateSnapshot.rows;
    const deltaSeconds = (now - liveRateSnapshot.timeMs) / 1000;
    if (deltaRows >= 0 && deltaSeconds > 0) {
      instantHz = deltaRows / deltaSeconds;
      liveRateHistory.push({ timeMs: now, rows });
    }
  } else {
    liveRateHistory = [{ timeMs: now, rows }];
  }
  liveRateSnapshot = { sessionId, rows, timeMs: now };

  const cutoff = now - 10000;
  liveRateHistory = liveRateHistory.filter((point) => point.timeMs >= cutoff);
  let averageHz = null;
  if (liveRateHistory.length >= 2) {
    const first = liveRateHistory[0];
    const last = liveRateHistory[liveRateHistory.length - 1];
    const deltaRows = last.rows - first.rows;
    const deltaSeconds = (last.timeMs - first.timeMs) / 1000;
    if (deltaRows >= 0 && deltaSeconds > 0) {
      averageHz = deltaRows / deltaSeconds;
    }
  }
  return { instantHz, averageHz };
}

function clearCanvas(canvas, message) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#f7f9fc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#61708a";
  ctx.font = "15px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function drawLiveScoreChart(canvas, results) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = { left: 58, right: 18, top: 20, bottom: 42 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!results.length) {
    clearCanvas(canvas, "Sin scores inferidos");
    return;
  }

  const scores = results.map((result) => Number(result.anomaly_score ?? 0));
  const maxScore = Math.max(1, ...scores) * 1.15;
  const threshold = Math.min(maxScore, 1);

  ctx.strokeStyle = "#e4eaf3";
  ctx.lineWidth = 1;
  ctx.font = "12px system-ui, sans-serif";
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    const value = maxScore - (maxScore * i) / 4;
    ctx.fillStyle = "#52627a";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(value.toFixed(2), padding.left - 8, y);
  }

  const thresholdY = padding.top + ((maxScore - threshold) / maxScore) * plotHeight;
  ctx.strokeStyle = "#ef4444";
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(padding.left, thresholdY);
  ctx.lineTo(width - padding.right, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#b91c1c";
  ctx.textAlign = "left";
  ctx.fillText("umbral anomalia", padding.left + 8, thresholdY - 8);

  const resultTimeMs = (result, index) => {
    const quality = result.quality || {};
    const value = Number(result.window_end_ms ?? quality.timestamp_end_ms);
    return Number.isFinite(value) ? value : index;
  };
  const firstTime = resultTimeMs(results[0], 0);
  const lastTime = resultTimeMs(results[results.length - 1], results.length - 1);
  const timeSpan = Math.max(1, lastTime - firstTime);
  const pointX = (index) => {
    const time = resultTimeMs(results[index], index);
    return padding.left + ((time - firstTime) / timeSpan) * plotWidth;
  };
  const pointY = (score) => padding.top + ((maxScore - score) / maxScore) * plotHeight;

  results.forEach((result, index) => {
    if (result.status !== "anomaly") return;
    const x = pointX(index);
    ctx.fillStyle = resultBreakdown(result).anomalyType === "slip" ? "rgba(249, 115, 22, 0.18)" : "rgba(239, 68, 68, 0.16)";
    ctx.fillRect(x - 4, padding.top, 8, plotHeight);
  });

  ctx.strokeStyle = "#1d4ed8";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  scores.forEach((score, index) => {
    const x = pointX(index);
    const y = pointY(score);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  results.forEach((result, index) => {
    const x = pointX(index);
    const y = pointY(scores[index]);
    ctx.fillStyle = resultPointColor(result);
    ctx.beginPath();
    ctx.arc(x, y, result.status === "anomaly" ? 4.2 : 3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.strokeStyle = "#b8c3d4";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  ctx.fillStyle = "#334155";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i += 1) {
    const x = padding.left + (plotWidth / 4) * i;
    const valueSeconds = ((firstTime + (timeSpan * i) / 4) / 1000).toFixed(1);
    ctx.fillText(`${valueSeconds}s`, x, height - padding.bottom + 10);
  }
  ctx.textBaseline = "bottom";
  ctx.fillText("tiempo de sesion (s)", padding.left + plotWidth / 2, height - 4);
  ctx.save();
  ctx.translate(16, padding.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("score de anomalia", 0, 0);
  ctx.restore();
}

async function renderLiveWindowAt(sessionId, index) {
  if (!liveWindows.length) {
    clearCanvas(liveWindowCanvas, "Sin ventana inferida");
    liveWindowLabel.textContent = "Sin ventana";
    liveRenderedWindowKey = "";
    if (liveWindowSlider) {
      liveWindowSlider.max = "0";
      liveWindowSlider.value = "0";
    }
    return;
  }

  liveWindowIndex = Math.max(0, Math.min(index, liveWindows.length - 1));
  const result = liveWindows[liveWindowIndex];
  if (!result) {
    clearCanvas(liveWindowCanvas, "Sin ventana inferida");
    liveWindowLabel.textContent = "Sin ventana";
    liveRenderedWindowKey = "";
    return;
  }

  const quality = result.quality || {};
  const breakdown = resultBreakdown(result);
  if (liveWindowSlider) {
    liveWindowSlider.max = String(Math.max(0, liveWindows.length - 1));
    liveWindowSlider.value = String(liveWindowIndex);
  }
  liveWindowLabel.textContent = `Ventana ${liveWindowIndex + 1}/${liveWindows.length} · seq ${quality.seq_start ?? "n/a"}..${quality.seq_end ?? "n/a"} · ${result.status} · ${resultDetailText(result)}`;
  const windowKey = `${sessionId}:${liveWindowIndex}:${quality.seq_start ?? ""}:${quality.seq_end ?? ""}:${result.status}:${result.anomaly_score ?? ""}:${windowYZoom}:${JSON.stringify(windowSeriesOptions)}`;
  if (windowKey === liveRenderedWindowKey) {
    return;
  }
  liveRenderedWindowKey = windowKey;
  const samples = await getWindowSamples(sessionId, result);
  const realData = result.metadata?.input_norm || normalizeSeries(samples);
  const reconstructionData = result.metadata?.reconstruction_norm || null;
  drawWindowChart(liveWindowCanvas, realData, reconstructionData, result.status, samples, quality);
  const isLatestWindow = liveWindowIndex === liveWindows.length - 1;
  liveWindowHint.textContent =
    result.status === "anomaly"
      ? `ALERTA: esta ventana${isLatestWindow ? " ultima" : ""} ha sido clasificada como anomala por ${breakdown.anomalyLabel}.`
      : `${isLatestWindow ? "Ultima ventana." : "Ventana historica."} Linea continua: vibracion real normalizada. ${
          reconstructionData
            ? "Linea discontinua: reconstruccion del VAE."
            : "Este resultado no incluye reconstruccion; en modo hibrido se prioriza payload ligero para tiempo real."
        }`;
}

function moveLiveWindow(delta) {
  if (!liveCurrentSessionId) return;
  if (liveAutoFollow) liveAutoFollow.checked = false;
  renderLiveWindowAt(liveCurrentSessionId, liveWindowIndex + delta).catch((error) => showToast(error.message));
}

function setLiveYZoom(nextZoom) {
  windowYZoom = Math.max(0.25, Math.min(nextZoom, 8));
  if (liveYZoomLabel) liveYZoomLabel.textContent = `${windowYZoom.toFixed(2)}x`;
  if (liveCurrentSessionId) {
    liveRenderedWindowKey = "";
    renderLiveWindowAt(liveCurrentSessionId, liveWindowIndex).catch((error) => showToast(error.message));
  }
}

async function renderLiveSnapshot(live, summary, results, flow = null) {
  const previousLiveSessionId = liveCurrentSessionId;
  liveCurrentSessionId = live.session_id;
  liveWindows = results || [];
  if (previousLiveSessionId && previousLiveSessionId !== liveCurrentSessionId) {
    liveWindowIndex = Math.max(0, liveWindows.length - 1);
    liveRenderedWindowKey = "";
    liveRateSnapshot = null;
    liveRateHistory = [];
  }
  const latest = latestResult(liveWindows);
  const latestStatus = latest?.status || "pending";
  const latestBreakdown = resultBreakdown(latest);
  const liveAge = ageMs(live.updated_at);
  const ageLabel = formatAge(liveAge);
  const flowData = flow || {};
  const acceptedRowsForRate = Number.isFinite(Number(flowData.accepted_rows_total))
    ? Number(flowData.accepted_rows_total)
    : live.rows;
  const { instantHz, averageHz } = updateLiveRate(live.session_id, acceptedRowsForRate);
  updateLiveCaptureControls({ session_id: live.session_id, capture_enabled: flowData.capture_enabled });
  liveSessionLabel.textContent = `${live.session_id} · ${live.rows} filas · actualizada ${formatDateTime(live.updated_at)} (${ageLabel})${liveSseConnected ? " · SSE" : ""}`;
  liveMetrics.innerHTML = `
    <div class="metric"><span>Sesion</span><strong>${live.session_id}</strong></div>
    <div class="metric"><span>Filas recibidas</span><strong>${live.rows}</strong></div>
    <div class="metric"><span>Tasa entrada · media 10s</span><strong>${averageHz === null ? "n/a" : `${averageHz.toFixed(1)} Hz`}</strong><small>inst ${instantHz === null ? "n/a" : `${instantHz.toFixed(1)} Hz`}</small></div>
    <div class="metric"><span>Ventanas inferidas</span><strong>${summary.window_count}</strong></div>
    <div class="metric"><span>Entrada ESP32</span><strong>${formatInteger(flowData.incoming_rows_total)}</strong><small>${formatInteger(flowData.incoming_batches_total)} batches · ${formatInteger(flowData.accepted_rows_total)} aceptadas</small></div>
    <div class="metric"><span>Flujo worker</span><strong>${formatInteger(flowData.samples_served_total)}</strong><small>muestras servidas</small></div>
    <div class="metric"><span>Resultados</span><strong>${formatInteger(flowData.results_total ?? summary.window_count)}</strong><small>ventanas devueltas</small></div>
    <div class="metric"><span>Descartadas</span><strong>${formatInteger(flowData.discarded_rows_total)}</strong><small>${flowData.capture_enabled === false ? "captura pausada" : "captura activa"}</small></div>
    <div class="metric"><span>Ultimo score</span><strong>${formatNumber(latest?.anomaly_score, 3)}</strong></div>
    <div class="metric"><span>Tipo ultima ventana</span><strong>${latestBreakdown.anomalyLabel}</strong></div>
    <div class="metric"><span>VAE</span><strong>${latestBreakdown.vaeStatus}</strong><small>score ${formatOptional(latestBreakdown.vaeScore)} · umbral ${formatOptional(latestBreakdown.vaeThreshold)}</small></div>
    <div class="metric"><span>CNN slip</span><strong>${latestBreakdown.slipStatus}</strong><small>prob ${formatOptional(latestBreakdown.slipProbability)} · umbral ${formatOptional(latestBreakdown.slipThreshold)}</small></div>
    <div class="metric"><span>Pico anomalia</span><strong>${formatNumber(summary.max_anomaly_score, 3)}</strong></div>
    <div class="metric"><span>Ratio anomalia</span><strong>${(summary.anomaly_ratio * 100).toFixed(2)}%</strong></div>
  `;

  if (flowData.capture_enabled === false) {
    setLiveAlert("warning", "Captura pausada", "El ESP32 puede seguir enviando, pero el VPS esta descartando muestras.");
  } else if (!liveWindows.length) {
    setLiveAlert("idle", "Recibiendo muestras", "Todavia no hay ventanas inferidas para esta sesion live.");
  } else if (latestStatus === "anomaly") {
    const title = latestBreakdown.anomalyType === "slip" ? "SLIP DETECTADO" : "ANOMALIA DETECTADA";
    setLiveAlert(
      "danger",
      title,
      `Ultima ventana: ${latestBreakdown.anomalyLabel}. VAE ${formatOptional(latestBreakdown.vaeScore)}, slip ${formatOptional(latestBreakdown.slipProbability)}.`,
    );
  } else if (summary.session_status === "anomaly") {
    setLiveAlert("warning", "Anomalias recientes", `La sesion acumula ${(summary.anomaly_ratio * 100).toFixed(2)}% de ventanas anomalas.`);
  } else {
    setLiveAlert("normal", "Circulacion normal", `Ultima ventana normal con score ${formatNumber(latest.anomaly_score, 3)}.`);
  }

  liveScoreLabel.textContent = `${liveWindows.length} ventanas recientes · estado ${sessionStatusLabel(summary.session_status)}`;
  drawLiveScoreChart(liveScoreCanvas, liveWindows);
  if (liveAutoFollow?.checked) {
    await renderLiveWindowAt(live.session_id, liveWindows.length - 1);
  } else {
    await renderLiveWindowAt(live.session_id, liveWindowIndex);
  }
}

function stopLiveEventStream() {
  if (liveEventSource) {
    liveEventSource.close();
  }
  liveEventSource = null;
  liveEventSessionId = "";
  liveSseConnected = false;
}

function startLiveEventStream(sessionId) {
  if (!window.EventSource || !sessionId) {
    return false;
  }
  if (liveEventSource && liveEventSessionId === sessionId) {
    return true;
  }
  stopLiveEventStream();
  liveEventSessionId = sessionId;
  liveEventSource = new EventSource(`/api/v1/live/${encodeURIComponent(sessionId)}/events?limit=120&interval=0.5`);
  liveEventSource.addEventListener("open", () => {
    liveSseConnected = true;
  });
  liveEventSource.addEventListener("snapshot", (event) => {
    liveSseConnected = true;
    const payload = JSON.parse(event.data);
    if (payload.flow?.active_session_id !== payload.session?.session_id) {
      stopLiveEventStream();
      updateSyncStatus(null);
      renderLiveEmpty("Registro live en espera. El VPS descartara datos hasta que inicies una sesion.", payload.flow);
      return;
    }
    updateSyncStatus(payload.session);
    renderLiveSnapshot(payload.session, payload.summary, payload.results || [], payload.flow).catch((error) => showToast(error.message));
  });
  liveEventSource.addEventListener("missing", () => {
    stopLiveEventStream();
    renderLiveEmpty();
  });
  liveEventSource.addEventListener("error", () => {
    liveSseConnected = false;
  });
  return true;
}

async function refreshLiveDashboard(manual = false) {
  const refreshSeq = ++liveRefreshSeq;
  if (manual && liveRefreshBtn) {
    liveRefreshBtn.disabled = true;
    liveRefreshBtn.textContent = "Actualizando...";
  }

  try {
    const [sessions, control] = await Promise.all([
      api("/api/v1/sessions"),
      api("/api/v1/live/control"),
    ]);
    if (refreshSeq !== liveRefreshSeq) return;
    if (control.active_session_id && liveControlSessionId && liveControlSessionId !== document.activeElement) {
      liveControlSessionId.value = control.active_session_id;
    }
    if (!control.active_session_id) {
      updateSyncStatus(null);
      stopLiveEventStream();
      renderLiveEmpty("Registro live en espera. El VPS descartara datos hasta que inicies una sesion.", control);
      return;
    }
    const live = sessions
      .filter((session) => session.mode === "live" && session.session_id === control.active_session_id)
      .sort((left, right) => (Date.parse(right.updated_at) || 0) - (Date.parse(left.updated_at) || 0))[0];

    updateSyncStatus(live);
    if (!live) {
      stopLiveEventStream();
      renderLiveEmpty(`Sesion ${control.active_session_id} iniciada. Esperando primeras muestras aceptadas.`, control);
      return;
    }

    const sseStarted = startLiveEventStream(live.session_id);
    if (!manual && sseStarted && liveSseConnected) {
      return;
    }

    const [summary, resultsData, flow] = await Promise.all([
      api(`/api/v1/sessions/${encodeURIComponent(live.session_id)}/summary`),
      api(`/api/v1/sessions/${encodeURIComponent(live.session_id)}/results?limit=120`),
      api(`/api/v1/live/${encodeURIComponent(live.session_id)}/control`),
    ]);
    if (refreshSeq !== liveRefreshSeq) return;
    await renderLiveSnapshot(live, summary, resultsData.results || [], flow);
  } finally {
    if (manual && refreshSeq === liveRefreshSeq && liveRefreshBtn) {
      liveRefreshBtn.disabled = false;
      liveRefreshBtn.textContent = "Actualizar live";
    }
  }
}

async function setLiveCapture(enabled) {
  const requestedSessionId = liveControlSessionId?.value.trim();
  if (enabled && (liveActiveSessionId || liveCaptureStartPending)) {
    showToast("Ya hay un registro live activo. Pausalo o finalizalo antes de iniciar otro.");
    return;
  }
  const sessionId = enabled
    ? requestedSessionId
    : liveCurrentSessionId || requestedSessionId || latestLiveSession()?.session_id;
  if (!sessionId) {
    showToast("Indica un nombre de sesion live.");
    return;
  }
  if (enabled) {
    liveCaptureStartPending = true;
    if (liveControlStartBtn) liveControlStartBtn.disabled = true;
  }
  try {
    const flow = await api("/api/v1/live/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, capture_enabled: enabled }),
    });
    updateLiveCaptureControls(flow);
    showToast(enabled ? "Registro live activado." : "Registro live pausado.");
    await refreshLiveDashboard(true);
  } catch (error) {
    if (enabled && !liveActiveSessionId && liveControlStartBtn) {
      liveControlStartBtn.disabled = false;
    }
    throw error;
  } finally {
    if (enabled) {
      liveCaptureStartPending = false;
    }
  }
}

async function finishLiveCapture() {
  const sessionId = liveCurrentSessionId || liveControlSessionId?.value.trim() || latestLiveSession()?.session_id;
  if (!sessionId) {
    showToast("No hay sesion live activa.");
    return;
  }
  const confirmed = window.confirm(`Finalizar el registro "${sessionId}"? El CSV quedara guardado en el historico.`);
  if (!confirmed) return;
  const flow = await api("/api/v1/live/control/finish", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, capture_enabled: false }),
  });
  updateLiveCaptureControls(flow);
  showToast(`Registro finalizado: ${sessionId}`);
  await loadSessions(false);
  await refreshLiveDashboard(true);
}

function populateWorkerSelect(workers) {
  const workersWithState = workers.map((worker) => ({
    ...worker,
    heartbeatAgeMs: ageMs(worker.last_seen_at),
    connectionState: stateFromAge(ageMs(worker.last_seen_at), workerGreenMs, workerYellowMs),
  }));
  const activeWorkers = workersWithState.filter((worker) => worker.connectionState === "ok");
  const degradedWorkers = workersWithState.filter((worker) => worker.connectionState === "pending");

  workerSelect.innerHTML = "";
  if (!workersWithState.length) {
    workerSelect.innerHTML = '<option value="">Sin workers disponibles</option>';
    setConnectionState(workerNode, "error");
    nodeText(workerNode, "Sin heartbeat");
    return;
  }

  for (const worker of workersWithState) {
    const option = document.createElement("option");
    option.value = worker.worker_id;
    option.disabled = worker.connectionState === "error";
    const label = worker.connectionState === "ok" ? "activo" : worker.connectionState === "pending" ? "degradado" : "desconectado";
    option.textContent = `${worker.worker_id} · ${label} · heartbeat ${formatAge(worker.heartbeatAgeMs)} · ${worker.capabilities.join(", ") || "sin capacidades"}`;
    workerSelect.appendChild(option);
  }

  if (activeWorkers.length) {
    workerSelect.value = activeWorkers[0].worker_id;
    setConnectionState(workerNode, "ok");
    nodeText(workerNode, `Conectado: ${activeWorkers[0].worker_id} · ${formatAge(activeWorkers[0].heartbeatAgeMs)}`);
  } else if (degradedWorkers.length) {
    workerSelect.value = degradedWorkers[0].worker_id;
    setConnectionState(workerNode, "pending");
    nodeText(workerNode, `Heartbeat antiguo: ${degradedWorkers[0].worker_id} · ${formatAge(degradedWorkers[0].heartbeatAgeMs)}`);
  } else {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Sin workers conectados";
    workerSelect.prepend(option);
    workerSelect.value = "";
    setConnectionState(workerNode, "error");
    nodeText(workerNode, "Servidor de inferencia desconectado");
  }
}

async function loadSessions(resetPage = true) {
  const loadSeq = ++sessionsLoadSeq;
  sessionsLoading = true;
  if (sessionsBtn) {
    sessionsBtn.disabled = true;
    sessionsBtn.textContent = "Cargando...";
  }

  try {
    const sessions = await api("/api/v1/sessions");
    const rows = await Promise.all(
      sessions.map(async (session) => {
        let summary = null;
        try {
          summary = await api(`/api/v1/sessions/${encodeURIComponent(session.session_id)}/summary`);
        } catch {
          summary = null;
        }
        return { session, summary };
      }),
    );

    if (loadSeq !== sessionsLoadSeq) {
      return;
    }

    sessionRows = rows;
    if (resetPage) {
      sessionsPage = 1;
    }
    renderSessions();
    renderLiveSessionsTable();
  } finally {
    if (loadSeq === sessionsLoadSeq) {
      sessionsLoading = false;
      if (sessionsBtn) {
        sessionsBtn.disabled = false;
        sessionsBtn.textContent = "Recargar";
      }
    }
  }
}

async function deleteSession(sessionId) {
  const confirmed = window.confirm(`Borrar la sesion "${sessionId}" y sus resultados? Esta accion no se puede deshacer.`);
  if (!confirmed) return;

  await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  showToast(`Sesion borrada: ${sessionId}`);
  if (liveRenderedWindowKey.startsWith(`${sessionId}:`)) {
    liveRenderedWindowKey = "";
  }
  await loadSessions(false);
  await refreshLiveDashboard();
}

function sortValue(row, key) {
  const { session, summary } = row;
  if (key === "session_id") return session.session_id.toLowerCase();
  if (key === "mode") return session.mode;
  if (key === "rows") return session.rows;
  if (key === "status") return sessionStatus(summary);
  if (key === "max_score") return summary?.max_anomaly_score ?? Number.NEGATIVE_INFINITY;
  if (key === "anomaly_ratio") return summary?.anomaly_ratio ?? Number.NEGATIVE_INFINITY;
  if (key === "updated_at") return Date.parse(session.updated_at) || 0;
  return "";
}

function compareRows(left, right) {
  const leftValue = sortValue(left, sessionSort.key);
  const rightValue = sortValue(right, sessionSort.key);
  const direction = sessionSort.direction === "asc" ? 1 : -1;

  if (typeof leftValue === "number" && typeof rightValue === "number") {
    return (leftValue - rightValue) * direction;
  }
  return String(leftValue).localeCompare(String(rightValue), "es", { numeric: true }) * direction;
}

function updateSortHeaders() {
  for (const header of sortHeaders) {
    const active = header.dataset.sort === sessionSort.key;
    header.classList.toggle("active", active);
    header.dataset.direction = active ? sessionSort.direction : "";
  }
}

function renderSessions() {
  sessionsBody.innerHTML = "";
  updateSortHeaders();
  const sortedRows = [...sessionRows].sort(compareRows);
  const totalPages = Math.max(1, Math.ceil(sortedRows.length / sessionsPageSize));
  sessionsPage = Math.min(Math.max(1, sessionsPage), totalPages);
  const start = (sessionsPage - 1) * sessionsPageSize;
  const visibleRows = sortedRows.slice(start, start + sessionsPageSize);

  sessionsPageInfo.textContent = `Pagina ${sessionsPage} de ${totalPages}`;
  prevSessionsPageBtn.disabled = sessionsPage <= 1;
  nextSessionsPageBtn.disabled = sessionsPage >= totalPages;

  for (const { session, summary } of visibleRows) {
    const status = sessionStatus(summary);
    const maxScore = summary?.max_anomaly_score;
    const anomalyRatio = summary ? `${(summary.anomaly_ratio * 100).toFixed(2)}%` : "n/a";
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${session.session_id}</td>
      <td>${session.mode}</td>
      <td>${session.rows}</td>
      <td><span class="badge ${status}">${sessionStatusLabel(status)}</span></td>
      <td>${formatNumber(maxScore, 3)}</td>
      <td>${anomalyRatio}</td>
      <td>${formatDateTime(session.updated_at)}</td>
      <td class="actions">
        <button class="secondary" data-action="summary" data-id="${session.session_id}">Resumen</button>
        <button class="secondary" data-action="samples" data-id="${session.session_id}">Muestras</button>
        <button class="secondary" data-action="results" data-id="${session.session_id}">Resultados</button>
        <button class="secondary" data-action="enqueue" data-id="${session.session_id}">Inferir</button>
        <a class="download" href="/api/v1/sessions/${session.session_id}/csv">CSV</a>
      </td>
    `;
    sessionsBody.appendChild(row);
  }
}

function setSessionSort(key) {
  if (sessionSort.key === key) {
    sessionSort.direction = sessionSort.direction === "asc" ? "desc" : "asc";
  } else {
    sessionSort = {
      key,
      direction: ["rows", "max_score", "anomaly_ratio", "updated_at"].includes(key) ? "desc" : "asc",
    };
  }
  sessionsPage = 1;
  renderSessions();
}

function changeSessionsPage(delta) {
  sessionsPage += delta;
  renderSessions();
}

async function uploadCsv(event) {
  event.preventDefault();
  const sessionId = document.querySelector("#uploadSessionId").value.trim();
  const fileInput = document.querySelector("#csvFile");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/csv`, {
    method: "POST",
    body: formData,
  });
  showToast(`CSV subido: ${data.rows_received} filas`);
  await loadSessions(true);
  await refreshJobs();
}

async function sendSyntheticBatch(event) {
  event.preventDefault();
  const sessionId = document.querySelector("#batchSessionId").value.trim();
  const start = Date.now() % 1000000;
  const samples = Array.from({ length: 50 }, (_, index) => ({
    timestamp_ms: start + index * 10,
    acc_x_g: 0.05 + Math.sin(index / 6) * 0.01,
    acc_y_g: 0.02,
    acc_z_g: 0.98 + Math.cos(index / 8) * 0.01,
  }));
  const data = await api("/api/v1/samples/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      device_id: "browser_test",
      session_id: sessionId,
      seq_start: Math.floor(Date.now() / 1000),
      sample_rate_hz: 100,
      samples,
    }),
  });
  showToast(`Lote enviado: ${data.rows_received} filas`);
  await loadSessions(true);
  await refreshLiveDashboard();
}

async function showSamples(sessionId) {
  const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/samples?after_seq=-1&limit=50`);
  const samples = data.samples || [];
  detailEl.innerHTML = `
    <div class="detail-section">
      <h3>Muestras de ${escapeHtml(sessionId)}</h3>
      <p class="hint">Primeras ${samples.length} muestras disponibles. Son datos crudos del acelerometro en g.</p>
      ${renderSamplesTable(samples)}
    </div>
  `;
}

async function showResults(sessionId) {
  const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/results?limit=500`);
  const results = data.results || [];
  detailEl.innerHTML = `
    <div class="detail-section">
      <h3>Resultados de ${escapeHtml(sessionId)}</h3>
      <p class="hint">Ultimas ${results.length} ventanas inferidas. Usa Resumen para navegar graficamente por las ventanas.</p>
      ${renderResultsTable(results)}
    </div>
  `;
}

function renderSamplesTable(samples) {
  if (!samples.length) return '<div class="status">No hay muestras disponibles.</div>';
  return `
    <div class="scroll-box">
      <table>
        <thead>
          <tr>
            <th>seq</th>
            <th>timestamp_ms</th>
            <th>X (g)</th>
            <th>Y (g)</th>
            <th>Z (g)</th>
          </tr>
        </thead>
        <tbody>
          ${samples
            .map(
              (sample) => `
                <tr>
                  <td>${escapeHtml(sample.seq)}</td>
                  <td>${escapeHtml(sample.timestamp_ms)}</td>
                  <td>${formatOptional(sample.acc_x_g, 5)}</td>
                  <td>${formatOptional(sample.acc_y_g, 5)}</td>
                  <td>${formatOptional(sample.acc_z_g, 5)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderResultsTable(results) {
  if (!results.length) return '<div class="status">No hay resultados disponibles.</div>';
  return `
    <div class="scroll-box tall">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>seq</th>
            <th>estado</th>
            <th>tipo</th>
            <th>score</th>
            <th>VAE</th>
            <th>CNN slip</th>
          </tr>
        </thead>
        <tbody>
          ${results
            .map((result, index) => {
              const quality = result.quality || {};
              const breakdown = resultBreakdown(result);
              return `
                <tr>
                  <td>${index + 1}</td>
                  <td>${escapeHtml(quality.seq_start ?? "n/a")}..${escapeHtml(quality.seq_end ?? "n/a")}</td>
                  <td>${statusBadge(result.status)}</td>
                  <td>${escapeHtml(breakdown.anomalyLabel)}</td>
                  <td>${formatOptional(result.anomaly_score)}</td>
                  <td>${escapeHtml(breakdown.vaeStatus)} · ${formatOptional(breakdown.vaeScore)}</td>
                  <td>${escapeHtml(breakdown.slipStatus)} · prob ${formatOptional(breakdown.slipProbability)}</td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function getWindowSamples(sessionId, windowResult) {
  const quality = windowResult.quality || {};
  const seqStart = Number.isFinite(quality.seq_start) ? quality.seq_start : -1;
  const expected = quality.samples_expected || quality.samples_received || 100;
  const data = await api(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/samples?after_seq=${seqStart - 1}&limit=${expected}`,
  );
  return data.samples;
}

function normalizeSeries(samples) {
  const columns = ["acc_x_g", "acc_y_g", "acc_z_g"];
  const values = samples.map((sample) => columns.map((column) => sample[column]));
  if (!values.length) return [];
  const minVals = columns.map((_, index) => Math.min(...values.map((row) => row[index])));
  const maxVals = columns.map((_, index) => Math.max(...values.map((row) => row[index])));
  return values.map((row) =>
    row.map((value, index) => {
      const denominator = maxVals[index] - minVals[index];
      return denominator === 0 ? 0.5 : (value - minVals[index]) / denominator;
    }),
  );
}

function drawWindowChart(canvas, realData, reconstructionData, status, samples, quality) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = { left: 58, right: 16, top: 18, bottom: 46 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = status === "anomaly" ? "#fdecec" : "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const channelNames = ["X", "Y", "Z"];
  ctx.strokeStyle = "#e4eaf3";
  ctx.lineWidth = 1;
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillStyle = "#52627a";
  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.strokeStyle = "#b8c3d4";
  ctx.stroke();

  const colors = ["#1d4ed8", "#16a34a", "#9333ea"];
  const reconColors = ["#93c5fd", "#86efac", "#d8b4fe"];
  const visibleSeries = [];
  const addSeries = (data, channel, key, label, color, dashed = false) => {
    if (!data?.length || !windowSeriesOptions[key]) return;
    visibleSeries.push({ data, channel, label, color, dashed });
  };

  for (let channel = 0; channel < 3; channel += 1) {
    addSeries(realData, channel, channelNames[channel].toLowerCase(), channelNames[channel], colors[channel]);
    addSeries(
      reconstructionData,
      channel,
      `r${channelNames[channel].toLowerCase()}`,
      `${channelNames[channel]} recon`,
      reconColors[channel],
      true,
    );
  }

  const values = visibleSeries.flatMap((series) =>
    series.data
      .map((row) => Number(row[series.channel]))
      .filter((value) => Number.isFinite(value)),
  );
  let yMin = values.length ? Math.min(...values) : -1;
  let yMax = values.length ? Math.max(...values) : 1;
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yCenter = (yMin + yMax) / 2;
  const baseHalfRange = ((yMax - yMin) / 2) * 1.12;
  const halfRange = baseHalfRange / windowYZoom;
  yMin = yCenter - halfRange;
  yMax = yCenter + halfRange;

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight / 4) * i;
    const value = yMax - ((yMax - yMin) * i) / 4;
    ctx.fillStyle = "#52627a";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(value.toFixed(2), padding.left - 8, y);
  }

  const durationMs =
    Number.isFinite(quality?.timestamp_end_ms) && Number.isFinite(quality?.timestamp_start_ms)
      ? Math.max(0, quality.timestamp_end_ms - quality.timestamp_start_ms)
      : Math.max(0, (realData?.length || 1) - 1) * 10;
  for (let i = 0; i <= 4; i += 1) {
    const x = padding.left + (plotWidth / 4) * i;
    const value = (durationMs * i) / 4;
    ctx.fillStyle = "#52627a";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`${Math.round(value)} ms`, x, height - padding.bottom + 10);
  }

  ctx.save();
  ctx.translate(16, padding.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#334155";
  ctx.fillText("aceleracion normalizada (z-score)", 0, 0);
  ctx.restore();
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillStyle = "#334155";
  ctx.fillText("tiempo dentro de la ventana (ms)", padding.left + plotWidth / 2, height - 4);

  const drawSeries = (series) => {
    const { data, channel, color, dashed } = series;
    ctx.strokeStyle = color;
    ctx.lineWidth = dashed ? 1.5 : 2;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.beginPath();
    data.forEach((row, index) => {
      const value = Number(row[channel]);
      if (!Number.isFinite(value)) return;
      const x = padding.left + (data.length === 1 ? 0 : (index / (data.length - 1)) * plotWidth);
      const y = padding.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  };

  for (const series of visibleSeries) {
    drawSeries(series);
  }

  let legendX = padding.left;
  const legendY = 10;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  for (const series of visibleSeries) {
    ctx.strokeStyle = series.color;
    ctx.lineWidth = series.dashed ? 1.5 : 2;
    ctx.setLineDash(series.dashed ? [5, 4] : []);
    ctx.beginPath();
    ctx.moveTo(legendX, legendY);
    ctx.lineTo(legendX + 22, legendY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#334155";
    ctx.fillText(series.label, legendX + 28, legendY);
    legendX += ctx.measureText(series.label).width + 58;
  }
}

async function renderWindowViewer(sessionId, index, targetEl = detailEl) {
  detailWindowIndex = Math.max(0, Math.min(index, detailWindows.length - 1));
  const result = detailWindows[detailWindowIndex];
  const slider = targetEl.querySelector("#windowSlider");
  const label = targetEl.querySelector("#windowLabel");
  const modelDetail = targetEl.querySelector("#windowModelDetail");
  const canvas = targetEl.querySelector("#windowCanvas");
  const warning = targetEl.querySelector("#windowWarning");
  if (!result || !slider || !label || !canvas || !warning) return;

  slider.value = String(detailWindowIndex);
  const quality = result.quality || {};
  label.textContent = `Ventana ${detailWindowIndex + 1}/${detailWindows.length} · seq ${quality.seq_start ?? "n/a"}..${quality.seq_end ?? "n/a"} · estado ${result.status} · score ${formatNumber(result.anomaly_score, 3)}`;
  if (modelDetail) {
    modelDetail.innerHTML = resultModelChips(result);
  }

  const samples = await getWindowSamples(sessionId, result);
  const realData = result.metadata?.input_norm || normalizeSeries(samples);
  const reconstructionData = result.metadata?.reconstruction_norm || null;
  warning.textContent = reconstructionData
    ? "Linea continua: vibracion real normalizada. Linea discontinua: reconstruccion del modelo."
    : "Esta inferencia no guarda reconstruccion. En el detector hibrido se conserva el score VAE y la CNN slip, pero no las series reconstruidas para reducir payload.";
  drawWindowChart(canvas, realData, reconstructionData, result.status, samples, quality);
}

function moveWindow(sessionId, delta, targetEl = detailEl) {
  renderWindowViewer(sessionId, detailWindowIndex + delta, targetEl).catch((error) => showToast(error.message));
}

function setWindowYZoom(sessionId, nextZoom, targetEl = detailEl) {
  windowYZoom = Math.max(0.25, Math.min(nextZoom, 8));
  const zoomLabel = targetEl.querySelector("#windowYZoomLabel");
  if (zoomLabel) zoomLabel.textContent = `${windowYZoom.toFixed(2)}x`;
  renderWindowViewer(sessionId, detailWindowIndex, targetEl).catch((error) => showToast(error.message));
}

async function renderSummary(summary, targetEl = detailEl) {
  const counts = summary.status_counts;
  const maxWindow = summary.max_score_result?.quality
    ? `${summary.max_score_result.quality.seq_start}..${summary.max_score_result.quality.seq_end}`
    : "n/a";
  const maxBreakdown = resultBreakdown(summary.max_score_result);
  const sessionId = summary.session_id;
  targetEl.innerHTML = `
    <div class="summary-grid">
      <div class="metric"><span>Ventanas</span><strong>${summary.window_count}</strong></div>
      <div class="metric"><span>Estado sesion</span><strong>${sessionStatusLabel(sessionStatus(summary))}</strong></div>
      <div class="metric"><span>Pico anomalia</span><strong>${formatNumber(summary.max_anomaly_score, 3)}</strong></div>
      <div class="metric"><span>Tipo pico</span><strong>${maxBreakdown.anomalyLabel}</strong></div>
      <div class="metric"><span>Pico VAE</span><strong>${formatOptional(maxBreakdown.vaeScore)}</strong></div>
      <div class="metric"><span>Pico CNN slip</span><strong>${formatOptional(maxBreakdown.slipProbability)}</strong></div>
      <div class="metric"><span>Error máximo</span><strong>${formatNumber(summary.max_reconstruction_error, 6)}</strong></div>
      <div class="metric"><span>Ratio anomalía</span><strong>${(summary.anomaly_ratio * 100).toFixed(2)}%</strong></div>
      <div class="metric"><span>Ventana pico</span><strong>${maxWindow}</strong></div>
    </div>
    <div class="window-viewer">
      <div class="viewer-header">
        <strong>Visor de ventanas</strong>
        <span id="windowLabel">Sin ventanas</span>
      </div>
      <div id="windowModelDetail"></div>
      <div class="viewer-controls">
        <div class="nav-buttons" aria-label="Navegacion de ventanas">
          <button class="secondary compact" data-window-step="-10" title="Retroceder 10 ventanas">«10</button>
          <button class="secondary compact" data-window-step="-1" title="Retroceder 1 ventana">‹1</button>
          <button class="secondary compact" data-window-step="1" title="Avanzar 1 ventana">1›</button>
          <button class="secondary compact" data-window-step="10" title="Avanzar 10 ventanas">10»</button>
        </div>
        <div class="scale-controls" aria-label="Escala vertical">
          <span>Escala Y</span>
          <button class="secondary compact" id="windowZoomOut" title="Ver mas rango vertical">−</button>
          <span id="windowYZoomLabel">${windowYZoom.toFixed(2)}x</span>
          <button class="secondary compact" id="windowZoomIn" title="Aumentar detalle vertical">+</button>
        </div>
      </div>
      <input id="windowSlider" type="range" min="0" max="0" value="0" />
      <div class="series-controls" aria-label="Series visibles">
        <label><input type="checkbox" data-series="x" ${windowSeriesOptions.x ? "checked" : ""}> X real</label>
        <label><input type="checkbox" data-series="y" ${windowSeriesOptions.y ? "checked" : ""}> Y real</label>
        <label><input type="checkbox" data-series="z" ${windowSeriesOptions.z ? "checked" : ""}> Z real</label>
        <label><input type="checkbox" data-series="rx" ${windowSeriesOptions.rx ? "checked" : ""}> X reconstruido</label>
        <label><input type="checkbox" data-series="ry" ${windowSeriesOptions.ry ? "checked" : ""}> Y reconstruido</label>
        <label><input type="checkbox" data-series="rz" ${windowSeriesOptions.rz ? "checked" : ""}> Z reconstruido</label>
      </div>
      <canvas id="windowCanvas" width="980" height="320"></canvas>
      <p id="windowWarning" class="hint"></p>
    </div>
    <pre>${JSON.stringify({ status_counts: counts, max_score_result: summary.max_score_result }, null, 2)}</pre>
  `;
  const resultsData = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/results?limit=5000`);
  detailWindows = resultsData.results || [];
  const slider = targetEl.querySelector("#windowSlider");
  if (!detailWindows.length) {
    targetEl.querySelector("#windowWarning").textContent = "Todavia no hay ventanas inferidas para esta sesion.";
    return;
  }
  slider.max = String(detailWindows.length - 1);
  slider.addEventListener("input", () => {
    renderWindowViewer(sessionId, Number(slider.value), targetEl).catch((error) => showToast(error.message));
  });
  targetEl.querySelectorAll("[data-window-step]").forEach((button) => {
    button.addEventListener("click", () => moveWindow(sessionId, Number(button.dataset.windowStep), targetEl));
  });
  targetEl.querySelector("#windowZoomOut")?.addEventListener("click", () => setWindowYZoom(sessionId, windowYZoom / 1.25, targetEl));
  targetEl.querySelector("#windowZoomIn")?.addEventListener("click", () => setWindowYZoom(sessionId, windowYZoom * 1.25, targetEl));
  targetEl.querySelectorAll("[data-series]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      windowSeriesOptions[checkbox.dataset.series] = checkbox.checked;
      renderWindowViewer(sessionId, detailWindowIndex, targetEl).catch((error) => showToast(error.message));
    });
  });
  await renderWindowViewer(sessionId, 0, targetEl);
}

async function showSummary(sessionId) {
  const summary = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/summary`);
  await renderSummary(summary, detailEl);
}

async function showLiveSummary(sessionId) {
  const summary = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/summary`);
  await renderSummary(summary, liveDetailEl);
  liveDetailEl?.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function enqueueJob(sessionId) {
  const confirmed = window.confirm(
    `Procesar de nuevo la sesion "${sessionId}" con el worker activo? Esto recalculara los resultados de inferencia para todo el CSV.`,
  );
  if (!confirmed) return;
  const data = await api(`/api/v1/sessions/${encodeURIComponent(sessionId)}/jobs`, {
    method: "POST",
  });
  showToast(`Job creado: ${data.job_id}`);
  await refreshJobs();
}

function statusBadge(status) {
  return `<span class="badge ${status || "pending"}">${sessionStatusLabel(status || "pending")}</span>`;
}

function renderWorkersVisual(workers) {
  if (!workers.length) {
    return '<div class="status">No hay servidores de inferencia registrados.</div>';
  }
  return `
    <div class="status-list">
      ${workers
        .map((worker) => {
          const heartbeatAge = ageMs(worker.last_seen_at);
          const state = stateFromAge(heartbeatAge, workerGreenMs, workerYellowMs);
          const label = state === "ok" ? "conectado" : state === "pending" ? "heartbeat antiguo" : "desconectado";
          return `
            <article class="worker-card">
              <header>
                <strong>${worker.worker_id}</strong>
                <span class="badge ${state === "ok" ? "normal" : state === "pending" ? "review" : "anomaly"}">${label}</span>
              </header>
              <div class="meta-row">
                <span>Ultimo heartbeat: ${formatAge(heartbeatAge)}</span>
                <span>Trabajo actual: ${worker.current_job_id || "ninguno"}</span>
              </div>
              <div class="chip-list">
                ${(worker.capabilities || []).map((capability) => `<span class="chip">${capability}</span>`).join("") || '<span class="chip">sin capacidades</span>'}
              </div>
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderJobsVisual(jobs) {
  if (!jobs.length) {
    return '<div class="status">No hay jobs registrados.</div>';
  }
  const recentJobs = [...jobs]
    .sort((left, right) => (Date.parse(right.updated_at) || 0) - (Date.parse(left.updated_at) || 0))
    .slice(0, 8);
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>job</th>
            <th>sesion</th>
            <th>estado</th>
            <th>worker</th>
            <th>actualizado</th>
          </tr>
        </thead>
        <tbody>
          ${recentJobs
            .map(
              (job) => `
                <tr>
                  <td>${job.job_id.slice(0, 8)}</td>
                  <td>${job.session_id}</td>
                  <td>${statusBadge(job.status)}</td>
                  <td>${job.claimed_by || "n/a"}</td>
                  <td>${formatDateTime(job.updated_at)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderArchitecturePanel(workers, jobs) {
  if (!architectureDetailEl) return;
  const activeWorkers = workers.filter((worker) => stateFromAge(ageMs(worker.last_seen_at), workerGreenMs, workerYellowMs) === "ok");
  const selectedWorker = activeWorkers[0] || null;
  const pendingJobs = jobs.filter((job) => job.status === "pending").length;
  const runningJobs = jobs.filter((job) => job.status === "running").length;
  const activeCapabilities = selectedWorker?.capabilities?.length ? selectedWorker.capabilities.join(", ") : "sin worker activo";
  architectureDetailEl.innerHTML = `
    <div class="architecture-grid">
      <article class="architecture-card">
        <strong>1. Captura</strong>
        <span>ESP32 + ADXL345</span>
        <p>Envia lotes HTTP de acelerometria triaxial al VPS cuando hay una sesion live activa.</p>
      </article>
      <article class="architecture-card">
        <strong>2. Orquestacion</strong>
        <span>VPS FastAPI</span>
        <p>Recibe CSV/live, conserva sesiones, publica jobs y muestra resultados en el dashboard.</p>
      </article>
      <article class="architecture-card">
        <strong>3. Inferencia</strong>
        <span>${escapeHtml(activeCapabilities)}</span>
        <p>Worker externo con detector hibrido: VAE para anomalias generales y CNN MIL para slip.</p>
      </article>
    </div>
    <div class="summary-grid">
      <div class="metric"><span>Workers activos</span><strong>${activeWorkers.length}</strong></div>
      <div class="metric"><span>Worker seleccionado</span><strong>${escapeHtml(selectedWorker?.worker_id || "n/a")}</strong></div>
      <div class="metric"><span>Jobs pendientes</span><strong>${pendingJobs}</strong></div>
      <div class="metric"><span>Jobs en ejecucion</span><strong>${runningJobs}</strong></div>
    </div>
  `;
}

async function refreshJobs() {
  const [workers, jobs] = await Promise.all([
    api("/api/v1/workers"),
    api("/api/v1/inference/jobs"),
  ]);
  latestWorkers = workers;
  latestJobs = jobs;
  populateWorkerSelect(workers);
  renderArchitecturePanel(workers, jobs);
  jobsDetailEl.innerHTML = `
    <div class="visual-block">
      <div class="summary-grid">
        <div class="metric"><span>Workers registrados</span><strong>${workers.length}</strong></div>
        <div class="metric"><span>Workers activos</span><strong>${workers.filter((worker) => stateFromAge(ageMs(worker.last_seen_at), workerGreenMs, workerYellowMs) === "ok").length}</strong></div>
        <div class="metric"><span>Jobs totales</span><strong>${jobs.length}</strong></div>
        <div class="metric"><span>Pendientes</span><strong>${jobs.filter((job) => job.status === "pending").length}</strong></div>
        <div class="metric"><span>En ejecucion</span><strong>${jobs.filter((job) => job.status === "running").length}</strong></div>
      </div>
      <h3>Servidores de inferencia</h3>
      ${renderWorkersVisual(workers)}
      <h3>Jobs recientes</h3>
      ${renderJobsVisual(jobs)}
    </div>
  `;
}

async function refreshStatus() {
  await Promise.all([
    refreshHealth(),
    refreshJobs(),
  ]);
}

async function refreshSessionsStatus() {
  await loadSessions(false);
}

for (const button of tabButtons) {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
}
document.querySelector("#refreshBtn").addEventListener("click", () => refreshStatus().catch((error) => showToast(error.message)));
sessionsBtn.addEventListener("click", () => {
  if (sessionsLoading) return;
  loadSessions(true).catch((error) => showToast(error.message));
});
document.querySelector("#jobsBtn").addEventListener("click", () => refreshJobs().catch((error) => showToast(error.message)));
document.querySelector("#uploadForm").addEventListener("submit", (event) => uploadCsv(event).catch((error) => showToast(error.message)));
document.querySelector("#batchForm").addEventListener("submit", (event) => sendSyntheticBatch(event).catch((error) => showToast(error.message)));
liveControlForm?.addEventListener("submit", (event) => {
  event.preventDefault();
  setLiveCapture(true).catch((error) => showToast(error.message));
});
liveControlPauseBtn?.addEventListener("click", () => setLiveCapture(!liveCaptureEnabled).catch((error) => showToast(error.message)));
liveControlFinishBtn?.addEventListener("click", () => finishLiveCapture().catch((error) => showToast(error.message)));
liveRefreshBtn.addEventListener("click", () => refreshLiveDashboard(true).catch((error) => showToast(error.message)));
liveSessionsRefreshBtn.addEventListener("click", () => loadSessions(false).catch((error) => showToast(error.message)));
liveWindowSlider?.addEventListener("input", () => {
  if (liveAutoFollow) liveAutoFollow.checked = false;
  renderLiveWindowAt(liveCurrentSessionId, Number(liveWindowSlider.value)).catch((error) => showToast(error.message));
});
liveAutoFollow?.addEventListener("change", () => {
  if (liveAutoFollow.checked && liveCurrentSessionId) {
    renderLiveWindowAt(liveCurrentSessionId, liveWindows.length - 1).catch((error) => showToast(error.message));
  }
});
document.querySelectorAll("[data-live-window-step]").forEach((button) => {
  button.addEventListener("click", () => moveLiveWindow(Number(button.dataset.liveWindowStep)));
});
document.querySelector("#liveZoomOut")?.addEventListener("click", () => setLiveYZoom(windowYZoom / 1.25));
document.querySelector("#liveZoomIn")?.addEventListener("click", () => setLiveYZoom(windowYZoom * 1.25));
document.querySelectorAll("[data-live-series]").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    windowSeriesOptions[checkbox.dataset.liveSeries] = checkbox.checked;
    liveRenderedWindowKey = "";
    if (liveCurrentSessionId) {
      renderLiveWindowAt(liveCurrentSessionId, liveWindowIndex).catch((error) => showToast(error.message));
    }
  });
});
prevSessionsPageBtn.addEventListener("click", () => changeSessionsPage(-1));
nextSessionsPageBtn.addEventListener("click", () => changeSessionsPage(1));
for (const header of sortHeaders) {
  header.addEventListener("click", () => setSessionSort(header.dataset.sort));
}
sessionsBody.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const sessionId = button.dataset.id;
  if (button.dataset.action === "summary") {
    showSummary(sessionId).catch((error) => showToast(error.message));
  } else if (button.dataset.action === "samples") {
    showSamples(sessionId).catch((error) => showToast(error.message));
  } else if (button.dataset.action === "results") {
    showResults(sessionId).catch((error) => showToast(error.message));
  } else {
    enqueueJob(sessionId).catch((error) => showToast(error.message));
  }
});
liveSessionsBody.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-live-action]");
  if (!button) return;
  const sessionId = button.dataset.id;
  if (button.dataset.liveAction === "delete") {
    deleteSession(sessionId).catch((error) => showToast(error.message));
  } else if (button.dataset.liveAction === "view") {
    showLiveSummary(sessionId).catch((error) => showToast(error.message));
  }
});

refreshStatus().catch((error) => showToast(error.message));
refreshSessionsStatus().catch((error) => showToast(error.message));
refreshLiveDashboard().catch((error) => showToast(error.message));
setInterval(() => {
  refreshStatus().catch((error) => showToast(error.message));
}, autoRefreshMs);
setInterval(() => {
  refreshLiveDashboard().catch((error) => showToast(error.message));
}, liveRefreshMs);
