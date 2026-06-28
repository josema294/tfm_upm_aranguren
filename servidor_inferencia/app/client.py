from __future__ import annotations

import time

import httpx


class VpsClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 10.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        headers = {"X-API-Key": api_key} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=headers)
        self.retries = max(1, retries)

    def close(self) -> None:
        self.client.close()

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_exc: httpx.HTTPError | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt >= self.retries:
                    break
                time.sleep(0.25 * attempt)
        assert last_exc is not None
        raise last_exc

    def get_samples(self, session_id: str, after_seq: int, limit: int) -> list[dict]:
        response = self._request(
            "GET",
            f"{self.base_url}/api/v1/sessions/{session_id}/samples",
            params={"after_seq": after_seq, "limit": limit},
        )
        return response.json()["samples"]

    def list_sessions(self) -> list[dict]:
        response = self._request("GET", f"{self.base_url}/api/v1/sessions")
        return response.json()

    def post_result(self, payload: dict) -> None:
        self._request("POST", f"{self.base_url}/api/v1/inference/results", json=payload)

    def heartbeat(self, worker_id: str, capabilities: list[str], current_job_id: str | None = None) -> dict:
        response = self._request(
            "POST",
            f"{self.base_url}/api/v1/workers/heartbeat",
            json={
                "worker_id": worker_id,
                "capabilities": capabilities,
                "current_job_id": current_job_id,
            },
        )
        return response.json()

    def claim_next_job(self, worker_id: str) -> dict | None:
        response = self._request(
            "POST",
            f"{self.base_url}/api/v1/inference/jobs/next",
            json={"worker_id": worker_id},
        )
        return response.json()

    def get_job(self, job_id: str) -> dict | None:
        response = self.client.get(f"{self.base_url}/api/v1/inference/jobs/{job_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def complete_job(self, job_id: str, worker_id: str) -> dict:
        response = self._request(
            "POST",
            f"{self.base_url}/api/v1/inference/jobs/{job_id}/complete",
            json={"worker_id": worker_id},
        )
        return response.json()

    def fail_job(self, job_id: str, worker_id: str, error: str) -> dict:
        response = self._request(
            "POST",
            f"{self.base_url}/api/v1/inference/jobs/{job_id}/fail",
            json={"worker_id": worker_id, "error": error},
        )
        return response.json()
