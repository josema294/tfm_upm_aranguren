from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import secrets
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, Response, status


SESSION_COOKIE_NAME = "tfm_session"
PBKDF2_ITERATIONS = 210_000


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


@dataclass(frozen=True)
class AuthSettings:
    username: str
    password_hash: str
    session_secret: str
    api_keys: tuple[str, ...]
    session_ttl_seconds: int
    secure_cookie: bool
    max_upload_bytes: int
    max_csv_rows: int


def load_auth_settings() -> AuthSettings:
    load_dotenv(os.getenv("TFM_ENV_FILE", ".env"))
    username = os.getenv("TFM_WEB_USERNAME", "admin")
    password_hash = os.getenv("TFM_WEB_PASSWORD_HASH", "")
    session_secret = os.getenv("TFM_SESSION_SECRET", "")
    api_keys = tuple(key.strip() for key in os.getenv("TFM_API_KEYS", "").split(",") if key.strip())

    placeholder_values = ("replace_", "change_me", "changeme")
    if not password_hash or password_hash.lower().startswith(placeholder_values):
        raise RuntimeError("TFM_WEB_PASSWORD_HASH is required")
    if len(session_secret) < 32 or session_secret.lower().startswith(placeholder_values):
        raise RuntimeError("TFM_SESSION_SECRET must be at least 32 characters")
    if not api_keys or any(key.lower().startswith(placeholder_values) for key in api_keys):
        raise RuntimeError("TFM_API_KEYS must include at least one API key")

    return AuthSettings(
        username=username,
        password_hash=password_hash,
        session_secret=session_secret,
        api_keys=api_keys,
        session_ttl_seconds=int(os.getenv("TFM_SESSION_TTL_SECONDS", "28800")),
        secure_cookie=os.getenv("TFM_SECURE_COOKIE", "false").lower() == "true",
        max_upload_bytes=int(os.getenv("TFM_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        max_csv_rows=int(os.getenv("TFM_MAX_CSV_ROWS", "250000")),
    )


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    encoded_salt = base64.urlsafe_b64encode(salt).decode("ascii")
    encoded_digest = base64.urlsafe_b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${encoded_salt}${encoded_digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_raw, encoded_salt, encoded_digest = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(encoded_digest.encode("ascii"))
    except (ValueError, TypeError):
        return False

    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


def _sign(value: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


def create_session_token(username: str, settings: AuthSettings) -> str:
    expires_at = int(time.time()) + settings.session_ttl_seconds
    payload = f"{username}|{expires_at}"
    signature = _sign(payload, settings.session_secret)
    token = f"{payload}|{signature}"
    return base64.urlsafe_b64encode(token.encode("utf-8")).decode("ascii")


def verify_session_token(token: str | None, settings: AuthSettings) -> bool:
    if not token:
        return False
    try:
        decoded = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        username, expires_raw, signature = decoded.rsplit("|", 2)
        expires_at = int(expires_raw)
    except (ValueError, TypeError):
        return False

    if username != settings.username or expires_at < int(time.time()):
        return False
    expected = _sign(f"{username}|{expires_at}", settings.session_secret)
    return hmac.compare_digest(signature, expected)


def request_has_valid_session(request: Request, settings: AuthSettings) -> bool:
    return verify_session_token(request.cookies.get(SESSION_COOKIE_NAME), settings)


def request_has_valid_api_key(request: Request, settings: AuthSettings) -> bool:
    provided = request.headers.get("x-api-key", "")
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        provided = auth_header[7:].strip()
    return bool(provided) and any(hmac.compare_digest(provided, key) for key in settings.api_keys)


def require_machine_or_web_auth(request: Request, settings: AuthSettings) -> None:
    if request_has_valid_session(request, settings) or request_has_valid_api_key(request, settings):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


def set_session_cookie(response: Response, token: str, settings: AuthSettings) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        httponly=True,
        secure=settings.secure_cookie,
        samesite="lax",
        max_age=settings.session_ttl_seconds,
        path="/",
    )


def clear_session_cookie(response: Response, settings: AuthSettings) -> None:
    response.delete_cookie(SESSION_COOKIE_NAME, httponly=True, secure=settings.secure_cookie, samesite="lax", path="/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Auth helpers for the TFM VPS server.")
    parser.add_argument("--hash-password", help="Generate a PBKDF2 hash for this password.")
    parser.add_argument("--random-secret", action="store_true", help="Generate a random session secret/API key.")
    args = parser.parse_args()

    if args.hash_password:
        print(hash_password(args.hash_password))
    if args.random_secret:
        print(secrets.token_urlsafe(48))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
