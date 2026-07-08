import hashlib
import os
import secrets
from datetime import timedelta
from pathlib import Path

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

import toktagger.api.config as config

ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60 * 24  # 24 hours
_SALT = "toktagger-auth-v1"

_serializer: URLSafeTimedSerializer | None = None
_internal_token: str | None = None


def get_internal_token() -> str:
    """Return a stable internal token for trusted server-to-server calls.

    Under multiple Gunicorn workers, all workers share one Ray cluster and
    call back into each other's server-to-server endpoints, so they must all
    use the same token. TOKTAGGER_INTERNAL_TOKEN is set once by the parent
    process (see `run_with_gunicorn` in main.py) and inherited by every
    worker; falling back to a random per-process token is only correct when
    there is a single process (e.g. --workers 1).
    """
    global _internal_token
    env_token = os.environ.get("TOKTAGGER_INTERNAL_TOKEN")
    if env_token:
        return env_token
    if _internal_token is None:
        _internal_token = secrets.token_urlsafe(32)
    return _internal_token


def _get_serializer() -> URLSafeTimedSerializer:
    global _serializer
    if _serializer is not None:
        return _serializer

    if config.settings.auth.secret_key:
        secret = config.settings.auth.secret_key
    else:
        cache_dir = Path(config.settings.server.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key_file = cache_dir / "secret.key"
        if key_file.exists():
            secret = key_file.read_text().strip()
        else:
            secret = secrets.token_hex(32)
            key_file.write_text(secret)

    _serializer = URLSafeTimedSerializer(secret, salt=_SALT)
    return _serializer


def _pbkdf2_hash(password: str, salt_hex: str) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), 260000
    )
    return dk.hex()


def hash_password(plain: str) -> str:
    salt = secrets.token_hex(16)
    hashed = _pbkdf2_hash(plain, salt)
    return f"pbkdf2:{salt}:{hashed}"


def verify_password(plain: str, stored: str) -> bool:
    if not stored.startswith("pbkdf2:"):
        return False
    try:
        _, salt, expected = stored.split(":")
    except ValueError:
        return False
    return secrets.compare_digest(_pbkdf2_hash(plain, salt), expected)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    return _get_serializer().dumps(data)


def decode_token(token: str) -> dict:
    try:
        return _get_serializer().loads(token, max_age=ACCESS_TOKEN_EXPIRE_SECONDS)
    except SignatureExpired:
        raise ValueError("Token has expired")
    except BadSignature:
        raise ValueError("Invalid token")
