import hashlib
import os
import secrets
from datetime import timedelta
from pathlib import Path

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from platformdirs import user_cache_dir

ACCESS_TOKEN_EXPIRE_SECONDS = 60 * 60 * 24  # 24 hours
_SALT = "toktagger-auth-v1"

_serializer: URLSafeTimedSerializer | None = None


def _get_serializer() -> URLSafeTimedSerializer:
    global _serializer
    if _serializer is not None:
        return _serializer

    env_key = os.environ.get("AUTH_SECRET_KEY")
    if env_key:
        secret = env_key
    else:
        cache_dir = Path(user_cache_dir("toktagger", "ukaea"))
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
