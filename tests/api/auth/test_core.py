"""Unit tests for toktagger.api.auth.core — no DB or network needed."""
import time
import pytest

from toktagger.api.auth.core import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    _get_serializer,
)


# ---------------------------------------------------------------------------
# hash_password / verify_password
# ---------------------------------------------------------------------------

def test_hash_password_format():
    h = hash_password("secret")
    parts = h.split(":")
    assert parts[0] == "pbkdf2"
    assert len(parts) == 3
    # salt and hash are non-empty hex strings
    assert len(parts[1]) > 0
    assert len(parts[2]) > 0


def test_hash_password_produces_unique_salts():
    h1 = hash_password("same")
    h2 = hash_password("same")
    # Different salts → different stored strings even for identical passwords
    assert h1 != h2


def test_verify_password_correct():
    stored = hash_password("correct_password")
    assert verify_password("correct_password", stored) is True


def test_verify_password_wrong_password():
    stored = hash_password("correct_password")
    assert verify_password("wrong_password", stored) is False


def test_verify_password_wrong_format_returns_false():
    assert verify_password("anything", "plaintext_hash") is False


def test_verify_password_empty_string():
    stored = hash_password("")
    assert verify_password("", stored) is True
    assert verify_password("notempty", stored) is False


# ---------------------------------------------------------------------------
# create_access_token / decode_token (round-trip)
# ---------------------------------------------------------------------------

def test_create_access_token_returns_string():
    token = create_access_token({"sub": "alice"})
    assert isinstance(token, str)
    assert len(token) > 0


def test_decode_token_round_trip():
    payload = {"sub": "alice", "role": "admin"}
    token = create_access_token(payload)
    decoded = decode_token(token)
    assert decoded["sub"] == "alice"
    assert decoded["role"] == "admin"


def test_decode_token_expired(monkeypatch):
    """Simulate an expired token by monkeypatching the serializer's loads to behave
    as if max_age has elapsed.  We achieve this by creating a token, then advancing
    the timestamp embedded in the signature past the expiry window."""
    from itsdangerous import SignatureExpired

    token = create_access_token({"sub": "expired_user"})

    serializer = _get_serializer()

    original_loads = serializer.loads

    def fake_loads(data, **kwargs):
        raise SignatureExpired("simulated expiry")

    monkeypatch.setattr(serializer, "loads", fake_loads)

    with pytest.raises(ValueError, match="expired"):
        decode_token(token)


def test_decode_token_invalid_raises():
    with pytest.raises(ValueError, match="Invalid"):
        decode_token("this.is.not.a.valid.token")


def test_decode_token_tampered_raises():
    token = create_access_token({"sub": "alice"})
    tampered = token[:-4] + "XXXX"
    with pytest.raises(ValueError):
        decode_token(tampered)
