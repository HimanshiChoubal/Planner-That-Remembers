"""Password hashing (bcrypt) and signed JWT access tokens."""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

from app.config import get_settings


def _prehash(password: str) -> bytes:
    """SHA-256 then base64 so any length is safe (bcrypt only reads the first 72 bytes)."""
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), hashed.encode("ascii"))
    except ValueError:
        return False


def create_access_token(user_id: str, role: str, *, expires_minutes: int | None = None) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    minutes = s.jwt_expire_minutes if expires_minutes is None else expires_minutes
    claims = {"sub": user_id, "role": role, "iat": now, "exp": now + timedelta(minutes=minutes)}
    return jwt.encode(claims, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any] | None:
    """Claims if the token is valid and unexpired, otherwise ``None``."""
    s = get_settings()
    try:
        return jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except JWTError:
        return None