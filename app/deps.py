"""Shared FastAPI dependencies.

The Meko scope is ALWAYS derived server-side from the authenticated user (or from a verified
advisor link), never from anything the client sends. That is what keeps each customer's
context isolated.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.security import decode_token
from app.config import get_settings
from app.db.models import AdvisorLink, User
from app.db.session import get_db
from app.meko.fake import FakeMeko
from app.meko.interfaces import MekoBackend, Scope


@lru_cache
def get_backend() -> MekoBackend:
    """The context backend. The fake is used until the real Meko client is wired in; the Meko
    URL and key (when set) stay server-side only."""
    return FakeMeko()


def scope_for(user_id: str) -> Scope:
    return Scope(user_id=user_id, datapack=f"dp_{user_id}")


def _token_from(request: Request) -> str | None:
    cookie = request.cookies.get(get_settings().cookie_name)
    if cookie:
        return cookie
    header = request.headers.get("Authorization", "")
    return header[7:] if header.lower().startswith("bearer ") else None


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = _token_from(request)
    claims = decode_token(token) if token else None
    user = db.get(User, claims["sub"]) if claims else None
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    return user


def require_role(*roles: str) -> Callable[[User], User]:
    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient permissions")
        return user

    return checker


def client_scope(advisor: User, client_id: str, db: Session) -> Scope:
    """Scope of a client's context, only if ``advisor`` is linked to that client."""
    link = db.scalar(select(AdvisorLink).where(AdvisorLink.advisor_id == advisor.id, AdvisorLink.client_id == client_id))
    if advisor.role != "advisor" or link is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "No access to this client")
    return scope_for(client_id)