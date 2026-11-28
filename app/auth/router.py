"""Authentication endpoints (cookie-based JWT sessions).

Security notes: bcrypt hashing, HTTP-only SameSite cookies, the same error for an unknown email
and a wrong password, and a dummy hash check so response time doesn't reveal which emails exist.
Not included yet: rate limiting / lockout (add before a public launch).
"""

from __future__ import annotations

import re
import secrets
from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth.security import create_access_token, hash_password, verify_password
from app.config import get_settings
from app.db.models import User, UserProfile, profile_to_dict
from app.db.session import get_db
from app.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DUMMY_HASH = hash_password("dummy-password-for-timing")
_INVALID = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")


class RegisterIn(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=80)
    region: Literal["us", "in"] = "us"

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL.match(v):
            raise ValueError("Invalid email address")
        return v


class LoginIn(BaseModel):
    email: str
    password: str


class GuestIn(BaseModel):
    persona_id: str | None = None


class UserOut(BaseModel):
    id: str
    email: str
    display_name: str
    region: str
    role: str
    persona_id: str | None = None

    @classmethod
    def of(cls, u: User) -> UserOut:
        return cls(id=u.id, email=u.email, display_name=u.display_name, region=u.region, role=u.role, persona_id=u.persona_id)


def _start_session(response: Response, user: User) -> None:
    s = get_settings()
    response.set_cookie(
        s.cookie_name, create_access_token(user.id, user.role), max_age=s.jwt_expire_minutes * 60,
        httponly=True, samesite="lax", secure=s.cookie_secure, path="/")


def ensure_demo_user(db: Session, persona_id: str | None = None) -> User:
    """Create (once) the passwordless demo account for a persona, with its stored profile."""
    import dataclasses

    from app.seed.generator import generate_history
    from app.seed.personas import PERSONAS, get_persona

    persona = get_persona(persona_id) if persona_id else next(iter(PERSONAS.values()))
    user = db.scalar(select(User).where(User.persona_id == persona.id))
    if user is None:
        user = User(email=persona.login, password_hash=hash_password(secrets.token_urlsafe(32)),
                    display_name=persona.display_name, region=persona.region, role="guest", persona_id=persona.id)
        db.add(user)
        db.flush()
        history = generate_history(persona, as_of=date.today(), months=18, seed=0)
        profile = dataclasses.replace(persona.profile, user_id=user.id, income_history=history.income)
        db.add(UserProfile(user_id=user.id, data=profile_to_dict(profile)))
        db.commit()
    return user


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, response: Response, db: Session = Depends(get_db)) -> UserOut:
    user = User(email=body.email, password_hash=hash_password(body.password),
                display_name=body.display_name.strip(), region=body.region, role="user")
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists") from exc
    _start_session(response, user)
    return UserOut.of(user)


@router.post("/login", response_model=UserOut)
def login(body: LoginIn, response: Response, db: Session = Depends(get_db)) -> UserOut:
    user = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    ok = verify_password(body.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok or not user.is_active or user.role == "guest":
        raise _INVALID
    _start_session(response, user)
    return UserOut.of(user)


@router.post("/guest", response_model=UserOut)
def guest(body: GuestIn, response: Response, db: Session = Depends(get_db)) -> UserOut:
    if not get_settings().guest_demo_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Guest demo is disabled")
    try:
        user = ensure_demo_user(db, body.persona_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    _start_session(response, user)
    return UserOut.of(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> None:
    response.delete_cookie(get_settings().cookie_name, path="/")


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.of(user)