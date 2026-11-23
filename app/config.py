"""Application settings, loaded from environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_INSECURE_DEFAULT_SECRET = "change-me-in-production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app ---
    app_name: str = "The Planner That Remembers"
    env: Literal["development", "production", "test"] = "development"
    debug: bool = False
    default_region: Literal["us", "in"] = "us"

    # --- database (Postgres) ---
    database_url: str = "postgresql+psycopg://postgres:dev@localhost:5432/planner"

    # --- auth ---
    jwt_secret: str = _INSECURE_DEFAULT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 8
    cookie_name: str = "planner_session"
    cookie_secure: bool = False  # set True behind HTTPS in production
    guest_demo_enabled: bool = True

    # --- Meko (endpoint + key stay server-side only) ---
    meko_url: str | None = None
    meko_api_key: str | None = None
    meko_timeout_seconds: float = 15.0
    meko_force_fake: bool = False  # use the in-memory fake even if credentials exist

    # --- optional LLM narrator (app must work without it) ---
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_cache_dir: str = ".cache/narrator"

    # --- RL ---
    rl_policy_path: str = "app/rl/models/policy.zip"

    @property
    def use_fake_meko(self) -> bool:
        """Fall back to the in-memory fake when Meko isn't configured."""
        return self.meko_force_fake or not (self.meko_url and self.meko_api_key)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key and self.llm_model)

    @model_validator(mode="after")
    def _check_production_secrets(self) -> Settings:
        if self.env == "production" and self.jwt_secret == _INSECURE_DEFAULT_SECRET:
            raise ValueError("JWT_SECRET must be set to a strong random value in production.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()