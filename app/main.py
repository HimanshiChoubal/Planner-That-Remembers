"""Application entry point.  Run:  uvicorn app.main:app --reload"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.auth.router import router as auth_router
from app.config import get_settings
from app.db.session import init_db
from app.web.routes.api import router as api_router

STATIC_DIR = Path(__file__).parent / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not initialise the database. Check DATABASE_URL "
            "(for local development you can use sqlite:///./dev.db)."
        ) from exc
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan, docs_url="/docs" if settings.debug else None, redoc_url=None)
    app.include_router(auth_router)
    app.include_router(api_router)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()