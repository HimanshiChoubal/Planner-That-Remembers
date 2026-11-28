"""JSON API used by the web UI. Every route requires an authenticated user, and the Meko scope
is derived from that user on the server."""

from __future__ import annotations

import dataclasses
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.context.schemas import Constraint, Dependent, Goal, LifeEvent, MemoryModel, RiskProfile
from app.db.models import AdvisorLink, User, profile_to_dict
from app.db.session import get_db
from app.deps import client_scope, get_current_user, require_role
from app.engine.money import Money
from app.web import whatif
from app.web.service import PlannerService, currency_for, get_service, merge_profile, money_value

router = APIRouter(prefix="/api", tags=["api"])
Money_ = Decimal | None


class LifeEventIn(BaseModel):
    event: Literal["new_job", "baby", "relocation", "job_loss", "medical_expense", "windfall", "marriage", "other"]
    description: str = Field(max_length=200, default="")
    occurred_on: date | None = None
    monthly_income_delta: Money_ = None
    monthly_expense_delta: Money_ = None
    one_off_amount: Money_ = None


class ChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    session_id: str | None = None
    life_event: LifeEventIn | None = None


class AssetIn(BaseModel):
    name: str
    kind: Literal["cash", "savings", "investments", "retirement", "property", "vehicle", "other"]
    value: Decimal = Field(ge=0)
    liquid: bool | None = None


class DebtIn(BaseModel):
    id: str | None = None
    name: str
    balance: Decimal = Field(ge=0)
    apr: Decimal = Field(ge=0, le=1, description="annual rate as a fraction, e.g. 0.2199")
    min_payment: Decimal = Field(ge=0)


class ProfileIn(BaseModel):
    age: int | None = Field(default=None, ge=16, le=110)
    retirement_age: int | None = Field(default=None, ge=30, le=100)
    monthly_gross_income: Money_ = Field(default=None, ge=0)
    monthly_expenses: Money_ = Field(default=None, ge=0)
    monthly_essential_expenses: Money_ = Field(default=None, ge=0)
    monthly_retirement_contribution: Money_ = Field(default=None, ge=0)
    employer_match_monthly: Money_ = Field(default=None, ge=0)
    match_captured: bool | None = None
    dependents: int | None = Field(default=None, ge=0, le=20)
    single_income: bool | None = None
    tax_regime: str | None = None
    insurance_covered: bool | None = None
    assets: list[AssetIn] | None = None
    debts: list[DebtIn] | None = None


class MemoryIn(BaseModel):
    kind: Literal["risk_profile", "goal", "constraint", "dependent"]
    tolerance: Literal["low", "medium", "high"] | None = None
    max_equity_share: float | None = Field(default=None, ge=0, le=1)
    horizon_years: int | None = Field(default=None, ge=0)
    name: str | None = None
    category: Literal["home", "education", "travel", "child", "retirement", "vehicle", "other"] = "other"
    target: Money_ = Field(default=None, gt=0)
    saved: Money_ = Field(default=None, ge=0)
    deadline: date | None = None
    priority: int = Field(default=3, ge=1, le=5)
    description: str | None = None
    subject: Literal["investing", "debt", "spending", "saving", "other"] = "other"
    hard: bool = True
    max_value: float | None = None
    relation: Literal["child", "parent", "partner", "other"] | None = None
    age_years: int | None = None


class WhatIfIn(BaseModel):
    scenarios: list[str] = Field(default_factory=lambda: ["job_loss"], max_length=4)
    episodes: int = Field(default=20, ge=5, le=60)


def _memory_model(body: MemoryIn, region: str) -> MemoryModel:
    cur = currency_for(region)
    need = lambda ok, msg: None if ok else (_ for _ in ()).throw(HTTPException(422, msg))  # noqa: E731
    if body.kind == "risk_profile":
        need(body.tolerance, "tolerance is required")
        return RiskProfile(tolerance=body.tolerance, horizon_years=body.horizon_years, max_equity_share=body.max_equity_share)  # type: ignore[arg-type]
    if body.kind == "goal":
        need(body.name and body.target, "name and target are required")
        return Goal(name=body.name, category=body.category, target=money_value(body.target, cur),  # type: ignore[arg-type]
                    deadline=body.deadline, saved=money_value(body.saved, cur), priority=body.priority)
    if body.kind == "constraint":
        need(body.description, "description is required")
        return Constraint(description=body.description, subject=body.subject, hard=body.hard, max_value=body.max_value)  # type: ignore[arg-type]
    need(body.relation, "relation is required")
    return Dependent(relation=body.relation, age_years=body.age_years)  # type: ignore[arg-type]


def _turn_json(turn, session_id: str) -> dict[str, Any]:
    plan = turn.plan or {}
    return {
        "session_id": session_id, "reply": turn.reply, "version": turn.version, "blocked": turn.blocked,
        "intent": turn.route.intent, "questions": list(turn.questions),
        "conflicts": [{"kind": c.kind, "severity": c.severity, "message": c.message} for c in turn.conflicts],
        "recommendations": plan.get("recommendations", [])[:5],
        "diff": None if turn.diff is None else {
            "changes": [dataclasses.asdict(c) for c in turn.diff.changes[:40]],
            "recommendations": [dataclasses.asdict(r) for r in turn.diff.recommendations]},
    }


# ----------------------------------------------------------------------- profile
@router.get("/profile")
def get_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db), svc: PlannerService = Depends(get_service)):
    return profile_to_dict(svc.load_profile(db, user))


@router.put("/profile")
def put_profile(body: ProfileIn, user: User = Depends(get_current_user), db: Session = Depends(get_db),
                svc: PlannerService = Depends(get_service)):
    data = body.model_dump(exclude_unset=True)
    profile = merge_profile(svc.load_profile(db, user), data)
    svc.save_profile(db, user, profile)
    return profile_to_dict(profile)


@router.post("/memories", status_code=status.HTTP_201_CREATED)
async def add_memory(body: MemoryIn, user: User = Depends(get_current_user), svc: PlannerService = Depends(get_service)):
    from app.meko.interfaces import Visibility

    await svc.ensure_seeded(user)
    model = _memory_model(body, user.region)
    record = await svc.bus(user).remember(model, visibility=Visibility.SHARED)
    return {"id": record.id, "kind": record.kind}


@router.get("/memories")
async def list_memories(user: User = Depends(get_current_user), svc: PlannerService = Depends(get_service)):
    await svc.ensure_seeded(user)
    return await svc.memories(user)


# -------------------------------------------------------------------------- chat
@router.post("/chat")
async def chat(body: ChatIn, user: User = Depends(get_current_user), db: Session = Depends(get_db),
               svc: PlannerService = Depends(get_service)):
    event = None
    if body.life_event is not None:
        e, cur = body.life_event, currency_for(user.region)
        event = LifeEvent(event=e.event, description=e.description or e.event.replace("_", " "), occurred_on=e.occurred_on,
                          monthly_income_delta=money_value(e.monthly_income_delta, cur),
                          monthly_expense_delta=money_value(e.monthly_expense_delta, cur),
                          one_off_amount=money_value(e.one_off_amount, cur))
    turn, session_id = await svc.chat(db, user, body.text, session_id=body.session_id, life_event=event)
    return _turn_json(turn, session_id)


@router.get("/sessions")
async def sessions(user: User = Depends(get_current_user), svc: PlannerService = Depends(get_service)):
    return await svc.sessions(user)


@router.get("/sessions/{session_id}/messages")
async def messages(session_id: str, user: User = Depends(get_current_user), svc: PlannerService = Depends(get_service)):
    return await svc.messages(user, session_id)


# -------------------------------------------------------------------- plan / why
@router.get("/plan")
async def plan(user: User = Depends(get_current_user), svc: PlannerService = Depends(get_service)):
    latest = await svc.latest_plan(user)
    if latest is None:
        raise HTTPException(404, "No plan yet")
    return latest


@router.get("/plan/versions")
async def plan_versions(user: User = Depends(get_current_user), svc: PlannerService = Depends(get_service)):
    return await svc.plan_versions(user)


@router.get("/plan/diff")
async def plan_diff(old: int = Query(ge=1), new: int = Query(ge=1), user: User = Depends(get_current_user),
                    svc: PlannerService = Depends(get_service)):
    diff = await svc.plan_diff(user, old, new)
    if diff is None:
        raise HTTPException(404, "Unknown plan version")
    return diff


@router.get("/why/{trace_id}")
async def why(trace_id: str, user: User = Depends(get_current_user), svc: PlannerService = Depends(get_service)):
    explanation = await svc.explain(user, trace_id)
    if explanation is None:
        raise HTTPException(404, "Unknown trace")
    return explanation


# ------------------------------------------------------------------------ what-if
@router.get("/whatif/scenarios")
def whatif_scenarios(user: User = Depends(get_current_user)):
    return [{"id": k, "label": v[0]} for k, v in whatif.PRESETS.items() if k != "baseline"]


@router.post("/whatif")
async def run_whatif(body: WhatIfIn, user: User = Depends(get_current_user), db: Session = Depends(get_db),
                     svc: PlannerService = Depends(get_service)):
    try:
        return await svc.what_if(db, user, body.scenarios, episodes=body.episodes)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# ------------------------------------------------------------------------ advisor
@router.get("/advisor/clients")
def advisor_clients(advisor: User = Depends(require_role("advisor")), db: Session = Depends(get_db)):
    rows = db.execute(select(User).join(AdvisorLink, AdvisorLink.client_id == User.id).where(AdvisorLink.advisor_id == advisor.id)).scalars()
    return [{"id": u.id, "display_name": u.display_name, "region": u.region} for u in rows]


@router.get("/advisor/clients/{client_id}/plan")
async def advisor_client_plan(client_id: str, advisor: User = Depends(require_role("advisor")), db: Session = Depends(get_db),
                              svc: PlannerService = Depends(get_service)):
    scope = client_scope(advisor, client_id, db)  # raises 403 unless linked
    art = await svc.backend.artifacts.get(scope, "plan")
    if art is None:
        raise HTTPException(404, "Client has no plan yet")
    return {"version": art.version, "plan": dict(art.payload)}