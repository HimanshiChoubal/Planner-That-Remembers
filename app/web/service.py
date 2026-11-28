"""Planner service: ties the database, the context backend and the orchestrator together.

Routes stay thin; everything that needs a user's scope goes through here, so scope is always
derived from the authenticated user.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from decimal import Decimal
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from app.agents.base import FinancialProfile
from app.agents.orchestrator import Orchestrator, TurnResult, apply_life_event
from app.context.bus import ContextBus
from app.context.schemas import LifeEvent, MoneyValue
from app.db.models import User, UserProfile, profile_from_dict, profile_to_dict
from app.deps import get_backend, scope_for
from app.engine.debt_payoff import Debt
from app.engine.money import INR, USD, Currency, Money
from app.engine.networth import Asset
from app.meko.interfaces import MekoBackend, Scope
from app.rl.env import params_from_profile
from app.seed.load import knowledge_for_region
from app.seed.personas import get_persona
from app.web import whatif


def currency_for(region: str) -> Currency:
    return INR if region == "in" else USD


class PlannerService:
    def __init__(self, backend: MekoBackend, orchestrator: Orchestrator | None = None) -> None:
        self.backend = backend
        self.orchestrator = orchestrator or Orchestrator()

    # ------------------------------------------------------------------ profile
    def load_profile(self, db: Session, user: User) -> FinancialProfile:
        row = db.get(UserProfile, user.id)
        if row is not None:
            return profile_from_dict(row.data)
        return FinancialProfile(user_id=user.id, region=user.region, currency=currency_for(user.region))

    def save_profile(self, db: Session, user: User, profile: FinancialProfile) -> None:
        row = db.get(UserProfile, user.id)
        data = profile_to_dict(dataclasses.replace(profile, user_id=user.id))
        if row is None:
            db.add(UserProfile(user_id=user.id, data=data))
        else:
            row.data = data
        db.commit()

    # ------------------------------------------------------------------ context
    async def ensure_seeded(self, user: User) -> Scope:
        """Make sure the user's context has knowledge docs (and persona memories for demo users)."""
        scope = scope_for(user.id)
        if await self.backend.knowledge.get(scope, "planning.emergency_fund") is None:
            await self.backend.knowledge.ingest(scope, knowledge_for_region(user.region))
        if user.persona_id and not await self.backend.memory.list(scope):
            onboarding = ContextBus(self.backend, scope, "onboarding", region=user.region)
            from app.meko.interfaces import Visibility

            for memory in get_persona(user.persona_id).onboarding:
                await onboarding.remember(memory, visibility=Visibility.SHARED)
        return scope

    def bus(self, user: User, session_id: str | None = None) -> ContextBus:
        return ContextBus(self.backend, scope_for(user.id), "planner", region=user.region, session_id=session_id)

    # --------------------------------------------------------------------- chat
    async def chat(
        self, db: Session, user: User, text: str, *, session_id: str | None = None, life_event: LifeEvent | None = None
    ) -> tuple[TurnResult, str]:
        await self.ensure_seeded(user)
        profile = self.load_profile(db, user)
        root = self.bus(user, session_id)
        if session_id is None:
            root = await root.with_new_session(text[:40] or "Chat")
        turn = await self.orchestrator.handle(root, profile, text, as_of=date.today(), life_event=life_event)
        if life_event is not None:  # persist the change so later sessions start from it
            self.save_profile(db, user, apply_life_event(profile, life_event))
        return turn, root.session_id or ""

    # --------------------------------------------------------------------- reads
    async def latest_plan(self, user: User) -> dict[str, Any] | None:
        art = await self.backend.artifacts.get(scope_for(user.id), "plan")
        return None if art is None else {"version": art.version, "note": art.note, "created_at": art.created_at.isoformat(), "plan": dict(art.payload)}

    async def plan_versions(self, user: User) -> list[dict[str, Any]]:
        versions = await self.backend.artifacts.list_versions(scope_for(user.id), "plan")
        return [{"version": v.version, "note": v.note, "created_at": v.created_at.isoformat(), "parent": v.parent_version} for v in versions]

    async def plan_diff(self, user: User, old: int, new: int) -> dict[str, Any] | None:
        from app.agents.planner import diff_plans

        scope = scope_for(user.id)
        a, b = await self.backend.artifacts.get(scope, "plan", old), await self.backend.artifacts.get(scope, "plan", new)
        if a is None or b is None:
            return None
        d = diff_plans(a.payload, b.payload)
        return {"from": old, "to": new,
                "changes": [dataclasses.asdict(c) for c in d.changes],
                "recommendations": [dataclasses.asdict(r) for r in d.recommendations]}

    async def explain(self, user: User, trace_id: str) -> dict[str, Any] | None:
        """Everything behind a recommendation: the trace, the memories it used and the rules cited."""
        scope = scope_for(user.id)
        trace = await self.backend.traces.get(scope, trace_id)
        if trace is None:
            return None
        records = {r.id: r for r in await self.backend.memory.list(scope)}
        docs = [await self.backend.knowledge.get(scope, kid) for kid in trace.knowledge_ids]
        return {
            "trace_id": trace.id, "agent": trace.agent, "recommendation": trace.recommendation, "reasoning": trace.reasoning,
            "inputs": dict(trace.inputs), "outputs": dict(trace.outputs),
            "tokens": {"in": trace.tokens_in, "out": trace.tokens_out},
            "memories": [{"id": r.id, "kind": r.kind, "agent": r.agent, "visibility": r.visibility.value, "content": dict(r.content)}
                         for mid in trace.memory_ids if (r := records.get(mid)) is not None],
            "knowledge": [{"id": d.id, "title": d.title, "text": d.text} for d in docs if d is not None],
        }

    async def memories(self, user: User) -> dict[str, Any]:
        records = await self.backend.memory.list(scope_for(user.id))
        rows = [{"id": r.id, "kind": r.kind, "agent": r.agent, "visibility": r.visibility.value,
                 "content": dict(r.content), "confidence": r.confidence, "created_at": r.created_at.isoformat()} for r in records]
        return {"shared": [r for r in rows if r["visibility"] == "shared"], "private": [r for r in rows if r["visibility"] == "private"]}

    async def sessions(self, user: User) -> list[dict[str, Any]]:
        out = await self.backend.conversations.list_sessions(scope_for(user.id))
        return [{"session_id": s.session_id, "title": s.title, "started_at": s.started_at.isoformat(), "messages": s.message_count} for s in out]

    async def messages(self, user: User, session_id: str) -> list[dict[str, Any]]:
        try:
            msgs = await self.backend.conversations.get_messages(scope_for(user.id), session_id)
        except KeyError:
            return []
        return [{"role": m.role, "author": m.author, "content": m.content, "metadata": dict(m.metadata), "created_at": m.created_at.isoformat()} for m in msgs]

    # ------------------------------------------------------------------- what-if
    async def what_if(self, db: Session, user: User, scenario_ids: list[str], *, episodes: int = 30) -> dict[str, Any]:
        await self.ensure_seeded(user)
        profile = self.load_profile(db, user)
        memories = [m.model for m in await self.bus(user).recall("", limit=200) if m.model is not None]
        params = params_from_profile(profile, memories, region=user.region, as_of=date.today())
        return whatif.simulate(params, scenario_ids, episodes=episodes)


# ------------------------------------------------------------------- input helpers
def money_value(amount: Decimal | None, currency: Currency) -> MoneyValue | None:
    return None if amount is None else MoneyValue.from_money(Money.of(amount, currency))


def merge_profile(base: FinancialProfile, data: dict[str, Any]) -> FinancialProfile:
    """Apply user-provided values (major currency units) on top of an existing profile."""
    cur = base.currency
    money_fields = ("monthly_gross_income", "monthly_expenses", "monthly_essential_expenses",
                    "monthly_retirement_contribution", "employer_match_monthly")
    changes: dict[str, Any] = {k: Money.of(data[k], cur) for k in money_fields if data.get(k) is not None}
    for k in ("age", "retirement_age", "dependents", "single_income", "tax_regime", "insurance_covered", "match_captured"):
        if data.get(k) is not None:
            changes[k] = data[k]
    if data.get("assets") is not None:
        changes["assets"] = tuple(Asset(a["name"], a["kind"], Money.of(a["value"], cur), a.get("liquid")) for a in data["assets"])
    if data.get("debts") is not None:
        changes["debts"] = tuple(
            Debt(d.get("id") or f"debt{i + 1}", d["name"], Money.of(d["balance"], cur), Decimal(str(d["apr"])), Money.of(d["min_payment"], cur))
            for i, d in enumerate(data["debts"]))
    return dataclasses.replace(base, **changes)


@lru_cache
def get_service() -> PlannerService:
    return PlannerService(get_backend())