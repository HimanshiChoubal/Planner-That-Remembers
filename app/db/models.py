"""Database models: users, stored financial profiles, advisor links and simulation runs.

Money is stored as integer minor units (never floats). Profiles are stored as JSON so the same
schema works on Postgres and SQLite.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.agents.base import FinancialProfile
from app.db.session import Base
from app.engine.debt_payoff import Debt
from app.engine.money import Money, get_currency
from app.engine.networth import Asset


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    display_name: Mapped[str] = mapped_column(String(80))
    region: Mapped[str] = mapped_column(String(8), default="us")
    role: Mapped[str] = mapped_column(String(16), default="user")  # user | advisor | guest
    persona_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict[str, Any]] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class AdvisorLink(Base):
    """A human advisor allowed to view one client's context (the collaboration feature)."""

    __tablename__ = "advisor_links"
    __table_args__ = (UniqueConstraint("advisor_id", "client_id", name="uq_advisor_client"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    advisor_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SimRun(Base):
    __tablename__ = "sim_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    scenario: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------- profile <-> JSON
def _m(money: Money | None) -> int | None:
    return None if money is None else money.minor


def profile_to_dict(p: FinancialProfile) -> dict[str, Any]:
    return {
        "user_id": p.user_id, "region": p.region, "currency": p.currency.code,
        "age": p.age, "retirement_age": p.retirement_age,
        "monthly_gross_income": _m(p.monthly_gross_income),
        "income_history": [m.minor for m in p.income_history],
        "monthly_expenses": _m(p.monthly_expenses),
        "monthly_essential_expenses": _m(p.monthly_essential_expenses),
        "assets": None if p.assets is None else [
            {"name": a.name, "kind": a.kind, "value": a.value.minor, "liquid": a.liquid} for a in p.assets],
        "debts": None if p.debts is None else [
            {"id": d.id, "name": d.name, "balance": d.balance.minor, "apr": str(d.apr),
             "min_payment": d.min_payment.minor} for d in p.debts],
        "dependents": p.dependents, "single_income": p.single_income, "tax_regime": p.tax_regime,
        "employer_match_monthly": _m(p.employer_match_monthly), "match_captured": p.match_captured,
        "insurance_covered": p.insurance_covered,
        "monthly_retirement_contribution": _m(p.monthly_retirement_contribution),
    }


def profile_from_dict(d: dict[str, Any]) -> FinancialProfile:
    cur = get_currency(d["currency"])

    def money(v: int | None) -> Money | None:
        return None if v is None else Money(int(v), cur)

    assets = None if d["assets"] is None else tuple(
        Asset(a["name"], a["kind"], Money(int(a["value"]), cur), a["liquid"]) for a in d["assets"])
    debts = None if d["debts"] is None else tuple(
        Debt(x["id"], x["name"], Money(int(x["balance"]), cur), Decimal(x["apr"]), Money(int(x["min_payment"]), cur))
        for x in d["debts"])
    return FinancialProfile(
        user_id=d["user_id"], region=d["region"], currency=cur, age=d["age"], retirement_age=d["retirement_age"],
        monthly_gross_income=money(d["monthly_gross_income"]),
        income_history=tuple(Money(int(v), cur) for v in d["income_history"]),
        monthly_expenses=money(d["monthly_expenses"]),
        monthly_essential_expenses=money(d["monthly_essential_expenses"]),
        assets=assets, debts=debts, dependents=d["dependents"], single_income=d["single_income"],
        tax_regime=d["tax_regime"], employer_match_monthly=money(d["employer_match_monthly"]),
        match_captured=d["match_captured"], insurance_covered=d["insurance_covered"],
        monthly_retirement_contribution=money(d["monthly_retirement_contribution"]))