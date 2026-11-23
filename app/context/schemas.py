"""Typed memory schemas.

Each model is one kind of long-term memory (goal, constraint, life event, ...). Memories are
stored in Meko as JSON ``content`` with a ``kind`` string; these models give them structure.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.engine.money import Money, get_currency


class MemoryKind(str, Enum):
    GOAL = "goal"
    RISK_PROFILE = "risk_profile"
    CONSTRAINT = "constraint"
    DEPENDENT = "dependent"
    INCOME_PATTERN = "income_pattern"
    LIFE_EVENT = "life_event"
    DECISION = "decision"
    REJECTED_DECISION = "rejected_decision"
    CATEGORY_CORRECTION = "category_correction"
    FINDING = "finding"


class MoneyValue(BaseModel):
    """JSON-safe money: integer minor units plus ISO currency code."""

    model_config = ConfigDict(frozen=True)

    minor: int
    currency: str

    @classmethod
    def from_money(cls, money: Money) -> MoneyValue:
        return cls(minor=money.minor, currency=money.currency.code)

    def to_money(self) -> Money:
        return Money(self.minor, get_currency(self.currency))


class MemoryModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    KIND: ClassVar[MemoryKind]


class Goal(MemoryModel):
    KIND: ClassVar[MemoryKind] = MemoryKind.GOAL

    name: str
    category: Literal["home", "education", "travel", "child", "retirement", "vehicle", "other"] = (
        "other"
    )
    target: MoneyValue
    deadline: date | None = None
    saved: MoneyValue | None = None  # already set aside toward this goal
    priority: int = Field(default=3, ge=1, le=5)  # 1 = highest


class RiskProfile(MemoryModel):
    KIND: ClassVar[MemoryKind] = MemoryKind.RISK_PROFILE

    tolerance: Literal["low", "medium", "high"]
    horizon_years: int | None = Field(default=None, ge=0)
    max_equity_share: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None


class Constraint(MemoryModel):
    """A rule the user set, e.g. 'no aggressive investing'. Hard constraints must never be
    violated by any recommendation or by the RL policy."""

    KIND: ClassVar[MemoryKind] = MemoryKind.CONSTRAINT

    description: str
    subject: Literal["investing", "debt", "spending", "saving", "other"] = "other"
    hard: bool = True
    max_value: float | None = None  # optional numeric cap, e.g. max equity share 0.3


class Dependent(MemoryModel):
    KIND: ClassVar[MemoryKind] = MemoryKind.DEPENDENT

    relation: Literal["child", "parent", "partner", "other"]
    age_years: int | None = Field(default=None, ge=0)
    expected_date: date | None = None  # e.g. due date for an expected child


class IncomePattern(MemoryModel):
    KIND: ClassVar[MemoryKind] = MemoryKind.INCOME_PATTERN

    kind: Literal["salaried", "irregular", "business"]
    monthly_mean: MoneyValue
    monthly_cv: float = Field(default=0.0, ge=0.0)  # coefficient of variation (std / mean)
    notes: str | None = None


class LifeEvent(MemoryModel):
    KIND: ClassVar[MemoryKind] = MemoryKind.LIFE_EVENT

    event: Literal[
        "new_job", "baby", "relocation", "job_loss", "medical_expense", "windfall", "marriage", "other"
    ]
    description: str
    occurred_on: date | None = None
    monthly_income_delta: MoneyValue | None = None
    monthly_expense_delta: MoneyValue | None = None
    one_off_amount: MoneyValue | None = None


class Decision(MemoryModel):
    KIND: ClassVar[MemoryKind] = MemoryKind.DECISION

    topic: str
    summary: str
    plan_version: int | None = None  # plan artifact version in which it was accepted
    rationale: str | None = None


class RejectedDecision(MemoryModel):
    KIND: ClassVar[MemoryKind] = MemoryKind.REJECTED_DECISION

    topic: str
    summary: str
    reason: str  # why the user rejected it, so it isn't proposed again


class CategoryCorrection(MemoryModel):
    """User fixed a transaction's category; the categorizer learns from these."""

    KIND: ClassVar[MemoryKind] = MemoryKind.CATEGORY_CORRECTION

    merchant_text: str
    category_id: str


class Finding(MemoryModel):
    """An agent's learned insight, promoted to shared memory so other agents benefit."""

    KIND: ClassVar[MemoryKind] = MemoryKind.FINDING

    topic: str  # e.g. "income_volatility"
    statement: str
    data: dict[str, Any] = Field(default_factory=dict)


MEMORY_MODELS: dict[MemoryKind, type[MemoryModel]] = {
    m.KIND: m
    for m in (
        Goal,
        RiskProfile,
        Constraint,
        Dependent,
        IncomePattern,
        LifeEvent,
        Decision,
        RejectedDecision,
        CategoryCorrection,
        Finding,
    )
}


def kind_of(model: MemoryModel) -> str:
    return model.KIND.value


def dump_content(model: MemoryModel) -> dict[str, Any]:
    """JSON-safe dict for storing as Meko memory content."""
    return model.model_dump(mode="json")


def parse_memory(kind: str, content: dict[str, Any] | Any) -> MemoryModel | None:
    """Rebuild a typed model from stored content; ``None`` for unknown kinds or bad data."""
    try:
        model_cls = MEMORY_MODELS[MemoryKind(kind)]
        return model_cls.model_validate(dict(content))
    except (ValueError, KeyError, TypeError):
        return None