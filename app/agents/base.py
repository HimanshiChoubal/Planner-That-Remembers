"""Agent base class.

Agents never talk to each other or to Meko directly: every read and write goes through their
ContextBus. ``execute`` enforces the rule that an agent asks instead of guessing: if required
context is missing it returns ``needs_input`` with questions rather than running.

All numbers in a recommendation must come from ``app.engine`` or ``app.rl``. Text is built from
those numbers (templates), never invented.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import date
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar

from app.context.bus import ContextBus, RecalledMemory
from app.context.gaps import Gap, blocking_gaps, find_gaps
from app.context.schemas import Dependent, Finding, LifeEvent, MemoryKind, MemoryModel
from app.engine import emergency_fund as ef
from app.engine import forecast as fc
from app.engine import priorities as pr
from app.engine.debt_payoff import Debt
from app.engine.money import Currency, Money
from app.engine.networth import Asset
from app.meko.interfaces import KnowledgeHit, MemoryRecord
from app.regions.base import RegionPack

M = TypeVar("M", bound=MemoryModel)


def money_json(money: Money | None) -> dict[str, Any] | None:
    """JSON-safe money for plan payloads: exact minor units plus a display string."""
    if money is None:
        return None
    return {"minor": money.minor, "currency": money.currency.code, "display": str(money)}


@dataclass(frozen=True)
class FinancialProfile:
    """Structured facts about a household. Stored in Postgres; ``None`` means unknown."""

    user_id: str
    region: str
    currency: Currency
    age: int | None = None
    retirement_age: int | None = None
    monthly_gross_income: Money | None = None
    income_history: tuple[Money, ...] = ()  # monthly, oldest first
    monthly_expenses: Money | None = None
    monthly_essential_expenses: Money | None = None
    assets: tuple[Asset, ...] | None = None
    debts: tuple[Debt, ...] | None = None  # empty tuple means "no debts"; None means unknown
    dependents: int = 0
    single_income: bool = False
    tax_regime: str | None = None
    employer_match_monthly: Money | None = None
    match_captured: bool = False
    insurance_covered: bool | None = None
    monthly_retirement_contribution: Money | None = None

    @property
    def liquid_savings(self) -> Money | None:
        if self.assets is None:
            return None
        total = Money.zero(self.currency)
        for a in self.assets:
            if a.is_liquid:
                total = total + a.value
        return total

    def facts(self) -> dict[str, Any]:
        """Known facts, keyed for gap detection (unknown values are omitted)."""
        raw = {
            "age": self.age,
            "retirement_age": self.retirement_age,
            "monthly_income": self.monthly_gross_income,
            "monthly_expenses": self.monthly_expenses,
            "monthly_essential_expenses": self.monthly_essential_expenses,
            "liquid_savings": self.liquid_savings,
            "debts": self.debts,
            "tax_regime": self.tax_regime,
            "insurance_covered": self.insurance_covered,
            "assets": self.assets,
        }
        return {k: v for k, v in raw.items() if v is not None}


@dataclass(frozen=True)
class Waterfall:
    surplus: Money
    items: tuple[pr.PriorityItem, ...]
    allocation: pr.Allocation
    assumptions: tuple[str, ...]
    source_memory_id: str | None  # shared finding the surplus came from

    def allocated_to(self, *steps: pr.Step) -> Money:
        total = Money.zero(self.surplus.currency)
        for step, amount in self.allocation.lines:
            if step in steps:
                total = total + amount
        return total

    def status_of(self, step: pr.Step) -> str:
        return next(i.status for i in self.items if i.step is step)


@dataclass(frozen=True)
class AgentContext:
    bus: ContextBus  # already bound to this agent
    pack: RegionPack
    profile: FinancialProfile
    memories: tuple[RecalledMemory, ...] = ()
    life_event: LifeEvent | None = None  # set when replanning after a life event
    as_of: date | None = None  # planning date; defaults to today

    @property
    def today(self) -> date:
        return self.as_of or date.today()

    @classmethod
    async def load(
        cls,
        bus: ContextBus,
        pack: RegionPack,
        profile: FinancialProfile,
        *,
        life_event: LifeEvent | None = None,
        as_of: date | None = None,
    ) -> AgentContext:
        memories = await bus.recall("", limit=200)
        return cls(bus, pack, profile, tuple(memories), life_event, as_of)

    def memories_of(self, kind: MemoryKind) -> list[RecalledMemory]:
        return [m for m in self.memories if m.record.kind == kind.value]

    def models(self, model_type: type[M]) -> list[M]:
        return [m.model for m in self.memories if isinstance(m.model, model_type)]

    def memory_ids(self, *model_types: type[MemoryModel]) -> tuple[str, ...]:
        return tuple(m.record.id for m in self.memories if isinstance(m.model, model_types))

    # ---- shared-context helpers: what other agents have already learned ----
    def latest_finding(self, topic: str) -> tuple[RecalledMemory, Finding] | None:
        """Newest visible finding on ``topic`` (typically promoted by another agent)."""
        found = [(m, m.model) for m in self.memories if isinstance(m.model, Finding) and m.model.topic == topic]
        return max(found, key=lambda t: t[0].record.created_at) if found else None

    def income_cv(self) -> float:
        """Income variability: a shared finding if present, else computed from history."""
        hit = self.latest_finding("income_volatility")
        if hit is not None and "cv" in hit[1].data:
            return float(hit[1].data["cv"])
        return fc.coefficient_of_variation(self.profile.income_history)

    def dependents(self) -> int:
        return max(self.profile.dependents, len(self.models(Dependent)))

    def monthly_surplus(self) -> tuple[Money | None, str | None]:
        """Monthly surplus after expenses (assumed to include minimum debt payments) and the
        memory ID it came from, if it was a shared finding."""
        hit = self.latest_finding("monthly_surplus")
        if hit is not None and hit[1].data.get("currency") == self.profile.currency.code:
            return Money(int(hit[1].data["surplus_minor"]), self.profile.currency), hit[0].record.id
        p = self.profile
        if p.monthly_gross_income is not None and p.monthly_expenses is not None:
            return p.monthly_gross_income - p.monthly_expenses, None
        return None, None

    def waterfall(self) -> Waterfall | None:
        """Priority waterfall for this month's surplus, or ``None`` if inputs are missing."""
        surplus, source = self.monthly_surplus()
        p = self.profile
        liquid, essential = p.liquid_savings, p.monthly_essential_expenses
        if surplus is None or liquid is None or essential is None:
            return None
        assumptions: list[str] = []
        debts = p.debts
        if debts is None:
            debts = ()
            assumptions.append("No debt was assumed because none was provided.")
        months, _ = ef.target_months(
            income_cv=self.income_cv(), dependents=self.dependents(), single_income=p.single_income
        )
        state = pr.PriorityState(
            currency=p.currency, liquid_savings=liquid, monthly_essential=essential,
            emergency_target_months=months, debts=debts,
            employer_match_monthly=p.employer_match_monthly, match_captured=p.match_captured,
        )
        items = tuple(pr.evaluate(state))
        return Waterfall(surplus, items, pr.allocate_surplus(items, surplus), tuple(assumptions), source)


@dataclass(frozen=True)
class Recommendation:
    id: str
    agent: str
    title: str
    summary: str
    metrics: Mapping[str, Any]
    memory_ids: tuple[str, ...]
    knowledge_ids: tuple[str, ...]
    trace_id: str  # resolves in the Why? panel
    priority: int = 3  # 1 = most important


@dataclass(frozen=True)
class AgentResult:
    agent: str
    status: Literal["ok", "needs_input"]
    recommendations: tuple[Recommendation, ...] = ()
    gaps: tuple[Gap, ...] = ()
    plan: Mapping[str, Any] = field(default_factory=dict)  # this agent's section of the plan
    findings: tuple[MemoryRecord, ...] = ()  # shared memories this run wrote

    @property
    def questions(self) -> list[str]:
        return [g.question for g in self.gaps]


class Agent(ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    async def context_for(
        self,
        bus: ContextBus,
        pack: RegionPack,
        profile: FinancialProfile,
        *,
        life_event: LifeEvent | None = None,
        as_of: date | None = None,
    ) -> AgentContext:
        """Build a context whose bus (and memory visibility) belongs to this agent."""
        return await AgentContext.load(
            bus.for_agent(self.name), pack, profile, life_event=life_event, as_of=as_of
        )

    async def execute(self, ctx: AgentContext) -> AgentResult:
        """Ask-before-guess wrapper around ``run``."""
        gaps = find_gaps(self.name, ctx.memories, ctx.profile.facts())
        if blocking_gaps(gaps):
            return AgentResult(agent=self.name, status="needs_input", gaps=tuple(gaps))
        result = await self.run(ctx)
        if gaps:  # non-blocking (low confidence) gaps ride along with the result
            return AgentResult(
                agent=result.agent, status=result.status, recommendations=result.recommendations,
                gaps=tuple(gaps), plan=result.plan, findings=result.findings,
            )
        return result

    @abstractmethod
    async def run(self, ctx: AgentContext) -> AgentResult:
        """Compute from ``ctx`` (engine/RL numbers only), then return recommendations."""

    # ---------------------------------------------------------------- helpers
    async def recommend(
        self,
        ctx: AgentContext,
        *,
        title: str,
        summary: str,
        reasoning: str,
        metrics: Mapping[str, Any],
        memory_ids: Sequence[str] = (),
        knowledge_ids: Sequence[str] = (),
        inputs: Mapping[str, Any] | None = None,
        priority: int = 3,
    ) -> Recommendation:
        """Record a decision trace, then build the recommendation that points at it."""
        trace = await ctx.bus.trace(
            title, reasoning,
            memory_ids=memory_ids, knowledge_ids=knowledge_ids,
            inputs=inputs, outputs=metrics,
        )
        return Recommendation(
            id=f"rec_{uuid.uuid4().hex[:12]}", agent=self.name, title=title, summary=summary,
            metrics=dict(metrics), memory_ids=tuple(memory_ids),
            knowledge_ids=tuple(knowledge_ids), trace_id=trace.id, priority=priority,
        )

    async def share_finding(
        self, ctx: AgentContext, topic: str, statement: str, data: Mapping[str, Any] | None = None
    ) -> MemoryRecord:
        """Promote something this agent learned so every other agent can use it."""
        return await ctx.bus.learn_shared(Finding(topic=topic, statement=statement, data=dict(data or {})))

    async def cite(self, ctx: AgentContext, query: str, *, limit: int = 2) -> list[KnowledgeHit]:
        """Region-filtered knowledge-base rules to cite in a trace."""
        return await ctx.bus.knowledge(query, limit=limit)