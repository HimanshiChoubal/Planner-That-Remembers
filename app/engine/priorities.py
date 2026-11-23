"""Financial priority ordering (the "waterfall").

Order of steps, each backed by a knowledge-base rule so agents can cite it in traces:

  1. starter buffer       one month of essential spending in liquid savings
  2. employer match       capture free money before anything else
  3. high-interest debt   debts at or above HIGH_INTEREST_APR
  4. full emergency fund  the volatility-adjusted target from ``emergency_fund``
  5. other debt           lower-interest debt (minimum payments are always made)
  6. goals & retirement   everything else

This is a planning HEURISTIC (a widely used rule of thumb), not a law. The first step still
open is "active"; later open steps are "pending". Hard user constraints are applied by agents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Literal

from app.engine.debt_payoff import Debt
from app.engine.money import Currency, Money

HIGH_INTEREST_APR = Decimal("0.10")
STARTER_BUFFER_MONTHS = Decimal("1")

Status = Literal["done", "active", "pending", "skipped"]


class Step(str, Enum):
    STARTER_BUFFER = "starter_buffer"
    EMPLOYER_MATCH = "employer_match"
    HIGH_INTEREST_DEBT = "high_interest_debt"
    FULL_EMERGENCY_FUND = "full_emergency_fund"
    OTHER_DEBT = "other_debt"
    GOALS_AND_RETIREMENT = "goals_and_retirement"


def rule_id(step: Step) -> str:
    return f"planning.priority.{step.value}"


@dataclass(frozen=True)
class PriorityState:
    currency: Currency
    liquid_savings: Money
    monthly_essential: Money
    emergency_target_months: Decimal
    debts: Sequence[Debt] = ()
    employer_match_monthly: Money | None = None  # monthly contribution that earns the full match
    match_captured: bool = False


@dataclass(frozen=True)
class PriorityItem:
    step: Step
    status: Status
    reason: str
    rule_id: str
    need: Money | None  # amount still needed to finish the step (None for open-ended steps)


@dataclass(frozen=True)
class Allocation:
    lines: tuple[tuple[Step, Money], ...]
    leftover: Money


def _positive(money: Money) -> Money:
    return Money.zero(money.currency) if money.is_negative() else money


def _total(items: Sequence[Money], currency: Currency) -> Money:
    total = Money.zero(currency)
    for m in items:
        total = total + m
    return total


def evaluate(state: PriorityState) -> list[PriorityItem]:
    """Status of every step, in priority order."""
    cur = state.currency
    liquid = state.liquid_savings
    starter_target = state.monthly_essential * STARTER_BUFFER_MONTHS
    full_target = state.monthly_essential * state.emergency_target_months
    open_debts = [d for d in state.debts if d.balance.minor > 0]
    high = [d for d in open_debts if d.apr >= HIGH_INTEREST_APR]
    low = [d for d in open_debts if d.apr < HIGH_INTEREST_APR]
    has_match = state.employer_match_monthly is not None and state.employer_match_monthly.minor > 0

    # (step, applicable, done, need, reason when open, reason when done / not applicable)
    rows: list[tuple[Step, bool, bool, Money | None, str, str]] = [
        (
            Step.STARTER_BUFFER, True, liquid >= starter_target,
            _positive(starter_target - liquid),
            f"Build a starter buffer of {starter_target} (1 month of essentials); you have {liquid}.",
            f"Starter buffer covered ({liquid} vs {starter_target}).",
        ),
        (
            Step.EMPLOYER_MATCH, has_match, state.match_captured,
            state.employer_match_monthly if has_match else None,
            "Contribute enough to capture the full employer match: it is an immediate return.",
            "Employer match captured." if has_match else "No employer match available.",
        ),
        (
            Step.HIGH_INTEREST_DEBT, bool(high), False,
            _total([d.balance for d in high], cur),
            f"Pay down {len(high)} debt(s) at {HIGH_INTEREST_APR:.0%} APR or more; "
            "their interest outweighs likely investment returns.",
            "No high-interest debt.",
        ),
        (
            Step.FULL_EMERGENCY_FUND, True, liquid >= full_target,
            _positive(full_target - liquid),
            f"Grow the emergency fund to {full_target} ({state.emergency_target_months} months); "
            f"you have {liquid}.",
            f"Emergency fund fully funded ({liquid} vs {full_target}).",
        ),
        (
            Step.OTHER_DEBT, bool(low), False,
            _total([d.balance for d in low], cur),
            f"Pay down {len(low)} lower-interest debt(s) after the emergency fund is in place.",
            "No other debt.",
        ),
        (
            Step.GOALS_AND_RETIREMENT, True, False, None,
            "Direct what is left to goals and retirement contributions.",
            "",
        ),
    ]

    items: list[PriorityItem] = []
    blocked = False
    for step, applicable, done, need, open_reason, closed_reason in rows:
        if not applicable:
            status: Status = "skipped"
            reason = closed_reason
        elif done:
            status, reason = "done", closed_reason
        elif not blocked:
            status, reason = "active", open_reason
            blocked = True
        else:
            status, reason = "pending", open_reason
        items.append(PriorityItem(step, status, reason, rule_id(step), None if status in ("done", "skipped") else need))
    return items


def next_focus(items: Sequence[PriorityItem]) -> PriorityItem | None:
    """The first step still open (always exists because goals never finish)."""
    return next((i for i in items if i.status == "active"), None)


def allocate_surplus(items: Sequence[PriorityItem], surplus: Money) -> Allocation:
    """Waterfall a monthly surplus down the open steps in priority order."""
    remaining = _positive(surplus)
    lines: list[tuple[Step, Money]] = []
    for item in items:
        if item.status not in ("active", "pending") or remaining.is_zero():
            continue
        amount = remaining if item.need is None else min(remaining, item.need)
        if amount.minor > 0:
            lines.append((item.step, amount))
            remaining = remaining - amount
    return Allocation(lines=tuple(lines), leftover=remaining)