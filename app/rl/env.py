"""Household finance simulator (Gymnasium environment).

Monthly steps in REAL (inflation-adjusted) terms. Each step: life events fire, income is drawn,
debt interest accrues, minimum payments are made, and the month's surplus is allocated by the
policy to four buckets (extra debt payment, emergency fund, goals, retirement) along with an
equity share for invested money. Deficits draw down savings, then fall onto an expensive
credit line. Goals resolve at their deadlines.

Observations are ratios to baseline income and are conditioned on remembered context (risk
level, income volatility, dependents, equity cap), so one trained policy serves every user.
Actions pass through ``project_action``, so hard constraints can never be violated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.agents.base import FinancialProfile
from app.context.schemas import Constraint, Goal, MemoryModel, RiskProfile
from app.engine import emergency_fund as ef
from app.engine import forecast as fc
from app.engine import retirement as rt
from app.rl.events import Event, EventRates, ScriptedEvent, sample_event
from app.rl.projection import Constraints, Projected, project_action
from app.rl.reward import RISK_LEVEL, RewardBreakdown, RewardWeights, step_reward

CREDIT_APR = 0.25  # expensive credit line that absorbs deficits
CREDIT_MIN_RATE = 0.03
HIGH_INTEREST_APR = 0.10
OBS_DIM = 16
STARTER_FLOOR_MONTHS = 1.0


@dataclass(frozen=True)
class DebtSpec:
    balance: float
    apr: float
    min_payment: float


@dataclass(frozen=True)
class GoalSpec:
    target: float
    deadline_month: int  # months from now
    saved: float
    priority: int = 3


@dataclass(frozen=True)
class HouseholdParams:
    """Plain floats in major currency units (the simulator is a statistical model)."""

    currency: str
    income: float  # baseline monthly income
    income_cv: float
    expenses: float  # monthly spending including minimum debt payments
    essential: float  # essential spending including minimum debt payments
    liquid: float
    debts: tuple[DebtSpec, ...]
    goals: tuple[GoalSpec, ...]
    retirement: float
    risk: str  # "low" | "medium" | "high"
    max_equity: float
    dependents: int
    emergency_months: float
    assumptions: rt.Assumptions
    horizon: int = 60

    @property
    def living_expenses(self) -> float:
        return self.expenses - sum(d.min_payment for d in self.debts)

    def reward_weights(self) -> RewardWeights:
        return RewardWeights.from_context(self.risk, self.income_cv, self.dependents)  # type: ignore[arg-type]


def _months_between(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def params_from_profile(
    profile: FinancialProfile,
    models: Sequence[MemoryModel],
    *,
    region: str,
    as_of: date,
    horizon: int = 60,
) -> HouseholdParams:
    """Build simulator parameters from the structured profile and remembered context."""
    if (profile.monthly_gross_income is None or profile.monthly_expenses is None
            or profile.monthly_essential_expenses is None or profile.assets is None):
        raise ValueError("Simulator needs income, expenses, essential expenses and assets")
    cv = fc.coefficient_of_variation(profile.income_history)
    income = float(profile.monthly_gross_income.amount)
    if len(profile.income_history) >= fc.MIN_OBSERVATIONS:
        income = sum(float(m.amount) for m in profile.income_history) / len(profile.income_history)

    risk_models = [m for m in models if isinstance(m, RiskProfile)]
    tolerance = risk_models[-1].tolerance if risk_models else "medium"
    caps = [m.max_equity_share for m in risk_models if m.max_equity_share is not None]
    caps += [c.max_value for c in models if isinstance(c, Constraint)
             and c.subject == "investing" and c.hard and c.max_value is not None]
    liquid = profile.liquid_savings
    retirement = sum(float(a.value.amount) for a in profile.assets if a.kind == "retirement")
    goals = tuple(
        GoalSpec(float(g.target.to_money().amount), max(1, _months_between(as_of, g.deadline)),
                 float(g.saved.to_money().amount) if g.saved else 0.0, g.priority)
        for g in models if isinstance(g, Goal) and g.deadline is not None and g.deadline > as_of)
    months, _ = ef.target_months(income_cv=cv, dependents=profile.dependents, single_income=profile.single_income)
    return HouseholdParams(
        currency=profile.currency.code, income=income, income_cv=cv,
        expenses=float(profile.monthly_expenses.amount), essential=float(profile.monthly_essential_expenses.amount),
        liquid=float(liquid.amount) if liquid else 0.0,
        debts=tuple(DebtSpec(float(d.balance.amount), float(d.apr), float(d.min_payment.amount)) for d in (profile.debts or ())),
        goals=goals, retirement=retirement, risk=tolerance, max_equity=min(caps) if caps else 1.0,
        dependents=profile.dependents, emergency_months=float(months), assumptions=rt.assumptions_for(region),
        horizon=horizon)


@dataclass
class SimState:
    t: int
    liquid: float
    balances: np.ndarray  # one slot per debt, plus a final credit-line slot
    goals_balance: float
    retirement: float
    perm_income: float = 1.0
    temp_income: float = 1.0
    temp_remaining: int = 0
    expense_add: float = 0.0
    dependents: int = 0
    goals_done: list[bool] = field(default_factory=list)
    goal_paid: float = 0.0
    interest_paid: float = 0.0
    deficit_months: int = 0
    violations: float = 0.0
    min_runway: float = float("inf")


class HouseholdEnv(gym.Env):
    metadata: dict[str, Any] = {}

    def __init__(
        self,
        params: HouseholdParams,
        *,
        rates: EventRates | None = None,
        scripted: Sequence[ScriptedEvent] = (),
        random_events: bool = True,
    ) -> None:
        super().__init__()
        self.params = params
        self.rates = rates or EventRates()
        self.scripted = tuple(scripted)
        self.random_events = random_events
        self.constraints = Constraints(params.max_equity, STARTER_FLOOR_MONTHS)
        self.weights = params.reward_weights()
        self.apr = np.array([d.apr for d in params.debts] + [CREDIT_APR])
        self.mins = np.array([d.min_payment for d in params.debts] + [0.0])
        self.goal_order = sorted(range(len(params.goals)), key=lambda i: params.goals[i].deadline_month)
        self.observation_space = spaces.Box(-1.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        self.state: SimState = self._initial_state()

    # ----------------------------------------------------------------- helpers
    def _initial_state(self) -> SimState:
        p = self.params
        return SimState(
            t=0, liquid=p.liquid, balances=np.array([d.balance for d in p.debts] + [0.0]),
            goals_balance=sum(g.saved for g in p.goals), retirement=p.retirement,
            dependents=p.dependents, goals_done=[False] * len(p.goals))

    def essential_now(self) -> float:
        return self.params.essential + self.state.expense_add

    def emergency_target(self) -> float:
        return self.essential_now() * self.params.emergency_months

    def adjusted_wealth(self) -> float:
        s = self.state
        return s.liquid + s.goals_balance + s.retirement - float(s.balances.sum()) + s.goal_paid

    def high_interest_debt(self) -> float:
        return float(self.state.balances[self.apr >= HIGH_INTEREST_APR].sum())

    def pending_goal_targets(self) -> float:
        return sum(g.target for g, done in zip(self.params.goals, self.state.goals_done, strict=True) if not done)

    def _obs(self) -> np.ndarray:
        p, s = self.params, self.state
        ess = max(self.essential_now(), 1e-9)
        target = max(self.emergency_target(), 1e-9)
        debt = float(s.balances.sum())
        pending = self.pending_goal_targets()
        open_deadlines = [g.deadline_month - s.t for g, d in zip(p.goals, s.goals_done, strict=True) if not d]
        avg_apr = float((s.balances * self.apr).sum() / debt) if debt > 1e-9 else 0.0
        income_now = p.income * s.perm_income * (s.temp_income if s.temp_remaining > 0 else 1.0)
        feats = [
            s.liquid / ess / 12.0,
            s.liquid / target,
            debt / (12.0 * p.income),
            avg_apr,
            self.high_interest_debt() / debt if debt > 1e-9 else 0.0,
            min(1.0, s.goals_balance / pending) if pending > 1e-9 else 1.0,
            min(1.0, max(0.0, min(open_deadlines) / p.horizon)) if open_deadlines else 1.0,
            s.retirement / (12.0 * p.income) / 10.0,
            (income_now - self.params.living_expenses - s.expense_add - float(np.minimum(self.mins, s.balances).sum())) / p.income,
            income_now / p.income,
            1.0 - s.t / p.horizon,
            RISK_LEVEL[p.risk],
            min(1.0, p.income_cv),
            min(1.0, s.dependents / 4.0),
            p.max_equity,
            1.0 if (s.temp_remaining > 0 and s.temp_income < 0.5) else 0.0,
        ]
        return np.clip(np.array(feats, dtype=np.float32), -1.0, 10.0)

    # --------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.state = self._initial_state()
        return self._obs(), {}

    def step(self, action: np.ndarray):
        p, s, rng = self.params, self.state, self.np_random
        prev_wealth = self.adjusted_wealth()

        # 1. life events
        events: list[Event] = [se.event for se in self.scripted if se.month == s.t]
        if self.random_events and (ev := sample_event(rng, self.rates, p.income_cv)) is not None:
            events.append(ev)
        one_off = 0.0
        for ev in events:
            if ev.duration > 0:
                s.temp_income, s.temp_remaining = ev.temp_income_factor, ev.duration
            s.perm_income *= ev.perm_income_factor
            s.expense_add += ev.perm_expense_fraction * p.income
            one_off += ev.one_off_fraction * p.income
            s.dependents += ev.dependents_delta

        # 2. income, interest and minimum payments
        sigma = float(np.sqrt(np.log(1.0 + p.income_cv**2)))
        noise = float(rng.lognormal(-sigma**2 / 2.0, sigma)) if sigma > 0 else 1.0
        income = p.income * s.perm_income * (s.temp_income if s.temp_remaining > 0 else 1.0) * noise
        if s.temp_remaining > 0:
            s.temp_remaining -= 1
        interest = s.balances * self.apr / 12.0
        s.balances = s.balances + interest
        s.interest_paid += float(interest.sum())
        mins = np.minimum(np.where(np.arange(len(s.balances)) == len(s.balances) - 1,
                                   CREDIT_MIN_RATE * s.balances, self.mins), s.balances)
        s.balances = s.balances - mins
        cash = income - (p.living_expenses + s.expense_add) - float(mins.sum()) - one_off

        # 3. allocate a surplus, or cover a deficit
        deficit = False  # True only when savings can't cover the gap and credit is used
        if cash < 0:
            need = -cash
            from_savings = min(s.liquid, need)
            s.liquid -= from_savings
            s.balances[-1] += need - from_savings
            deficit = need > from_savings
            s.deficit_months += int(deficit)
            projected = project_action(np.asarray(action, dtype=float), surplus=0.0, liquid=s.liquid,
                                       essential=self.essential_now(), debt_total=float(s.balances.sum()),
                                       constraints=self.constraints)
        else:
            projected = project_action(np.asarray(action, dtype=float), surplus=cash, liquid=s.liquid,
                                       essential=self.essential_now(), debt_total=float(s.balances.sum()),
                                       constraints=self.constraints)
            self._apply_allocation(projected)
        s.violations += projected.violation

        # 4. investment growth (real terms) on goals and retirement money
        mu, sigma_a = rt.portfolio_params(projected.equity_share, p.assumptions)
        real_mu = mu - p.assumptions.inflation_mean
        r = max(float(rng.normal(real_mu / 12.0, sigma_a / np.sqrt(12.0))), -0.5)
        s.goals_balance *= 1.0 + r
        s.retirement *= 1.0 + r

        # 5. advance time, resolve goals, reward
        s.t += 1
        fractions = self._resolve_goals()
        runway = s.liquid / max(self.essential_now(), 1e-9)
        s.min_runway = min(s.min_runway, runway)
        target = self.emergency_target()
        breakdown: RewardBreakdown = step_reward(
            prev_wealth=prev_wealth, cur_wealth=self.adjusted_wealth(), income_scale=p.income,
            shortfall_ratio=max(0.0, 1.0 - s.liquid / target) if target > 0 else 0.0,
            resolved_goal_fractions=fractions, n_goals=len(p.goals), deficit=deficit,
            violation=projected.violation, weights=self.weights)
        terminated = s.t >= p.horizon
        info = {"reward": breakdown.as_dict(), "allocation": projected.allocation, "equity_share": projected.equity_share,
                "notes": projected.notes, "events": [e.kind for e in events], "net_worth": self.adjusted_wealth(),
                "deficit": deficit}
        return self._obs(), float(breakdown.total), terminated, False, info

    # --------------------------------------------------------------- mechanics
    def _apply_allocation(self, projected: Projected) -> None:
        s, a = self.state, projected.allocation
        remaining = a.debt
        for i in np.argsort(-self.apr):  # extra payments go to the highest APR first
            pay = min(remaining, float(s.balances[i]))
            s.balances[i] -= pay
            remaining -= pay
        s.liquid += a.emergency + remaining  # float residue stays in cash
        s.goals_balance += a.goals
        s.retirement += a.retirement

    def _resolve_goals(self) -> list[float]:
        s = self.state
        fractions: list[float] = []
        for i in self.goal_order:
            g = self.params.goals[i]
            if not s.goals_done[i] and g.deadline_month <= s.t:
                payout = min(s.goals_balance, g.target)
                s.goals_balance -= payout
                s.goal_paid += payout
                s.goals_done[i] = True
                fractions.append(payout / g.target if g.target > 0 else 1.0)
        return fractions