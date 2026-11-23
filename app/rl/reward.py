"""Reward for the household simulator.

The weights are CONDITIONED ON MEMORY: a remembered low risk tolerance, income volatility and
dependents raise the penalty for a thin emergency fund, so the same life event yields different
best policies for different remembered users. Every step returns a component breakdown so the
Why? panel can say what drove a recommendation (wealth growth, goal success, safety net,
deficits, constraint pressure). All terms are normalised by baseline income, so one policy
serves any household and currency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Tolerance = Literal["low", "medium", "high"]
RISK_LEVEL: dict[str, float] = {"low": 0.0, "medium": 0.5, "high": 1.0}

_WEALTH = {"low": 0.6, "medium": 0.8, "high": 1.0}
_SHORTFALL = {"low": 0.10, "medium": 0.07, "high": 0.04}
GOAL_BONUS = 3.0
DEFICIT_PENALTY = 0.5
VIOLATION_PENALTY = 1.0


@dataclass(frozen=True)
class RewardWeights:
    wealth: float
    shortfall: float
    goal: float = GOAL_BONUS
    deficit: float = DEFICIT_PENALTY
    violation: float = VIOLATION_PENALTY

    @classmethod
    def from_context(cls, tolerance: Tolerance, income_cv: float = 0.0, dependents: int = 0) -> RewardWeights:
        """Weights derived from the remembered risk profile, income volatility and dependents."""
        shortfall = _SHORTFALL[tolerance] + 0.015 * dependents + 0.05 * income_cv
        return cls(wealth=_WEALTH[tolerance], shortfall=shortfall)


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    wealth: float
    shortfall: float
    goals: float
    deficit: float
    violation: float

    def as_dict(self) -> dict[str, float]:
        return {"total": self.total, "wealth": self.wealth, "shortfall": self.shortfall,
                "goals": self.goals, "deficit": self.deficit, "violation": self.violation}


def step_reward(
    *,
    prev_wealth: float,
    cur_wealth: float,
    income_scale: float,
    shortfall_ratio: float,
    resolved_goal_fractions: Sequence[float],
    n_goals: int,
    deficit: bool,
    violation: float,
    weights: RewardWeights,
) -> RewardBreakdown:
    """One month's reward.

    ``wealth`` values are adjusted net worth (net worth plus goal payouts already made).
    ``shortfall_ratio`` is how far liquid savings sit below the emergency target (0 to 1).
    ``resolved_goal_fractions`` lists the funded fraction of each goal whose deadline passed.
    """
    wealth = weights.wealth * (cur_wealth - prev_wealth) / income_scale
    shortfall = -weights.shortfall * shortfall_ratio
    goals = weights.goal * sum(resolved_goal_fractions) / max(1, n_goals)
    deficit_term = -weights.deficit if deficit else 0.0
    violation_term = -weights.violation * violation
    total = wealth + shortfall + goals + deficit_term + violation_term
    return RewardBreakdown(total, wealth, shortfall, goals, deficit_term, violation_term)