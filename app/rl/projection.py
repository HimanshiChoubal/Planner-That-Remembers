"""Action projection.

Whatever a policy outputs, this layer maps it to a feasible allocation, so a recommendation
can never break a hard rule:

* allocations are non-negative and sum exactly to the month's surplus;
* the equity share never exceeds the user's maximum (their remembered hard limit);
* no money goes to goals or retirement while liquid savings are below the starter buffer;
* nothing is paid to debt beyond what is owed (the excess is redirected).

``violation`` measures how far the raw action was from feasible; the reward penalises it so
the policy learns to propose feasible actions, while the projection guarantees safety.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BUCKETS = ("debt", "emergency", "goals", "retirement")
_BLOCKED_PENALTY = 0.5  # weight on the share of the action blocked by the starter-buffer rule
_OVERPAY_PENALTY = 0.0  # redirecting an excess debt payment is harmless, so it is not penalised


@dataclass(frozen=True)
class Constraints:
    max_equity_share: float = 1.0
    starter_floor_months: float = 1.0  # months of essentials that must be liquid before investing


@dataclass(frozen=True)
class Allocation:
    debt: float
    emergency: float
    goals: float
    retirement: float

    def total(self) -> float:
        return self.debt + self.emergency + self.goals + self.retirement


@dataclass(frozen=True)
class Projected:
    allocation: Allocation
    equity_share: float
    violation: float  # >= 0; zero when the raw action was already feasible
    notes: tuple[str, ...]


def project_action(
    action: np.ndarray,
    *,
    surplus: float,
    liquid: float,
    essential: float,
    debt_total: float,
    constraints: Constraints,
) -> Projected:
    """Project a raw 5-dim action ``[w_debt, w_emergency, w_goals, w_retire, equity]``."""
    raw = np.asarray(action, dtype=float)
    if raw.shape != (5,):
        raise ValueError("action must have shape (5,)")
    notes: list[str] = []
    violation = 0.0

    equity = float(np.clip(raw[4], 0.0, 1.0))
    if equity > constraints.max_equity_share:
        violation += equity - constraints.max_equity_share
        notes.append(f"equity share capped at {constraints.max_equity_share:.0%}")
        equity = constraints.max_equity_share

    if surplus <= 0:
        return Projected(Allocation(0.0, 0.0, 0.0, 0.0), equity, violation, tuple(notes))

    w = np.clip(raw[:4], 0.0, None)
    w = w / w.sum() if w.sum() > 0 else np.array([0.0, 1.0, 0.0, 0.0])  # no preference: build the buffer

    floor = constraints.starter_floor_months * essential
    blocked = liquid < floor
    if blocked and (w[2] + w[3]) > 0:
        violation += _BLOCKED_PENALTY * float(w[2] + w[3])
        notes.append("goals and retirement paused until the starter buffer is in place")
        w = np.array([w[0], w[1] + w[2] + w[3], 0.0, 0.0])

    amounts = w * surplus
    if amounts[0] > debt_total:  # never pay more than is owed; redirect the excess
        excess = float(amounts[0] - debt_total)
        violation += _OVERPAY_PENALTY * excess / surplus
        notes.append("payment above the amount owed redirected")
        amounts[0] = debt_total
        rest = np.array([0.0, w[1], w[2], w[3]])
        if rest.sum() > 0:
            amounts += excess * rest / rest.sum()
        else:
            amounts[1] += excess

    amounts[int(np.argmax(amounts))] += surplus - float(amounts.sum())  # remove float residue
    return Projected(Allocation(*(float(a) for a in amounts)), equity, violation, tuple(notes))