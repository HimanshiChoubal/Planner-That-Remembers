"""Memory conflict detection.

Surfaces contradictions between what the user asks for now and what they told us before:
a proposal that breaks a stated limit, exceeds their risk tolerance, or repeats something they
rejected. Also flags when a new memory changes an old one (risk profile, goal, income pattern).
Numeric limits are checked exactly; free-text rules can only be flagged for confirmation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.context.bus import RecalledMemory
from app.context.schemas import (
    Constraint,
    Goal,
    IncomePattern,
    MemoryModel,
    RejectedDecision,
    RiskProfile,
)

Severity = Literal["block", "warn", "info"]
Topic = Literal["investing", "debt", "spending", "saving", "other"]
Tolerance = Literal["low", "medium", "high"]

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
_SEVERITY_ORDER = {"block": 0, "warn": 1, "info": 2}
_SIMILARITY_THRESHOLD = 0.5
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and the to of for in on my our with is it be this that".split())


@dataclass(frozen=True)
class Proposal:
    """What is being proposed or requested, in structured form."""

    topic: Topic
    description: str
    value: float | None = None  # numeric size, e.g. proposed equity share 0-1
    risk: Tolerance | None = None


@dataclass(frozen=True)
class Conflict:
    kind: str
    severity: Severity
    message: str
    memory_ids: tuple[str, ...]
    existing: MemoryModel


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP}


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def check_proposal(proposal: Proposal, memories: Sequence[RecalledMemory]) -> list[Conflict]:
    """Conflicts between ``proposal`` and remembered constraints, risk profile and rejections."""
    found: list[Conflict] = []
    for mem in memories:
        model = mem.model
        rid = (mem.record.id,)

        if isinstance(model, Constraint) and model.subject == proposal.topic:
            if model.max_value is not None and proposal.value is not None:
                if proposal.value > model.max_value:
                    found.append(Conflict(
                        "constraint_violated",
                        "block" if model.hard else "warn",
                        f"You set a limit: \"{model.description}\" (max {model.max_value:g}), "
                        f"but this proposal is {proposal.value:g}.",
                        rid, model))
            elif model.hard:
                found.append(Conflict(
                    "constraint_related", "warn",
                    f"You told me: \"{model.description}\". This is in the same area, so please "
                    "confirm the proposal respects it.",
                    rid, model))

        elif isinstance(model, RiskProfile):
            if proposal.risk and _RISK_ORDER[proposal.risk] > _RISK_ORDER[model.tolerance]:
                found.append(Conflict(
                    "risk_mismatch", "warn",
                    f"Your saved risk tolerance is {model.tolerance}, but this proposal is "
                    f"{proposal.risk} risk.",
                    rid, model))
            if (
                proposal.topic == "investing"
                and proposal.value is not None
                and model.max_equity_share is not None
                and proposal.value > model.max_equity_share
            ):
                found.append(Conflict(
                    "equity_above_profile", "warn",
                    f"Proposed equity share {proposal.value:.0%} exceeds your saved maximum of "
                    f"{model.max_equity_share:.0%}.",
                    rid, model))

        elif isinstance(model, RejectedDecision):
            if _similarity(model.summary, proposal.description) >= _SIMILARITY_THRESHOLD:
                found.append(Conflict(
                    "previously_rejected", "warn",
                    f"You rejected something similar before (\"{model.summary}\"). "
                    f"Your reason: {model.reason}",
                    rid, model))

    return sorted(found, key=lambda c: _SEVERITY_ORDER[c.severity])


def detect_memory_conflicts(new: MemoryModel, memories: Sequence[RecalledMemory]) -> list[Conflict]:
    """Informational conflicts when ``new`` changes something previously remembered."""
    found: list[Conflict] = []
    for mem in memories:
        old, rid = mem.model, (mem.record.id,)
        if isinstance(new, RiskProfile) and isinstance(old, RiskProfile):
            if new.tolerance != old.tolerance:
                found.append(Conflict(
                    "risk_profile_changed", "info",
                    f"Risk tolerance changed from {old.tolerance} to {new.tolerance}.", rid, old))
        elif isinstance(new, Goal) and isinstance(old, Goal):
            if new.name.strip().lower() == old.name.strip().lower() and (
                new.target != old.target or new.deadline != old.deadline
            ):
                found.append(Conflict(
                    "goal_changed", "info",
                    f"The goal \"{old.name}\" was already saved with a different target or deadline.",
                    rid, old))
        elif isinstance(new, IncomePattern) and isinstance(old, IncomePattern):
            if new.kind != old.kind:
                found.append(Conflict(
                    "income_pattern_changed", "info",
                    f"Income pattern changed from {old.kind} to {new.kind}.", rid, old))
    return sorted(found, key=lambda c: _SEVERITY_ORDER[c.severity])


def blocking(conflicts: Sequence[Conflict]) -> list[Conflict]:
    return [c for c in conflicts if c.severity == "block"]