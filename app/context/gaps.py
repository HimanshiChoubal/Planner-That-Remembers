"""Context-gap detection.

Each planning agent declares what it needs to know. If a requirement is missing (or only held
with low confidence), the agent asks the user instead of guessing. A requirement is satisfied
by a stored memory of a given kind, or by a known profile fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from app.context.bus import RecalledMemory
from app.context.schemas import MemoryKind
from app.meko.interfaces import ContextResult

MIN_CONFIDENCE = 0.5
MAX_MISSING_TERMS = 3


@dataclass(frozen=True)
class Requirement:
    key: str
    question: str
    memory_kind: MemoryKind | None = None  # satisfied by a stored memory of this kind
    profile_key: str | None = None  # or by a known (non-None) profile fact


@dataclass(frozen=True)
class Gap:
    key: str
    question: str
    reason: Literal["missing", "low_confidence", "weak_retrieval"]
    agent: str

    @property
    def blocking(self) -> bool:
        """Missing information blocks an agent; low confidence only warns."""
        return self.reason != "low_confidence"


REQUIREMENTS: dict[str, tuple[Requirement, ...]] = {
    "budget": (
        Requirement("income", "What is your typical monthly income, and does it vary much?",
                    MemoryKind.INCOME_PATTERN, "monthly_income"),
        Requirement("expenses", "Roughly how much do you spend each month?",
                    None, "monthly_expenses"),
    ),
    "debt": (
        Requirement("debts", "Do you have any debts? Please share balances, interest rates and minimum payments.",
                    None, "debts"),
    ),
    "savings": (
        Requirement("essential_expenses", "How much of your monthly spending is essential (housing, food, bills)?",
                    None, "monthly_essential_expenses"),
        Requirement("liquid_savings", "How much do you hold in cash or easily accessible savings?",
                    None, "liquid_savings"),
    ),
    "goals": (
        Requirement("goal", "What are you saving toward, and by when?", MemoryKind.GOAL, None),
    ),
    "tax": (
        Requirement("income", "What is your yearly income before tax?", None, "monthly_income"),
        Requirement("tax_regime", "Which tax regime do you file under?", None, "tax_regime"),
    ),
    "retirement": (
        Requirement("age", "How old are you?", None, "age"),
        Requirement("retirement_age", "At what age would you like to retire?", None, "retirement_age"),
        Requirement("expenses", "Roughly how much do you spend each month?", None, "monthly_expenses"),
        Requirement("assets", "What have you saved so far, including retirement accounts?", None, "assets"),
        Requirement("risk", "How comfortable are you with investment ups and downs: low, medium or high?",
                    MemoryKind.RISK_PROFILE, None),
    ),
    "risk": (
        Requirement("insurance", "Do you have health and life insurance coverage?", None, "insurance_covered"),
    ),
}


def find_gaps(
    agent: str,
    memories: Sequence[RecalledMemory],
    profile_facts: Mapping[str, Any],
    *,
    min_confidence: float = MIN_CONFIDENCE,
) -> list[Gap]:
    """Unmet requirements for ``agent``, in the order the requirements are declared."""
    gaps: list[Gap] = []
    for req in REQUIREMENTS.get(agent, ()):
        if req.profile_key and profile_facts.get(req.profile_key) is not None:
            continue
        if req.memory_kind is not None:
            held = [m for m in memories if m.record.kind == req.memory_kind.value]
            if any(m.record.confidence >= min_confidence for m in held):
                continue
            reason: Literal["missing", "low_confidence"] = "low_confidence" if held else "missing"
        else:
            reason = "missing"
        gaps.append(Gap(req.key, req.question, reason, agent))
    return gaps


def retrieval_gap(agent: str, result: ContextResult, threshold: float) -> Gap | None:
    """A gap when a free-form retrieval came back too weak to answer from."""
    if result.quality >= threshold:
        return None
    terms = ", ".join(result.missing[:MAX_MISSING_TERMS])
    detail = f" about {terms}" if terms else ""
    return Gap(
        "retrieval",
        f"I couldn't find enough saved context{detail}. Can you tell me more?",
        "weak_retrieval",
        agent,
    )


def blocking_gaps(gaps: Sequence[Gap]) -> list[Gap]:
    return [g for g in gaps if g.blocking]


def questions(gaps: Sequence[Gap]) -> list[str]:
    return [g.question for g in gaps]