"""Intent routing.

Deterministic keyword rules (no LLM, so it works offline and never hallucinates a route):
which planning areas a message is about, whether it describes a life event, and whether it
contains an investing proposal that should be checked against remembered constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.context.conflicts import Proposal

AGENT_ORDER = ("budget", "tax", "savings", "debt", "goals", "retirement", "risk")
_NEEDS_BUDGET = {"savings", "debt", "goals", "retirement"}  # these read the budget agent's shared findings

Intent = Literal["plan", "life_event", "area", "explain", "what_if"]

_LIFE_EVENTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("baby", re.compile(r"\b(baby|pregnan\w*|newborn|expecting a (child|baby)|having a (child|kid|baby))\b", re.I)),
    ("job_loss", re.compile(r"\b(laid off|lost my job|job loss|fired|redundan\w*|unemployed)\b", re.I)),
    ("new_job", re.compile(r"\b(new job|got a job|promotion|new role|switch\w* jobs?|job offer)\b", re.I)),
    ("relocation", re.compile(r"\b(moving|relocat\w*|new city|move to)\b", re.I)),
    ("medical_expense", re.compile(r"\b(surgery|hospital|medical (bill|expense))\b", re.I)),
    ("windfall", re.compile(r"\b(inherit\w*|windfall|bonus|lottery)\b", re.I)),
    ("marriage", re.compile(r"\b(getting married|got married|my wedding|marriage)\b", re.I)),
)
_AREAS: dict[str, re.Pattern[str]] = {
    "budget": re.compile(r"\b(budget|spend\w*|expens\w*|cash ?flow|income)\b", re.I),
    "debt": re.compile(r"\b(debts?|loans?|credit cards?|emi|mortgage|owe)\b", re.I),
    "savings": re.compile(r"\b(emergency|savings?|rainy day|buffer)\b", re.I),
    "goals": re.compile(r"\b(goals?|house|home|deposit|down payment|car|vacation|trip|education|college)\b", re.I),
    "tax": re.compile(r"\b(tax\w*|deduct\w*|regime|80c|401\(?k\)?|ira)\b", re.I),
    "retirement": re.compile(r"\b(retire\w*|pension|nps|corpus)\b", re.I),
    "risk": re.compile(r"\b(insurance|insure\w*|coverage|protect\w*)\b", re.I),
}
_WHAT_IF = re.compile(r"\b(what if|what-if|scenario|simulate|compare)\b", re.I)
_EXPLAIN = re.compile(r"\b(why|explain|how did you)\b", re.I)
_FULL_PLAN = re.compile(r"\b(plan|overview|review|everything|full|where do i stand|update)\b", re.I)
_INVESTING = re.compile(r"\b(invest\w*|stocks?|equit\w*|crypto|shares|mutual funds?|index funds?|aggressive)\b", re.I)
_RISKY = re.compile(r"\b(aggressive|crypto|all in|high[- ]risk)\b", re.I)
_PERCENT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%")


@dataclass(frozen=True)
class Route:
    intent: Intent
    agents: tuple[str, ...]  # requested agents, before dependency resolution
    life_event_kind: str | None = None
    proposal: Proposal | None = None


def resolve_agents(requested: tuple[str, ...] | list[str]) -> list[str]:
    """Order requested agents and add the budget agent when others depend on its findings."""
    wanted = set(requested)
    if wanted & _NEEDS_BUDGET:
        wanted.add("budget")
    return [a for a in AGENT_ORDER if a in wanted]


def detect_proposal(text: str) -> Proposal | None:
    """An investing proposal, only when the message states a size or a risky style."""
    if not _INVESTING.search(text):
        return None
    pct = _PERCENT.search(text)
    value = float(pct.group(1)) / 100 if pct and float(pct.group(1)) <= 100 else None
    risky = bool(_RISKY.search(text))
    if value is None and not risky:
        return None
    if risky or (value is not None and value >= 0.7):
        risk = "high"
    elif value is not None and value >= 0.4:
        risk = "medium"
    else:
        risk = "low"
    return Proposal("investing", text.strip(), value=value, risk=risk)  # type: ignore[arg-type]


def route(text: str) -> Route:
    proposal = detect_proposal(text)
    event = next((kind for kind, pat in _LIFE_EVENTS if pat.search(text)), None)
    if event:
        return Route("life_event", AGENT_ORDER, event, proposal)
    if _WHAT_IF.search(text):
        areas = tuple(a for a in AGENT_ORDER if _AREAS[a].search(text)) or AGENT_ORDER
        return Route("what_if", areas, None, proposal)
    if _EXPLAIN.search(text):
        return Route("explain", (), None, proposal)
    areas = tuple(a for a in AGENT_ORDER if _AREAS[a].search(text))
    if areas and not _FULL_PLAN.search(text):
        return Route("area", areas, None, proposal)
    if proposal and not areas:
        return Route("area", ("risk",), None, proposal)  # an investing request still gets a protection check
    return Route("plan", AGENT_ORDER, None, proposal)