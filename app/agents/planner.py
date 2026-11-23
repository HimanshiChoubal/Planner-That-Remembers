"""Plan assembly and diffing.

``build_plan`` merges agent results into one JSON-safe payload (saved as a versioned Meko
artifact). Agents that were not re-run keep their previous section and recommendations, so a
narrow question doesn't wipe the rest of the plan. ``diff_plans`` compares two payloads and
links each change back to the decision traces that explain it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from app.agents.base import AgentResult, FinancialProfile
from app.engine import forecast as fc
from app.engine import metrics as mt
from app.engine import networth as nw
from app.engine.money import Money, sum_money
from app.regions.base import RegionPack

DISCLAIMER = "Education and demonstration only. Synthetic data. Not financial advice."
AGENT_ORDER = ("budget", "tax", "savings", "debt", "goals", "retirement", "risk")
_MONEY_KEYS = {"minor", "currency", "display"}
_NUMBER_RUN = re.compile(r"[\d$\u20b9,.%]+")


def _rec_key(rec: Mapping[str, Any]) -> tuple[str, str]:
    """Stable identity for a recommendation across plan versions: its agent plus its title with
    numbers masked, so "to 4 months" and "to 4.5 months" count as the same (changed) item."""
    return rec["agent"], _NUMBER_RUN.sub("#", rec["title"])


def headline_metrics(profile: FinancialProfile, as_of: date) -> list[dict[str, Any]]:
    """Dashboard metrics, or an empty list when the profile lacks the inputs."""
    if profile.monthly_gross_income is None or profile.monthly_expenses is None or profile.assets is None:
        return []
    cur = profile.currency
    debts = profile.debts or ()
    liabilities = [nw.Liability(d.name, d.balance, d.apr, d.min_payment) for d in debts]
    hm = mt.compute_metrics(
        currency=cur, as_of=as_of, monthly_gross_income=profile.monthly_gross_income,
        monthly_expenses=profile.monthly_expenses,
        monthly_essential=profile.monthly_essential_expenses or Money.zero(cur),
        monthly_debt_payments=sum_money([d.min_payment for d in debts], cur),
        assets=profile.assets, liabilities=liabilities,
        income_cv=fc.coefficient_of_variation(profile.income_history),
        dependents=profile.dependents, single_income=profile.single_income)
    return [{"key": m.key, "label": m.label, "display": m.display, "status": m.status, "rule_id": m.rule_id}
            for m in hm.metrics]


def build_plan(
    results: Mapping[str, AgentResult],
    *,
    profile: FinancialProfile,
    pack: RegionPack,
    as_of: date,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sections: dict[str, Any] = dict(previous["sections"]) if previous else {}
    recs = [r for r in (previous["recommendations"] if previous else []) if r["agent"] not in results]
    questions = [q for q in (previous["open_questions"] if previous else []) if q["agent"] not in results]

    for agent, result in results.items():
        if result.status == "ok":
            sections[agent] = dict(result.plan)
        else:
            sections.pop(agent, None)  # inputs changed and it can't run: don't keep a stale section
        for r in result.recommendations:
            recs.append({"id": r.id, "agent": r.agent, "title": r.title, "summary": r.summary,
                         "priority": r.priority, "trace_id": r.trace_id, "metrics": dict(r.metrics)})
        for g in result.gaps:
            questions.append({"agent": g.agent, "key": g.key, "question": g.question, "blocking": g.blocking})

    order = {name: i for i, name in enumerate(AGENT_ORDER)}
    recs.sort(key=lambda r: (r["priority"], order.get(r["agent"], 99)))
    return {
        "as_of": as_of.isoformat(),
        "region": pack.code,
        "currency": profile.currency.code,
        "sections": sections,
        "recommendations": recs,
        "open_questions": questions,
        "headline": headline_metrics(profile, as_of),
        "disclaimer": DISCLAIMER,
    }


# --------------------------------------------------------------------------- diff
@dataclass(frozen=True)
class PlanChange:
    agent: str
    path: str
    kind: Literal["changed", "added", "removed"]
    before: str | None
    after: str | None


@dataclass(frozen=True)
class RecChange:
    kind: Literal["added", "removed", "changed"]
    agent: str
    title: str
    before: str | None
    after: str | None
    trace_id: str | None  # trace explaining the new/changed recommendation


@dataclass(frozen=True)
class PlanDiff:
    changes: tuple[PlanChange, ...]
    recommendations: tuple[RecChange, ...]

    @property
    def is_empty(self) -> bool:
        return not self.changes and not self.recommendations

    def trace_ids(self, agent: str | None = None) -> tuple[str, ...]:
        return tuple(r.trace_id for r in self.recommendations
                     if r.trace_id and (agent is None or r.agent == agent))


def _flatten(value: Any, prefix: str = "") -> dict[str, str]:
    """Leaf values keyed by path; a money dict is one leaf shown by its display string."""
    if isinstance(value, dict):
        if value and set(value) == _MONEY_KEYS:
            return {prefix: str(value["display"])}
        out: dict[str, str] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(value, list):
        out = {}
        for i, v in enumerate(value):
            out.update(_flatten(v, f"{prefix}[{i}]"))
        return out
    return {prefix: "" if value is None else str(value)}


def diff_plans(old: Mapping[str, Any], new: Mapping[str, Any]) -> PlanDiff:
    changes: list[PlanChange] = []
    for agent in sorted(set(old["sections"]) | set(new["sections"]), key=lambda a: AGENT_ORDER.index(a) if a in AGENT_ORDER else 99):
        before = _flatten(old["sections"].get(agent, {}))
        after = _flatten(new["sections"].get(agent, {}))
        for path in sorted(set(before) | set(after)):
            b, a = before.get(path), after.get(path)
            if b == a:
                continue
            kind: Literal["changed", "added", "removed"] = "added" if b is None else "removed" if a is None else "changed"
            changes.append(PlanChange(agent, f"{agent}.{path}", kind, b, a))

    old_recs = {_rec_key(r): r for r in old["recommendations"]}
    new_recs = {_rec_key(r): r for r in new["recommendations"]}
    rec_changes: list[RecChange] = []
    for key, r in new_recs.items():
        if key not in old_recs:
            rec_changes.append(RecChange("added", r["agent"], r["title"], None, r["summary"], r["trace_id"]))
        elif old_recs[key]["summary"] != r["summary"] or old_recs[key]["title"] != r["title"]:
            rec_changes.append(RecChange("changed", r["agent"], r["title"], old_recs[key]["summary"], r["summary"], r["trace_id"]))
    for key, r in old_recs.items():
        if key not in new_recs:
            rec_changes.append(RecChange("removed", r["agent"], r["title"], r["summary"], None, None))
    return PlanDiff(tuple(changes), tuple(rec_changes))