"""What-if scenario planner.

Runs the household simulator under scripted life events and compares planning strategies,
reporting ranges (10th / 50th / 90th percentile) rather than single numbers. Random life events
are switched off so the effect of the scripted event is isolated; market returns and income
noise still vary across episodes. Results are SIMULATION-based on synthetic assumptions.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from app.rl.baselines import BASELINES, Policy, summarize_episode
from app.rl.env import HouseholdEnv, HouseholdParams
from app.rl.events import ScriptedEvent, baby, income_jump, job_loss, medical

PRESETS: dict[str, tuple[str, tuple[ScriptedEvent, ...]]] = {
    "baseline": ("No life event", ()),
    "job_loss": ("Lose your job in month 12 (6 months at 20% income)", (ScriptedEvent(12, job_loss(6, 0.2)),)),
    "baby": ("Baby arrives in month 6", (ScriptedEvent(6, baby()),)),
    "medical": ("Large medical bill in month 9 (about 3 months of income)", (ScriptedEvent(9, medical(3.0)),)),
    "raise": ("20% raise in month 12", (ScriptedEvent(12, income_jump(0.2)),)),
}
STRATEGIES = ("waterfall", "debt_first", "fixed_split", "goals_first")


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def _run(params: HouseholdParams, scripted: tuple[ScriptedEvent, ...], policy: Policy, seed: int) -> tuple[Any, list[float]]:
    env = HouseholdEnv(params, scripted=scripted, random_events=False)
    env.reset(seed=seed)
    wealth = [env.adjusted_wealth()]
    total, done = 0.0, False
    while not done:
        _, reward, done, _, _ = env.step(policy(env))
        total += reward
        wealth.append(env.adjusted_wealth())
    return summarize_episode(env, total), wealth


def simulate(
    params: HouseholdParams,
    scenario_ids: Sequence[str],
    *,
    strategies: Sequence[str] = STRATEGIES,
    episodes: int = 30,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare strategies across scenarios. Money values are in the household's currency."""
    unknown = [s for s in scenario_ids if s not in PRESETS] + [s for s in strategies if s not in BASELINES]
    if unknown:
        raise ValueError(f"Unknown scenario or strategy: {', '.join(unknown)}")
    scenario_ids = list(dict.fromkeys(["baseline", *scenario_ids]))  # always compare against baseline
    out: dict[str, Any] = {"currency": params.currency, "horizon_months": params.horizon, "episodes": episodes,
                           "note": "Simulation on synthetic assumptions. Not financial advice.", "scenarios": {}}
    for sid in scenario_ids:
        label, scripted = PRESETS[sid]
        per_strategy: dict[str, Any] = {}
        for name in strategies:
            results, curves = [], []
            for ep in range(episodes):
                res, wealth = _run(params, scripted, BASELINES[name], seed + ep)
                results.append(res)
                curves.append(wealth)
            arr = np.array(curves)
            per_strategy[name] = {
                "final_net_worth": _percentiles([r.final_net_worth for r in results]),
                "goals_funded": float(np.mean([r.goals_funded_fraction for r in results])),
                "min_runway_months": float(np.mean([min(r.min_runway_months, 60.0) for r in results])),
                "credit_months": float(np.mean([r.deficit_months for r in results])),
                "interest_paid": float(np.mean([r.interest_paid for r in results])),
                "trajectory": {k: np.percentile(arr, q, axis=0).round(2).tolist() for k, q in (("p10", 10), ("p50", 50), ("p90", 90))},
            }
        out["scenarios"][sid] = {"label": label, "strategies": per_strategy}

    base = out["scenarios"]["baseline"]["strategies"]
    for sid, sc in out["scenarios"].items():  # impact versus the no-event baseline, per strategy
        for name, st in sc["strategies"].items():
            ref = base[name]
            st["impact_vs_baseline"] = {
                "net_worth_p50": st["final_net_worth"]["p50"] - ref["final_net_worth"]["p50"],
                "min_runway_months": st["min_runway_months"] - ref["min_runway_months"],
                "credit_months": st["credit_months"] - ref["credit_months"],
            }
    return out