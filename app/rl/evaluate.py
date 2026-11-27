"""Evaluation of policies on held-out households.

Every policy is run on the SAME households and random seeds (paired comparison), and reported as
its improvement over the waterfall baseline with a bootstrap 95% confidence interval and a win
rate. A result is only claimed when the interval excludes zero. ``MemoryMasked`` supports an
ablation: the same policy with the remembered-context inputs hidden.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from app.rl.baselines import EpisodeResult, run_episode
from app.rl.env import HouseholdEnv, HouseholdParams, params_from_profile

Policy = Callable[[HouseholdEnv], np.ndarray]


@dataclass(frozen=True)
class PolicyScore:
    name: str
    mean_reward: float
    improvement: float  # mean paired reward difference versus the baseline
    ci_low: float
    ci_high: float
    win_rate: float  # share of paired episodes with a higher reward than the baseline
    net_worth_months: float  # final adjusted net worth, in months of income
    interest_months: float  # interest paid, in months of income
    goals_funded: float  # mean funded fraction of goals that reached their deadline
    min_runway: float  # lowest emergency runway in months
    credit_months: float  # months that needed the credit line
    violations: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


@dataclass(frozen=True)
class EvalReport:
    label: str
    n_households: int
    n_episodes: int
    baseline: str
    scores: tuple[PolicyScore, ...]

    def to_markdown(self) -> str:
        head = (f"**{self.label}** ({self.n_households} households, {self.n_episodes} paired episodes; "
                f"improvement is versus `{self.baseline}`, 95% bootstrap CI)\n\n"
                "| Policy | Mean reward | Improvement | 95% CI | Win rate | Net worth (months of income) | "
                "Interest (months) | Goals funded | Min runway (months) | Credit months | Violations |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n")
        rows = []
        for s in self.scores:
            imp = "(baseline)" if s.name == self.baseline else f"{s.improvement:+.2f}{'*' if s.significant else ''}"
            ci = "-" if s.name == self.baseline else f"[{s.ci_low:+.2f}, {s.ci_high:+.2f}]"
            win = "-" if s.name == self.baseline else f"{s.win_rate:.0%}"
            rows.append(f"| {s.name} | {s.mean_reward:.2f} | {imp} | {ci} | {win} | {s.net_worth_months:.1f} | "
                        f"{s.interest_months:.2f} | {s.goals_funded:.0%} | {s.min_runway:.1f} | "
                        f"{s.credit_months:.2f} | {s.violations:.2f} |")
        return head + "\n".join(rows) + "\n\n`*` = the confidence interval excludes zero.\n"

    def to_dict(self) -> dict:
        return {"label": self.label, "n_households": self.n_households, "n_episodes": self.n_episodes,
                "baseline": self.baseline, "scores": [dataclasses.asdict(s) for s in self.scores]}


def evaluate(
    policies: Mapping[str, Policy],
    households: Sequence[HouseholdParams],
    *,
    label: str = "evaluation",
    episodes_per_household: int = 5,
    seed: int = 0,
    baseline: str = "waterfall",
    n_boot: int = 2000,
) -> EvalReport:
    if baseline not in policies:
        raise ValueError(f"baseline '{baseline}' must be one of the policies")
    results: dict[str, list[EpisodeResult]] = {name: [] for name in policies}
    for hi, params in enumerate(households):
        for ep in range(episodes_per_household):
            ep_seed = seed + 1000 * hi + ep
            for name, policy in policies.items():
                results[name].append(run_episode(HouseholdEnv(params), policy, seed=ep_seed))
    incomes = np.array([h.income for h in households for _ in range(episodes_per_household)])
    base = np.array([r.total_reward for r in results[baseline]])
    rng = np.random.default_rng(seed)
    scores = []
    for name, rs in results.items():
        rewards = np.array([r.total_reward for r in rs])
        diff = rewards - base
        boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        scores.append(PolicyScore(
            name=name, mean_reward=float(rewards.mean()), improvement=float(diff.mean()),
            ci_low=float(lo), ci_high=float(hi), win_rate=float(np.mean(diff > 1e-9)),
            net_worth_months=float(np.mean([r.final_net_worth for r in rs] / incomes)),
            interest_months=float(np.mean([r.interest_paid for r in rs] / incomes)),
            goals_funded=float(np.mean([r.goals_funded_fraction for r in rs])),
            min_runway=float(np.mean([min(r.min_runway_months, 60.0) for r in rs])),
            credit_months=float(np.mean([r.deficit_months for r in rs])),
            violations=float(np.mean([r.violations for r in rs]))))
    return EvalReport(label, len(households), len(households) * episodes_per_household, baseline, tuple(scores))


def persona_households(as_of: date, *, horizon: int = 60) -> dict[str, HouseholdParams]:
    """Simulator parameters for the six demo personas (held out from training)."""
    from app.seed.generator import generate_history
    from app.seed.personas import PERSONAS

    out = {}
    for pid, persona in PERSONAS.items():
        history = generate_history(persona, as_of=as_of, months=18, seed=0)
        profile = dataclasses.replace(persona.profile, income_history=history.income)
        out[pid] = params_from_profile(profile, persona.onboarding, region=persona.region, as_of=as_of, horizon=horizon)
    return out