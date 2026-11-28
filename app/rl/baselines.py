"""Baseline policies and an episode runner.

The RL policy must beat these on held-out scenarios to earn its place:

* ``waterfall``      the deterministic priority rules the agents already apply (starter buffer,
                     high-interest debt, full emergency fund, other debt, then goals/retirement);
* ``debt_first``     all surplus to debt until it is gone, then emergency fund, goals, retirement;
* ``fixed_split``    a static 25/25/25/25 split (a "rule of thumb" stand-in);
* ``goals_first``    everything to goals;
* ``random``         uniform random actions (a sanity floor).

Every baseline uses the same action space and passes through the same projection layer, so the
comparison is fair. Equity share follows the remembered risk tolerance, capped by the user's limit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.engine.retirement import START_EQUITY
from app.rl.env import HouseholdEnv

Policy = Callable[[HouseholdEnv], np.ndarray]


def _equity(env: HouseholdEnv) -> float:
    return min(env.params.max_equity, START_EQUITY[env.params.risk])


def _action(env: HouseholdEnv, debt: float = 0.0, emergency: float = 0.0, goals: float = 0.0, retirement: float = 0.0) -> np.ndarray:
    return np.array([debt, emergency, goals, retirement, _equity(env)], dtype=np.float32)


def _goals_or_retirement(env: HouseholdEnv) -> np.ndarray:
    pending = env.pending_goal_targets() > 0
    return _action(env, goals=0.6, retirement=0.4) if pending else _action(env, retirement=1.0)


def waterfall(env: HouseholdEnv) -> np.ndarray:
    s, p = env.state, env.params
    if s.liquid < p.essential:  # 1. starter buffer
        return _action(env, emergency=1.0)
    if env.high_interest_debt() > 0:  # 2. high-interest debt
        return _action(env, debt=1.0)
    if s.liquid < env.emergency_target():  # 3. full emergency fund
        return _action(env, emergency=1.0)
    if float(s.balances.sum()) > 0:  # 4. other debt
        return _action(env, debt=1.0)
    return _goals_or_retirement(env)  # 5. goals and retirement


def debt_first(env: HouseholdEnv) -> np.ndarray:
    if float(env.state.balances.sum()) > 0:
        return _action(env, debt=1.0)
    if env.state.liquid < env.emergency_target():
        return _action(env, emergency=1.0)
    return _goals_or_retirement(env)


def fixed_split(env: HouseholdEnv) -> np.ndarray:
    return _action(env, 0.25, 0.25, 0.25, 0.25)


def goals_first(env: HouseholdEnv) -> np.ndarray:
    return _action(env, goals=1.0)


def make_random(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)

    def policy(env: HouseholdEnv) -> np.ndarray:
        return rng.random(5).astype(np.float32)

    return policy


BASELINES: dict[str, Policy] = {
    "waterfall": waterfall,
    "debt_first": debt_first,
    "fixed_split": fixed_split,
    "goals_first": goals_first,
    "random": make_random(0),
}


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    final_net_worth: float  # adjusted: net worth plus goal payouts
    interest_paid: float
    goals_funded_fraction: float  # mean funded fraction over goals that reached their deadline
    min_runway_months: float
    deficit_months: int
    violations: float
    high_interest_left: float


def summarize_episode(env: HouseholdEnv, total_reward: float) -> EpisodeResult:
    """Summarise a finished episode from the environment's final state."""
    s = env.state
    resolved = [i for i, d in enumerate(s.goals_done) if d]
    funded = s.goal_paid / sum(env.params.goals[i].target for i in resolved) if resolved else 1.0
    return EpisodeResult(
        total_reward=total_reward, final_net_worth=env.adjusted_wealth(), interest_paid=s.interest_paid,
        goals_funded_fraction=min(1.0, funded), min_runway_months=s.min_runway,
        deficit_months=s.deficit_months, violations=s.violations,
        high_interest_left=env.high_interest_debt())


def run_episode(env: HouseholdEnv, policy: Policy, *, seed: int) -> EpisodeResult:
    """Run one episode with a policy and summarise it."""
    env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        _, reward, done, _, _ = env.step(policy(env))
        total += reward
    return summarize_episode(env, total)