"""Training: evolution strategies (OpenAI-ES style) in plain numpy.

* Households are sampled from a scale-free generator that is independent of the demo personas,
  so personas stay genuinely held out for evaluation.
* Fitness is the policy's improvement over the deterministic waterfall baseline on the same
  household and random seed (common random numbers), so the policy is directly optimised to beat
  the rules it must justify itself against.
* Antithetic sampling, centred-rank fitness shaping and Adam keep the gradient estimate stable.
* The best policy on a fixed validation set is returned, not just the last iterate.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.engine import emergency_fund as ef
from app.engine import retirement as rt
from app.rl.baselines import waterfall
from app.rl.env import DebtSpec, GoalSpec, HouseholdEnv, HouseholdParams
from app.rl.policy import MLPPolicy, PolicyMeta

Policy = Callable[[HouseholdEnv], np.ndarray]


def sample_household(rng: np.random.Generator, horizon: int = 60) -> HouseholdParams:
    """A random household in scale-free units (income = 1000)."""
    income = 1000.0
    cv = float(rng.choice([rng.uniform(0.01, 0.06), rng.uniform(0.15, 0.45)], p=[0.55, 0.45]))
    dependents = int(rng.choice([0, 1, 2, 3], p=[0.4, 0.25, 0.25, 0.1]))
    single_income = bool(rng.random() < 0.3)

    debts = []
    for _ in range(int(rng.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2]))):
        balance = income * float(rng.uniform(0.5, 30.0))
        apr = float(rng.choice([rng.uniform(0.03, 0.09), rng.uniform(0.10, 0.36)]))
        minimum = max(balance * float(rng.uniform(0.015, 0.04)), balance * apr / 12.0 * 1.05)
        debts.append(DebtSpec(balance, apr, minimum))
    mins = sum(d.min_payment for d in debts)

    essential = max(income * float(rng.uniform(0.35, 0.65)), mins + 0.15 * income)
    expenses = max(essential + income * float(rng.uniform(0.05, 0.3)), essential)
    liquid = essential * float(rng.uniform(0.0, 8.0))
    goals = tuple(
        GoalSpec(target=income * float(rng.uniform(3, 60)), deadline_month=int(rng.integers(6, 72)),
                 saved=0.0, priority=int(rng.integers(1, 5)))
        for _ in range(int(rng.choice([0, 1, 2, 3], p=[0.15, 0.4, 0.3, 0.15]))))
    goals = tuple(GoalSpec(g.target, g.deadline_month, g.target * float(rng.uniform(0.0, 0.5)), g.priority) for g in goals)

    months, _ = ef.target_months(income_cv=cv, dependents=dependents, single_income=single_income)
    return HouseholdParams(
        currency="SIM", income=income, income_cv=cv, expenses=expenses, essential=essential, liquid=liquid,
        debts=tuple(debts), goals=goals, retirement=income * float(rng.uniform(0, 120)),
        risk=str(rng.choice(["low", "medium", "high"])),
        max_equity=1.0 if rng.random() < 0.55 else float(rng.uniform(0.2, 0.9)),
        dependents=dependents, emergency_months=float(months),
        assumptions=rt.US_ASSUMPTIONS if rng.random() < 0.5 else rt.IN_ASSUMPTIONS, horizon=horizon)


def total_reward(env: HouseholdEnv, policy: Policy, seed: int) -> float:
    env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        _, reward, done, _, _ = env.step(policy(env))
        total += reward
    return total


@dataclass(frozen=True)
class ESConfig:
    generations: int = 100
    population: int = 32  # perturbations per generation (antithetic pairs)
    households_per_gen: int = 8
    sigma: float = 0.08
    lr: float = 0.05
    hidden: tuple[int, ...] = (32, 32)
    init_scale: float = 0.1
    weight_decay: float = 0.002
    validation_households: int = 40
    validate_every: int = 10
    seed: int = 0
    residual: bool = True  # learn a bounded correction to the waterfall instead of a policy from scratch


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


def validate(policy: Policy, households: list[HouseholdParams], baseline_rewards: np.ndarray, seed: int) -> float:
    """Mean improvement over the baseline on a fixed household set."""
    rewards = np.array([total_reward(HouseholdEnv(h), policy, seed + i) for i, h in enumerate(households)])
    return float(np.mean(rewards - baseline_rewards))


def train_es(
    cfg: ESConfig = ESConfig(),
    *,
    log: Callable[[dict], None] | None = None,
    init: MLPPolicy | None = None,
) -> tuple[MLPPolicy, list[dict]]:
    """Train from scratch, or continue from ``init`` (same architecture)."""
    if cfg.population % 2:
        raise ValueError("population must be even (antithetic pairs)")
    master = np.random.default_rng(cfg.seed)
    policy = init if init is not None else MLPPolicy.random(cfg.hidden, master, cfg.init_scale, residual=cfg.residual)
    theta = policy.params.copy()
    n = theta.size
    m, v = np.zeros(n), np.zeros(n)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    val_rng = np.random.default_rng(10_000 + cfg.seed)
    val_households = [sample_household(val_rng) for _ in range(cfg.validation_households)]
    val_seed = 777_000
    val_baseline = np.array([total_reward(HouseholdEnv(h), waterfall, val_seed + i) for i, h in enumerate(val_households)])
    best_theta, best_val = theta.copy(), validate(policy.with_params(theta), val_households, val_baseline, val_seed)
    history: list[dict] = [{"generation": 0, "validation_improvement": best_val}]
    started = time.time()

    for gen in range(1, cfg.generations + 1):
        rng = np.random.default_rng(cfg.seed * 100_003 + gen)
        households = [sample_household(rng) for _ in range(cfg.households_per_gen)]
        seeds = [int(s) for s in rng.integers(0, 2**31 - 1, cfg.households_per_gen)]
        envs = [HouseholdEnv(h) for h in households]
        base = np.array([total_reward(e, waterfall, s) for e, s in zip(envs, seeds, strict=True)])

        noise = rng.standard_normal((cfg.population // 2, n))
        perturbations = np.concatenate([noise, -noise])
        fitness = np.empty(cfg.population)
        for i, eps_i in enumerate(perturbations):
            cand = policy.with_params(theta + cfg.sigma * eps_i)
            rewards = np.array([total_reward(e, cand, s) for e, s in zip(envs, seeds, strict=True)])
            fitness[i] = float(np.mean(rewards - base))

        weights = _centered_ranks(fitness)
        grad = (weights @ perturbations) / (cfg.population * cfg.sigma) - cfg.weight_decay * theta
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        theta = theta + cfg.lr * (m / (1 - beta1**gen)) / (np.sqrt(v / (1 - beta2**gen)) + eps)

        record = {"generation": gen, "mean_fitness": float(fitness.mean()), "best_fitness": float(fitness.max()),
                  "elapsed_s": round(time.time() - started, 1)}
        if gen % cfg.validate_every == 0 or gen == cfg.generations:
            val = validate(policy.with_params(theta), val_households, val_baseline, val_seed)
            record["validation_improvement"] = val
            if val > best_val:
                best_val, best_theta = val, theta.copy()
        history.append(record)
        if log:
            log(record)

    meta = PolicyMeta(generations=cfg.generations, seed=cfg.seed, hidden=cfg.hidden, residual=policy.meta.residual,
                      notes="Evolution strategies on a residual-over-waterfall policy; fitness = improvement over the waterfall.",
                      extra={"best_validation_improvement": best_val, "population": cfg.population})
    return MLPPolicy(best_theta, policy.sizes, meta), history