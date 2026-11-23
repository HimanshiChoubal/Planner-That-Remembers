"""Retirement planning via Monte Carlo.

Yearly simulation with random returns and inflation:

* Accumulation: ``balance = balance * (1 + return) + contribution`` (contribution at year end,
  growing at ``contribution_growth``).
* Retirement: spending (today's money, scaled by simulated inflation, minus other income) is
  withdrawn at the start of each year; the rest grows. A path fails when it runs out.
* Return assumptions follow an equity glide path: equity share falls toward retirement.

All assumptions are SYNTHETIC and illustrative, set per region, and returns/inflation are
independent across years. Output is a probability and ranges, not a single forecast.
Floats are used internally (statistical estimate); results convert back to ``Money``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.engine.money import Money

Tolerance = Literal["low", "medium", "high"]

START_EQUITY: dict[str, float] = {"low": 0.40, "medium": 0.60, "high": 0.80}
GLIDE_YEARS = 20  # equity share declines over this many years before retirement
FINAL_EQUITY_FACTOR = 0.5  # equity share at retirement as a fraction of the starting share
MIN_RETURN = -0.90


@dataclass(frozen=True)
class Assumptions:
    equity_return: float
    equity_vol: float
    bond_return: float
    bond_vol: float
    correlation: float
    inflation_mean: float
    inflation_vol: float


US_ASSUMPTIONS = Assumptions(0.07, 0.16, 0.035, 0.06, 0.20, 0.025, 0.010)
IN_ASSUMPTIONS = Assumptions(0.11, 0.20, 0.070, 0.04, 0.10, 0.050, 0.015)
_BY_REGION = {"us": US_ASSUMPTIONS, "in": IN_ASSUMPTIONS}


def assumptions_for(region: str | None) -> Assumptions:
    return _BY_REGION.get((region or "us").lower(), US_ASSUMPTIONS)


@dataclass(frozen=True)
class RetirementInputs:
    current_age: int
    retirement_age: int
    life_expectancy: int  # planning horizon; plan to a conservative age
    current_savings: Money
    monthly_contribution: Money
    desired_monthly_spend: Money  # in today's money
    other_monthly_income: Money | None = None  # pension etc., in today's money
    contribution_growth: float = 0.03
    risk_tolerance: Tolerance = "medium"
    use_glide_path: bool = True
    max_equity_share: float | None = None  # hard user limit on the equity share


@dataclass(frozen=True)
class RetirementOutcome:
    probability_funded: float  # P(money lasts to life_expectancy)
    corpus_p10: Money  # nominal corpus at retirement
    corpus_p50: Money
    corpus_p90: Money
    corpus_real_p50: Money  # median corpus in today's money
    median_depletion_age: int | None  # among failing paths; None if none fail
    years_to_retirement: int
    years_in_retirement: int
    n_sims: int
    seed: int


def equity_share(
    age: int,
    retirement_age: int,
    tolerance: Tolerance,
    use_glide_path: bool = True,
    max_equity: float | None = None,
) -> float:
    """Equity share of the portfolio at ``age`` (glide path toward retirement), never above
    ``max_equity`` when the user has set a hard limit."""
    start = START_EQUITY[tolerance]
    if not use_glide_path:
        share = start
    else:
        final = start * FINAL_EQUITY_FACTOR
        years_left = retirement_age - age
        if years_left <= 0:
            share = final
        elif years_left >= GLIDE_YEARS:
            share = start
        else:
            share = final + (start - final) * (years_left / GLIDE_YEARS)
    return share if max_equity is None else min(share, max_equity)


def portfolio_params(w: float, a: Assumptions) -> tuple[float, float]:
    """Expected return and volatility of an equity/bond mix with equity share ``w``."""
    mu = w * a.equity_return + (1 - w) * a.bond_return
    var = (
        (w * a.equity_vol) ** 2
        + ((1 - w) * a.bond_vol) ** 2
        + 2 * w * (1 - w) * a.correlation * a.equity_vol * a.bond_vol
    )
    return mu, float(np.sqrt(var))


def _validate(i: RetirementInputs) -> None:
    if not 0 <= i.current_age < i.life_expectancy:
        raise ValueError("current_age must be non-negative and below life_expectancy")
    if i.retirement_age < i.current_age:
        raise ValueError("retirement_age must not be below current_age")
    if i.life_expectancy <= i.retirement_age:
        raise ValueError("life_expectancy must be above retirement_age")
    cur = i.current_savings.currency
    others = [i.monthly_contribution, i.desired_monthly_spend]
    if i.other_monthly_income is not None:
        others.append(i.other_monthly_income)
    if any(m.currency != cur for m in others):
        raise ValueError("All amounts must share one currency")


@dataclass(frozen=True)
class _Draws:
    returns: np.ndarray  # (n, years)
    inflation: np.ndarray  # (n, years)


def _draw(i: RetirementInputs, a: Assumptions, n_sims: int, seed: int) -> _Draws:
    years = i.life_expectancy - i.current_age
    rng = np.random.default_rng(seed)
    params = [
        portfolio_params(equity_share(i.current_age + y, i.retirement_age, i.risk_tolerance, i.use_glide_path, i.max_equity_share), a)
        for y in range(years)
    ]
    mu = np.array([p[0] for p in params])
    sigma = np.array([p[1] for p in params])
    returns = np.maximum(rng.normal(mu, sigma, size=(n_sims, years)), MIN_RETURN)
    inflation = np.clip(rng.normal(a.inflation_mean, a.inflation_vol, size=(n_sims, years)), -0.02, 0.25)
    return _Draws(returns, inflation)


@dataclass(frozen=True)
class _Run:
    success: np.ndarray
    corpus: np.ndarray  # nominal, at retirement
    corpus_real: np.ndarray
    depletion_age: np.ndarray  # NaN where the path did not deplete


def _run(i: RetirementInputs, d: _Draws, annual_contribution: float) -> _Run:
    n, years = d.returns.shape
    retire_idx = i.retirement_age - i.current_age
    balance = np.full(n, float(i.current_savings.amount))
    price = np.ones(n)  # price level index versus today
    alive = np.ones(n, dtype=bool)
    depletion_age = np.full(n, np.nan)
    corpus = balance.copy()
    corpus_real = balance.copy()
    spend = float(i.desired_monthly_spend.amount) * 12.0
    other = float(i.other_monthly_income.amount) * 12.0 if i.other_monthly_income else 0.0

    for y in range(years):
        if y < retire_idx:
            contribution = annual_contribution * (1.0 + i.contribution_growth) ** y
            balance = balance * (1.0 + d.returns[:, y]) + contribution
        else:
            if y == retire_idx:
                corpus = balance.copy()
                corpus_real = balance / price
            need = max(0.0, spend - other) * price
            balance = balance - need
            newly_depleted = alive & (balance < 0)
            depletion_age[newly_depleted] = i.current_age + y
            alive &= ~newly_depleted
            balance = np.where(alive, balance * (1.0 + d.returns[:, y]), 0.0)
        price = price * (1.0 + d.inflation[:, y])

    if retire_idx == 0:  # already at retirement age: corpus is today's savings
        corpus = np.full(n, float(i.current_savings.amount))
        corpus_real = corpus.copy()
    return _Run(alive, corpus, corpus_real, depletion_age)


def _money(x: float, like: Money) -> Money:
    return Money.of(f"{x:.4f}", like.currency)


def project(
    inputs: RetirementInputs,
    assumptions: Assumptions | None = None,
    *,
    n_sims: int = 4000,
    seed: int = 0,
) -> RetirementOutcome:
    _validate(inputs)
    a = assumptions or US_ASSUMPTIONS
    draws = _draw(inputs, a, n_sims, seed)
    run = _run(inputs, draws, float(inputs.monthly_contribution.amount) * 12.0)
    p10, p50, p90 = np.percentile(run.corpus, [10, 50, 90])
    failed = run.depletion_age[~run.success]
    cur = inputs.current_savings
    return RetirementOutcome(
        probability_funded=float(run.success.mean()),
        corpus_p10=_money(float(p10), cur),
        corpus_p50=_money(float(p50), cur),
        corpus_p90=_money(float(p90), cur),
        corpus_real_p50=_money(float(np.median(run.corpus_real)), cur),
        median_depletion_age=int(np.median(failed)) if failed.size else None,
        years_to_retirement=inputs.retirement_age - inputs.current_age,
        years_in_retirement=inputs.life_expectancy - inputs.retirement_age,
        n_sims=n_sims,
        seed=seed,
    )


def required_monthly_contribution(
    inputs: RetirementInputs,
    assumptions: Assumptions | None = None,
    *,
    target_probability: float = 0.8,
    n_sims: int = 3000,
    seed: int = 0,
) -> Money | None:
    """Smallest starting monthly contribution reaching ``target_probability`` of being funded.

    Uses common random numbers so success is monotonic in the contribution, then bisects.
    Returns ``None`` if no feasible contribution is found.
    """
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must be strictly between 0 and 1")
    _validate(inputs)
    a = assumptions or US_ASSUMPTIONS
    draws = _draw(inputs, a, n_sims, seed)

    def funded(annual: float) -> float:
        return float(_run(inputs, draws, annual).success.mean())

    cur = inputs.current_savings
    if funded(0.0) >= target_probability:
        return Money.zero(cur.currency)
    hi = max(1.0, float(inputs.monthly_contribution.amount) * 12.0)
    for _ in range(60):
        if funded(hi) >= target_probability:
            break
        hi *= 2.0
    else:
        return None
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if funded(mid) >= target_probability:
            hi = mid
        else:
            lo = mid
    return _money(hi / 12.0, cur)