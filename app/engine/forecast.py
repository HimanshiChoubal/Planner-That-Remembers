"""Probabilistic cash-flow forecasting.

Model (per monthly series such as income or essential spending):

    value_t = level * season[month_t] * eps_t

* ``level``: exponentially weighted average of the deseasonalized history.
* ``season``: calendar-month factors, fitted only with 24+ months, kept only when a one-way
  ANOVA says months really differ (p < 0.05), and shrunk toward 1 by the strength of evidence.
* ``eps``: bootstrap of the model's own one-step-ahead ratio errors, so irregular incomes get
  wide intervals and steady salaries get narrow ones, with no distribution assumed.
* A per-path level shock reflects uncertainty in the level estimate itself.

Forecasts are returned as ranges (low / median / high), never single numbers. Use
``backtest_coverage`` to check that an 80% interval really covers about 80% of outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats

from app.engine.money import Currency, Money

MIN_OBSERVATIONS = 6
MIN_SEASONAL_OBSERVATIONS = 24
SEASON_ALPHA = 0.05  # significance level for keeping seasonality
MIN_HONEST_RATIOS = 6  # out-of-sample errors needed before trusting them over in-sample ones
SALARIED_CV_THRESHOLD = 0.15  # below this coefficient of variation, income counts as steady


@dataclass(frozen=True, eq=False)
class SeriesModel:
    currency: Currency
    level: float  # deseasonalized level in major units
    residual_ratios: np.ndarray = field(repr=False)
    season: Mapping[int, float]  # calendar month (1-12) -> factor; empty if not seasonal
    last_month: int | None  # calendar month of the last observation
    n_obs: int
    cv: float  # spread of one-step ratio errors (about the coefficient of variation)

    @property
    def is_seasonal(self) -> bool:
        return bool(self.season)


@dataclass(frozen=True)
class Forecast:
    months: int
    low: tuple[Money, ...]
    median: tuple[Money, ...]
    high: tuple[Money, ...]
    mean: tuple[Money, ...]
    interval: float  # nominal coverage of [low, high], e.g. 0.8
    cv: float
    seasonal: bool


@dataclass(frozen=True)
class BacktestResult:
    coverage: float  # observed share of actuals inside the interval
    nominal: float
    n_points: int
    mean_width_ratio: float  # mean interval width / mean actual (sharpness)


def _major(values: Sequence[Money]) -> np.ndarray:
    return np.array([float(m.amount) for m in values], dtype=float)


def _money(x: float, currency: Currency) -> Money:
    return Money.of(f"{x:.4f}", currency)


def coefficient_of_variation(values: Sequence[Money]) -> float:
    """Sample standard deviation divided by the mean (0 for fewer than 2 values or mean <= 0)."""
    x = _major(values)
    if len(x) < 2 or x.mean() <= 0:
        return 0.0
    return float(x.std(ddof=1) / x.mean())


def classify_income(values: Sequence[Money], threshold: float = SALARIED_CV_THRESHOLD) -> Literal["salaried", "irregular"]:
    return "salaried" if coefficient_of_variation(values) < threshold else "irregular"


def _calendar_month(start_month: int, offset: int) -> int:
    return (start_month - 1 + offset) % 12 + 1


def _fit_season(x: np.ndarray, months: np.ndarray) -> tuple[dict[int, float], float]:
    """Calendar-month factors and the shrinkage weight used; ``({}, 0.0)`` if unsupported.

    One-way ANOVA across the 12 calendar months; if significant, each month's factor is shrunk
    toward 1 by ``w = 1 - 1/F`` so strong seasonality is kept and weak seasonality is damped.
    """
    groups = [x[months == m] for m in range(1, 13)]
    if any(len(g) < 2 for g in groups):
        return {}, 0.0
    f_stat, p_value = stats.f_oneway(*groups)
    if not np.isfinite(f_stat) or p_value >= SEASON_ALPHA:
        return {}, 0.0
    weight = max(0.0, 1.0 - 1.0 / float(f_stat))
    overall = float(x.mean())
    factors = {
        m: 1.0 + weight * (float(g.mean()) / overall - 1.0)
        for m, g in zip(range(1, 13), groups, strict=True)
    }
    return factors, weight


def _ewma_level(xd: np.ndarray, alpha: float) -> float:
    """End-of-series exponentially weighted level (initialised from the first 3 values)."""
    prev = float(xd[:3].mean())
    for t in range(3, len(xd)):
        prev = alpha * float(xd[t]) + (1.0 - alpha) * prev
    return prev


def _honest_ratios(x: np.ndarray, months: np.ndarray, alpha: float) -> list[float]:
    """Out-of-sample one-step ratio errors for a seasonal model.

    Refits seasonality and level on each expanding window and compares the next actual with
    the forecast, so the errors include parameter uncertainty (in-sample residuals of a model
    with 12 fitted factors are far too small).
    """
    ratios: list[float] = []
    for t in range(MIN_SEASONAL_OBSERVATIONS, len(x)):
        season, _ = _fit_season(x[:t], months[:t])
        f = np.array([season.get(int(m), 1.0) for m in months[: t + 1]])
        predicted = _ewma_level(x[:t] / f[:t], alpha) * f[t]
        if predicted > 0:
            ratios.append(max(0.0, float(x[t] / predicted)))
    return ratios


def fit_series(
    values: Sequence[Money],
    *,
    start_month: int | None = None,
    alpha: float = 0.3,
) -> SeriesModel:
    """Fit the level/seasonality/error model. ``start_month`` is the calendar month (1-12) of
    the first observation; without it no seasonality is fitted."""
    n = len(values)
    if n < MIN_OBSERVATIONS:
        raise ValueError(f"Need at least {MIN_OBSERVATIONS} monthly observations, got {n}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if start_month is not None and not 1 <= start_month <= 12:
        raise ValueError("start_month must be 1-12")
    currency = values[0].currency
    x = _major(values)

    season: dict[int, float] = {}
    factors = np.ones(n)
    dof_scale = 1.0
    months: np.ndarray | None = None
    if start_month is not None and n >= MIN_SEASONAL_OBSERVATIONS and x.mean() > 0:
        months = np.array([_calendar_month(start_month, i) for i in range(n)])
        season, weight = _fit_season(x, months)
        if season:
            factors = np.array([season[int(m)] for m in months])
            # Fitting 12 factors to the same data understates in-sample errors; rescale by
            # sqrt(n / (n - k_eff)) so intervals stay honest (k_eff counts shrunk parameters).
            dof_scale = float(np.sqrt(n / max(n - 11.0 * weight, 1.0)))
    xd = x / factors

    prev = float(xd[:3].mean())
    ratios: list[float] = []
    for t in range(3, n):
        if prev > 0:
            ratios.append(max(0.0, float(xd[t] / prev)))
        prev = alpha * float(xd[t]) + (1.0 - alpha) * prev
    level = max(0.0, prev)
    if not ratios or level <= 0:
        ratios = [1.0]
    arr = np.array(ratios)
    if season and months is not None:
        honest = _honest_ratios(x, months, alpha)
        if len(honest) >= MIN_HONEST_RATIOS:
            arr = np.array(honest)  # out-of-sample errors already include parameter uncertainty
        elif dof_scale != 1.0:  # short history: fall back to a degrees-of-freedom correction
            arr = np.maximum(arr.mean() + (arr - arr.mean()) * dof_scale, 0.0)
    last_month = _calendar_month(start_month, n - 1) if start_month is not None else None
    return SeriesModel(
        currency=currency,
        level=level,
        residual_ratios=arr,
        season=season,
        last_month=last_month,
        n_obs=n,
        cv=float(arr.std()),
    )


def simulate_paths(
    model: SeriesModel, horizon: int, *, n_sims: int = 2000, seed: int = 0
) -> np.ndarray:
    """Simulated future values in major units, shape ``(n_sims, horizon)``."""
    if horizon <= 0 or n_sims <= 0:
        raise ValueError("horizon and n_sims must be positive")
    rng = np.random.default_rng(seed)
    ratios = model.residual_ratios
    eps = ratios[rng.integers(0, len(ratios), size=(n_sims, horizon))]
    shock_sd = model.cv / np.sqrt(len(ratios))
    shock = np.maximum(rng.normal(1.0, shock_sd, size=(n_sims, 1)), 0.0)
    if model.is_seasonal and model.last_month is not None:
        season = np.array(
            [model.season[_calendar_month(model.last_month, h + 1)] for h in range(horizon)]
        )
    else:
        season = np.ones(horizon)
    return np.maximum(model.level * season[None, :] * eps * shock, 0.0)


def forecast(
    model: SeriesModel,
    horizon: int,
    *,
    interval: float = 0.8,
    n_sims: int = 2000,
    seed: int = 0,
) -> Forecast:
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must be strictly between 0 and 1")
    paths = simulate_paths(model, horizon, n_sims=n_sims, seed=seed)
    lo_q, hi_q = (1.0 - interval) / 2.0, 1.0 - (1.0 - interval) / 2.0
    low, med, high = np.quantile(paths, [lo_q, 0.5, hi_q], axis=0)
    mean = paths.mean(axis=0)
    cur = model.currency
    return Forecast(
        months=horizon,
        low=tuple(_money(float(v), cur) for v in low),
        median=tuple(_money(float(v), cur) for v in med),
        high=tuple(_money(float(v), cur) for v in high),
        mean=tuple(_money(float(v), cur) for v in mean),
        interval=interval,
        cv=model.cv,
        seasonal=model.is_seasonal,
    )


def backtest_coverage(
    values: Sequence[Money],
    *,
    interval: float = 0.8,
    min_train: int = 12,
    start_month: int | None = None,
    alpha: float = 0.3,
    n_sims: int = 1000,
    seed: int = 0,
) -> BacktestResult:
    """Rolling-origin check: forecast each next month from the data before it and count how
    often the actual value lands inside the interval."""
    n = len(values)
    if n <= min_train or min_train < MIN_OBSERVATIONS:
        raise ValueError("Need more history than min_train (and min_train >= 6)")
    x = _major(values)
    hits = 0
    widths: list[float] = []
    points = 0
    for origin in range(min_train, n):
        model = fit_series(values[:origin], start_month=start_month, alpha=alpha)
        paths = simulate_paths(model, 1, n_sims=n_sims, seed=seed + origin)[:, 0]
        lo, hi = np.quantile(paths, [(1 - interval) / 2, 1 - (1 - interval) / 2])
        hits += int(lo <= x[origin] <= hi)
        widths.append(float(hi - lo))
        points += 1
    mean_actual = float(x[min_train:].mean())
    return BacktestResult(
        coverage=hits / points,
        nominal=interval,
        n_points=points,
        mean_width_ratio=float(np.mean(widths) / mean_actual) if mean_actual > 0 else 0.0,
    )