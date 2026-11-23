"""Anomaly detection on transactions.

Robust statistics (median / MAD "modified z-scores") rather than a black-box model, so every
flag can be explained in the Why? panel: "this amount is 5.2 robust deviations above your usual
spend in this category". Also detects duplicate charges and month-level category spikes.

This module also defines ``Transaction``, the shared transaction type.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np

from app.engine.categorize import normalize_merchant
from app.engine.money import Currency, Money

OUTLIER_Z = 3.5  # Iglewicz-Hoaglin recommended cutoff for modified z-scores
SATURATED_SCORE = 99.0  # score when history has zero spread but the value differs


@dataclass(frozen=True)
class Transaction:
    id: str
    date: date
    amount: Money  # always a positive magnitude
    kind: Literal["expense", "income"] = "expense"
    category: str = "other"
    merchant: str = ""


@dataclass(frozen=True)
class Anomaly:
    kind: Literal["outlier", "duplicate", "monthly_spike"]
    txn_ids: tuple[str, ...]
    category: str | None
    score: float
    amount: Money
    baseline: Money | None
    month: str | None  # "YYYY-MM" for monthly spikes
    detail: str


def modified_zscore(value: float, history: Sequence[float]) -> float:
    """Modified z-score of ``value`` against ``history`` (median/MAD based)."""
    h = np.asarray(history, dtype=float)
    if h.size == 0:
        return 0.0
    med = float(np.median(h))
    mad = float(np.median(np.abs(h - med)))
    if mad > 0:
        return 0.6745 * (value - med) / mad
    mean_ad = float(np.mean(np.abs(h - med)))
    if mean_ad > 0:
        return (value - med) / (1.253314 * mean_ad)
    if value == med:
        return 0.0
    return SATURATED_SCORE if value > med else -SATURATED_SCORE


def _money(x: float, currency: Currency) -> Money:
    return Money.of(f"{x:.4f}", currency)


def detect_spend_outliers(
    txns: Sequence[Transaction],
    *,
    z_threshold: float = OUTLIER_Z,
    min_history: int = 8,
    min_ratio: float = 1.5,
) -> list[Anomaly]:
    """Unusually large expenses relative to their category's history."""
    by_cat: dict[str, list[Transaction]] = defaultdict(list)
    for t in txns:
        if t.kind == "expense":
            by_cat[t.category].append(t)
    found: list[Anomaly] = []
    for category, items in by_cat.items():
        if len(items) < min_history:
            continue
        amounts = [float(t.amount.amount) for t in items]
        median = float(np.median(amounts))
        for t, a in zip(items, amounts, strict=True):
            z = modified_zscore(a, amounts)
            if z > z_threshold and a > median * min_ratio:
                found.append(
                    Anomaly(
                        kind="outlier",
                        txn_ids=(t.id,),
                        category=category,
                        score=z,
                        amount=t.amount,
                        baseline=_money(median, t.amount.currency),
                        month=None,
                        detail=(
                            f"{t.amount} at '{t.merchant}' is {z:.1f} robust deviations above "
                            f"the typical {category} expense."
                        ),
                    )
                )
    return sorted(found, key=lambda x: x.score, reverse=True)


def detect_duplicates(txns: Sequence[Transaction], *, window_days: int = 3) -> list[Anomaly]:
    """Same merchant and same amount charged again within ``window_days``."""
    groups: dict[tuple[str, int], list[Transaction]] = defaultdict(list)
    for t in txns:
        if t.kind == "expense" and t.merchant:
            groups[(normalize_merchant(t.merchant), t.amount.minor)].append(t)
    found: list[Anomaly] = []
    for items in groups.values():
        items = sorted(items, key=lambda t: t.date)
        for earlier, later in zip(items, items[1:], strict=False):
            gap = (later.date - earlier.date).days
            if gap <= window_days:
                found.append(
                    Anomaly(
                        kind="duplicate",
                        txn_ids=(earlier.id, later.id),
                        category=later.category,
                        score=float(window_days - gap + 1),
                        amount=later.amount,
                        baseline=None,
                        month=None,
                        detail=(
                            f"{later.amount} at '{later.merchant}' was charged twice within "
                            f"{gap} day(s)."
                        ),
                    )
                )
    return sorted(found, key=lambda x: x.score, reverse=True)


def _month_key(d: date) -> tuple[int, int]:
    return d.year, d.month


def _month_range(first: tuple[int, int], last: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    y, m = first
    while (y, m) <= last:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def detect_monthly_spikes(
    txns: Sequence[Transaction],
    *,
    z_threshold: float = OUTLIER_Z,
    min_months: int = 4,
    min_ratio: float = 1.25,
) -> list[Anomaly]:
    """Latest month's spend in a category versus its own earlier months."""
    expenses = [t for t in txns if t.kind == "expense"]
    if not expenses:
        return []
    currency = expenses[0].amount.currency
    months = _month_range(
        min(_month_key(t.date) for t in expenses), max(_month_key(t.date) for t in expenses)
    )
    if len(months) < min_months + 1:
        return []
    totals: dict[str, dict[tuple[int, int], float]] = defaultdict(lambda: defaultdict(float))
    for t in expenses:
        totals[t.category][_month_key(t.date)] += float(t.amount.amount)
    latest = months[-1]
    found: list[Anomaly] = []
    for category, per_month in totals.items():
        prior = [per_month.get(m, 0.0) for m in months[:-1]]
        current = per_month.get(latest, 0.0)
        baseline = float(np.median(prior))
        z = modified_zscore(current, prior)
        if z > z_threshold and current > baseline * min_ratio:
            found.append(
                Anomaly(
                    kind="monthly_spike",
                    txn_ids=(),
                    category=category,
                    score=z,
                    amount=_money(current, currency),
                    baseline=_money(baseline, currency),
                    month=f"{latest[0]:04d}-{latest[1]:02d}",
                    detail=(
                        f"New {category} spending of {_money(current, currency)} in "
                        f"{latest[0]:04d}-{latest[1]:02d}, with none in most earlier months."
                        if baseline == 0
                        else f"{category} spending of {_money(current, currency)} in "
                        f"{latest[0]:04d}-{latest[1]:02d} is well above the usual "
                        f"{_money(baseline, currency)} per month."
                    ),
                )
            )
    return sorted(found, key=lambda x: x.score, reverse=True)


def detect_all(txns: Sequence[Transaction]) -> list[Anomaly]:
    """Every detector combined, highest score first."""
    combined = [
        *detect_spend_outliers(txns),
        *detect_duplicates(txns),
        *detect_monthly_spikes(txns),
    ]
    return sorted(combined, key=lambda a: a.score, reverse=True)