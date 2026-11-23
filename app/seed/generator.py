"""Synthetic income and transaction generator.

Reproducible (fixed seed) and consistent with each persona's stated profile: monthly spending
lands near ``monthly_expenses``, essential categories near ``monthly_essential_expenses``, and
debt minimums are paid exactly. Because the data is generated, the true category of every
transaction is known (``Transaction.category``) while ``merchant`` is a messy bank string, which
gives real accuracy numbers for the categorizer. Optionally injects a duplicate charge and an
outlier in the latest month for the anomaly detector to find.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

import numpy as np

from app.engine.anomaly import Transaction
from app.engine.money import Money
from app.regions.registry import get_pack
from app.seed.calibration import CALIBRATION, RegionCalibration
from app.seed.personas import Persona

OUTLIER_MULTIPLIER = 6.0  # injected outlier = this many times the category's median amount
SPLIT_CONCENTRATION = 3.0  # Dirichlet concentration: higher means more even transaction sizes
_FIXED_ESSENTIAL = ("debt_payment", "childcare")


@dataclass(frozen=True)
class SyntheticHistory:
    months: int
    first_month: date
    income: tuple[Money, ...]  # monthly income, oldest first
    transactions: tuple[Transaction, ...]  # sorted by date
    injected_ids: tuple[str, ...] = ()  # the duplicate and outlier added for anomaly demos


def month_starts(as_of: date, months: int) -> list[date]:
    """First day of each of the ``months`` complete months before ``as_of``, oldest first."""
    out = []
    for back in range(months, 0, -1):
        y, m = as_of.year, as_of.month - back
        while m <= 0:
            m += 12
            y -= 1
        out.append(date(y, m, 1))
    return out


def noisy_description(name: str, rng: np.random.Generator) -> str:
    """Turn a clean merchant name into a messy bank-statement string."""
    text = name
    if rng.random() < 0.5:
        text = "POS DEBIT " + text
    if rng.random() < 0.5:
        text += f" {int(rng.integers(100, 99999))}"
    if rng.random() < 0.25 and len(text) > 6:
        i = int(rng.integers(2, len(text) - 2))
        text = text[:i] + text[i + 1:]  # dropped character
    if rng.random() < 0.25:
        text = text.lower()
    return text


def _stable_offset(text: str) -> int:
    return sum(ord(ch) * (i + 1) for i, ch in enumerate(text)) % 10_000


def _income_series(persona: Persona, starts: list[date], cal: RegionCalibration, rng: np.random.Generator) -> list[Money]:
    cur = persona.profile.currency
    mean = float(persona.profile.monthly_gross_income.amount)  # type: ignore[union-attr]
    cv = persona.income_cv
    sigma = float(np.sqrt(np.log(1 + cv**2)))
    out: list[Money] = []
    for start in starts:
        factor = float(rng.lognormal(-sigma**2 / 2, sigma)) if sigma > 0 else 1.0
        if persona.income_kind == "business":
            factor *= cal.business_season.get(start.month, 1.0)
        out.append(Money.of(f"{mean * factor:.2f}", cur))
    return out


def generate_history(
    persona: Persona,
    *,
    as_of: date,
    months: int = 18,
    seed: int = 0,
    inject_anomalies: bool = True,
) -> SyntheticHistory:
    if months < 6:
        raise ValueError("months must be at least 6")
    p = persona.profile
    pack = get_pack(p.region)
    cal = CALIBRATION[p.region]
    cur = p.currency
    rng = np.random.default_rng(seed + _stable_offset(persona.id))
    starts = month_starts(as_of, months)
    income = _income_series(persona, starts, cal, rng)

    expenses = float(p.monthly_expenses.amount)  # type: ignore[union-attr]
    essential = float(p.monthly_essential_expenses.amount)  # type: ignore[union-attr]
    debt_fixed = sum(float(d.min_payment.amount) for d in (p.debts or ()))
    childcare = float(persona.childcare_monthly.amount) if persona.childcare_monthly else 0.0
    ess_pool = max(0.0, essential - debt_fixed - childcare)
    disc_pool = max(0.0, expenses - essential)
    categories = pack.categories()
    ess_ids = [c.id for c in categories if c.essential and c.id not in _FIXED_ESSENTIAL]
    disc_ids = [c.id for c in categories if not c.essential]
    merchants = pack.merchants()

    def shares(ids: list[str]) -> dict[str, float]:
        total = sum(cal.category_shares.get(i, 0.0) for i in ids)
        return {i: cal.category_shares.get(i, 0.0) / total for i in ids}

    pools = {**{k: v * ess_pool for k, v in shares(ess_ids).items()}, **{k: v * disc_pool for k, v in shares(disc_ids).items()}}
    txns: list[Transaction] = []
    counter = 0

    def add(day: date, amount: float, kind: str, category: str, merchant: str) -> Transaction | None:
        nonlocal counter
        if amount < 0.01:
            return None
        counter += 1
        txn = Transaction(f"{persona.id}-{counter:05d}", day, Money.of(f"{amount:.2f}", cur), kind, category, merchant)  # type: ignore[arg-type]
        txns.append(txn)
        return txn

    for start, inc in zip(starts, income, strict=True):
        dim = calendar.monthrange(start.year, start.month)[1]
        pay_day = 1 if persona.income_kind == "salaried" else int(rng.integers(1, dim + 1))
        employer = "PAYROLL ACME CORP" if persona.income_kind == "salaried" else "CLIENT PAYMENT"
        add(start.replace(day=pay_day), float(inc.amount), "income", "income", employer)

        for cat, pool in pools.items():
            sigma = cal.monthly_noise
            total = pool * cal.seasonal.get(start.month, {}).get(cat, 1.0) * float(rng.lognormal(-sigma**2 / 2, sigma))
            n = cal.txns_per_month.get(cat, 1)
            parts = rng.dirichlet(np.full(n, SPLIT_CONCENTRATION)) * total if n > 1 else np.array([total])
            fixed_day = cal.fixed_days.get(cat)
            days = [fixed_day] * n if fixed_day else sorted(int(d) for d in rng.integers(1, dim + 1, size=n))
            for amount, day in zip(parts, days, strict=True):
                name = str(rng.choice(list(merchants[cat])))
                add(start.replace(day=min(day, dim)), float(amount), "expense", cat, noisy_description(name, rng))
        for i, debt in enumerate(p.debts or ()):
            name = merchants["debt_payment"][i % len(merchants["debt_payment"])]
            add(start.replace(day=min(20, dim)), float(debt.min_payment.amount), "expense", "debt_payment", noisy_description(name, rng))
        if persona.childcare_monthly:
            add(start.replace(day=min(3, dim)), childcare, "expense", "childcare", noisy_description(merchants["childcare"][0], rng))

    injected: list[str] = []
    if inject_anomalies:
        last = starts[-1]
        in_last = [t for t in txns if (t.date.year, t.date.month) == (last.year, last.month)]
        dining = next((t for t in in_last if t.category == "dining" and t.kind == "expense"), None)
        if dining is not None:
            dup = add(min(dining.date.replace(day=min(dining.date.day + 1, 28)), as_of), float(dining.amount.amount), "expense", "dining", dining.merchant)
            if dup:
                injected.append(dup.id)
        grocery_txns = [t for t in in_last if t.category == "groceries"]
        if grocery_txns:
            median = float(np.median([float(t.amount.amount) for t in grocery_txns]))
            groceries = grocery_txns[0]
            idx = txns.index(groceries)
            big = Transaction(groceries.id, groceries.date, Money.of(f"{median * OUTLIER_MULTIPLIER:.2f}", cur), "expense", "groceries", groceries.merchant)
            txns[idx] = big
            injected.append(big.id)

    txns.sort(key=lambda t: (t.date, t.id))
    return SyntheticHistory(months, starts[0], tuple(income), tuple(txns), tuple(injected))