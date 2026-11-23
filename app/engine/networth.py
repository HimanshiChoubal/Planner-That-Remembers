"""Net worth, trend and core household ratios (savings rate, debt-to-income)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from app.engine.money import Currency, Money

AssetKind = Literal["cash", "savings", "investments", "retirement", "property", "vehicle", "other"]
_LIQUID_KINDS: frozenset[str] = frozenset({"cash", "savings"})


@dataclass(frozen=True)
class Asset:
    name: str
    kind: AssetKind
    value: Money
    liquid: bool | None = None  # override; defaults to True only for cash and savings

    @property
    def is_liquid(self) -> bool:
        return self.kind in _LIQUID_KINDS if self.liquid is None else self.liquid


@dataclass(frozen=True)
class Liability:
    name: str
    balance: Money
    apr: Decimal = Decimal(0)
    min_payment: Money | None = None


@dataclass(frozen=True)
class NetWorthSnapshot:
    as_of: date
    assets: Money
    liabilities: Money
    net_worth: Money
    liquid_assets: Money
    by_kind: Mapping[str, Money]


@dataclass(frozen=True)
class NetWorthChange:
    absolute: Money
    percent: Decimal | None  # relative to the starting magnitude; None if it started at zero
    months: int
    monthly_average: Money


def snapshot(
    assets: Iterable[Asset],
    liabilities: Iterable[Liability],
    as_of: date,
    currency: Currency,
) -> NetWorthSnapshot:
    zero = Money.zero(currency)
    total_assets = zero
    liquid = zero
    by_kind: dict[str, Money] = defaultdict(lambda: zero)
    for a in assets:
        if a.value.currency != currency:
            raise ValueError("All assets must be in the snapshot currency")
        total_assets = total_assets + a.value
        by_kind[a.kind] = by_kind[a.kind] + a.value
        if a.is_liquid:
            liquid = liquid + a.value
    total_liabilities = zero
    for liab in liabilities:
        if liab.balance.currency != currency:
            raise ValueError("All liabilities must be in the snapshot currency")
        total_liabilities = total_liabilities + liab.balance
    return NetWorthSnapshot(
        as_of=as_of,
        assets=total_assets,
        liabilities=total_liabilities,
        net_worth=total_assets - total_liabilities,
        liquid_assets=liquid,
        by_kind=dict(by_kind),
    )


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def trend(snapshots: Sequence[NetWorthSnapshot]) -> list[tuple[date, Money]]:
    """Net worth over time, oldest first."""
    return [(s.as_of, s.net_worth) for s in sorted(snapshots, key=lambda s: s.as_of)]


def change(first: NetWorthSnapshot, last: NetWorthSnapshot) -> NetWorthChange:
    """Net worth change between two snapshots (``first`` should be the earlier one)."""
    if last.as_of < first.as_of:
        raise ValueError("last snapshot must not be earlier than first")
    absolute = last.net_worth - first.net_worth
    months = _months_between(first.as_of, last.as_of)
    percent = (
        None
        if first.net_worth.minor == 0
        else Decimal(absolute.minor) / Decimal(abs(first.net_worth.minor))
    )
    monthly = absolute / months if months > 0 else absolute
    return NetWorthChange(absolute=absolute, percent=percent, months=months, monthly_average=monthly)


def savings_rate(monthly_income: Money, monthly_expenses: Money) -> Decimal:
    """(income - expenses) / income. Zero when there is no income. May be negative."""
    if monthly_income.minor <= 0:
        return Decimal(0)
    return (monthly_income - monthly_expenses).ratio(monthly_income)


def debt_to_income(monthly_debt_payments: Money, monthly_gross_income: Money) -> Decimal:
    """Monthly debt payments divided by gross monthly income. Zero when there is no income."""
    if monthly_gross_income.minor <= 0:
        return Decimal(0)
    return monthly_debt_payments.ratio(monthly_gross_income)