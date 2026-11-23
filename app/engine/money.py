"""Exact money arithmetic.

Amounts are stored as integer minor units (cents / paise) plus a currency, and all
intermediate math uses ``Decimal``. Floats are never used for money.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import total_ordering
from typing import Literal

Number = int | Decimal | str | float


@dataclass(frozen=True)
class Currency:
    code: str
    symbol: str
    minor_units: int  # digits after the decimal point
    grouping: Literal["western", "indian"]


USD = Currency("USD", "$", 2, "western")
INR = Currency("INR", "₹", 2, "indian")

_CURRENCIES: dict[str, Currency] = {c.code: c for c in (USD, INR)}


def get_currency(code: str) -> Currency:
    try:
        return _CURRENCIES[code.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported currency: {code}") from exc


class CurrencyMismatchError(ValueError):
    """Raised when combining or comparing money in different currencies."""


def _to_decimal(value: Number) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a valid numeric amount")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))  # repr avoids binary-float artifacts like 0.1 -> 0.1000000000000000055
    return Decimal(value)


def _round_half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _group(whole: int, grouping: str) -> str:
    digits = str(whole)
    if len(digits) <= 3:
        return digits
    if grouping == "indian":
        head, tail = digits[:-3], digits[-3:]
        parts: list[str] = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        return ",".join([*parts, tail])
    return f"{whole:,}"


@total_ordering
@dataclass(frozen=True)
class Money:
    minor: int
    currency: Currency

    # ---- construction ----
    @classmethod
    def zero(cls, currency: Currency) -> Money:
        return cls(0, currency)

    @classmethod
    def of(cls, amount: Number, currency: Currency) -> Money:
        """Build from a major-unit amount, e.g. ``Money.of("1234.56", USD)``."""
        scaled = _to_decimal(amount) * (Decimal(10) ** currency.minor_units)
        return cls(_round_half_up(scaled), currency)

    # ---- views ----
    @property
    def amount(self) -> Decimal:
        return Decimal(self.minor) / (Decimal(10) ** self.currency.minor_units)

    def is_zero(self) -> bool:
        return self.minor == 0

    def is_negative(self) -> bool:
        return self.minor < 0

    def __abs__(self) -> Money:
        return Money(abs(self.minor), self.currency)

    def __neg__(self) -> Money:
        return Money(-self.minor, self.currency)

    # ---- arithmetic ----
    def _check(self, other: Money) -> None:
        if not isinstance(other, Money):
            raise TypeError(f"Expected Money, got {type(other).__name__}")
        if other.currency != self.currency:
            raise CurrencyMismatchError(f"{self.currency.code} vs {other.currency.code}")

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.minor + other.minor, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(self.minor - other.minor, self.currency)

    def __mul__(self, factor: Number) -> Money:
        return Money(_round_half_up(Decimal(self.minor) * _to_decimal(factor)), self.currency)

    __rmul__ = __mul__

    def __truediv__(self, divisor: Number) -> Money:
        d = _to_decimal(divisor)
        if d == 0:
            raise ZeroDivisionError("Cannot divide Money by zero")
        return Money(_round_half_up(Decimal(self.minor) / d), self.currency)

    def ratio(self, other: Money) -> Decimal:
        """Dimensionless ratio self / other (e.g. debt-to-income)."""
        self._check(other)
        if other.minor == 0:
            raise ZeroDivisionError("Cannot take ratio against zero Money")
        return Decimal(self.minor) / Decimal(other.minor)

    def allocate(self, ratios: Sequence[Number]) -> list[Money]:
        """Split into parts proportional to ``ratios`` with no minor unit lost.

        Uses the largest-remainder method, so the parts always sum exactly to ``self``.
        """
        weights = [_to_decimal(r) for r in ratios]
        if not weights or any(w < 0 for w in weights):
            raise ValueError("ratios must be a non-empty sequence of non-negative numbers")
        total = sum(weights)
        if total == 0:
            raise ValueError("ratios must not all be zero")

        sign = -1 if self.minor < 0 else 1
        magnitude = abs(self.minor)
        exact = [Decimal(magnitude) * w / total for w in weights]
        floors = [int(x) for x in exact]
        remainder = magnitude - sum(floors)
        order = sorted(range(len(weights)), key=lambda i: exact[i] - floors[i], reverse=True)
        for i in order[:remainder]:
            floors[i] += 1
        return [Money(sign * f, self.currency) for f in floors]

    # ---- ordering ----
    def __lt__(self, other: Money) -> bool:
        self._check(other)
        return self.minor < other.minor

    # ---- display ----
    def format(self, *, symbol: bool = True, decimals: int | None = None) -> str:
        places = self.currency.minor_units if decimals is None else decimals
        value = self.amount.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
        sign = "-" if value < 0 else ""
        whole_part, _, frac_part = f"{abs(value):f}".partition(".")
        text = _group(int(whole_part), self.currency.grouping)
        if places > 0:
            text += "." + frac_part.ljust(places, "0")[:places]
        prefix = self.currency.symbol if symbol else ""
        return f"{sign}{prefix}{text}"

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return f"Money({self.amount} {self.currency.code})"


def sum_money(items: Iterable[Money], currency: Currency) -> Money:
    """Sum an iterable of Money; returns zero for an empty iterable."""
    total = Money.zero(currency)
    for item in items:
        total = total + item
    return total