"""Region pack interface.

A RegionPack supplies everything country-specific: currency, tax rules, retirement
instruments, expense categories and synthetic merchants. All rules are SIMPLIFIED and
SYNTHETIC. They exist for education/demonstration and are not real tax law.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path

from app.engine.money import Currency, Money


@dataclass(frozen=True)
class TaxBracket:
    """Progressive bracket. ``upper=None`` means no upper bound."""

    upper: Money | None
    rate: Decimal


@dataclass(frozen=True)
class BracketLine:
    lower: Money
    upper: Money | None
    rate: Decimal
    taxed_amount: Money
    tax: Money


@dataclass(frozen=True)
class TaxResult:
    regime: str
    gross_income: Money
    deductions: Money
    taxable_income: Money
    tax: Money
    effective_rate: Decimal
    marginal_rate: Decimal
    breakdown: tuple[BracketLine, ...]
    rule_ids: tuple[str, ...]  # knowledge-doc IDs to cite in decision traces


class InstrumentKind(str, Enum):
    PRE_TAX = "pre_tax"  # contribution reduces taxable income
    POST_TAX = "post_tax"  # contribution from taxed income, tax-advantaged growth
    EMPLOYER = "employer"  # employer-funded or matched


@dataclass(frozen=True)
class RetirementInstrument:
    id: str
    name: str
    kind: InstrumentKind
    annual_limit: Money | None
    description: str
    rule_id: str  # knowledge-doc ID backing this instrument


@dataclass(frozen=True)
class ExpenseCategory:
    id: str
    label: str
    essential: bool  # essential spending counts toward emergency-fund runway


@dataclass(frozen=True)
class RegionPack(ABC):
    code: str
    name: str
    currency: Currency
    locale: str
    extras: Mapping[str, str] = field(default_factory=dict)

    # ---- tax configuration ----
    @abstractmethod
    def tax_regimes(self) -> Sequence[str]:
        """Names of selectable regimes (e.g. ``["standard"]`` or ``["old", "new"]``)."""

    @abstractmethod
    def brackets(self, regime: str) -> Sequence[TaxBracket]:
        """Ascending brackets for the regime; last bracket should have ``upper=None``."""

    @abstractmethod
    def standard_deduction(self, regime: str) -> Money:
        ...

    # ---- planning configuration ----
    @abstractmethod
    def retirement_instruments(self) -> Sequence[RetirementInstrument]:
        ...

    @abstractmethod
    def categories(self) -> Sequence[ExpenseCategory]:
        ...

    @abstractmethod
    def merchants(self) -> Mapping[str, Sequence[str]]:
        """Category id -> synthetic merchant names, used by the data generator."""

    @abstractmethod
    def knowledge_dir(self) -> Path:
        """Folder of rule docs ingested into the Meko knowledge base."""

    # ---- shared behaviour ----
    def allows_instrument_deductions(self, regime: str) -> bool:
        """Whether contributions to pre-tax retirement instruments reduce taxable income
        under ``regime``. Packs override this when a regime disallows them."""
        return True

    def default_regime(self) -> str:
        return self.tax_regimes()[0]

    def rule_id(self, topic: str, regime: str | None = None) -> str:
        parts = [self.code, topic] + ([regime] if regime else [])
        return ".".join(parts)

    def compute_tax(
        self,
        gross_income: Money,
        regime: str | None = None,
        extra_deductions: Money | None = None,
    ) -> TaxResult:
        """Annual progressive tax on ``gross_income`` after deductions."""
        regime = regime or self.default_regime()
        if regime not in self.tax_regimes():
            raise ValueError(f"Unknown regime '{regime}' for region '{self.code}'")
        if gross_income.currency != self.currency:
            raise ValueError("gross_income currency does not match the region's currency")

        zero = Money.zero(self.currency)
        extra = extra_deductions or zero
        deductions = self.standard_deduction(regime) + extra
        taxable = gross_income - deductions
        if taxable.is_negative():
            taxable = zero

        lines: list[BracketLine] = []
        remaining = taxable
        lower = zero
        total_tax = Decimal(0)  # accumulate exactly, round once at the end
        marginal = Decimal(0)

        for bracket in self.brackets(regime):
            if remaining.is_zero():
                break
            width = remaining if bracket.upper is None else min(remaining, bracket.upper - lower)
            if width.is_negative() or width.is_zero():
                lower = bracket.upper if bracket.upper is not None else lower
                continue
            exact = Decimal(width.minor) * bracket.rate
            total_tax += exact
            marginal = bracket.rate
            lines.append(
                BracketLine(
                    lower=lower,
                    upper=bracket.upper,
                    rate=bracket.rate,
                    taxed_amount=width,
                    tax=Money(int(exact.to_integral_value(rounding="ROUND_HALF_UP")), self.currency),
                )
            )
            remaining = remaining - width
            lower = lower + width

        tax = Money(int(total_tax.to_integral_value(rounding="ROUND_HALF_UP")), self.currency)
        effective = (
            Decimal(tax.minor) / Decimal(gross_income.minor)
            if gross_income.minor > 0
            else Decimal(0)
        )
        return TaxResult(
            regime=regime,
            gross_income=gross_income,
            deductions=deductions,
            taxable_income=taxable,
            tax=tax,
            effective_rate=effective,
            marginal_rate=marginal,
            breakdown=tuple(lines),
            rule_ids=(self.rule_id("tax", regime),),
        )

    def compare_regimes(
        self, gross_income: Money, extra_deductions: Money | None = None
    ) -> dict[str, TaxResult]:
        """Compute tax under every regime so the Tax agent can recommend the cheaper one."""
        return {r: self.compute_tax(gross_income, r, extra_deductions) for r in self.tax_regimes()}