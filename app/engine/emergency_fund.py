"""Emergency fund planning.

The target is a PLANNING HEURISTIC (documented in the knowledge base as a rule), not a law:
a 3-month base, plus months for income volatility, dependents and single-income households,
rounded to the nearest half month and clamped to a sensible range.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from app.engine.money import Money

BASE_MONTHS = Decimal("3")
MIN_MONTHS = Decimal("3")
MAX_MONTHS = Decimal("12")
VOLATILITY_MONTHS_PER_CV = Decimal("6")  # +6 months per 1.0 of income coefficient of variation
VOLATILITY_CAP = Decimal("3")
DEPENDENT_MONTHS_EACH = Decimal("0.5")
DEPENDENT_CAP = Decimal("2")
SINGLE_INCOME_MONTHS = Decimal("1")

RULE_ID = "planning.emergency_fund"


@dataclass(frozen=True)
class EmergencyFundTarget:
    months: Decimal
    components: dict[str, Decimal]  # for traces: why this many months
    monthly_essential: Money
    amount: Money
    rule_id: str = RULE_ID


@dataclass(frozen=True)
class EmergencyFundStatus:
    liquid_savings: Money
    runway_months: Decimal
    target_months: Decimal
    target_amount: Money
    shortfall: Money  # zero when fully funded
    funded_ratio: Decimal  # 1.0 = fully funded (capped at 1)
    is_funded: bool


def _round_half_month(value: Decimal) -> Decimal:
    return (value * 2).quantize(Decimal(1), rounding=ROUND_HALF_UP) / 2


def target_months(
    *,
    income_cv: float = 0.0,
    dependents: int = 0,
    single_income: bool = False,
) -> tuple[Decimal, dict[str, Decimal]]:
    """Recommended months of essential expenses, plus the components behind the number."""
    if income_cv < 0 or dependents < 0:
        raise ValueError("income_cv and dependents must not be negative")
    volatility = min(VOLATILITY_CAP, Decimal(str(income_cv)) * VOLATILITY_MONTHS_PER_CV)
    dependent = min(DEPENDENT_CAP, DEPENDENT_MONTHS_EACH * dependents)
    single = SINGLE_INCOME_MONTHS if single_income else Decimal(0)
    raw = BASE_MONTHS + volatility + dependent + single
    months = max(MIN_MONTHS, min(MAX_MONTHS, _round_half_month(raw)))
    cents = Decimal("0.01")  # components are for display; rounding them doesn't change ``months``
    components = {
        "base": BASE_MONTHS,
        "income_volatility": volatility.quantize(cents),
        "dependents": dependent.quantize(cents),
        "single_income": single.quantize(cents),
    }
    return months, components


def build_target(
    monthly_essential: Money,
    *,
    income_cv: float = 0.0,
    dependents: int = 0,
    single_income: bool = False,
) -> EmergencyFundTarget:
    months, components = target_months(
        income_cv=income_cv, dependents=dependents, single_income=single_income
    )
    return EmergencyFundTarget(
        months=months,
        components=components,
        monthly_essential=monthly_essential,
        amount=monthly_essential * months,
    )


def runway_months(liquid_savings: Money, monthly_essential: Money) -> Decimal:
    """How many months essential expenses are covered by liquid savings."""
    if monthly_essential.minor <= 0:
        raise ValueError("monthly_essential must be positive")
    if liquid_savings.minor <= 0:
        return Decimal(0)
    return liquid_savings.ratio(monthly_essential)


def status(liquid_savings: Money, target: EmergencyFundTarget) -> EmergencyFundStatus:
    runway = runway_months(liquid_savings, target.monthly_essential)
    zero = Money.zero(liquid_savings.currency)
    shortfall = target.amount - liquid_savings
    if shortfall.is_negative():
        shortfall = zero
    ratio = (
        Decimal(1)
        if target.amount.minor == 0
        else min(Decimal(1), max(Decimal(0), liquid_savings.ratio(target.amount)))
    )
    return EmergencyFundStatus(
        liquid_savings=liquid_savings,
        runway_months=runway,
        target_months=target.months,
        target_amount=target.amount,
        shortfall=shortfall,
        funded_ratio=ratio,
        is_funded=shortfall.is_zero(),
    )


def months_to_fund(shortfall: Money, monthly_contribution: Money) -> int | None:
    """Months of saving needed to close the shortfall; ``None`` if contributions are zero."""
    if shortfall.minor <= 0:
        return 0
    if monthly_contribution.minor <= 0:
        return None
    return -(-shortfall.minor // monthly_contribution.minor)  # exact integer ceiling


def required_monthly_contribution(shortfall: Money, months: int) -> Money:
    """Monthly amount needed to close the shortfall within ``months`` (rounded up)."""
    if months <= 0:
        raise ValueError("months must be positive")
    if shortfall.minor <= 0:
        return Money.zero(shortfall.currency)
    return Money(-(-shortfall.minor // months), shortfall.currency)