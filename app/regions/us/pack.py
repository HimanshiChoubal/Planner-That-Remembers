"""United States region pack.

SYNTHETIC AND SIMPLIFIED. Figures are an illustrative snapshot (approximately 2024, single
filer) for demonstration only. They are not tax advice or current law. Not modelled: FICA,
state/local tax, credits, AMT, capital gains, phase-outs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from app.engine.money import USD, Currency, Money
from app.regions.base import (
    ExpenseCategory,
    InstrumentKind,
    RegionPack,
    RetirementInstrument,
    TaxBracket,
)


def _usd(amount: int) -> Money:
    return Money.of(amount, USD)


_BRACKETS: tuple[TaxBracket, ...] = (
    TaxBracket(_usd(11_600), Decimal("0.10")),
    TaxBracket(_usd(47_150), Decimal("0.12")),
    TaxBracket(_usd(100_525), Decimal("0.22")),
    TaxBracket(_usd(191_950), Decimal("0.24")),
    TaxBracket(_usd(243_725), Decimal("0.32")),
    TaxBracket(_usd(609_350), Decimal("0.35")),
    TaxBracket(None, Decimal("0.37")),
)

_INSTRUMENTS: tuple[RetirementInstrument, ...] = (
    RetirementInstrument(
        id="401k",
        name="401(k) (traditional)",
        kind=InstrumentKind.PRE_TAX,
        annual_limit=_usd(23_000),
        description="Employer plan; contributions reduce taxable income, growth is tax-deferred.",
        rule_id="us.retirement.401k",
    ),
    RetirementInstrument(
        id="401k_match",
        name="Employer 401(k) match",
        kind=InstrumentKind.EMPLOYER,
        annual_limit=None,
        description="Employer contribution matched to employee deferrals; capture it first.",
        rule_id="us.retirement.401k_match",
    ),
    RetirementInstrument(
        id="ira_traditional",
        name="Traditional IRA",
        kind=InstrumentKind.PRE_TAX,
        annual_limit=_usd(7_000),
        description="Individual account; deduction may be limited by income and workplace plan.",
        rule_id="us.retirement.ira_traditional",
    ),
    RetirementInstrument(
        id="ira_roth",
        name="Roth IRA",
        kind=InstrumentKind.POST_TAX,
        annual_limit=_usd(7_000),
        description="Contributions from taxed income; qualified withdrawals are tax-free.",
        rule_id="us.retirement.ira_roth",
    ),
)

_CATEGORIES: tuple[ExpenseCategory, ...] = (
    ExpenseCategory("housing", "Housing (rent / mortgage)", True),
    ExpenseCategory("utilities", "Utilities", True),
    ExpenseCategory("groceries", "Groceries", True),
    ExpenseCategory("transport", "Transport", True),
    ExpenseCategory("healthcare", "Healthcare", True),
    ExpenseCategory("insurance", "Insurance", True),
    ExpenseCategory("childcare", "Childcare", True),
    ExpenseCategory("debt_payment", "Debt payments", True),
    ExpenseCategory("dining", "Dining out", False),
    ExpenseCategory("subscriptions", "Subscriptions", False),
    ExpenseCategory("entertainment", "Entertainment", False),
    ExpenseCategory("shopping", "Shopping", False),
    ExpenseCategory("travel", "Travel", False),
    ExpenseCategory("education", "Education", False),
    ExpenseCategory("other", "Other", False),
)

_MERCHANTS: Mapping[str, Sequence[str]] = {
    "housing": ["MAPLE RIDGE APTS", "HARBORVIEW PROPERTY MGMT", "CITYSIDE MORTGAGE PMT"],
    "utilities": ["BRIGHTGRID ELECTRIC", "CLEARWATER UTILITIES", "FIBERLINE INTERNET"],
    "groceries": ["FRESHMART #214", "GREEN BASKET MARKET", "VALUE FOODS 0873", "HARVEST CO-OP"],
    "transport": ["QUICKFUEL 5531", "METRO TRANSIT", "RIDEWAY *TRIP", "PARKSMART GARAGE"],
    "healthcare": ["WELLNESS PHARMACY", "NORTHSIDE CLINIC", "VISIONPLUS OPTICAL"],
    "insurance": ["SAFEHARBOR INSURANCE", "AUTOGUARD PREMIUM"],
    "childcare": ["LITTLE STEPS DAYCARE", "BRIGHT FUTURES ACADEMY"],
    "debt_payment": ["STUDENTLOAN SERVICING", "CARDCO PAYMENT", "AUTO FINANCE PMT"],
    "dining": ["CORNER BISTRO", "NOODLE HOUSE 22", "SUNRISE CAFE", "TACO LOCO"],
    "subscriptions": ["STREAMBOX MONTHLY", "TUNEWAVE PREMIUM", "CLOUDDRIVE STORAGE"],
    "entertainment": ["CINEPLEX 9", "ARCADE ZONE", "LIVE NATION*TIX"],
    "shopping": ["SHOPSPHERE ONLINE", "HOMESTYLE GOODS", "URBAN THREADS"],
    "travel": ["SKYWAY AIRLINES", "STAYWELL HOTELS", "ROADTRIP RENTALS"],
    "education": ["OPENCOURSE LEARNING", "CAMPUS BOOKSTORE"],
    "other": ["MISC PAYMENT", "ATM WITHDRAWAL", "TRANSFER OUT"],
}


@dataclass(frozen=True)
class USPack(RegionPack):
    code: str = "us"
    name: str = "United States"
    currency: Currency = USD
    locale: str = "en-US"

    def tax_regimes(self) -> Sequence[str]:
        return ("standard",)

    def brackets(self, regime: str) -> Sequence[TaxBracket]:
        return _BRACKETS

    def standard_deduction(self, regime: str) -> Money:
        return _usd(14_600)

    def retirement_instruments(self) -> Sequence[RetirementInstrument]:
        return _INSTRUMENTS

    def categories(self) -> Sequence[ExpenseCategory]:
        return _CATEGORIES

    def merchants(self) -> Mapping[str, Sequence[str]]:
        return _MERCHANTS

    def knowledge_dir(self) -> Path:
        return Path(__file__).parent / "knowledge"


US_PACK = USPack()