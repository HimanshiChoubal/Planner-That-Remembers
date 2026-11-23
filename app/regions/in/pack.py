"""India region pack.

SYNTHETIC AND SIMPLIFIED. Figures are an illustrative snapshot (approximately FY2025-26,
individual resident) for demonstration only. They are not tax advice or current law. Not
modelled: section 87A rebate, surcharge, health and education cess, capital gains, HRA, and
the detailed deduction sections. Deductions under the "old" regime are supplied by callers
via ``extra_deductions``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from app.engine.money import INR, Currency, Money
from app.regions.base import (
    ExpenseCategory,
    InstrumentKind,
    RegionPack,
    RetirementInstrument,
    TaxBracket,
)

LAKH = 100_000


def _inr(amount: int) -> Money:
    return Money.of(amount, INR)


_NEW_BRACKETS: tuple[TaxBracket, ...] = (
    TaxBracket(_inr(4 * LAKH), Decimal("0.00")),
    TaxBracket(_inr(8 * LAKH), Decimal("0.05")),
    TaxBracket(_inr(12 * LAKH), Decimal("0.10")),
    TaxBracket(_inr(16 * LAKH), Decimal("0.15")),
    TaxBracket(_inr(20 * LAKH), Decimal("0.20")),
    TaxBracket(_inr(24 * LAKH), Decimal("0.25")),
    TaxBracket(None, Decimal("0.30")),
)

_OLD_BRACKETS: tuple[TaxBracket, ...] = (
    TaxBracket(_inr(250_000), Decimal("0.00")),
    TaxBracket(_inr(5 * LAKH), Decimal("0.05")),
    TaxBracket(_inr(10 * LAKH), Decimal("0.20")),
    TaxBracket(None, Decimal("0.30")),
)

_STANDARD_DEDUCTION = {"new": _inr(75_000), "old": _inr(50_000)}

_INSTRUMENTS: tuple[RetirementInstrument, ...] = (
    RetirementInstrument(
        id="epf",
        name="Employees' Provident Fund (EPF)",
        kind=InstrumentKind.PRE_TAX,
        annual_limit=None,
        description="Salaried retirement fund; employee share is deductible under the old regime.",
        rule_id="in.retirement.epf",
    ),
    RetirementInstrument(
        id="epf_employer",
        name="Employer EPF contribution",
        kind=InstrumentKind.EMPLOYER,
        annual_limit=None,
        description="Employer share of provident fund contributions.",
        rule_id="in.retirement.epf_employer",
    ),
    RetirementInstrument(
        id="ppf",
        name="Public Provident Fund (PPF)",
        kind=InstrumentKind.PRE_TAX,
        annual_limit=_inr(150_000),
        description="Long-term government-backed savings; deductible under the old regime only.",
        rule_id="in.retirement.ppf",
    ),
    RetirementInstrument(
        id="nps",
        name="National Pension System (NPS, Tier I)",
        kind=InstrumentKind.PRE_TAX,
        annual_limit=_inr(50_000),
        description="Market-linked pension account; extra deduction cap shown is illustrative.",
        rule_id="in.retirement.nps",
    ),
)

_CATEGORIES: tuple[ExpenseCategory, ...] = (
    ExpenseCategory("housing", "Housing (rent / EMI)", True),
    ExpenseCategory("utilities", "Utilities", True),
    ExpenseCategory("groceries", "Groceries", True),
    ExpenseCategory("transport", "Transport", True),
    ExpenseCategory("healthcare", "Healthcare", True),
    ExpenseCategory("insurance", "Insurance", True),
    ExpenseCategory("childcare", "Childcare and schooling", True),
    ExpenseCategory("domestic_help", "Domestic help", True),
    ExpenseCategory("debt_payment", "Loan EMIs and card payments", True),
    ExpenseCategory("dining", "Dining out", False),
    ExpenseCategory("subscriptions", "Subscriptions", False),
    ExpenseCategory("entertainment", "Entertainment", False),
    ExpenseCategory("shopping", "Shopping", False),
    ExpenseCategory("travel", "Travel", False),
    ExpenseCategory("education", "Education", False),
    ExpenseCategory("other", "Other", False),
)

_MERCHANTS: Mapping[str, Sequence[str]] = {
    "housing": ["SUNRISE HOMES RENT", "LAKEVIEW SOCIETY MAINT", "HOMELOAN EMI AUTO-DEBIT"],
    "utilities": ["CITYPOWER BILL PAY", "AQUAFLOW WATER BOARD", "FIBERNET BROADBAND"],
    "groceries": ["FRESHBASKET GROCERY", "DAILY NEEDS MART", "VEGGIE HUB", "SABZI WALA UPI"],
    "transport": ["PETROLPLUS 4421", "METRO SMARTCARD", "RIDEGO *TRIP", "AUTO RICKSHAW UPI"],
    "healthcare": ["CAREWELL PHARMACY", "LIFELINE DIAGNOSTICS", "CITY CLINIC OPD"],
    "insurance": ["SECURELIFE PREMIUM", "HEALTHSHIELD POLICY"],
    "childcare": ["LITTLE SCHOLARS SCHOOL FEE", "KIDS CORNER CRECHE"],
    "domestic_help": ["HOMEHELP SERVICES", "MAID PAYMENT UPI"],
    "debt_payment": ["CARDBANK PAYMENT", "AUTOLOAN EMI", "PERSONALLOAN EMI"],
    "dining": ["SPICE ROUTE CAFE", "TIFFIN CENTRE", "PIZZA CORNER", "CHAI POINT"],
    "subscriptions": ["STREAMBOX INDIA", "TUNEWAVE PREMIUM", "CLOUDDRIVE STORAGE"],
    "entertainment": ["MULTIPLEX TICKETS", "GAMEZONE ARCADE"],
    "shopping": ["SHOPSPHERE ONLINE", "FASHION STREET", "HOMESTYLE GOODS"],
    "travel": ["RAILWAY BOOKING", "SKYWAY AIRLINES", "STAYWELL HOTELS"],
    "education": ["OPENCOURSE LEARNING", "COACHING INSTITUTE FEE"],
    "other": ["ATM WITHDRAWAL", "UPI TRANSFER", "MISC PAYMENT"],
}


@dataclass(frozen=True)
class IndiaPack(RegionPack):
    code: str = "in"
    name: str = "India"
    currency: Currency = INR
    locale: str = "en-IN"

    def tax_regimes(self) -> Sequence[str]:
        return ("new", "old")

    def brackets(self, regime: str) -> Sequence[TaxBracket]:
        return _NEW_BRACKETS if regime == "new" else _OLD_BRACKETS

    def standard_deduction(self, regime: str) -> Money:
        return _STANDARD_DEDUCTION[regime]

    def allows_instrument_deductions(self, regime: str) -> bool:
        return regime == "old"  # the simplified new regime disallows these deductions

    def retirement_instruments(self) -> Sequence[RetirementInstrument]:
        return _INSTRUMENTS

    def categories(self) -> Sequence[ExpenseCategory]:
        return _CATEGORIES

    def merchants(self) -> Mapping[str, Sequence[str]]:
        return _MERCHANTS

    def knowledge_dir(self) -> Path:
        return Path(__file__).parent / "knowledge"


INDIA_PACK = IndiaPack()