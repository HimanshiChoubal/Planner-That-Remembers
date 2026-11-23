"""Dashboard health metrics.

Combines net worth, savings rate, debt-to-income and emergency-fund runway into one object
for the dashboard. Each metric carries a status (good / watch / risk) and the rule ID behind
its threshold. Thresholds are ILLUSTRATIVE rules of thumb, not advice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

from app.engine import emergency_fund as ef
from app.engine import networth as nw
from app.engine.money import Currency, Money

Status = Literal["good", "watch", "risk"]

SAVINGS_GOOD = Decimal("0.20")
SAVINGS_WATCH = Decimal("0.10")
DTI_GOOD = Decimal("0.20")
DTI_WATCH = Decimal("0.36")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    value: Decimal | None
    display: str
    status: Status
    rule_id: str
    note: str


@dataclass(frozen=True)
class HouseholdMetrics:
    metrics: tuple[Metric, ...]
    net_worth: nw.NetWorthSnapshot
    emergency_target: ef.EmergencyFundTarget | None

    def get(self, key: str) -> Metric:
        for m in self.metrics:
            if m.key == key:
                return m
        raise KeyError(key)

    def worst_status(self) -> Status:
        order = {"good": 0, "watch": 1, "risk": 2}
        return max((m.status for m in self.metrics), key=lambda s: order[s])


def _pct(value: Decimal) -> str:
    return f"{value * 100:.1f}%"


def compute_metrics(
    *,
    currency: Currency,
    as_of: date,
    monthly_gross_income: Money,
    monthly_expenses: Money,
    monthly_essential: Money,
    monthly_debt_payments: Money,
    assets: Sequence[nw.Asset],
    liabilities: Sequence[nw.Liability],
    income_cv: float = 0.0,
    dependents: int = 0,
    single_income: bool = False,
) -> HouseholdMetrics:
    snap = nw.snapshot(assets, liabilities, as_of, currency)

    rate = nw.savings_rate(monthly_gross_income, monthly_expenses)
    if rate >= SAVINGS_GOOD:
        rate_status: Status = "good"
    elif rate >= SAVINGS_WATCH:
        rate_status = "watch"
    else:
        rate_status = "risk"

    dti = nw.debt_to_income(monthly_debt_payments, monthly_gross_income)
    if dti <= DTI_GOOD:
        dti_status: Status = "good"
    elif dti <= DTI_WATCH:
        dti_status = "watch"
    else:
        dti_status = "risk"

    nw_value = Decimal(snap.net_worth.amount)
    nw_status: Status = "good" if snap.net_worth.minor > 0 else "watch" if snap.net_worth.minor == 0 else "risk"

    metrics = [
        Metric("savings_rate", "Savings rate", rate, _pct(rate), rate_status,
               "planning.metric.savings_rate", f"Good at {_pct(SAVINGS_GOOD)}+, watch below {_pct(SAVINGS_WATCH)}."),
        Metric("debt_to_income", "Debt-to-income", dti, _pct(dti), dti_status,
               "planning.metric.debt_to_income", f"Good up to {_pct(DTI_GOOD)}, risk above {_pct(DTI_WATCH)}."),
        Metric("net_worth", "Net worth", nw_value, str(snap.net_worth), nw_status,
               "planning.metric.net_worth", "Assets minus liabilities."),
    ]

    target: ef.EmergencyFundTarget | None = None
    if monthly_essential.minor > 0:
        target = ef.build_target(
            monthly_essential, income_cv=income_cv, dependents=dependents, single_income=single_income
        )
        runway = ef.runway_months(snap.liquid_assets, monthly_essential)
        if runway >= target.months:
            run_status: Status = "good"
        elif runway >= target.months / 2:
            run_status = "watch"
        else:
            run_status = "risk"
        metrics.append(
            Metric("emergency_runway", "Emergency runway", runway, f"{runway:.1f} months", run_status,
                   ef.RULE_ID, f"Target is {target.months} months of essential spending.")
        )
    else:
        metrics.append(
            Metric("emergency_runway", "Emergency runway", None, "n/a", "watch", ef.RULE_ID,
                   "Essential monthly spending is unknown.")
        )

    return HouseholdMetrics(metrics=tuple(metrics), net_worth=snap, emergency_target=target)