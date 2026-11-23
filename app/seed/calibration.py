"""Calibration parameters for the synthetic data generator.

ILLUSTRATIVE. The category shares, seasonality and noise levels are set by hand to look
plausible; before citing them as realistic, calibrate against public household surveys
(for example consumer-expenditure surveys) and verify each source's license. All generated
data is synthetic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegionCalibration:
    code: str
    # Share of the (essential or discretionary) pool each category receives; renormalised
    # within its group, so only the ratios matter. Debt payments and childcare are fixed.
    category_shares: Mapping[str, float]
    monthly_noise: float = 0.08  # lognormal sigma on each category's monthly total
    txns_per_month: Mapping[str, int] = field(default_factory=dict)
    fixed_days: Mapping[str, int] = field(default_factory=dict)  # day of month for fixed charges
    seasonal: Mapping[int, Mapping[str, float]] = field(default_factory=dict)  # month -> category multiplier
    business_season: Mapping[int, float] = field(default_factory=dict)  # month -> income multiplier


_TXNS = {"groceries": 8, "dining": 6, "transport": 6, "shopping": 3, "entertainment": 2,
         "travel": 1, "healthcare": 1, "education": 1, "other": 3, "utilities": 2, "subscriptions": 2}
_DAYS = {"housing": 1, "childcare": 3, "domestic_help": 4, "insurance": 5, "utilities": 10, "subscriptions": 8}
_BUSINESS = {1: 0.85, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.9, 7: 0.9, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.25, 12: 1.2}

CALIBRATION: dict[str, RegionCalibration] = {
    "us": RegionCalibration(
        code="us",
        category_shares={"housing": 0.34, "utilities": 0.06, "groceries": 0.12, "transport": 0.11,
                         "healthcare": 0.05, "insurance": 0.05, "dining": 0.07, "subscriptions": 0.02,
                         "entertainment": 0.04, "shopping": 0.06, "travel": 0.03, "education": 0.01, "other": 0.04},
        txns_per_month=_TXNS, fixed_days=_DAYS,
        seasonal={11: {"shopping": 1.25}, 12: {"shopping": 1.5, "travel": 1.4, "entertainment": 1.2, "dining": 1.2}},
        business_season=_BUSINESS,
    ),
    "in": RegionCalibration(
        code="in",
        category_shares={"housing": 0.28, "utilities": 0.06, "groceries": 0.15, "transport": 0.09,
                         "healthcare": 0.05, "insurance": 0.05, "domestic_help": 0.03, "dining": 0.06,
                         "subscriptions": 0.02, "entertainment": 0.03, "shopping": 0.07, "travel": 0.04,
                         "education": 0.02, "other": 0.05},
        txns_per_month={**_TXNS, "domestic_help": 1}, fixed_days=_DAYS,
        seasonal={5: {"travel": 1.2}, 6: {"travel": 1.2}, 10: {"shopping": 1.5, "travel": 1.2},
                  11: {"shopping": 1.5, "entertainment": 1.2}},
        business_season=_BUSINESS,
    ),
}
