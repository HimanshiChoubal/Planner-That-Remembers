"""Demo personas (all fictional, all synthetic).

Each persona has a structured profile (income history is filled in by the generator) and the
memories that onboarding would have captured: risk profile, hard constraints and goals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from app.agents.base import FinancialProfile
from app.context.schemas import Constraint, Goal, MemoryModel, MoneyValue, RiskProfile
from app.engine.debt_payoff import Debt
from app.engine.money import INR, USD, Currency, Money
from app.engine.networth import Asset


@dataclass(frozen=True)
class Persona:
    id: str
    display_name: str
    region: str
    summary: str
    income_kind: str  # "salaried" | "irregular" | "business"
    income_cv: float
    profile: FinancialProfile  # income_history left empty
    onboarding: tuple[MemoryModel, ...]
    childcare_monthly: Money | None = None

    @property
    def login(self) -> str:
        return f"{self.id.replace('_', '.')}@demo.example"


def _goal(cur: Currency, name: str, category: str, target: int, deadline: date, saved: int, priority: int = 3) -> Goal:
    return Goal(name=name, category=category, target=MoneyValue.from_money(Money.of(target, cur)),
                deadline=deadline, saved=MoneyValue.from_money(Money.of(saved, cur)), priority=priority)  # type: ignore[arg-type]


def _debt(cur: Currency, id_: str, name: str, balance: int, apr: str, minimum: int) -> Debt:
    return Debt(id_, name, Money.of(balance, cur), Decimal(apr), Money.of(minimum, cur))


def _us() -> list[Persona]:
    c = USD
    m = lambda x: Money.of(x, c)  # noqa: E731
    salaried = Persona(
        "us_salaried", "Jordan Ellis", "us", "Salaried household with a mortgage and two goals.", "salaried", 0.02,
        FinancialProfile(
            "us_salaried", "us", c, age=38, retirement_age=65, monthly_gross_income=m(7800),
            monthly_expenses=m(5600), monthly_essential_expenses=m(4300),
            assets=(Asset("Checking", "cash", m(3500)), Asset("Savings", "savings", m(14000)),
                    Asset("401(k)", "retirement", m(82000)), Asset("Brokerage", "investments", m(9000))),
            debts=(_debt(c, "mortgage", "Mortgage", 215000, "0.0425", 1320), _debt(c, "car", "Car loan", 11000, "0.071", 340)),
            dependents=1, tax_regime="standard", employer_match_monthly=m(390), match_captured=True,
            insurance_covered=True, monthly_retirement_contribution=m(780)),
        (RiskProfile(tolerance="medium", horizon_years=27, max_equity_share=0.7),
         Constraint(description="No more than 70% in equities", subject="investing", hard=True, max_value=0.7),
         _goal(c, "College fund", "education", 60000, date(2038, 9, 1), 8000, 2),
         _goal(c, "Kitchen remodel", "home", 25000, date(2028, 6, 1), 6000, 3)))
    freelancer = Persona(
        "us_freelancer", "Sam Rivera", "us", "Freelancer with irregular income, card debt and no insurance.", "irregular", 0.38,
        FinancialProfile(
            "us_freelancer", "us", c, age=31, retirement_age=67, monthly_gross_income=m(5200),
            monthly_expenses=m(3600), monthly_essential_expenses=m(2700),
            assets=(Asset("Savings", "savings", m(4200)), Asset("IRA", "retirement", m(12500))),
            debts=(_debt(c, "card", "Credit card", 6800, "0.2399", 190), _debt(c, "student", "Student loan", 14500, "0.055", 165)),
            single_income=True, tax_regime="standard", insurance_covered=False, monthly_retirement_contribution=m(200)),
        (RiskProfile(tolerance="medium", horizon_years=36, max_equity_share=0.6),
         Constraint(description="No individual stocks or crypto", subject="investing", hard=True),
         _goal(c, "House deposit", "home", 45000, date(2031, 6, 1), 3000, 1)))
    student = Persona(
        "us_student_debt", "Alex Morgan", "us", "Young professional paying down student and card debt.", "salaried", 0.02,
        FinancialProfile(
            "us_student_debt", "us", c, age=26, retirement_age=65, monthly_gross_income=m(4300),
            monthly_expenses=m(3000), monthly_essential_expenses=m(2300),
            assets=(Asset("Checking", "cash", m(1600)), Asset("Savings", "savings", m(2400)), Asset("401(k)", "retirement", m(5200))),
            debts=(_debt(c, "student", "Student loan", 28000, "0.0675", 320), _debt(c, "card", "Credit card", 2400, "0.2199", 80)),
            tax_regime="standard", employer_match_monthly=m(129), match_captured=False,
            insurance_covered=True, monthly_retirement_contribution=m(100)),
        (RiskProfile(tolerance="high", horizon_years=39, max_equity_share=0.9),
         _goal(c, "Replace the car", "vehicle", 12000, date(2028, 3, 1), 500, 2)))
    return [salaried, freelancer, student]


def _in() -> list[Persona]:
    c = INR
    m = lambda x: Money.of(x, c)  # noqa: E731
    family = Persona(
        "in_family", "Ananya Sharma", "in", "Salaried family with a home loan and a child on the way.", "salaried", 0.03,
        FinancialProfile(
            "in_family", "in", c, age=32, retirement_age=60, monthly_gross_income=m(140000),
            monthly_expenses=m(78000), monthly_essential_expenses=m(60000),
            assets=(Asset("Savings account", "savings", m(320000)), Asset("Fixed deposit", "savings", m(200000)),
                    Asset("EPF", "retirement", m(650000)), Asset("PPF", "retirement", m(180000))),
            debts=(_debt(c, "home", "Home loan", 2800000, "0.0865", 24600), _debt(c, "personal", "Personal loan", 210000, "0.125", 9500)),
            tax_regime="new", insurance_covered=True, monthly_retirement_contribution=m(12000)),
        (RiskProfile(tolerance="medium", horizon_years=28, max_equity_share=0.5),
         Constraint(description="Keep equity at or below 50%", subject="investing", hard=True, max_value=0.5),
         _goal(c, "Child's education", "education", 2500000, date(2044, 6, 1), 100000, 2),
         _goal(c, "Family vacation", "travel", 300000, date(2028, 5, 1), 40000, 4)))
    business = Persona(
        "in_business", "Vikram Patel", "in", "Small-business owner with variable income and a business loan.", "business", 0.35,
        FinancialProfile(
            "in_business", "in", c, age=41, retirement_age=60, monthly_gross_income=m(210000),
            monthly_expenses=m(105000), monthly_essential_expenses=m(78000),
            assets=(Asset("Savings account", "savings", m(450000)), Asset("Mutual funds", "investments", m(300000)),
                    Asset("PPF", "retirement", m(420000))),
            debts=(_debt(c, "business", "Business loan", 600000, "0.145", 21000), _debt(c, "card", "Credit card", 85000, "0.36", 6000)),
            dependents=2, single_income=True, tax_regime="old", insurance_covered=False,
            monthly_retirement_contribution=m(10000)),
        (RiskProfile(tolerance="medium", horizon_years=19, max_equity_share=0.6),
         Constraint(description="Never touch the PPF balance", subject="saving", hard=True),
         _goal(c, "Shop expansion", "other", 800000, date(2028, 3, 1), 150000, 1)))
    saver = Persona(
        "in_saver", "Priya Nair", "in", "High-saving salaried professional comparing tax regimes.", "salaried", 0.02,
        FinancialProfile(
            "in_saver", "in", c, age=29, retirement_age=55, monthly_gross_income=m(160000),
            monthly_expenses=m(60000), monthly_essential_expenses=m(42000),
            assets=(Asset("Savings account", "savings", m(550000)), Asset("Mutual funds", "investments", m(420000)),
                    Asset("EPF", "retirement", m(310000)), Asset("NPS", "retirement", m(90000))),
            debts=(), tax_regime="new", insurance_covered=True, monthly_retirement_contribution=m(25000)),
        (RiskProfile(tolerance="high", horizon_years=26, max_equity_share=0.85),
         _goal(c, "Home down payment", "home", 2000000, date(2030, 3, 1), 600000, 1),
         _goal(c, "Europe trip", "travel", 400000, date(2027, 9, 1), 100000, 4)))
    return [family, business, saver]


PERSONAS: dict[str, Persona] = {p.id: p for p in (*_us(), *_in())}


def get_persona(persona_id: str) -> Persona:
    try:
        return PERSONAS[persona_id]
    except KeyError as exc:
        raise ValueError(f"Unknown persona '{persona_id}'. Available: {', '.join(PERSONAS)}") from exc