"""Scripted three-session stories, one per persona.

Session 1 builds the plan from onboarding. Session 2 tests memory: a narrow question, or a
proposal that conflicts with something remembered. Session 3 is a life event that replans
every area and produces a plan diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.context.schemas import LifeEvent, MoneyValue
from app.engine.money import INR, USD, Currency, Money


@dataclass(frozen=True)
class ScenarioStep:
    title: str
    text: str
    life_event: LifeEvent | None = None
    expects: str = ""  # what the demo should show, for the demo script and tests


@dataclass(frozen=True)
class Scenario:
    persona_id: str
    steps: tuple[ScenarioStep, ...]


def _mv(cur: Currency, amount: int) -> MoneyValue:
    return MoneyValue.from_money(Money.of(amount, cur))


_FULL = "Give me a full plan"


def _scenarios() -> list[Scenario]:
    u, r = USD, INR
    return [
        Scenario("us_salaried", (
            ScenarioStep("Session 1: first plan", _FULL, expects="Plan v1 from onboarding memories"),
            ScenarioStep("Session 2: memory check", "Should I put 85% of our savings into stocks?",
                         expects="Blocked: exceeds the remembered 70% equity limit"),
            ScenarioStep("Session 3: bonus", "I received a work bonus", LifeEvent(
                event="windfall", description="Annual bonus", occurred_on=date(2026, 12, 15),
                one_off_amount=_mv(u, 15000)), expects="Plan v2: liquid savings and emergency runway rise"))),
        Scenario("us_freelancer", (
            ScenarioStep("Session 1: first plan", _FULL, expects="Plans on lower-income months; insurance gap flagged"),
            ScenarioStep("Session 2: memory check", "I want to go all in on crypto",
                         expects="Warns: conflicts with the 'no crypto' rule and the risk profile"),
            ScenarioStep("Session 3: lost a client", "I lost my biggest client", LifeEvent(
                event="job_loss", description="Lost the largest client", occurred_on=date(2026, 12, 1),
                monthly_income_delta=_mv(u, -2000)), expects="Plan v2: deficit, goals deferred"))),
        Scenario("us_student_debt", (
            ScenarioStep("Session 1: first plan", _FULL, expects="Match capture and card debt ahead of loans"),
            ScenarioStep("Session 2: debt question", "How should I tackle my student loan and credit card debt?",
                         expects="Only budget and debt re-run; other sections kept"),
            ScenarioStep("Session 3: new job", "I got a new job with a raise", LifeEvent(
                event="new_job", description="New job, higher salary", occurred_on=date(2027, 1, 10),
                monthly_income_delta=_mv(u, 1500)), expects="Plan v2: bigger surplus, faster payoff"))),
        Scenario("in_family", (
            ScenarioStep("Session 1: first plan", _FULL, expects="Plan v1 with tax regime and goals"),
            ScenarioStep("Session 2: tax question", "Is the new tax regime better for us?",
                         expects="Only the tax agent re-runs; regime comparison with breakeven"),
            ScenarioStep("Session 3: baby", "We're having a baby in March", LifeEvent(
                event="baby", description="Baby due in March", occurred_on=date(2027, 3, 15),
                monthly_expense_delta=_mv(r, 18000), one_off_amount=_mv(r, 120000)),
                expects="Plan v2: dependents, higher emergency target, thin-safety-net flag"))),
        Scenario("in_business", (
            ScenarioStep("Session 1: first plan", _FULL, expects="Variable income, 36% card debt, insurance gap"),
            ScenarioStep("Session 2: buffer question", "How much should I keep as an emergency fund?",
                         expects="Volatility-adjusted target from the budget agent's shared finding"),
            ScenarioStep("Session 3: medical bill", "We had a large hospital bill", LifeEvent(
                event="medical_expense", description="Hospital bill", occurred_on=date(2027, 1, 20),
                one_off_amount=_mv(r, 250000)), expects="Plan v2: runway drops, buffer priority"))),
        Scenario("in_saver", (
            ScenarioStep("Session 1: first plan", _FULL, expects="Regime comparison, retirement at 55, goals on track"),
            ScenarioStep("Session 2: memory check", "Put 90% of my savings in equity funds",
                         expects="Warns: above the remembered 85% maximum equity share"),
            ScenarioStep("Session 3: promotion", "I got a promotion", LifeEvent(
                event="new_job", description="Promotion with a raise", occurred_on=date(2027, 2, 1),
                monthly_income_delta=_mv(r, 40000)), expects="Plan v2: tax and retirement update"))),
    ]


SCENARIOS: dict[str, Scenario] = {s.persona_id: s for s in _scenarios()}


def get_scenario(persona_id: str) -> Scenario:
    try:
        return SCENARIOS[persona_id]
    except KeyError as exc:
        raise ValueError(f"No scenario for persona '{persona_id}'") from exc