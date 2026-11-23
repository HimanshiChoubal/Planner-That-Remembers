"""Tax agent.

Works entirely from the region pack's SIMPLIFIED, SYNTHETIC rules (not real tax law): compares
regimes, finds the deduction level at which a deduction-friendly regime starts to win, and
estimates the tax saved by contributing to pre-tax retirement instruments.
"""

from __future__ import annotations

from decimal import Decimal

from app.agents.base import Agent, AgentContext, AgentResult, Recommendation, money_json
from app.engine.money import Money
from app.engine.priorities import Step
from app.regions.base import InstrumentKind, RegionPack, TaxResult

MAX_INSTRUMENT_RECS = 2


def breakeven_deduction(pack: RegionPack, annual: Money, regime: str, other_tax: Money) -> Money | None:
    """Smallest extra deduction under ``regime`` that makes its tax no higher than ``other_tax``.
    Zero if it already is; ``None`` if no deduction up to the full income gets there."""
    zero = Money.zero(annual.currency)
    if pack.compute_tax(annual, regime).tax <= other_tax:
        return zero
    if pack.compute_tax(annual, regime, annual).tax > other_tax:
        return None
    lo, hi = 0, annual.minor
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pack.compute_tax(annual, regime, Money(mid, annual.currency)).tax <= other_tax:
            hi = mid
        else:
            lo = mid
    unit = 10 ** annual.currency.minor_units
    return Money(min(-(-hi // unit) * unit, annual.minor), annual.currency)  # whole currency units


def _pct(rate: Decimal) -> str:
    return f"{rate:.1%}"


class TaxAgent(Agent):
    name = "tax"
    description = "Tax regime choice and tax-aware contribution advice (synthetic rules)"

    async def run(self, ctx: AgentContext) -> AgentResult:
        p, pack = ctx.profile, ctx.pack
        if p.monthly_gross_income is None:
            raise ValueError("TaxAgent requires monthly income (gap check should prevent this)")
        annual = p.monthly_gross_income * 12
        results = pack.compare_regimes(annual)
        current = p.tax_regime if p.tax_regime in results else pack.default_regime()
        cur_result = results[current]
        best = min(results.values(), key=lambda r: r.tax.minor)
        recs: list[Recommendation] = []

        # Regime choice (only meaningful when a region has more than one regime).
        breakevens: dict[str, Money | None] = {}
        if len(results) > 1:
            flexible = [r for r in results if pack.allows_instrument_deductions(r)]
            fixed = [r for r in results if not pack.allows_instrument_deductions(r)]
            for regime in flexible:
                if fixed:
                    rival = min((results[f] for f in fixed), key=lambda r: r.tax.minor)
                    breakevens[regime] = breakeven_deduction(pack, annual, regime, rival.tax)
            saving = cur_result.tax - best.tax
            rule_ids = [*cur_result.rule_ids, *best.rule_ids]
            if best.regime != current and saving.minor > 0:
                extra = ""
                for regime, be in breakevens.items():
                    if be is not None and be.minor > 0:
                        extra = f" The {regime} regime only wins if you claim at least {be} in deductions."
                recs.append(await self.recommend(
                    ctx, title=f"Consider the {best.regime} tax regime",
                    summary=f"On your income the {best.regime} regime has lower tax ({best.tax} vs {cur_result.tax}), saving about {saving} a year.{extra}",
                    reasoning=f"Computed tax: {current} {cur_result.tax}, {best.regime} {best.tax} on gross {annual}.",
                    metrics={"current_tax": str(cur_result.tax), "best_tax": str(best.tax), "saving": str(saving)},
                    knowledge_ids=rule_ids, priority=2))
            else:
                be_text = " ".join(f"The {r} regime would only win with at least {be} in deductions." for r, be in breakevens.items() if be is not None and be.minor > 0)
                recs.append(await self.recommend(
                    ctx, title=f"The {current} regime is the cheaper choice",
                    summary=f"Estimated tax is {cur_result.tax} a year under the {current} regime.{(' ' + be_text) if be_text else ''}",
                    reasoning=f"Computed tax across regimes: " + ", ".join(f"{k} {v.tax}" for k, v in results.items()),
                    metrics={"current_tax": str(cur_result.tax)}, knowledge_ids=rule_ids, priority=4))

        # Contribution advice: pre-tax instruments that reduce taxable income under the chosen regime.
        advice: list[dict] = []
        if pack.allows_instrument_deductions(current):
            for inst in pack.retirement_instruments():
                if inst.kind is not InstrumentKind.PRE_TAX or inst.annual_limit is None:
                    continue
                reduced = pack.compute_tax(annual, current, inst.annual_limit)
                saved = cur_result.tax - reduced.tax
                if saved.minor > 0:
                    advice.append({"instrument": inst, "saved": saved, "tax_after": reduced.tax})
            advice.sort(key=lambda a: a["saved"].minor, reverse=True)
            wf = ctx.waterfall()
            waits = wf is not None and wf.status_of(Step.GOALS_AND_RETIREMENT) == "pending"
            wait_note = " Only worth doing once higher-priority steps (emergency buffer, high-interest debt) are covered." if waits else ""
            for a in advice[:MAX_INSTRUMENT_RECS]:
                inst = a["instrument"]
                recs.append(await self.recommend(
                    ctx, title=f"Contribute to {inst.name} to cut tax",
                    summary=f"Contributing up to {inst.annual_limit} a year to {inst.name} could lower your tax by about {a['saved']} (marginal rate {_pct(cur_result.marginal_rate)}). Estimates for different instruments overlap, so don't add them together.{wait_note}",
                    reasoning=f"Tax falls from {cur_result.tax} to {a['tax_after']} when {inst.annual_limit} is deducted under the {current} regime.",
                    metrics={"instrument": inst.id, "limit": str(inst.annual_limit), "tax_saved": str(a["saved"])},
                    knowledge_ids=[*cur_result.rule_ids, inst.rule_id], priority=4 if waits else 3))

        finding = await self.share_finding(
            ctx, "marginal_tax_rate",
            f"Marginal tax rate is {_pct(cur_result.marginal_rate)} (effective {_pct(cur_result.effective_rate)}) under the {current} regime.",
            {"marginal_rate": float(cur_result.marginal_rate), "effective_rate": float(cur_result.effective_rate), "regime": current})

        plan = {
            "annual_gross_income": money_json(annual),
            "regime": current,
            "best_regime": best.regime,
            "by_regime": {k: _result_json(v) for k, v in results.items()},
            "breakeven_deduction": {k: money_json(v) for k, v in breakevens.items()},
            "instrument_savings": [
                {"instrument": a["instrument"].id, "limit": money_json(a["instrument"].annual_limit), "tax_saved": money_json(a["saved"])}
                for a in advice],
            "synthetic_rules": True,
        }
        return AgentResult(self.name, "ok", tuple(recs), plan=plan, findings=(finding,))


def _result_json(r: TaxResult) -> dict:
    return {"tax": money_json(r.tax), "taxable_income": money_json(r.taxable_income),
            "effective_rate": round(float(r.effective_rate), 4), "marginal_rate": float(r.marginal_rate)}