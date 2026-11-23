"""Seed the context store and play scripted scenarios.

Works against any ``MekoBackend`` (the fake today, the real client later). Per persona it
ingests knowledge docs, writes the onboarding memories as shared memory, and generates the
synthetic financial history. ``play_scenario`` replays the three-session story through the
orchestrator, which is also the basis of the end-to-end test and the demo video.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.agents.base import FinancialProfile
from app.agents.orchestrator import Orchestrator, TurnResult, apply_life_event
from app.context.bus import ContextBus
from app.meko.interfaces import KnowledgeDoc, MekoBackend, Scope, Visibility
from app.regions.registry import get_pack
from app.seed.generator import SyntheticHistory, generate_history
from app.seed.personas import PERSONAS, Persona
from app.seed.scenarios import Scenario, get_scenario

PLANNING_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge" / "planning"


@dataclass(frozen=True)
class SeededPersona:
    persona: Persona
    scope: Scope
    profile: FinancialProfile  # includes the generated income history
    history: SyntheticHistory
    scenario: Scenario


def parse_knowledge_doc(text: str, default_region: str | None = None) -> KnowledgeDoc:
    """Parse a markdown doc with a simple front matter block (id, title, tags, region)."""
    if not text.startswith("---"):
        raise ValueError("Knowledge doc must start with a front matter block")
    _, front, body = text.split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    if "id" not in meta or "title" not in meta:
        raise ValueError("Knowledge doc front matter needs 'id' and 'title'")
    tags = tuple(t.strip() for t in meta.get("tags", "").split(",") if t.strip())
    region = meta.get("region") or default_region
    return KnowledgeDoc(id=meta["id"], title=meta["title"], text=body.strip(), tags=tags, region=region)


def load_knowledge_dir(directory: Path, region: str | None = None) -> list[KnowledgeDoc]:
    return [parse_knowledge_doc(p.read_text(encoding="utf-8"), region) for p in sorted(directory.glob("*.md"))]


def knowledge_for_region(region: str) -> list[KnowledgeDoc]:
    docs = load_knowledge_dir(PLANNING_KNOWLEDGE_DIR, None)
    docs += load_knowledge_dir(get_pack(region).knowledge_dir(), region)
    return docs


async def seed_persona(
    backend: MekoBackend,
    persona: Persona,
    *,
    as_of: date,
    months: int = 18,
    seed: int = 0,
) -> SeededPersona:
    scope = Scope(user_id=persona.id, datapack=f"dp_{persona.id}")
    await backend.knowledge.ingest(scope, knowledge_for_region(persona.region))
    onboarding = ContextBus(backend, scope, "onboarding", region=persona.region)
    for memory in persona.onboarding:
        await onboarding.remember(memory, visibility=Visibility.SHARED)
    history = generate_history(persona, as_of=as_of, months=months, seed=seed)
    profile = dataclasses.replace(persona.profile, income_history=history.income)
    return SeededPersona(persona, scope, profile, history, get_scenario(persona.id))


async def seed_all(backend: MekoBackend, *, as_of: date, months: int = 18, seed: int = 0) -> dict[str, SeededPersona]:
    return {pid: await seed_persona(backend, p, as_of=as_of, months=months, seed=seed) for pid, p in PERSONAS.items()}


async def play_scenario(
    orchestrator: Orchestrator,
    backend: MekoBackend,
    seeded: SeededPersona,
    *,
    as_of: date,
) -> list[TurnResult]:
    """Replay the scripted sessions in order, carrying life-event changes forward."""
    root = ContextBus(backend, seeded.scope, "planner", region=seeded.persona.region)
    profile = seeded.profile
    turns: list[TurnResult] = []
    for step in seeded.scenario.steps:
        session_bus = await root.with_new_session(step.title)
        turns.append(await orchestrator.handle(session_bus, profile, step.text, as_of=as_of, life_event=step.life_event))
        if step.life_event is not None:
            profile = apply_life_event(profile, step.life_event)
    return turns