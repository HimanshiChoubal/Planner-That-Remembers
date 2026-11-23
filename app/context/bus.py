"""ContextBus: the single gateway between agents and Meko.

Agents never talk to each other or to Meko directly. Everything they read or write goes
through a bus bound to one (scope, agent, session). That keeps all sharing visible in Meko,
makes traces and token accounting uniform, and lets tests swap in the fake backend.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.context.schemas import MemoryKind, MemoryModel, dump_content, kind_of, parse_memory
from app.meko.interfaces import (
    ArtifactVersion,
    ContextResult,
    KnowledgeHit,
    MekoBackend,
    MemoryRecord,
    Message,
    Scope,
    SessionInfo,
    Trace,
    Visibility,
)

GAP_QUALITY_THRESHOLD = 0.35  # below this retrieval quality, ask the user instead of guessing


@dataclass
class _Meter:
    """Token accounting shared by every bus derived from the same root bus."""

    tokens_retrieved: int = 0
    retrievals: int = 0


@dataclass(frozen=True)
class RecalledMemory:
    record: MemoryRecord
    score: float
    model: MemoryModel | None  # None when the kind is unknown or the content is invalid


@dataclass
class ContextBus:
    backend: MekoBackend
    scope: Scope
    agent: str
    session_id: str | None = None
    region: str | None = None
    _meter: _Meter = field(default_factory=_Meter, repr=False)

    # ------------------------------------------------------------ derived buses
    def for_agent(self, agent: str) -> ContextBus:
        """Same scope, session and token meter, different agent identity."""
        return ContextBus(self.backend, self.scope, agent, self.session_id, self.region, self._meter)

    async def with_new_session(self, title: str | None = None) -> ContextBus:
        info: SessionInfo = await self.backend.conversations.start_session(self.scope, title)
        return ContextBus(self.backend, self.scope, self.agent, info.session_id, self.region, self._meter)

    # ------------------------------------------------------------------- memory
    async def remember(
        self,
        model: MemoryModel,
        *,
        visibility: Visibility = Visibility.PRIVATE,
        confidence: float = 1.0,
        trace_id: str | None = None,
    ) -> MemoryRecord:
        return await self.backend.memory.write(
            self.scope,
            agent=self.agent,
            kind=kind_of(model),
            content=dump_content(model),
            visibility=visibility,
            confidence=confidence,
            source_trace_id=trace_id,
            session_id=self.session_id,
        )

    async def share(self, record: MemoryRecord) -> MemoryRecord:
        """Promote a private memory to shared, so every agent and human can use it."""
        return await self.backend.memory.promote(self.scope, record.id)

    async def learn_shared(
        self,
        model: MemoryModel,
        *,
        confidence: float = 1.0,
        trace_id: str | None = None,
    ) -> MemoryRecord:
        """Write privately first, then promote, so private -> shared lineage is explicit."""
        private = await self.remember(model, confidence=confidence, trace_id=trace_id)
        return await self.share(private)

    async def recall(
        self,
        query: str = "",
        *,
        kinds: Sequence[MemoryKind | str] | None = None,
        limit: int = 10,
    ) -> list[RecalledMemory]:
        kind_values = [k.value if isinstance(k, MemoryKind) else k for k in kinds] if kinds else None
        hits = await self.backend.memory.search(
            self.scope, query, agent=self.agent, kinds=kind_values, limit=limit
        )
        return [
            RecalledMemory(h.record, h.score, parse_memory(h.record.kind, h.record.content))
            for h in hits
        ]

    # ------------------------------------------------------- retrieval / knowledge
    async def retrieve(
        self,
        query: str,
        *,
        sources: Sequence[str] | None = None,
        limit: int = 10,
    ) -> ContextResult:
        result = await self.backend.search.context_search(
            self.scope, query, agent=self.agent, sources=sources, limit=limit
        )
        self._meter.tokens_retrieved += result.tokens_used
        self._meter.retrievals += 1
        return result

    @staticmethod
    def has_gap(result: ContextResult, threshold: float = GAP_QUALITY_THRESHOLD) -> bool:
        """True when retrieval was too weak to answer reliably (ask the user instead)."""
        return result.quality < threshold

    async def knowledge(self, query: str, *, limit: int = 3) -> list[KnowledgeHit]:
        return await self.backend.knowledge.search(
            self.scope, query, region=self.region, limit=limit
        )

    @property
    def tokens_retrieved(self) -> int:
        return self._meter.tokens_retrieved

    @property
    def retrievals(self) -> int:
        return self._meter.retrievals

    # ------------------------------------------------------------------- traces
    async def trace(
        self,
        recommendation: str,
        reasoning: str,
        *,
        memory_ids: Sequence[str] = (),
        knowledge_ids: Sequence[str] = (),
        inputs: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> Trace:
        trace = Trace(
            id=f"trc_{uuid.uuid4().hex[:12]}",
            scope=self.scope,
            agent=self.agent,
            recommendation=recommendation,
            reasoning=reasoning,
            memory_ids=tuple(memory_ids),
            knowledge_ids=tuple(knowledge_ids),
            inputs=dict(inputs or {}),
            outputs=dict(outputs or {}),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            session_id=self.session_id,
        )
        return await self.backend.traces.record(trace)

    # ---------------------------------------------------------------- artifacts
    async def save_artifact(
        self,
        artifact_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
        note: str | None = None,
        trace_ids: Sequence[str] = (),
    ) -> ArtifactVersion:
        return await self.backend.artifacts.save(
            self.scope,
            artifact_id,
            kind=kind,
            payload=payload,
            created_by=self.agent,
            note=note,
            trace_ids=trace_ids,
        )

    async def latest_artifact(self, artifact_id: str) -> ArtifactVersion | None:
        return await self.backend.artifacts.get(self.scope, artifact_id)

    # ------------------------------------------------------------ conversations
    async def say(
        self, role: str, content: str, *, metadata: Mapping[str, Any] | None = None
    ) -> Message:
        if self.session_id is None:
            raise ValueError("No active session. Call with_new_session() first.")
        message = Message(
            session_id=self.session_id,
            role=role,
            content=content,
            author=self.agent,
            metadata=dict(metadata or {}),
        )
        return await self.backend.conversations.append(self.scope, message)

    async def history(self, session_id: str | None = None) -> list[Message]:
        sid = session_id or self.session_id
        if sid is None:
            raise ValueError("No session specified.")
        return await self.backend.conversations.get_messages(self.scope, sid)