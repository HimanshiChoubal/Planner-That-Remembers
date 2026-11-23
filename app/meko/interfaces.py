"""Backend-agnostic interfaces for the Meko context layer.

PROVISIONAL: Meko's real MCP tool names, memory visibility semantics and multi-datapack
support have not been verified against its docs. All Meko-specific details must stay
behind these interfaces (implemented in ``client.py`` and ``fake.py``), so adjusting to
the real API only touches ``app/meko/``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- scope
@dataclass(frozen=True)
class Scope:
    """Isolation boundary. Resolved server-side from the authenticated user, never from
    client input. ``datapack`` maps to a Meko datapack or namespace."""

    user_id: str
    datapack: str


# ------------------------------------------------------------------------- memory
class Visibility(str, Enum):
    PRIVATE = "private"  # visible to the writing agent only
    SHARED = "shared"  # visible to every agent and human in the scope


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    scope: Scope
    agent: str  # writer, e.g. "debt"
    kind: str  # e.g. "goal", "constraint", "decision", "rejected_decision"
    content: Mapping[str, Any]
    visibility: Visibility
    created_at: datetime = field(default_factory=utcnow)
    confidence: float = 1.0
    source_trace_id: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class MemoryHit:
    record: MemoryRecord
    score: float  # relevance in [0, 1]


@runtime_checkable
class MemoryStore(Protocol):
    async def write(
        self,
        scope: Scope,
        *,
        agent: str,
        kind: str,
        content: Mapping[str, Any],
        visibility: Visibility = Visibility.PRIVATE,
        confidence: float = 1.0,
        source_trace_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryRecord: ...

    async def promote(self, scope: Scope, memory_id: str) -> MemoryRecord:
        """Private -> shared. Returns the shared record."""
        ...

    async def search(
        self,
        scope: Scope,
        query: str,
        *,
        agent: str,
        kinds: Sequence[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryHit]:
        """Records visible to ``agent``: its own private ones plus all shared ones."""
        ...

    async def list(
        self,
        scope: Scope,
        *,
        agent: str | None = None,
        kinds: Sequence[str] | None = None,
        visibility: Visibility | None = None,
    ) -> list[MemoryRecord]: ...

    async def delete(self, scope: Scope, memory_id: str) -> None: ...


# ---------------------------------------------------------------------- knowledge
@dataclass(frozen=True)
class KnowledgeDoc:
    id: str  # stable ID cited in traces, e.g. "us.tax.standard"
    title: str
    text: str
    tags: tuple[str, ...] = ()
    region: str | None = None


@dataclass(frozen=True)
class KnowledgeHit:
    doc: KnowledgeDoc
    score: float  # relevance in [0, 1]
    snippet: str


@runtime_checkable
class KnowledgeStore(Protocol):
    async def ingest(self, scope: Scope, docs: Sequence[KnowledgeDoc]) -> int:
        """Upsert docs by ID. Returns the number ingested."""
        ...

    async def search(
        self,
        scope: Scope,
        query: str,
        *,
        region: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 5,
    ) -> list[KnowledgeHit]: ...

    async def get(self, scope: Scope, doc_id: str) -> KnowledgeDoc | None: ...


# ---------------------------------------------------------------------- artifacts
@dataclass(frozen=True)
class ArtifactVersion:
    artifact_id: str  # logical artifact, e.g. "plan"
    version: int  # 1, 2, 3...
    scope: Scope
    kind: str  # "plan", "report", "simulation"
    payload: Mapping[str, Any]
    created_at: datetime = field(default_factory=utcnow)
    created_by: str = "planner"
    note: str | None = None  # e.g. "Replan after: baby"
    parent_version: int | None = None
    trace_ids: tuple[str, ...] = ()


@runtime_checkable
class ArtifactStore(Protocol):
    async def save(
        self,
        scope: Scope,
        artifact_id: str,
        *,
        kind: str,
        payload: Mapping[str, Any],
        created_by: str = "planner",
        note: str | None = None,
        trace_ids: Sequence[str] = (),
    ) -> ArtifactVersion:
        """Always creates a new version; never overwrites."""
        ...

    async def get(
        self, scope: Scope, artifact_id: str, version: int | None = None
    ) -> ArtifactVersion | None:
        """Latest version when ``version`` is None."""
        ...

    async def list_versions(self, scope: Scope, artifact_id: str) -> list[ArtifactVersion]: ...


# ------------------------------------------------------------------------- traces
@dataclass(frozen=True)
class Trace:
    id: str
    scope: Scope
    agent: str
    recommendation: str  # short label, e.g. "Raise emergency fund target to 6 months"
    reasoning: str  # templated or LLM-narrated explanation
    memory_ids: tuple[str, ...] = ()
    knowledge_ids: tuple[str, ...] = ()
    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    session_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)


@runtime_checkable
class TraceStore(Protocol):
    async def record(self, trace: Trace) -> Trace: ...

    async def get(self, scope: Scope, trace_id: str) -> Trace | None: ...

    async def list(
        self,
        scope: Scope,
        *,
        agent: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[Trace]: ...


# ------------------------------------------------------------------ conversations
@dataclass(frozen=True)
class Message:
    session_id: str
    role: str  # "user" | "assistant" | "agent" | "advisor"
    content: str
    author: str | None = None  # agent name or advisor id
    created_at: datetime = field(default_factory=utcnow)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    scope: Scope
    started_at: datetime
    title: str | None = None
    message_count: int = 0


@runtime_checkable
class ConversationStore(Protocol):
    async def start_session(self, scope: Scope, title: str | None = None) -> SessionInfo: ...

    async def append(self, scope: Scope, message: Message) -> Message: ...

    async def get_messages(self, scope: Scope, session_id: str) -> list[Message]:
        """Verbatim history, oldest first."""
        ...

    async def list_sessions(self, scope: Scope) -> list[SessionInfo]: ...


# --------------------------------------------------------------- context retrieval
@dataclass(frozen=True)
class ContextItem:
    source: str  # "memory" | "knowledge" | "conversation" | "artifact"
    ref_id: str
    text: str
    score: float


@dataclass(frozen=True)
class ContextResult:
    """Scored retrieval. ``missing`` powers context-gap detection: when overall quality
    is low, agents ask the user instead of guessing."""

    items: tuple[ContextItem, ...]
    quality: float  # overall retrieval quality in [0, 1]
    missing: tuple[str, ...] = ()
    tokens_used: int = 0


@runtime_checkable
class ContextSearch(Protocol):
    async def context_search(
        self,
        scope: Scope,
        query: str,
        *,
        agent: str,
        sources: Sequence[str] | None = None,
        limit: int = 10,
    ) -> ContextResult: ...


# ----------------------------------------------------------------------- bundle
@runtime_checkable
class MekoBackend(Protocol):
    """Everything the app needs from Meko, behind one object."""

    memory: MemoryStore
    knowledge: KnowledgeStore
    artifacts: ArtifactStore
    traces: TraceStore
    conversations: ConversationStore
    search: ContextSearch

    async def health(self) -> bool: ...