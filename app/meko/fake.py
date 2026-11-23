"""In-memory fake of the Meko backend.

Used for development, tests and as an offline demo fallback. It enforces the same rules the
real backend must: scope isolation, private/shared visibility, append-only artifact versions.
Relevance is simple token overlap, which is good enough to exercise gap detection.
"""

from __future__ import annotations

import dataclasses
import re
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from app.meko.interfaces import (
    ArtifactVersion,
    ContextItem,
    ContextResult,
    KnowledgeDoc,
    KnowledgeHit,
    MemoryHit,
    MemoryRecord,
    Message,
    Scope,
    SessionInfo,
    Trace,
    Visibility,
    utcnow,
)

_List = list  # stores define a ``list`` method, which shadows the builtin inside class bodies

_STOPWORDS = frozenset(
    "a an and are as at be by do for from has have how i in is it me my of on or our so that "
    "the their this to was we what when which with you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> _List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def relevance(query: str, text: str) -> float:
    """Fraction of query terms present in ``text``. Empty queries match everything at 0.5."""
    q = set(_tokens(query))
    if not q:
        return 0.5
    return len(q & set(_tokens(text))) / len(q)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _content_text(content: Mapping[str, Any]) -> str:
    return " ".join(f"{k} {v}" for k, v in content.items())


# ------------------------------------------------------------------------- memory
class FakeMemoryStore:
    def __init__(self) -> None:
        self._records: dict[Scope, dict[str, MemoryRecord]] = defaultdict(dict)

    def _get(self, scope: Scope, memory_id: str) -> MemoryRecord:
        try:
            return self._records[scope][memory_id]
        except KeyError as exc:
            raise KeyError(f"Unknown memory id: {memory_id}") from exc

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
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=_new_id("mem"),
            scope=scope,
            agent=agent,
            kind=kind,
            content=dict(content),
            visibility=visibility,
            confidence=confidence,
            source_trace_id=source_trace_id,
            session_id=session_id,
        )
        self._records[scope][record.id] = record
        return record

    async def promote(self, scope: Scope, memory_id: str) -> MemoryRecord:
        shared = dataclasses.replace(self._get(scope, memory_id), visibility=Visibility.SHARED)
        self._records[scope][memory_id] = shared
        return shared

    def _visible(self, scope: Scope, agent: str) -> _List[MemoryRecord]:
        return [
            r
            for r in self._records[scope].values()
            if r.visibility is Visibility.SHARED or r.agent == agent
        ]

    async def search(
        self,
        scope: Scope,
        query: str,
        *,
        agent: str,
        kinds: Sequence[str] | None = None,
        limit: int = 10,
    ) -> _List[MemoryHit]:
        hits = []
        for r in self._visible(scope, agent):
            if kinds and r.kind not in kinds:
                continue
            score = relevance(query, f"{r.kind} {_content_text(r.content)}")
            if score > 0:
                hits.append(MemoryHit(record=r, score=score))
        hits.sort(key=lambda h: (h.score, h.record.created_at), reverse=True)
        return hits[:limit]

    async def list(
        self,
        scope: Scope,
        *,
        agent: str | None = None,
        kinds: Sequence[str] | None = None,
        visibility: Visibility | None = None,
    ) -> _List[MemoryRecord]:
        out = [
            r
            for r in self._records[scope].values()
            if (agent is None or r.agent == agent)
            and (not kinds or r.kind in kinds)
            and (visibility is None or r.visibility is visibility)
        ]
        return sorted(out, key=lambda r: r.created_at)

    async def delete(self, scope: Scope, memory_id: str) -> None:
        self._get(scope, memory_id)
        del self._records[scope][memory_id]


# ---------------------------------------------------------------------- knowledge
class FakeKnowledgeStore:
    def __init__(self) -> None:
        self._docs: dict[Scope, dict[str, KnowledgeDoc]] = defaultdict(dict)

    async def ingest(self, scope: Scope, docs: Sequence[KnowledgeDoc]) -> int:
        for doc in docs:
            self._docs[scope][doc.id] = doc
        return len(docs)

    async def search(
        self,
        scope: Scope,
        query: str,
        *,
        region: str | None = None,
        tags: Sequence[str] | None = None,
        limit: int = 5,
    ) -> _List[KnowledgeHit]:
        hits = []
        for doc in self._docs[scope].values():
            if region and doc.region not in (None, region):
                continue
            if tags and not set(tags) & set(doc.tags):
                continue
            score = relevance(query, f"{doc.title} {' '.join(doc.tags)} {doc.text}")
            if score > 0:
                hits.append(KnowledgeHit(doc=doc, score=score, snippet=doc.text[:240]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    async def get(self, scope: Scope, doc_id: str) -> KnowledgeDoc | None:
        return self._docs[scope].get(doc_id)


# ---------------------------------------------------------------------- artifacts
class FakeArtifactStore:
    def __init__(self) -> None:
        self._versions: dict[tuple[Scope, str], _List[ArtifactVersion]] = defaultdict(_List)

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
        history = self._versions[(scope, artifact_id)]
        version = ArtifactVersion(
            artifact_id=artifact_id,
            version=len(history) + 1,
            scope=scope,
            kind=kind,
            payload=dict(payload),
            created_by=created_by,
            note=note,
            parent_version=history[-1].version if history else None,
            trace_ids=tuple(trace_ids),
        )
        history.append(version)
        return version

    async def get(
        self, scope: Scope, artifact_id: str, version: int | None = None
    ) -> ArtifactVersion | None:
        history = self._versions.get((scope, artifact_id), [])
        if not history:
            return None
        if version is None:
            return history[-1]
        return next((v for v in history if v.version == version), None)

    async def list_versions(self, scope: Scope, artifact_id: str) -> _List[ArtifactVersion]:
        return list(self._versions.get((scope, artifact_id), []))


# ------------------------------------------------------------------------- traces
class FakeTraceStore:
    def __init__(self) -> None:
        self._traces: dict[Scope, dict[str, Trace]] = defaultdict(dict)

    async def record(self, trace: Trace) -> Trace:
        self._traces[trace.scope][trace.id] = trace
        return trace

    async def get(self, scope: Scope, trace_id: str) -> Trace | None:
        return self._traces[scope].get(trace_id)

    async def list(
        self,
        scope: Scope,
        *,
        agent: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> _List[Trace]:
        out = [
            t
            for t in self._traces[scope].values()
            if (agent is None or t.agent == agent)
            and (session_id is None or t.session_id == session_id)
        ]
        out.sort(key=lambda t: t.created_at, reverse=True)
        return out[:limit]


# ------------------------------------------------------------------ conversations
class FakeConversationStore:
    def __init__(self) -> None:
        self._sessions: dict[Scope, dict[str, SessionInfo]] = defaultdict(dict)
        self._messages: dict[tuple[Scope, str], _List[Message]] = defaultdict(_List)

    async def start_session(self, scope: Scope, title: str | None = None) -> SessionInfo:
        info = SessionInfo(
            session_id=_new_id("ses"), scope=scope, started_at=utcnow(), title=title
        )
        self._sessions[scope][info.session_id] = info
        return info

    async def append(self, scope: Scope, message: Message) -> Message:
        if message.session_id not in self._sessions[scope]:
            raise KeyError(f"Unknown session: {message.session_id}")
        self._messages[(scope, message.session_id)].append(message)
        return message

    async def get_messages(self, scope: Scope, session_id: str) -> _List[Message]:
        if session_id not in self._sessions[scope]:
            raise KeyError(f"Unknown session: {session_id}")
        return list(self._messages[(scope, session_id)])

    async def list_sessions(self, scope: Scope) -> _List[SessionInfo]:
        sessions = [
            dataclasses.replace(s, message_count=len(self._messages[(scope, s.session_id)]))
            for s in self._sessions[scope].values()
        ]
        return sorted(sessions, key=lambda s: s.started_at)


# --------------------------------------------------------------- context retrieval
class FakeContextSearch:
    """Scored cross-source retrieval with missing-term reporting (for gap detection).

    Supported sources: ``memory``, ``knowledge``, ``conversation``. Default: memory + knowledge.
    """

    def __init__(
        self,
        memory: FakeMemoryStore,
        knowledge: FakeKnowledgeStore,
        conversations: FakeConversationStore,
    ) -> None:
        self._memory = memory
        self._knowledge = knowledge
        self._conversations = conversations

    async def context_search(
        self,
        scope: Scope,
        query: str,
        *,
        agent: str,
        sources: Sequence[str] | None = None,
        limit: int = 10,
    ) -> ContextResult:
        wanted = set(sources) if sources else {"memory", "knowledge"}
        items: _List[ContextItem] = []

        if "memory" in wanted:
            for hit in await self._memory.search(scope, query, agent=agent, limit=limit):
                text = f"{hit.record.kind}: {_content_text(hit.record.content)}"
                items.append(ContextItem("memory", hit.record.id, text, hit.score))
        if "knowledge" in wanted:
            for khit in await self._knowledge.search(scope, query, limit=limit):
                items.append(ContextItem("knowledge", khit.doc.id, khit.snippet, khit.score))
        if "conversation" in wanted:
            for session in await self._conversations.list_sessions(scope):
                messages = await self._conversations.get_messages(scope, session.session_id)
                for i, msg in enumerate(messages):
                    score = relevance(query, msg.content)
                    if score > 0:
                        ref = f"{session.session_id}:{i}"
                        items.append(ContextItem("conversation", ref, msg.content, score))

        items.sort(key=lambda it: it.score, reverse=True)
        items = items[:limit]

        top = [it.score for it in items[:3]]
        quality = sum(top) / len(top) if top else 0.0
        covered = {t for it in items for t in _tokens(it.text)}
        missing = tuple(dict.fromkeys(t for t in _tokens(query) if t not in covered))
        return ContextResult(
            items=tuple(items),
            quality=quality,
            missing=missing,
            tokens_used=sum(_estimate_tokens(it.text) for it in items),
        )


# ----------------------------------------------------------------------- bundle
class FakeMeko:
    """Drop-in ``MekoBackend`` implementation backed by process memory."""

    def __init__(self) -> None:
        self.memory = FakeMemoryStore()
        self.knowledge = FakeKnowledgeStore()
        self.artifacts = FakeArtifactStore()
        self.traces = FakeTraceStore()
        self.conversations = FakeConversationStore()
        self.search = FakeContextSearch(self.memory, self.knowledge, self.conversations)

    async def health(self) -> bool:
        return True