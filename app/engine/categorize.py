"""Transaction categorization.

Weighted k-nearest-neighbour over text embeddings of cleaned merchant strings. The default
embedder is character n-gram TF-IDF, which runs offline with no model download and handles the
typos and noise of bank descriptions well. A sentence-transformer embedder can be swapped in.

The categorizer learns: every user correction is added as a high-weight example, and low
confidence predictions are flagged so the app can ask the user instead of guessing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.regions.base import RegionPack

UNKNOWN_CATEGORY = "other"
CORRECTION_WEIGHT = 3.0
MIN_SIMILARITY = 0.05
_NON_LETTERS = re.compile(r"[^A-Z\s]")
_GENERIC_TOKENS = frozenset({"POS", "DEBIT", "PURCHASE", "CHECKCARD", "TXN", "REF", "NO", "UPI"})


def normalize_merchant(text: str) -> str:
    """Uppercase, drop digits/punctuation and generic bank tokens, collapse whitespace."""
    upper = text.upper()
    letters = _NON_LETTERS.sub(" ", upper)
    tokens = [t for t in letters.split() if t not in _GENERIC_TOKENS]
    cleaned = " ".join(tokens)
    return cleaned or " ".join(upper.split())


@dataclass(frozen=True)
class Example:
    text: str
    category: str
    weight: float = 1.0


@dataclass(frozen=True)
class Prediction:
    category: str
    confidence: float  # 0-1; vote share scaled by neighbour similarity
    similarity: float  # similarity of the closest neighbour
    runner_up: str | None
    needs_review: bool  # True when the app should ask the user to confirm


class Embedder(Protocol):
    def fit(self, texts: Sequence[str]) -> None: ...

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised dense rows, one per text."""
        ...


class TfidfCharEmbedder:
    """Character n-gram TF-IDF (default): offline, fast, robust to typos."""

    def __init__(self, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True)

    def fit(self, texts: Sequence[str]) -> None:
        self._vec.fit(list(texts))

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._vec.transform(list(texts)).todense(), dtype=float)


class SentenceTransformerEmbedder:
    """Optional semantic embedder (needs the sentence-transformers package and model files)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self._model = SentenceTransformer(model_name)

    def fit(self, texts: Sequence[str]) -> None:
        return None  # pretrained

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True), dtype=float)


class Categorizer:
    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        k: int = 5,
        review_threshold: float = 0.5,
        strong_similarity: float = 0.6,
    ) -> None:
        self._embedder: Embedder = embedder or TfidfCharEmbedder()
        self._k = k
        self._review_threshold = review_threshold
        self._strong = strong_similarity
        self._examples: list[Example] = []
        self._matrix: np.ndarray | None = None

    # ---------------------------------------------------------------- training
    @property
    def n_examples(self) -> int:
        return len(self._examples)

    def fit(self, examples: Iterable[Example]) -> Categorizer:
        self._examples = [
            Example(normalize_merchant(e.text), e.category, e.weight) for e in examples
        ]
        self._refit()
        return self

    def add_examples(self, examples: Iterable[Example]) -> None:
        self._examples.extend(
            Example(normalize_merchant(e.text), e.category, e.weight) for e in examples
        )
        self._refit()

    def learn(self, text: str, category: str, weight: float = CORRECTION_WEIGHT) -> None:
        """Learn from a user correction (weighted above ordinary examples)."""
        self.add_examples([Example(text, category, weight)])

    def _refit(self) -> None:
        if not self._examples:
            self._matrix = None
            return
        texts = [e.text for e in self._examples]
        self._embedder.fit(texts)
        self._matrix = self._embedder.transform(texts)

    # -------------------------------------------------------------- prediction
    def predict(self, text: str) -> Prediction:
        if self._matrix is None:
            raise RuntimeError("Categorizer has no training examples; call fit() first")
        q = self._embedder.transform([normalize_merchant(text)])[0]
        sims = self._matrix @ q
        top = np.argsort(-sims)[: self._k]
        votes: dict[str, float] = defaultdict(float)
        for i in top:
            s = float(sims[i])
            if s > MIN_SIMILARITY:
                votes[self._examples[i].category] += s * s * self._examples[i].weight
        if not votes:
            return Prediction(UNKNOWN_CATEGORY, 0.0, 0.0, None, True)
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        best, best_score = ranked[0]
        share = best_score / sum(votes.values())
        top_sim = float(sims[top[0]])
        confidence = share * min(1.0, top_sim / self._strong)
        return Prediction(
            category=best,
            confidence=confidence,
            similarity=top_sim,
            runner_up=ranked[1][0] if len(ranked) > 1 else None,
            needs_review=confidence < self._review_threshold,
        )

    def predict_many(self, texts: Sequence[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]

    def accuracy(self, labelled: Iterable[tuple[str, str]]) -> float:
        """Share of (text, true_category) pairs predicted correctly."""
        pairs = list(labelled)
        if not pairs:
            return 0.0
        return sum(self.predict(t).category == c for t, c in pairs) / len(pairs)


def seed_examples(pack: RegionPack) -> list[Example]:
    """Starter training set built from a region pack's synthetic merchants."""
    return [
        Example(name, category)
        for category, names in pack.merchants().items()
        for name in names
    ]