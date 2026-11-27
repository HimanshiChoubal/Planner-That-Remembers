"""Trained policy: a small numpy MLP, no PyTorch needed at inference time.

The policy maps the 16-dim observation (ratios to income plus remembered context) to the 5-dim
action; the projection layer then guarantees feasibility. The file format is a single ``.npz``
so it can be committed and loaded on a free CPU host in microseconds. If the file is missing,
``load_or_fallback`` returns the deterministic waterfall baseline so the app never breaks.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.rl.baselines import waterfall
from app.rl.env import OBS_DIM, HouseholdEnv

ACT_DIM = 5
DEFAULT_POLICY_PATH = Path(__file__).parent / "models" / "policy.npz"
# Observation indices that carry remembered context (risk level, income cv, dependents, equity cap).
MEMORY_FEATURES = (11, 12, 13, 14)
RESIDUAL_SCALE = 0.5  # a residual policy can shift each action component by at most this much


@dataclass(frozen=True)
class PolicyMeta:
    algorithm: str = "evolution_strategies"
    version: int = 1
    generations: int = 0
    seed: int = 0
    hidden: tuple[int, ...] = (32, 32)
    notes: str = ""
    residual: bool = False  # True: action = waterfall action + a bounded learned correction
    extra: dict = field(default_factory=dict)


def layer_sizes(hidden: Sequence[int]) -> tuple[int, ...]:
    return (OBS_DIM, *hidden, ACT_DIM)


def num_params(sizes: Sequence[int]) -> int:
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:], strict=True))


class MLPPolicy:
    def __init__(self, params: np.ndarray, sizes: Sequence[int], meta: PolicyMeta | None = None) -> None:
        self.sizes = tuple(sizes)
        if params.shape != (num_params(self.sizes),):
            raise ValueError(f"Expected {num_params(self.sizes)} parameters, got {params.shape}")
        self.params = params.astype(np.float64)
        self.meta = meta or PolicyMeta(hidden=self.sizes[1:-1])
        self._layers = self._unpack()

    @classmethod
    def random(cls, hidden: Sequence[int], rng: np.random.Generator, scale: float = 0.1, residual: bool = False) -> MLPPolicy:
        sizes = layer_sizes(hidden)
        params = rng.standard_normal(num_params(sizes)) * scale
        if residual:  # zero the output layer so the policy starts exactly at the waterfall
            params[num_params(sizes[:-1]):] = 0.0
        return cls(params, sizes, PolicyMeta(hidden=tuple(hidden), residual=residual))

    def _unpack(self) -> list[tuple[np.ndarray, np.ndarray]]:
        layers, i = [], 0
        for a, b in zip(self.sizes[:-1], self.sizes[1:], strict=True):
            w = self.params[i:i + a * b].reshape(a, b)
            i += a * b
            layers.append((w, self.params[i:i + b]))
            i += b
        return layers

    def with_params(self, params: np.ndarray) -> MLPPolicy:
        return MLPPolicy(params, self.sizes, self.meta)

    def _raw(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        for w, b in self._layers[:-1]:
            x = np.tanh(x @ w + b)
        w, b = self._layers[-1]
        return x @ w + b

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Direct action from an observation (non-residual policies)."""
        return (1.0 / (1.0 + np.exp(-self._raw(obs)))).astype(np.float32)

    def decide(self, env: HouseholdEnv, obs: np.ndarray | None = None) -> np.ndarray:
        """Action for the environment's current state (``obs`` can be overridden for ablations)."""
        raw = self._raw(env.observation() if obs is None else obs)
        if self.meta.residual:
            base = waterfall(env).astype(np.float64)
            return np.clip(base + RESIDUAL_SCALE * np.tanh(raw), 0.0, 1.0).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)

    def __call__(self, env: HouseholdEnv) -> np.ndarray:
        return self.decide(env)

    # ------------------------------------------------------------------ files
    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(asdict(self.meta))
        np.savez(path, params=self.params, sizes=np.array(self.sizes), meta=np.array(meta))

    @classmethod
    def load(cls, path: Path | str) -> MLPPolicy:
        data = np.load(Path(path), allow_pickle=False)
        meta_dict = json.loads(str(data["meta"]))
        meta_dict["hidden"] = tuple(meta_dict.get("hidden", ()))
        return cls(data["params"], tuple(int(s) for s in data["sizes"]), PolicyMeta(**meta_dict))


class MemoryMasked:
    """Wraps a policy and hides the remembered-context features (for ablations)."""

    def __init__(self, policy: MLPPolicy, neutral: Sequence[float] = (0.5, 0.0, 0.0, 1.0)) -> None:
        self.policy, self.neutral = policy, np.array(neutral, dtype=np.float32)

    def __call__(self, env: HouseholdEnv) -> np.ndarray:
        obs = env.observation().copy()
        obs[list(MEMORY_FEATURES)] = self.neutral
        return self.policy.decide(env, obs)


def load_or_fallback(path: Path | str | None = None) -> tuple[Callable[[HouseholdEnv], np.ndarray], bool]:
    """The trained policy if its file exists, else the waterfall baseline. Returns
    ``(policy, is_trained)`` so callers can say which one produced a recommendation."""
    target = Path(path) if path else DEFAULT_POLICY_PATH
    if target.exists():
        try:
            return MLPPolicy.load(target), True
        except (OSError, ValueError, KeyError):
            pass
    return waterfall, False