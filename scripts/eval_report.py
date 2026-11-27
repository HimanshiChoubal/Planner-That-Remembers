"""Evaluate the trained policy against the baselines and write docs/evaluation.md.

    python scripts/eval_report.py --policy app/rl/models/policy.npz --out docs/evaluation.md

Two held-out sets: random households from a sampler different from the training seed, and the six
demo personas (never used in training). Results are paired by household and random seed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rl.baselines import BASELINES  # noqa: E402
from app.rl.evaluate import evaluate, persona_households  # noqa: E402
from app.rl.policy import DEFAULT_POLICY_PATH, MemoryMasked, MLPPolicy  # noqa: E402
from app.rl.train import sample_household  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the policy against baselines.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--households", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=5, help="episodes per household")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation.md"))
    args = parser.parse_args()

    policies = {k: v for k, v in BASELINES.items()}
    if args.policy.exists():
        trained = MLPPolicy.load(args.policy)
        policies["rl"] = trained
        policies["rl_without_memory"] = MemoryMasked(trained)
    else:
        print(f"No policy at {args.policy}; evaluating baselines only.")

    rng = np.random.default_rng(args.seed)
    random_set = [sample_household(rng) for _ in range(args.households)]
    personas = list(persona_households(date.today()).values())
    reports = [
        evaluate(policies, random_set, label="Random held-out households", episodes_per_household=args.episodes, seed=args.seed),
        evaluate(policies, personas, label="Demo personas (never used in training)", episodes_per_household=args.episodes * 4, seed=args.seed),
    ]
    text = "# Policy evaluation\n\nSynthetic simulation results. Not financial advice.\n\n" + "\n".join(r.to_markdown() for r in reports)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()