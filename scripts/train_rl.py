"""Train the RL policy (evolution strategies, numpy only) and save it.

    python scripts/train_rl.py --generations 120 --out app/rl/models/policy.npz

No GPU and no PyTorch are needed. A log of every generation is written next to the policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rl.policy import DEFAULT_POLICY_PATH, MLPPolicy  # noqa: E402
from app.rl.train import ESConfig, train_es  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the household-planning policy.")
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--households", type=int, default=8, help="households per generation")
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--resume", type=Path, default=None, help="continue training from this policy file")
    args = parser.parse_args()

    cfg = ESConfig(generations=args.generations, population=args.population, households_per_gen=args.households,
                   sigma=args.sigma, lr=args.lr, seed=args.seed)

    def log(rec: dict) -> None:
        val = rec.get("validation_improvement")
        extra = f" | validation improvement {val:+.3f}" if val is not None else ""
        print(f"gen {rec['generation']:4d} | mean fitness {rec['mean_fitness']:+.3f} | best {rec['best_fitness']:+.3f}"
              f" | {rec['elapsed_s']:.0f}s{extra}", flush=True)

    init = MLPPolicy.load(args.resume) if args.resume else None
    policy, history = train_es(cfg, log=log, init=init)
    policy.save(args.out)
    args.out.with_suffix(".log.json").write_text(json.dumps(history, indent=2))
    print(f"saved {args.out} (best validation improvement "
          f"{policy.meta.extra['best_validation_improvement']:+.3f} over the waterfall)")


if __name__ == "__main__":
    main()