from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cooperative_pretraining.envs.public_eavesdrop_env import PublicEavesdropParallelEnv
from cooperative_pretraining.maddpg import MADDPGConfig, run_screen_and_confirm_maddpg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-episodes", type=int, default=2500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--screening-seeds", type=int, default=5)
    parser.add_argument("--confirmation-seeds", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    config = MADDPGConfig(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        device=args.device,
    )
    results = run_screen_and_confirm_maddpg(
        env_factory=lambda: PublicEavesdropParallelEnv(),
        screening_seeds=args.screening_seeds,
        confirmation_seeds=args.confirmation_seeds,
        base_seed=args.base_seed,
        config=config,
    )
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
