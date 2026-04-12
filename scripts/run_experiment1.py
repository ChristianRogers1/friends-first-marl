from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from cooperative_pretraining.envs.public_eavesdrop_env import (
    ADVERSARY,
    LISTENER,
    SPEAKER,
    PublicEavesdropParallelEnv,
)
from cooperative_pretraining.policies import heuristic_actions, random_actions


def rollout_episode(env: PublicEavesdropParallelEnv, phase: str, policy: str, rng: np.random.Generator):
    observations, infos = env.reset(seed=int(rng.integers(0, 1_000_000)), options={"phase": phase})

    if policy == "heuristic":
        actions = heuristic_actions(observations)
    else:
        actions = random_actions(observations, rng)
    observations, rewards, terminations, truncations, infos_step0 = env.step(actions)

    if policy == "heuristic":
        actions = heuristic_actions(observations)
    else:
        actions = random_actions(observations, rng)
    observations, rewards, terminations, truncations, infos_step1 = env.step(actions)

    del observations, terminations, truncations, infos, infos_step0
    return rewards, infos_step1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-sample", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = PublicEavesdropParallelEnv(render_mode="ansi" if args.show_sample else None)

    if args.show_sample:
        observations, infos = env.reset(seed=args.seed, options={"phase": "public_adversarial"})
        print("Initial state")
        print(env.render())
        print(f"speaker_obs={observations[SPEAKER]}")
        print(f"listener_obs={observations[LISTENER]}")
        print(f"adversary_obs={observations[ADVERSARY]}")
        print(f"infos={infos}")
        print()

    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for phase in ("cooperative_pretraining", "public_adversarial"):
        for _ in range(args.episodes):
            rewards, infos = rollout_episode(env, phase=phase, policy=args.policy, rng=rng)
            metrics[phase]["listener_accuracy"].append(float(infos[LISTENER]["listener_correct"]))
            metrics[phase]["adversary_attribute_accuracy"].append(
                float(infos[ADVERSARY]["adversary_attribute_accuracy"])
            )
            metrics[phase]["adversary_exact_accuracy"].append(
                float(infos[ADVERSARY]["adversary_exact"])
            )
            metrics[phase]["speaker_reward"].append(float(rewards[SPEAKER]))
            metrics[phase]["adversary_reward"].append(float(rewards[ADVERSARY]))

    print(f"Policy: {args.policy}")
    print("Note: this evaluates fixed policies in the proposal-aligned domain. It does not train agents.")
    print()
    for phase in ("cooperative_pretraining", "public_adversarial"):
        print(phase)
        print(_format_metrics(metrics[phase]))
        print()


def _format_metrics(metrics: dict[str, list[float]]) -> str:
    ordered_keys = (
        "listener_accuracy",
        "adversary_attribute_accuracy",
        "adversary_exact_accuracy",
        "speaker_reward",
        "adversary_reward",
    )
    parts = []
    for key in ordered_keys:
        values = np.asarray(metrics[key], dtype=np.float64)
        parts.append(f"{key}: mean={values.mean():.3f} std={values.std():.3f}")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
