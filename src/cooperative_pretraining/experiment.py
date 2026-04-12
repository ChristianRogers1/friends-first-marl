from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import erf, sqrt
from typing import Any

import numpy as np

from cooperative_pretraining.envs.public_eavesdrop_env import (
    ADVERSARY,
    LISTENER,
    SPEAKER,
    PublicEavesdropParallelEnv,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float64)
    out[index] = 1.0
    return out


@dataclass(frozen=True)
class TrainingConfig:
    train_episodes: int = 4000
    eval_episodes: int = 300
    listener_lr: float = 0.08
    adversary_lr: float = 0.08
    speaker_lr: float = 0.03
    leak_penalty: float = 0.5
    entropy_bonus: float = 0.002
    hidden_attributes_default: int = 0


class TrainableAgents:
    def __init__(self, num_candidates: int, num_attributes: int, attribute_cardinality: int, vocab_size: int, message_length: int):
        self.num_candidates = num_candidates
        self.num_attributes = num_attributes
        self.attribute_cardinality = attribute_cardinality
        self.vocab_size = vocab_size
        self.message_length = message_length

        speaker_input_dim = num_attributes * attribute_cardinality
        message_dim = message_length * vocab_size
        listener_input_dim = num_candidates * num_attributes * attribute_cardinality + message_dim
        adversary_input_dim = message_dim

        self.speaker_weights = np.zeros((message_length, speaker_input_dim, vocab_size), dtype=np.float64)
        self.listener_weights = np.zeros((listener_input_dim, num_candidates), dtype=np.float64)
        self.adversary_weights = np.zeros(
            (num_attributes, adversary_input_dim, attribute_cardinality), dtype=np.float64
        )

    def clone(self) -> "TrainableAgents":
        other = TrainableAgents(
            self.num_candidates,
            self.num_attributes,
            self.attribute_cardinality,
            self.vocab_size,
            self.message_length,
        )
        other.speaker_weights = self.speaker_weights.copy()
        other.listener_weights = self.listener_weights.copy()
        other.adversary_weights = self.adversary_weights.copy()
        return other

    def speaker_distribution(self, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self._encode_target(target)
        probs = []
        for token_idx in range(self.message_length):
            logits = x @ self.speaker_weights[token_idx]
            probs.append(_softmax(logits))
        return x, np.stack(probs)

    def sample_speaker_message(self, target: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        x, probs = self.speaker_distribution(target)
        message = np.array([rng.choice(self.vocab_size, p=probs[i]) for i in range(self.message_length)], dtype=np.int64)
        return message, {"speaker_input": x, "speaker_probs": probs}

    def listener_distribution(self, candidates: np.ndarray, message: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self._encode_listener(candidates, message)
        logits = x @ self.listener_weights
        return x, _softmax(logits)

    def sample_listener_action(self, candidates: np.ndarray, message: np.ndarray, rng: np.random.Generator) -> tuple[int, dict[str, Any]]:
        x, probs = self.listener_distribution(candidates, message)
        action = int(rng.choice(self.num_candidates, p=probs))
        return action, {"listener_input": x, "listener_probs": probs}

    def adversary_distributions(self, message: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self._encode_message(message)
        probs = []
        for attr_idx in range(self.num_attributes):
            logits = x @ self.adversary_weights[attr_idx]
            probs.append(_softmax(logits))
        return x, np.stack(probs)

    def sample_adversary_action(self, message: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        x, probs = self.adversary_distributions(message)
        reconstruction = np.array(
            [rng.choice(self.attribute_cardinality, p=probs[attr_idx]) for attr_idx in range(self.num_attributes)],
            dtype=np.int64,
        )
        return reconstruction, {"adversary_input": x, "adversary_probs": probs}

    def greedy_actions(self, observations: dict[str, dict[str, Any]]) -> dict[str, Any]:
        step_id = int(observations[SPEAKER]["step_id"])
        if step_id == 0:
            _, probs = self.speaker_distribution(np.asarray(observations[SPEAKER]["speaker_target"], dtype=np.int64))
            message = np.argmax(probs, axis=1).astype(np.int64)
            return {
                SPEAKER: message,
                LISTENER: 0,
                ADVERSARY: np.zeros(self.num_attributes, dtype=np.int64),
            }

        _, listener_probs = self.listener_distribution(
            np.asarray(observations[LISTENER]["listener_candidates"], dtype=np.int64),
            np.asarray(observations[LISTENER]["public_message"], dtype=np.int64),
        )
        _, adversary_probs = self.adversary_distributions(
            np.asarray(observations[ADVERSARY]["public_message"], dtype=np.int64)
        )
        return {
            SPEAKER: np.zeros(self.message_length, dtype=np.int64),
            LISTENER: int(np.argmax(listener_probs)),
            ADVERSARY: np.argmax(adversary_probs, axis=1).astype(np.int64),
        }

    def _encode_target(self, target: np.ndarray) -> np.ndarray:
        pieces = [_one_hot(int(value), self.attribute_cardinality) for value in target]
        return np.concatenate(pieces)

    def _encode_message(self, message: np.ndarray) -> np.ndarray:
        pieces = [_one_hot(int(value), self.vocab_size) for value in message]
        return np.concatenate(pieces)

    def _encode_listener(self, candidates: np.ndarray, message: np.ndarray) -> np.ndarray:
        candidate_bits = []
        for candidate in candidates:
            candidate_bits.extend(_one_hot(int(value), self.attribute_cardinality) for value in candidate)
        return np.concatenate(candidate_bits + [_one_hot(int(value), self.vocab_size) for value in message])


def train_agents(
    env: PublicEavesdropParallelEnv,
    pretraining_ratio: float,
    seed: int,
    config: TrainingConfig,
) -> TrainableAgents:
    rng = np.random.default_rng(seed)
    agents = TrainableAgents(
        num_candidates=env.action_space(LISTENER).n,
        num_attributes=env.action_space(ADVERSARY).nvec.shape[0],
        attribute_cardinality=int(env.action_space(ADVERSARY).nvec[0]),
        vocab_size=int(env.action_space(SPEAKER).nvec[0]),
        message_length=env.action_space(SPEAKER).nvec.shape[0],
    )

    pretrain_episodes = int(config.train_episodes * pretraining_ratio)
    public_episodes = config.train_episodes - pretrain_episodes

    for _ in range(pretrain_episodes):
        _run_training_episode(env, agents, "cooperative_pretraining", rng, config, train_adversary=False)
    for _ in range(public_episodes):
        _run_training_episode(env, agents, "public_adversarial", rng, config, train_adversary=True)

    return agents


def _run_training_episode(
    env: PublicEavesdropParallelEnv,
    agents: TrainableAgents,
    phase: str,
    rng: np.random.Generator,
    config: TrainingConfig,
    train_adversary: bool,
) -> None:
    observations, _ = env.reset(seed=int(rng.integers(0, 1_000_000)), options={"phase": phase})
    target = np.asarray(observations[SPEAKER]["speaker_target"], dtype=np.int64)

    message, speaker_cache = agents.sample_speaker_message(target, rng)
    observations, _, _, _, _ = env.step(
        {
            SPEAKER: message,
            LISTENER: 0,
            ADVERSARY: np.zeros(agents.num_attributes, dtype=np.int64),
        }
    )

    candidates = np.asarray(observations[LISTENER]["listener_candidates"], dtype=np.int64)
    listener_guess, listener_cache = agents.sample_listener_action(candidates, message, rng)
    adversary_reconstruction, adversary_cache = agents.sample_adversary_action(message, rng)
    _, rewards, _, _, infos = env.step(
        {
            SPEAKER: np.zeros(agents.message_length, dtype=np.int64),
            LISTENER: listener_guess,
            ADVERSARY: adversary_reconstruction,
        }
    )

    cooperative_reward = float(rewards[SPEAKER])
    adversary_reward = float(rewards.get(ADVERSARY, 0.0))
    target_index = int(infos[LISTENER]["target_index"])

    listener_target = _one_hot(target_index, agents.num_candidates)
    listener_error = listener_target - listener_cache["listener_probs"]
    agents.listener_weights += config.listener_lr * np.outer(listener_cache["listener_input"], listener_error)

    speaker_advantage = cooperative_reward + config.entropy_bonus
    speaker_input = speaker_cache["speaker_input"]
    speaker_probs = speaker_cache["speaker_probs"]
    for token_idx, token in enumerate(message):
        token_error = _one_hot(int(token), agents.vocab_size) - speaker_probs[token_idx]
        agents.speaker_weights[token_idx] += config.speaker_lr * speaker_advantage * np.outer(
            speaker_input, token_error
        )

    if train_adversary and phase == "public_adversarial":
        adversary_input = adversary_cache["adversary_input"]
        adversary_probs = adversary_cache["adversary_probs"]
        for attr_idx, value in enumerate(target):
            attr_error = _one_hot(int(value), agents.attribute_cardinality) - adversary_probs[attr_idx]
            agents.adversary_weights[attr_idx] += config.adversary_lr * np.outer(
                adversary_input, attr_error
            )

        leak_pressure = float(infos[ADVERSARY]["adversary_attribute_accuracy"])
        speaker_penalty = config.leak_penalty * leak_pressure
        for token_idx, token in enumerate(message):
            token_error = _one_hot(int(token), agents.vocab_size) - speaker_probs[token_idx]
            agents.speaker_weights[token_idx] -= config.speaker_lr * speaker_penalty * np.outer(
                speaker_input, token_error
            )


def evaluate_agents(
    env: PublicEavesdropParallelEnv,
    agents: TrainableAgents,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    listener_hits = []
    adversary_attr_acc = []
    cooperative_rewards = []
    cic_values = []

    for _ in range(episodes):
        observations, _ = env.reset(seed=int(rng.integers(0, 1_000_000)), options={"phase": "public_adversarial"})
        actions = agents.greedy_actions(observations)
        observations, _, _, _, _ = env.step(actions)
        actions = agents.greedy_actions(observations)
        _, rewards, _, _, infos = env.step(actions)

        listener_correct = float(infos[LISTENER]["listener_correct"])
        adv_acc = float(infos[ADVERSARY]["adversary_attribute_accuracy"])
        listener_hits.append(listener_correct)
        adversary_attr_acc.append(adv_acc)
        cooperative_rewards.append(float(rewards[SPEAKER]))

        candidates = np.asarray(observations[LISTENER]["listener_candidates"], dtype=np.int64)
        message = np.asarray(observations[LISTENER]["public_message"], dtype=np.int64)
        cic_values.append(estimate_cic(agents, candidates, message))

    return {
        "listener_accuracy": np.asarray(listener_hits, dtype=np.float64),
        "adversary_accuracy": np.asarray(adversary_attr_acc, dtype=np.float64),
        "cooperative_reward": np.asarray(cooperative_rewards, dtype=np.float64),
        "cic": np.asarray(cic_values, dtype=np.float64),
    }


def estimate_cic(agents: TrainableAgents, candidates: np.ndarray, observed_message: np.ndarray) -> float:
    _, base_probs = agents.listener_distribution(candidates, observed_message)
    total = 0.0
    count = 0
    for counterfactual in _counterfactual_messages(agents.vocab_size, agents.message_length, observed_message, limit=20):
        _, cf_probs = agents.listener_distribution(candidates, counterfactual)
        total += 0.5 * np.abs(base_probs - cf_probs).sum()
        count += 1
    return total / max(count, 1)


def _counterfactual_messages(
    vocab_size: int, message_length: int, observed_message: np.ndarray, limit: int
):
    count = 0
    observed_tuple = tuple(int(x) for x in observed_message)
    for values in product(range(vocab_size), repeat=message_length):
        if values == observed_tuple:
            continue
        yield np.asarray(values, dtype=np.int64)
        count += 1
        if count >= limit:
            return


def run_screen_and_confirm(
    env_factory,
    screening_seeds: int = 5,
    confirmation_seeds: int = 12,
    ratios: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75),
    base_seed: int = 7,
    config: TrainingConfig | None = None,
) -> dict[str, Any]:
    config = TrainingConfig() if config is None else config
    screening = {}

    for ratio in ratios:
        runs = []
        for offset in range(screening_seeds):
            seed = base_seed + offset
            env = env_factory()
            agents = train_agents(env, pretraining_ratio=ratio, seed=seed, config=config)
            metrics = evaluate_agents(env, agents, episodes=config.eval_episodes, seed=seed + 10_000)
            env.close()
            runs.append(metrics)
        screening[ratio] = summarize_run_group(runs)

    best_ratio = max(ratios, key=lambda ratio: screening[ratio]["listener_accuracy"]["mean"])

    baseline_runs = []
    pretrained_runs = []
    for offset in range(confirmation_seeds):
        seed = base_seed + 100 + offset
        env = env_factory()
        baseline_agents = train_agents(env, pretraining_ratio=0.0, seed=seed, config=config)
        baseline_metrics = evaluate_agents(env, baseline_agents, episodes=config.eval_episodes, seed=seed + 20_000)
        env.close()
        baseline_runs.append(baseline_metrics)

        env = env_factory()
        pretrained_agents = train_agents(env, pretraining_ratio=best_ratio, seed=seed, config=config)
        pretrained_metrics = evaluate_agents(
            env, pretrained_agents, episodes=config.eval_episodes, seed=seed + 30_000
        )
        env.close()
        pretrained_runs.append(pretrained_metrics)

    baseline_summary = summarize_run_group(baseline_runs)
    pretrained_summary = summarize_run_group(pretrained_runs)

    return {
        "screening": screening,
        "selected_ratio": best_ratio,
        "confirmation": {
            "baseline": baseline_summary,
            "pretrained": pretrained_summary,
            "comparisons": compare_groups(
                baseline_runs,
                pretrained_runs,
                keys=("listener_accuracy", "cic", "adversary_accuracy", "leakage_ratio"),
            ),
        },
    }


def summarize_run_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for key in runs[0]:
        per_seed = np.array([run[key].mean() for run in runs], dtype=np.float64)
        ci_low, ci_high = mean_confidence_interval(per_seed)
        summary[key] = {
            "mean": float(per_seed.mean()),
            "std": float(per_seed.std(ddof=0)),
            "ci95": [float(ci_low), float(ci_high)],
            "per_seed": per_seed.tolist(),
        }
    leakage_per_seed = []
    for run in runs:
        listener_mean = float(run["listener_accuracy"].mean())
        adversary_mean = float(run["adversary_accuracy"].mean())
        leakage_per_seed.append(adversary_mean / max(listener_mean, 1e-6))
    leakage_per_seed = np.asarray(leakage_per_seed, dtype=np.float64)
    ci_low, ci_high = mean_confidence_interval(leakage_per_seed)
    summary["leakage_ratio"] = {
        "mean": float(leakage_per_seed.mean()),
        "std": float(leakage_per_seed.std(ddof=0)),
        "ci95": [float(ci_low), float(ci_high)],
        "per_seed": leakage_per_seed.tolist(),
    }
    return summary


def compare_groups(
    baseline_runs: list[dict[str, Any]],
    pretrained_runs: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    comparisons = {}
    for key in keys:
        if key == "leakage_ratio":
            baseline = np.array(
                [
                    run["adversary_accuracy"].mean() / max(run["listener_accuracy"].mean(), 1e-6)
                    for run in baseline_runs
                ],
                dtype=np.float64,
            )
            pretrained = np.array(
                [
                    run["adversary_accuracy"].mean() / max(run["listener_accuracy"].mean(), 1e-6)
                    for run in pretrained_runs
                ],
                dtype=np.float64,
            )
        else:
            baseline = np.array([run[key].mean() for run in baseline_runs], dtype=np.float64)
            pretrained = np.array([run[key].mean() for run in pretrained_runs], dtype=np.float64)
        diff = pretrained - baseline
        ci_low, ci_high = mean_confidence_interval(diff)
        comparisons[key] = {
            "baseline_mean": float(baseline.mean()),
            "pretrained_mean": float(pretrained.mean()),
            "difference_mean": float(diff.mean()),
            "difference_ci95": [float(ci_low), float(ci_high)],
            "cohens_d": float(cohens_d(pretrained, baseline)),
            "p_value_normal_approx": float(two_sided_pvalue(diff)),
        }
    return comparisons


def mean_confidence_interval(values: np.ndarray, z: float = 1.96) -> tuple[float, float]:
    if values.size == 1:
        value = float(values[0])
        return value, value
    mean = float(values.mean())
    stderr = float(values.std(ddof=1) / np.sqrt(values.size))
    return mean - z * stderr, mean + z * stderr


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return 0.0
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0:
        return 0.0
    return float((x.mean() - y.mean()) / np.sqrt(pooled))


def two_sided_pvalue(diff: np.ndarray) -> float:
    if diff.size < 2:
        return 1.0
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    if std == 0:
        return 0.0 if mean != 0 else 1.0
    z = abs(mean / (std / np.sqrt(diff.size)))
    return float(2 * (1 - _normal_cdf(z)))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
