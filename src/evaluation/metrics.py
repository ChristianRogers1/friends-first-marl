"""
Evaluation metrics for emergent communication experiments.

Implements the three key metrics from the project proposal:
  H1 - Task Performance: Listener accuracy with effect size and CIs.
  H2 - Communication Robustness: Causal Influence of Communication (CIC).
  H3 - Information Leakage: Eavesdropper accuracy and leakage ratio.

CIC Reference:
    Lowe et al., "On the Pitfalls of Measuring Emergent Communication" (2019)
"""

import numpy as np
import torch
from scipy import stats

from ..environments.adversarial_crypto import AdversarialCryptoEnv
from ..agents.maddpg import MADDPGAgent


def evaluate_task_performance(
    env: AdversarialCryptoEnv,
    agents: dict[str, MADDPGAgent],
    num_episodes: int = 500,
    phase: int = 2,
    config: dict | None = None,
) -> dict:
    """Evaluate listener task accuracy (H1).

    Args:
        env: The environment instance.
        agents: Dictionary of agents.
        num_episodes: Number of evaluation episodes.
        phase: Training phase (1=cooperative, 2=adversarial).
        config: Configuration dictionary.

    Returns:
        Dictionary with accuracy, confidence interval, and per-episode results.
    """
    if config is None:
        config = {"env": {"vocab_size": 10, "msg_length": 3, "num_values": 5, "num_attributes": 3}}

    env.set_phase(phase)
    for agent in agents.values():
        agent.eval_mode()

    results = []
    for _ in range(num_episodes):
        obs = env.reset()
        speaker_obs = obs["speaker"]

        # Speaker
        speaker_action = agents["speaker"].select_action(speaker_obs, explore=False)
        msg_discrete = _onehot_to_discrete(
            speaker_action, config["env"]["vocab_size"], config["env"]["msg_length"]
        )
        post_obs = env.step_speaker(msg_discrete)

        # Listener
        listener_obs = post_obs["listener"]
        listener_action = agents["listener"].select_action(listener_obs, explore=False)
        listener_selection = np.argmax(listener_action)
        env.step_listener(listener_selection)
        results.append(env._listener_correct)

    results = np.array(results)
    accuracy = results.mean()
    ci = stats.t.interval(
        0.95,
        df=len(results) - 1,
        loc=accuracy,
        scale=stats.sem(results) if results.std() > 0 else 1e-10,
    )

    return {
        "accuracy": float(accuracy),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "std": float(results.std()),
        "per_episode": results.tolist(),
    }


def compute_cic(
    env: AdversarialCryptoEnv,
    agents: dict[str, MADDPGAgent],
    num_episodes: int = 200,
    num_counterfactuals: int = 10,
    config: dict | None = None,
) -> dict:
    """Compute Causal Influence of Communication (H2).

    Uses counterfactual message substitution: for each episode, replace the
    speaker's actual message with random alternative messages and measure
    how much the listener's action distribution changes.

    CIC > 0 confirms genuine communication rather than epiphenomenal correlation.

    Args:
        env: The environment instance.
        agents: Dictionary of agents.
        num_episodes: Number of episodes to evaluate.
        num_counterfactuals: Number of counterfactual messages per episode.
        config: Configuration dictionary.

    Returns:
        Dictionary with mean CIC, per-episode CIC values.
    """
    if config is None:
        config = {"env": {"vocab_size": 10, "msg_length": 3, "num_values": 5, "num_attributes": 3}}

    vocab_size = config["env"]["vocab_size"]
    msg_length = config["env"]["msg_length"]

    env.set_phase(2)  # CIC measured in adversarial phase
    for agent in agents.values():
        agent.eval_mode()

    cic_values = []

    for _ in range(num_episodes):
        obs = env.reset()
        speaker_obs = obs["speaker"]

        # Get actual message
        speaker_action = agents["speaker"].select_action(speaker_obs, explore=False)
        actual_msg = _onehot_to_discrete(speaker_action, vocab_size, msg_length)

        # Get listener's action distribution with actual message
        post_obs = env.step_speaker(actual_msg)
        listener_obs_actual = post_obs["listener"]
        with torch.no_grad():
            obs_t = torch.FloatTensor(listener_obs_actual).unsqueeze(0)
            actual_action_dist = agents["listener"].actor(
                obs_t.to(agents["listener"].device)
            ).cpu().numpy().flatten()

        # Generate counterfactual messages and measure action distributions
        kl_divergences = []
        for _ in range(num_counterfactuals):
            # Random counterfactual message
            cf_msg = np.random.randint(0, vocab_size, size=msg_length)

            # Get listener observation with counterfactual message
            cf_post_obs = env.step_speaker(cf_msg)
            cf_listener_obs = cf_post_obs["listener"]

            with torch.no_grad():
                cf_obs_t = torch.FloatTensor(cf_listener_obs).unsqueeze(0)
                cf_action_dist = agents["listener"].actor(
                    cf_obs_t.to(agents["listener"].device)
                ).cpu().numpy().flatten()

            # KL divergence: D_KL(actual || counterfactual)
            kl = _kl_divergence(actual_action_dist, cf_action_dist)
            kl_divergences.append(kl)

        # CIC for this episode = average KL divergence across counterfactuals
        cic_values.append(np.mean(kl_divergences))

    cic_values = np.array(cic_values)
    return {
        "mean_cic": float(cic_values.mean()),
        "std_cic": float(cic_values.std()),
        "per_episode": cic_values.tolist(),
    }


def compute_information_leakage(
    env: AdversarialCryptoEnv,
    agents: dict[str, MADDPGAgent],
    num_episodes: int = 500,
    config: dict | None = None,
) -> dict:
    """Compute information leakage metrics (H3).

    Measures eavesdropper accuracy and the ratio of eavesdropper accuracy
    to listener accuracy.

    Args:
        env: The environment instance.
        agents: Dictionary of agents.
        num_episodes: Number of evaluation episodes.
        config: Configuration dictionary.

    Returns:
        Dictionary with adversary accuracy, listener accuracy, leakage ratio.
    """
    if config is None:
        config = {"env": {"vocab_size": 10, "msg_length": 3, "num_values": 5, "num_attributes": 3}}

    env.set_phase(2)
    for agent in agents.values():
        agent.eval_mode()

    listener_results = []
    adversary_results = []

    for _ in range(num_episodes):
        obs = env.reset()
        speaker_obs = obs["speaker"]

        # Speaker
        speaker_action = agents["speaker"].select_action(speaker_obs, explore=False)
        msg_discrete = _onehot_to_discrete(
            speaker_action, config["env"]["vocab_size"], config["env"]["msg_length"]
        )
        post_obs = env.step_speaker(msg_discrete)

        # Listener
        listener_obs = post_obs["listener"]
        listener_action = agents["listener"].select_action(listener_obs, explore=False)
        listener_selection = np.argmax(listener_action)
        env.step_listener(listener_selection)
        listener_results.append(env._listener_correct)

        # Adversary
        if "adversary" in post_obs:
            adversary_obs = post_obs["adversary"]
            adversary_action = agents["adversary"].select_action(adversary_obs, explore=False)
            adversary_recon = _onehot_to_discrete(
                adversary_action,
                config["env"]["num_values"],
                config["env"]["num_attributes"],
            )
            adversary_acc = float(np.mean(adversary_recon == env._target))
            adversary_results.append(adversary_acc)

    listener_accuracy = np.mean(listener_results)
    adversary_accuracy = np.mean(adversary_results) if adversary_results else 0.0

    # Leakage ratio: adversary_accuracy / listener_accuracy
    leakage_ratio = (
        adversary_accuracy / listener_accuracy
        if listener_accuracy > 0
        else float("inf")
    )

    return {
        "listener_accuracy": float(listener_accuracy),
        "adversary_accuracy": float(adversary_accuracy),
        "leakage_ratio": float(leakage_ratio),
        "listener_per_episode": listener_results,
        "adversary_per_episode": adversary_results,
    }


def compute_effect_size(
    pretrained_results: list[float],
    baseline_results: list[float],
) -> dict:
    """Compute Cohen's d effect size between pretrained and baseline.

    Args:
        pretrained_results: Per-episode results from pretrained agents.
        baseline_results: Per-episode results from baseline agents.

    Returns:
        Dictionary with Cohen's d, pooled std, and means.
    """
    pre = np.array(pretrained_results)
    base = np.array(baseline_results)

    mean_pre = pre.mean()
    mean_base = base.mean()

    # Pooled standard deviation
    n1, n2 = len(pre), len(base)
    pooled_std = np.sqrt(
        ((n1 - 1) * pre.var(ddof=1) + (n2 - 1) * base.var(ddof=1))
        / (n1 + n2 - 2)
    )

    cohens_d = (mean_pre - mean_base) / pooled_std if pooled_std > 0 else 0.0

    return {
        "cohens_d": float(cohens_d),
        "pooled_std": float(pooled_std),
        "pretrained_mean": float(mean_pre),
        "baseline_mean": float(mean_base),
    }


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence D_KL(p || q) with numerical stability."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _onehot_to_discrete(
    onehot: np.ndarray, num_classes: int, num_slots: int
) -> np.ndarray:
    """Convert flat one-hot vector to discrete indices."""
    reshaped = onehot.reshape(num_slots, num_classes)
    return np.argmax(reshaped, axis=-1)
