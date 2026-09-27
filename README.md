# Friends First: Cooperative Pretraining for Robust Emergent Communication Under Adversarial Pressure

Christian A. Rogers (Christian.Rogers@utah.edu) and Nathan Jensen (u1493643@utah.edu)
---

## Overview

Emergent communication in multi-agent reinforcement learning (MARL) degrades in mixed cooperative-competitive environments when messages are transmitted over public channels. Adversarial agents that can observe cooperative communication disrupt learned protocols, reducing cooperative performance to levels comparable to no communication at all.

This project investigates whether a **cooperative pretraining stage** can improve the resilience of learned communication. By first training agents in a purely cooperative setting (Phase 1), we allow them to develop communication conventions before exposing them to adversarial conditions (Phase 2). We hypothesize that this two-phase curriculum produces more robust communication protocols than training all agents simultaneously from the start.

### Hypotheses

1. **H1 - Task Performance:** Cooperatively pretrained agents will achieve higher listener accuracy than simultaneously trained agents when deployed in an adversarial environment with public communication.

2. **H2 - Communication Robustness:** Communication protocols established during cooperative pretraining will exhibit graceful degradation under adversarial pressure, rather than the protocol collapse observed in simultaneously trained agents. Measured via Causal Influence of Communication (CIC).

3. **H3 - Information Leakage:** Pretrained cooperative agents will leak less actionable information to adversarial eavesdroppers, as measured by eavesdropper accuracy and the leakage ratio.

---

## Environment: Adversarial Crypto

A custom multi-agent referential communication game inspired by PettingZoo's `simple_crypto`, modified to support a team-based public-eavesdropping setting:

```
                 +-----------+
  Target State   |  Speaker  | --- discrete message ---> PUBLIC CHANNEL
  (3 attr x 5)   +-----------+                              |
                                                    +-------+-------+
                                                    |               |
                                              +----------+    +-----------+
                      candidates + message -->| Listener |    | Adversary |<-- message only
                                              +----------+    +-----------+
                                                    |               |
                                               select target   reconstruct
                                                                  target
```

**Agents:**

| Agent | Observes | Action | Objective |
|-------|----------|--------|-----------|
| **Speaker** | Target state (3 attributes, 5 values each, one-hot encoded) | Discrete message: 3 symbols from vocabulary of 10 | Maximize listener accuracy, minimize information leakage |
| **Listener** | Shuffled candidates (1 target + 2 distractors) + message | Select 1 of 3 candidates | Correctly identify the target |
| **Adversary** | Message only (public channel) | Reconstruct target (predict each attribute) | Maximize reconstruction accuracy |

**Reward Structure:**

- *Phase 1 (cooperative):* `speaker_reward = listener_reward = listener_correct`
- *Phase 2 (adversarial):* `speaker_reward = listener_reward = listener_correct - 1.0 * adversary_accuracy`; `adversary_reward = adversary_accuracy`

**Training Phases:**

| Phase | Adversary | Channel | Objective |
|-------|-----------|---------|-----------|
| Phase 1 (cooperative pretraining) | Absent | Private | Develop communication protocols |
| Phase 2 (adversarial) | Active | Public | Maintain communication under eavesdropping |

The transition between phases is controlled by the `pretraining_ratio` parameter. For example, with `pretraining_ratio = 0.25` and 50,000 total episodes, Phase 1 runs for 12,500 episodes and Phase 2 for 37,500.

---

## Agent Architecture

Agents are trained using **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient), which provides centralized critics during training while permitting decentralized execution.

### Networks

| Component | Architecture | Input | Output |
|-----------|-------------|-------|--------|
| **Actor** | 2-layer MLP (128 hidden, ReLU) | Agent observation | Action probabilities (softmax) |
| **Critic** | 2-layer MLP (128 hidden, ReLU) | All observations + all actions (centralized) | Q-value |
| **Communication Module** (speaker only) | 2-layer MLP (64 hidden, ReLU) | Speaker observation | Discrete message via Gumbel-Softmax |

The **Gumbel-Softmax** relaxation enables gradient-based optimization of discrete messages during training, with argmax discretization at evaluation time.

---

## Project Structure

```
Friends-First-Marl/
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── default.yaml            # Default experiment configuration
│   ├── screening.yaml          # Screening stage (multiple pretraining ratios)
│   └── baseline.yaml           # Baseline: no pretraining (ratio = 0.0)
├── scripts/
│   ├── train.py                # Train a single agent configuration
│   ├── run_screening.py        # Run the full screening stage
│   └── evaluate.py             # Evaluate trained agents on H1/H2/H3
├── src/
│   ├── environments/
│   │   └── adversarial_crypto.py   # Custom multi-agent environment
│   ├── agents/
│   │   ├── networks.py             # Actor, Critic, CommunicationModule
│   │   └── maddpg.py               # MADDPG agent with communication
│   ├── training/
│   │   └── trainer.py              # Two-phase training pipeline
│   ├── evaluation/
│   │   └── metrics.py              # Task performance, CIC, leakage metrics
│   └── utils/
│       ├── config.py               # YAML config loader with defaults
│       └── replay_buffer.py        # Experience replay buffer
├── analysis/
│   ├── analyze_results.py          # Does the main analysis for the paper
│   └── attractor_analysis.py       # Verifies that the results are unimodal
├── tests/
│   ├── test_environment.py         # 9 environment tests
│   ├── test_agents.py              # 9 agent/network tests
│   └── test_training.py            # 3 training pipeline tests
└── slurm/
    ├── run_pipeline.sh             # Runs the full screen-validate pipeline on the U of U CHPC
    └── train_single.sh             # Trains a single environment/seed and outputs results on the U of U CHPC
```

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0

### Setup

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

All 21 tests should pass, confirming the environment, agents, and training pipeline are correctly set up.

---

## Usage

### Single Training Run

```bash
# Train with default configuration (pretraining_ratio = 0.25)
python scripts/train.py --config configs/default.yaml --seed 42

# Train the baseline (no pretraining, ratio = 0.0)
python scripts/train.py --config configs/baseline.yaml --seed 42

# Override pretraining ratio from the command line
python scripts/train.py --config configs/default.yaml --seed 42 --pretraining-ratio 0.5

# Use GPU if available
python scripts/train.py --config configs/default.yaml --seed 42 --device cuda
```

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir runs/
```

### Screening Stage

The screening stage trains agents across multiple pretraining ratios (0.1, 0.25, 0.5, 0.75) with 5 seeds each to identify the best configuration:

```bash
python scripts/run_screening.py --config configs/screening.yaml
```

This produces `results/screening/screening_results.json` with per-ratio performance summaries.

### Confirmation Stage

After selecting the best pretraining ratio from screening, run the confirmation stage with 12 seeds for both the selected configuration and the baseline:

```bash
# Pretrained agents (replace 0.25 with the selected ratio)
for seed in $(seq 0 11); do
    python scripts/train.py --config configs/default.yaml --seed $seed --pretraining-ratio 0.25
done

# Baseline agents (no pretraining)
for seed in $(seq 0 11); do
    python scripts/train.py --config configs/baseline.yaml --seed $seed
done
```

### Evaluation

Evaluate trained agents on all three hypothesis metrics:

```bash
python scripts/evaluate.py \
    --config configs/default.yaml \
    --checkpoint-dir checkpoints/seed_42_ratio0.25 \
    --num-episodes 500
```

This reports:

- **H1:** Listener accuracy with 95% confidence intervals
- **H2:** Mean Causal Influence of Communication (CIC) via counterfactual message substitution
- **H3:** Adversary accuracy and leakage ratio (adversary accuracy / listener accuracy)

Results are saved to `evaluation_results.json` in the checkpoint directory.

---

## Configuration

All configuration is managed through YAML files in `configs/`. Any parameter can be overridden. The full set of parameters with defaults:

### Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_attributes` | 3 | Number of attributes composing the target state |
| `num_values` | 5 | Number of possible values per attribute |
| `num_distractors` | 2 | Number of distractor targets (candidates = 1 + distractors) |
| `vocab_size` | 10 | Size of the discrete communication vocabulary |
| `msg_length` | 3 | Number of symbols per message |
| `cooperative_reward_weight` | 1.0 | Scaling factor for listener accuracy reward |
| `leakage_penalty_weight` | 0.5 | Scaling factor for adversary accuracy penalty |

### Agent Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | Hidden layer size for actor and critic networks |
| `comm_hidden_dim` | 64 | Hidden layer size for communication module |
| `lr_actor` | 0.001 | Adam learning rate for actor |
| `lr_critic` | 0.001 | Adam learning rate for critic |
| `gamma` | 0.95 | Discount factor |
| `tau` | 0.01 | Soft update coefficient for target networks |
| `gumbel_temperature` | 1.0 | Gumbel-Softmax temperature for discrete messages |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_episodes` | 50,000 | Total training episodes across both phases |
| `pretraining_ratio` | 0.25 | Fraction of episodes for cooperative pretraining (Phase 1) |
| `batch_size` | 256 | Minibatch size for network updates |
| `buffer_capacity` | 100,000 | Replay buffer capacity |
| `update_every` | 4 | Update networks every N episodes |
| `eval_every` | 500 | Run evaluation every N episodes |
| `eval_episodes` | 100 | Number of episodes per evaluation |
| `checkpoint_every` | 5,000 | Save checkpoints every N episodes |
| `noise_scale` | 0.1 | Initial exploration noise magnitude |
| `noise_decay` | 0.9999 | Multiplicative noise decay per episode |
| `min_noise` | 0.01 | Minimum exploration noise |

### Experiment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `num_seeds` | 5 | Number of seeds (for screening) |
| `device` | `"cpu"` | Compute device (`"cpu"` or `"cuda"`) |
| `log_dir` | `"runs"` | TensorBoard log directory |
| `checkpoint_dir` | `"checkpoints"` | Model checkpoint directory |
| `results_dir` | `"results"` | Results output directory |

---

## Evaluation Metrics

### H1 - Task Performance

Proportion of evaluation episodes where the listener correctly identifies the target. Reported with 95% confidence intervals and Cohen's d effect size comparing pretrained vs. baseline.

### H2 - Causal Influence of Communication (CIC)

Measures whether messages *causally influence* the listener's behavior using counterfactual message substitution (Lowe et al., 2019). For each episode, the speaker's actual message is replaced with random alternative messages and the KL divergence between the listener's resulting action distributions is computed. CIC > 0 confirms genuine communication rather than epiphenomenal correlation.

### H3 - Information Leakage

Two sub-metrics:
- **Eavesdropper accuracy:** Fraction of target attributes the adversary correctly reconstructs from the public message.
- **Leakage ratio:** Eavesdropper accuracy divided by listener accuracy. Lower values indicate the cooperative team communicates effectively while limiting adversary information gain.

---

## Experimental Workflow

```
1. Screening Stage                    2. Confirmation Stage              3. Analysis
+------------------------------+      +---------------------------+      +---------------------+
| For each ratio in             |      | 12 seeds x best ratio    |      | Compare pretrained  |
| {0.1, 0.25, 0.5, 0.75}:     |      | 12 seeds x baseline      |      | vs. baseline:       |
|   Train 5 seeds              | ---> | (pretraining_ratio=0.0)  | ---> | - H1: accuracy + CI |
|   Evaluate listener accuracy |      |                           |      | - H2: CIC values    |
|   Select best ratio          |      | Full evaluation on H1-H3 |      | - H3: leakage ratio |
+------------------------------+      +---------------------------+      | - Cohen's d         |
                                                                          +---------------------+
```

---

## Key References

- Lowe et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.* arXiv:1706.02275
- Vanneste et al. (2021). *Mixed Cooperative-Competitive Communication Using Multi-Agent Reinforcement Learning.* Springer.
- Blumenkamp & Prorok (2020). *The Emergence of Adversarial Communication in Multi-Agent Reinforcement Learning.* arXiv:2008.02616
- Lowe et al. (2019). *On the Pitfalls of Measuring Emergent Communication.* arXiv:1903.05168
- Tilbury et al. (2023). *Revisiting the Gumbel-Softmax in MADDPG.* arXiv:2302.11793
- Farrell & Gibbons (1989). *Cheap Talk with Two Audiences.* American Economic Review, 79:1214--1223
- Abadi & Andersen (2016). *Learning to Protect Communications with Adversarial Neural Cryptography.* arXiv:1610.06918
- Yu, Mu, & Goodman (2022). *Emergent Covert Signaling in Adversarial Reference Games.* EmeCom Workshop, ICLR 2022
- Bengio et al. (2009). *Curriculum Learning.* ICML.

---

## Attribution

This project was originally developed as a final project for Spring 2026 - CS 6955, Advanced AI (crosslisted with CS 5955) at the University of Utah under the Kahlert School of Computing.

We want to express our gratitude for the use of the University's Center for High Performance Computing resources, as this project could not have been completed without their GPU allocations.
