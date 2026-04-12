from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

SPEAKER = "speaker_0"
LISTENER = "listener_0"
ADVERSARY = "adversary_0"


@dataclass(frozen=True)
class PublicEavesdropConfig:
    vocab_size: int = 10
    message_length: int = 3
    num_candidates: int = 4
    num_attributes: int = 5
    attribute_cardinality: int = 10
    distractor_mutations: int = 2
    leak_penalty: float = 0.5
    phase: str = "public_adversarial"
    render_size: int = 720


class PublicEavesdropParallelEnv(ParallelEnv[str, dict[str, Any], Any]):
    """Modified MPE-style public eavesdropping domain for experiment 1.

    The environment follows the proposal's structure:
    - a speaker observes the target state
    - a listener identifies the target from distractors
    - an adversary reconstructs the target from the public message
    - messages are discrete sequences of length 3 over a vocabulary of size 10

    The environment is implemented as a two-step PettingZoo parallel episode:
    1. the speaker emits the message
    2. the listener and adversary act on the public message
    """

    metadata = {
        "name": "public_eavesdrop_parallel_v0",
        "render_modes": ["human", "rgb_array", "ansi"],
        "is_parallelizable": True,
        "render_fps": 8,
    }

    def __init__(
        self,
        config: PublicEavesdropConfig | None = None,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self.config = PublicEavesdropConfig(**kwargs) if config is None else config
        self.render_mode = render_mode
        self.possible_agents = [SPEAKER, LISTENER, ADVERSARY]
        self.agents = self.possible_agents[:]
        self.rng = np.random.default_rng()
        self._episode_state: dict[str, Any] | None = None

        attr_high = self.config.attribute_cardinality - 1
        message_high = self.config.vocab_size - 1
        candidate_shape = (self.config.num_candidates, self.config.num_attributes)
        position_shape = (self.config.num_candidates, 2)

        self.observation_spaces = {
            SPEAKER: spaces.Dict(
                {
                    "phase_id": spaces.Discrete(2),
                    "step_id": spaces.Discrete(2),
                    "speaker_target": spaces.Box(
                        low=0,
                        high=attr_high,
                        shape=(self.config.num_attributes,),
                        dtype=np.int64,
                    ),
                    "candidate_positions": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=position_shape,
                        dtype=np.float32,
                    ),
                    "public_message": spaces.Box(
                        low=0,
                        high=message_high,
                        shape=(self.config.message_length,),
                        dtype=np.int64,
                    ),
                    "has_public_message": spaces.Discrete(2),
                }
            ),
            LISTENER: spaces.Dict(
                {
                    "phase_id": spaces.Discrete(2),
                    "step_id": spaces.Discrete(2),
                    "listener_candidates": spaces.Box(
                        low=0,
                        high=attr_high,
                        shape=candidate_shape,
                        dtype=np.int64,
                    ),
                    "candidate_positions": spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=position_shape,
                        dtype=np.float32,
                    ),
                    "public_message": spaces.Box(
                        low=0,
                        high=message_high,
                        shape=(self.config.message_length,),
                        dtype=np.int64,
                    ),
                    "has_public_message": spaces.Discrete(2),
                }
            ),
            ADVERSARY: spaces.Dict(
                {
                    "phase_id": spaces.Discrete(2),
                    "step_id": spaces.Discrete(2),
                    "public_message": spaces.Box(
                        low=0,
                        high=message_high,
                        shape=(self.config.message_length,),
                        dtype=np.int64,
                    ),
                    "has_public_message": spaces.Discrete(2),
                }
            ),
        }
        self.action_spaces = {
            SPEAKER: spaces.MultiDiscrete([self.config.vocab_size] * self.config.message_length),
            LISTENER: spaces.Discrete(self.config.num_candidates),
            ADVERSARY: spaces.MultiDiscrete(
                [self.config.attribute_cardinality] * self.config.num_attributes
            ),
        }

        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.canvas = pygame.Surface((self.config.render_size, self.config.render_size))

    def _active_agents_for_phase(self, phase: str) -> list[str]:
        if phase == "cooperative_pretraining":
            return [SPEAKER, LISTENER]
        if phase == "public_adversarial":
            return [SPEAKER, LISTENER, ADVERSARY]
        raise ValueError(f"Unsupported phase: {phase}")

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        phase = (options or {}).get("phase", self.config.phase)
        self.agents = self._active_agents_for_phase(phase)
        candidates, target_index = self._sample_candidates()
        positions = self._sample_positions()
        target = candidates[target_index].copy()

        self._episode_state = {
            "phase": phase,
            "step_id": 0,
            "target": target,
            "target_index": int(target_index),
            "candidates": candidates,
            "candidate_positions": positions,
            "public_message": np.zeros(self.config.message_length, dtype=np.int64),
            "has_public_message": 0,
        }

        observations = self._build_observations()
        infos = {agent: {"role": agent.split("_")[0], "phase": phase} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, Any]):
        if self._episode_state is None:
            raise RuntimeError("Call reset() before step().")

        phase = str(self._episode_state["phase"])
        step_id = int(self._episode_state["step_id"])

        if step_id == 0:
            message = np.asarray(actions[SPEAKER], dtype=np.int64)
            if not self.action_spaces[SPEAKER].contains(message):
                raise ValueError(f"Invalid speaker action: {message}")

            self._episode_state["public_message"] = message.copy()
            self._episode_state["has_public_message"] = 1
            self._episode_state["step_id"] = 1

            observations = self._build_observations()
            rewards = {agent: 0.0 for agent in self.agents}
            terminations = {agent: False for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            infos = {
                SPEAKER: {"step": "message_emitted", "message": message.copy(), "phase": phase},
                LISTENER: {"step": "message_received", "phase": phase},
            }
            if ADVERSARY in self.agents:
                infos[ADVERSARY] = {"step": "message_received", "phase": phase}
            return observations, rewards, terminations, truncations, infos

        listener_guess = int(actions[LISTENER])
        if not self.action_spaces[LISTENER].contains(listener_guess):
            raise ValueError(f"Invalid listener action: {listener_guess}")

        target = np.asarray(self._episode_state["target"], dtype=np.int64)
        target_index = int(self._episode_state["target_index"])
        message = np.asarray(self._episode_state["public_message"], dtype=np.int64)

        listener_correct = int(listener_guess == target_index)
        if phase == "cooperative_pretraining":
            adversary_reconstruction = None
            adversary_attribute_accuracy = 0.0
            adversary_exact = False
        else:
            adversary_reconstruction = np.asarray(actions[ADVERSARY], dtype=np.int64)
            if not self.action_spaces[ADVERSARY].contains(adversary_reconstruction):
                raise ValueError(f"Invalid adversary action: {adversary_reconstruction}")
            attr_matches = adversary_reconstruction == target
            adversary_attribute_accuracy = float(attr_matches.mean())
            adversary_exact = bool(attr_matches.all())

        cooperative_reward = float(
            listener_correct - self.config.leak_penalty * adversary_attribute_accuracy
        )
        adversary_reward = float(adversary_attribute_accuracy)

        observations = {
            agent: self._zero_observation(self.observation_spaces[agent])
            for agent in self.agents
        }
        rewards = {SPEAKER: cooperative_reward, LISTENER: cooperative_reward}
        if ADVERSARY in self.agents:
            rewards[ADVERSARY] = adversary_reward
        terminations = {agent: True for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {
            SPEAKER: {
                "phase": phase,
                "message": message.copy(),
                "listener_correct": bool(listener_correct),
                "target": target.copy(),
            },
            LISTENER: {
                "phase": phase,
                "message": message.copy(),
                "target_index": target_index,
                "listener_correct": bool(listener_correct),
            },
        }
        if ADVERSARY in self.agents:
            assert adversary_reconstruction is not None
            infos[ADVERSARY] = {
                "phase": phase,
                "message": message.copy(),
                "target": target.copy(),
                "reconstruction": adversary_reconstruction.copy(),
                "adversary_attribute_accuracy": adversary_attribute_accuracy,
                "adversary_exact": adversary_exact,
            }

        self.agents = []
        if self.render_mode == "human":
            self.render()
        self._episode_state = None
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()

        if self._episode_state is None:
            return None

        self._init_render()
        assert self.font is not None

        size = self.config.render_size
        self.canvas.fill((247, 244, 236))

        positions = np.asarray(self._episode_state["candidate_positions"], dtype=np.float32)
        candidates = np.asarray(self._episode_state["candidates"], dtype=np.int64)
        target_index = int(self._episode_state["target_index"])
        message = np.asarray(self._episode_state["public_message"], dtype=np.int64)
        step_id = int(self._episode_state["step_id"])
        phase = str(self._episode_state["phase"])

        for idx, (pos, attrs) in enumerate(zip(positions, candidates)):
            x, y = self._world_to_screen(pos)
            radius = 34 if idx == target_index else 28
            color = self._attributes_to_color(attrs)
            border = (20, 20, 20) if idx == target_index else (80, 80, 80)
            pygame.draw.circle(self.canvas, color, (x, y), radius)
            pygame.draw.circle(self.canvas, border, (x, y), radius, 3)
            label = self.font.render(f"{idx}", True, (0, 0, 0))
            self.canvas.blit(label, (x - 6, y - 10))

        header = self.font.render(f"phase: {phase}  step: {step_id}", True, (20, 20, 20))
        self.canvas.blit(header, (20, 20))
        msg_text = self.font.render(
            f"public message: [{message[0]} {message[1]} {message[2]}]",
            True,
            (20, 20, 20),
        )
        self.canvas.blit(msg_text, (20, 55))
        legend = self.font.render("* larger circle = true target", True, (20, 20, 20))
        self.canvas.blit(legend, (20, 90))

        if self.render_mode == "human":
            assert self.window is not None and self.clock is not None
            self.window.blit(self.canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        frame = pygame.surfarray.array3d(self.canvas)
        return np.transpose(frame, axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()

    def state(self) -> np.ndarray:
        if self._episode_state is None:
            return np.zeros(
                self.config.num_attributes
                + self.config.num_candidates * self.config.num_attributes
                + self.config.num_candidates * 2
                + self.config.message_length
                + 2,
                dtype=np.float32,
            )
        phase_id = self._phase_id(str(self._episode_state["phase"]))
        step_id = int(self._episode_state["step_id"])
        target = np.asarray(self._episode_state["target"], dtype=np.float32)
        candidates = np.asarray(self._episode_state["candidates"], dtype=np.float32).reshape(-1)
        positions = np.asarray(self._episode_state["candidate_positions"], dtype=np.float32).reshape(-1)
        message = np.asarray(self._episode_state["public_message"], dtype=np.float32)
        return np.concatenate(
            [
                np.array([phase_id, step_id], dtype=np.float32),
                target,
                candidates,
                positions,
                message,
            ]
        )

    def _build_observations(self) -> dict[str, dict[str, Any]]:
        assert self._episode_state is not None
        phase_id = self._phase_id(str(self._episode_state["phase"]))
        step_id = int(self._episode_state["step_id"])
        target = np.asarray(self._episode_state["target"], dtype=np.int64)
        candidates = np.asarray(self._episode_state["candidates"], dtype=np.int64)
        positions = np.asarray(self._episode_state["candidate_positions"], dtype=np.float32)
        message = np.asarray(self._episode_state["public_message"], dtype=np.int64)
        has_public_message = int(self._episode_state["has_public_message"])

        all_obs = {
            SPEAKER: {
                "phase_id": phase_id,
                "step_id": step_id,
                "speaker_target": target.copy(),
                "candidate_positions": positions.copy(),
                "public_message": message.copy(),
                "has_public_message": has_public_message,
            },
            LISTENER: {
                "phase_id": phase_id,
                "step_id": step_id,
                "listener_candidates": candidates.copy(),
                "candidate_positions": positions.copy(),
                "public_message": message.copy(),
                "has_public_message": has_public_message,
            },
            ADVERSARY: {
                "phase_id": phase_id,
                "step_id": step_id,
                "public_message": message.copy(),
                "has_public_message": has_public_message,
            },
        }
        return {agent: all_obs[agent] for agent in self.agents}

    def _sample_candidates(self) -> tuple[np.ndarray, int]:
        target = self.rng.integers(
            low=0,
            high=self.config.attribute_cardinality,
            size=(self.config.num_attributes,),
            dtype=np.int64,
        )
        candidates = [target.copy()]
        seen = {tuple(int(x) for x in target)}

        while len(candidates) < self.config.num_candidates:
            distractor = target.copy()
            mutate_indices = self.rng.choice(
                self.config.num_attributes,
                size=self.config.distractor_mutations,
                replace=False,
            )
            for idx in mutate_indices:
                original = int(distractor[idx])
                replacement = int(self.rng.integers(0, self.config.attribute_cardinality - 1))
                if replacement >= original:
                    replacement += 1
                distractor[idx] = replacement
            key = tuple(int(x) for x in distractor)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(distractor)

        self.rng.shuffle(candidates)
        candidate_array = np.stack(candidates)
        matches = np.all(candidate_array == target, axis=1)
        target_index = int(np.flatnonzero(matches)[0])
        return candidate_array, target_index

    def _sample_positions(self) -> np.ndarray:
        positions = []
        while len(positions) < self.config.num_candidates:
            candidate = self.rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
            if all(np.linalg.norm(candidate - existing) > 0.35 for existing in positions):
                positions.append(candidate)
        return np.stack(positions)

    def _init_render(self) -> None:
        if self.font is None:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("dejavusansmono", 24)
        if self.render_mode == "human" and self.window is None:
            self.window = pygame.display.set_mode((self.config.render_size, self.config.render_size))
            self.clock = pygame.time.Clock()

    def _render_ansi(self) -> str:
        if self._episode_state is None:
            return "Episode complete. Reset required."

        target = np.asarray(self._episode_state["target"], dtype=np.int64)
        candidates = np.asarray(self._episode_state["candidates"], dtype=np.int64)
        positions = np.asarray(self._episode_state["candidate_positions"], dtype=np.float32)
        message = np.asarray(self._episode_state["public_message"], dtype=np.int64)
        lines = [
            f"phase={self._episode_state['phase']}",
            f"step_id={self._episode_state['step_id']}",
            f"target_index={self._episode_state['target_index']}",
            f"target={target.tolist()}",
            f"public_message={message.tolist()}",
            "candidates:",
        ]
        for idx, (candidate, pos) in enumerate(zip(candidates, positions)):
            marker = "*" if idx == int(self._episode_state["target_index"]) else " "
            lines.append(f"  {marker} {idx}: attrs={candidate.tolist()} pos={pos.round(3).tolist()}")
        return "\n".join(lines)

    def _world_to_screen(self, pos: np.ndarray) -> tuple[int, int]:
        size = self.config.render_size
        x = int((pos[0] + 1.0) * 0.5 * size)
        y = int((1.0 - (pos[1] + 1.0) * 0.5) * size)
        return x, y

    @staticmethod
    def _attributes_to_color(attrs: np.ndarray) -> tuple[int, int, int]:
        vals = np.asarray(attrs[:3], dtype=np.float32) / 9.0
        color = np.clip(vals * 205 + 30, 0, 255).astype(np.uint8)
        return int(color[0]), int(color[1]), int(color[2])

    @staticmethod
    def _zero_observation(space):
        if isinstance(space, spaces.Dict):
            return {
                key: PublicEavesdropParallelEnv._zero_observation(subspace)
                for key, subspace in space.spaces.items()
            }
        if isinstance(space, spaces.Box):
            return np.zeros(space.shape, dtype=space.dtype)
        if isinstance(space, spaces.Discrete):
            return 0
        if isinstance(space, spaces.MultiDiscrete):
            return np.zeros(space.shape, dtype=space.dtype)
        raise TypeError(f"Unsupported space type: {type(space)}")

    @staticmethod
    def _phase_id(phase: str) -> int:
        if phase == "cooperative_pretraining":
            return 0
        if phase == "public_adversarial":
            return 1
        raise ValueError(f"Unsupported phase: {phase}")
