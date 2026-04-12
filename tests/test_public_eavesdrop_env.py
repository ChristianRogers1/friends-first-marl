import unittest

import numpy as np

from cooperative_pretraining.envs.public_eavesdrop_env import (
    ADVERSARY,
    LISTENER,
    SPEAKER,
    PublicEavesdropParallelEnv,
)


class PublicEavesdropEnvTests(unittest.TestCase):
    def test_reset_exposes_distinct_agent_views(self):
        env = PublicEavesdropParallelEnv()
        observations, infos = env.reset(seed=1, options={"phase": "public_adversarial"})

        self.assertIn("speaker_target", observations[SPEAKER])
        self.assertIn("listener_candidates", observations[LISTENER])
        self.assertNotIn("speaker_target", observations[ADVERSARY])
        self.assertEqual(int(observations[SPEAKER]["step_id"]), 0)
        self.assertEqual(infos[SPEAKER]["phase"], "public_adversarial")

    def test_private_phase_blocks_adversary_reward(self):
        env = PublicEavesdropParallelEnv()
        observations, _ = env.reset(seed=2, options={"phase": "cooperative_pretraining"})
        message = observations[SPEAKER]["speaker_target"][:3].copy()

        observations, _, _, _, _ = env.step(
            {
                SPEAKER: message,
                LISTENER: 0,
                ADVERSARY: np.zeros(5, dtype=np.int64),
            }
        )

        candidates = observations[LISTENER]["listener_candidates"]
        target = observations[SPEAKER]["speaker_target"]
        target_index = int(np.flatnonzero(np.all(candidates == target, axis=1))[0])

        _, rewards, terminations, truncations, infos = env.step(
            {
                SPEAKER: np.zeros(3, dtype=np.int64),
                LISTENER: target_index,
                ADVERSARY: target.copy(),
            }
        )

        self.assertTrue(all(terminations.values()))
        self.assertTrue(not any(truncations.values()))
        self.assertNotIn(ADVERSARY, rewards)
        self.assertNotIn(ADVERSARY, infos)

    def test_public_phase_scores_partial_reconstruction(self):
        env = PublicEavesdropParallelEnv()
        observations, _ = env.reset(seed=3, options={"phase": "public_adversarial"})
        message = observations[SPEAKER]["speaker_target"][:3].copy()

        observations, _, _, _, _ = env.step(
            {
                SPEAKER: message,
                LISTENER: 0,
                ADVERSARY: np.zeros(5, dtype=np.int64),
            }
        )

        candidates = observations[LISTENER]["listener_candidates"]
        target = observations[SPEAKER]["speaker_target"]
        target_index = int(np.flatnonzero(np.all(candidates == target, axis=1))[0])
        reconstruction = target.copy()
        reconstruction[-1] = (reconstruction[-1] + 1) % 10

        _, rewards, _, _, infos = env.step(
            {
                SPEAKER: np.zeros(3, dtype=np.int64),
                LISTENER: target_index,
                ADVERSARY: reconstruction,
            }
        )

        self.assertAlmostEqual(rewards[SPEAKER], 0.6)
        self.assertAlmostEqual(rewards[ADVERSARY], 0.8)
        self.assertTrue(infos[LISTENER]["listener_correct"])
        self.assertAlmostEqual(infos[ADVERSARY]["adversary_attribute_accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
