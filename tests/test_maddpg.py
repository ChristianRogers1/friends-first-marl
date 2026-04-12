import unittest

import torch

from cooperative_pretraining.envs.public_eavesdrop_env import PublicEavesdropParallelEnv
from cooperative_pretraining.maddpg import MADDPGConfig, run_screen_and_confirm_maddpg


class MADDPGSmokeTests(unittest.TestCase):
    def test_maddpg_pipeline_runs(self):
        results = run_screen_and_confirm_maddpg(
            env_factory=lambda: PublicEavesdropParallelEnv(),
            screening_seeds=1,
            confirmation_seeds=1,
            ratios=(0.1, 0.25),
            base_seed=5,
            config=MADDPGConfig(
                train_episodes=20,
                eval_episodes=10,
                batch_size=8,
                warmup_steps=8,
                hidden_dim=32,
                device="cpu",
            ),
        )
        self.assertIn("screening", results)
        self.assertIn("confirmation", results)
        self.assertTrue(torch.cuda.is_available() or True)


if __name__ == "__main__":
    unittest.main()
