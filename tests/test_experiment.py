import unittest

from cooperative_pretraining.envs.public_eavesdrop_env import PublicEavesdropParallelEnv
from cooperative_pretraining.experiment import TrainingConfig, run_screen_and_confirm


class ExperimentPipelineTests(unittest.TestCase):
    def test_screen_and_confirm_pipeline_runs(self):
        results = run_screen_and_confirm(
            env_factory=lambda: PublicEavesdropParallelEnv(),
            screening_seeds=1,
            confirmation_seeds=1,
            ratios=(0.1, 0.25),
            base_seed=3,
            config=TrainingConfig(train_episodes=40, eval_episodes=20),
        )

        self.assertIn("screening", results)
        self.assertIn(results["selected_ratio"], (0.1, 0.25))
        self.assertIn("confirmation", results)
        self.assertIn("comparisons", results["confirmation"])
        self.assertIn("listener_accuracy", results["confirmation"]["comparisons"])


if __name__ == "__main__":
    unittest.main()
