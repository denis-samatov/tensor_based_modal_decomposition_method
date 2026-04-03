import unittest
from TBMD.utils.Analytics import ExperimentConfig, ExperimentRunner

class TestExperimentConfig(unittest.TestCase):
    def test_default_initialization(self):
        config = ExperimentConfig()
        self.assertEqual(config.solver_method, "triangular")
        self.assertEqual(config.device, 'cpu')
        self.assertEqual(config.confidence_level, 0.95)

    def test_custom_initialization(self):
        config = ExperimentConfig(solver_method="svd", device="cuda", confidence_level=0.99)
        self.assertEqual(config.solver_method, "svd")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.confidence_level, 0.99)

    def test_post_init_validation_correction(self):
        # 0.8 is invalid, should fallback to 0.95
        config = ExperimentConfig(confidence_level=0.80)
        self.assertEqual(config.confidence_level, 0.95)

class TestExperimentRunner(unittest.TestCase):
    def test_initialization_with_default_config(self):
        runner = ExperimentRunner()
        self.assertIsNotNone(runner.config)
        self.assertEqual(runner.config.confidence_level, 0.95)
        # Verify setup interval is called by checking z_scores structure
        self.assertIn(0.95, runner.z_scores)
        self.assertEqual(runner.z_scores[0.95], 1.96)

    def test_initialization_with_custom_config(self):
        config = ExperimentConfig(confidence_level=0.99)
        runner = ExperimentRunner(config)
        self.assertEqual(runner.config.confidence_level, 0.99)
        self.assertEqual(runner.z_scores[0.99], 2.576)

if __name__ == '__main__':
    unittest.main()
