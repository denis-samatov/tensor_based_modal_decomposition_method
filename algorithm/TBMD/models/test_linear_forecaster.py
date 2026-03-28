import itertools
import unittest
import numpy as np
import torch
import os
import tempfile

from algorithm.TBMD.models.LinearForecaster import LinearForecaster


class TestLinearForecaster(unittest.TestCase):

    def setUp(self):
        """Set up a simple testing environment."""
        self.in_dim = 3
        self.t_steps = 20
        # Create a known transformation matrix
        self.true_m = np.array([
            [0.1, 0.2, 0.7],
            [0.4, 0.5, 0.1],
            [0.8, 0.1, 0.1]
        ])
        # Generate synthetic data
        self.x_history = np.array(list(itertools.accumulate(
            itertools.repeat(self.true_m, self.t_steps - 1),
            np.matmul,
            initial=np.array([1., 2., 3.])
        )))

    def test_training_metrics_numpy(self):
        """Test if the training metrics are calculated correctly with NumPy."""
        forecaster = LinearForecaster(use_torch=False)
        metrics = forecaster.train(self.x_history, verbose=False)

        # Re-calculate the expected predictions
        x_input = self.x_history[:-1]
        expected_predictions = x_input @ forecaster.M

        # Manually calculate MSE
        expected_mse = np.mean((self.x_history[1:] - expected_predictions)**2)

        # The reported MSE should be close to our manually calculated one
        self.assertAlmostEqual(metrics['mse'], expected_mse, places=5,
                               msg="NumPy MSE metric is incorrect.")

    def test_training_metrics_torch(self):
        """Test if the training metrics are calculated correctly with PyTorch."""
        forecaster = LinearForecaster(use_torch=True)
        metrics = forecaster.train(self.x_history, verbose=False)

        # Re-calculate the expected predictions
        x_input = torch.tensor(self.x_history[:-1], dtype=torch.float32, device=forecaster.device)
        expected_predictions = x_input @ forecaster.M

        # Manually calculate MSE
        expected_mse = torch.mean((torch.tensor(self.x_history[1:], dtype=torch.float32, device=forecaster.device) - expected_predictions)**2).item()

        # The reported MSE should be close to our manually calculated one
        self.assertAlmostEqual(metrics['mse'], expected_mse, places=5,
                               msg="PyTorch MSE metric is incorrect.")

    def test_evaluation_metrics_numpy(self):
        """Test if the evaluation metrics are calculated correctly with NumPy."""
        forecaster = LinearForecaster(use_torch=False)
        forecaster.train(self.x_history, verbose=False)
        metrics = forecaster.evaluate(self.x_history)

        # Re-calculate the expected predictions
        x_input = self.x_history[:-1]
        expected_predictions = x_input @ forecaster.M

        # Manually calculate MSE
        expected_mse = np.mean((self.x_history[1:] - expected_predictions)**2)

        self.assertAlmostEqual(metrics['mse'], expected_mse, places=5,
                               msg="NumPy evaluation MSE is incorrect.")

    def test_evaluation_metrics_torch(self):
        """Test if the evaluation metrics are calculated correctly with PyTorch."""
        forecaster = LinearForecaster(use_torch=True)
        forecaster.train(self.x_history, verbose=False)
        metrics = forecaster.evaluate(self.x_history)

        # Re-calculate the expected predictions
        x_input = torch.tensor(self.x_history[:-1], dtype=torch.float32, device=forecaster.device)
        expected_predictions = x_input @ forecaster.M

        # Manually calculate MSE
        expected_mse = torch.mean((torch.tensor(self.x_history[1:], dtype=torch.float32, device=forecaster.device) - expected_predictions)**2).item()

        self.assertAlmostEqual(metrics['mse'], expected_mse, places=5,
                               msg="PyTorch evaluation MSE is incorrect.")

    def test_predict_sequence_torch(self):
        """Test if predict_sequence works correctly with PyTorch."""
        forecaster = LinearForecaster(use_torch=True)
        forecaster.train(self.x_history, verbose=False)

        x_start = self.x_history[0]
        n_steps = 5

        # Optimized sequence prediction
        seq = forecaster.predict_sequence(x_start, n_steps)

        # Manual verification
        manual_seq = []
        x_curr = torch.tensor(x_start, dtype=torch.float32, device=forecaster.device)
        for _ in range(n_steps):
            x_curr = x_curr @ forecaster.M
            manual_seq.append(x_curr.detach().cpu().numpy())
        manual_seq = np.array(manual_seq)

        np.testing.assert_allclose(seq, manual_seq, atol=1e-5, err_msg="predict_sequence output mismatch")

    def test_save_load_model_numpy(self):
        """Test if the model is saved and loaded correctly with NumPy."""
        forecaster = LinearForecaster(use_torch=False)
        forecaster.train(self.x_history, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            forecaster.save_model(tmp_path)

            loaded_forecaster = LinearForecaster(use_torch=False)
            loaded_forecaster.load_model(tmp_path)

            self.assertTrue(loaded_forecaster.trained)
            self.assertFalse(loaded_forecaster.use_torch)
            self.assertEqual(forecaster.metrics['mse'], loaded_forecaster.metrics['mse'])
            np.testing.assert_allclose(forecaster.M, loaded_forecaster.M)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_load_model_torch(self):
        """Test if the model is saved and loaded correctly with PyTorch."""
        forecaster = LinearForecaster(use_torch=True)
        forecaster.train(self.x_history, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            forecaster.save_model(tmp_path)

            loaded_forecaster = LinearForecaster(use_torch=True)
            loaded_forecaster.load_model(tmp_path)

            self.assertTrue(loaded_forecaster.trained)
            self.assertTrue(loaded_forecaster.use_torch)
            self.assertEqual(forecaster.metrics['mse'], loaded_forecaster.metrics['mse'])
            np.testing.assert_allclose(
                forecaster.M.detach().cpu().numpy(),
                loaded_forecaster.M.detach().cpu().numpy()
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_model_untrained(self):
        """Test if saving an untrained model raises a RuntimeError."""
        forecaster = LinearForecaster()
        with self.assertRaises(RuntimeError):
            forecaster.save_model("any_path.npz")

    def test_load_model_nonexistent(self):
        """Test if loading a non-existent file raises a FileNotFoundError."""
        forecaster = LinearForecaster()
        with self.assertRaises(FileNotFoundError):
            forecaster.load_model("non_existent_file.npz")

if __name__ == '__main__':
    unittest.main()
