import unittest
from unittest.mock import patch
import numpy as np
import torch
import os
import tempfile
import sys

# Ensure algorithm directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from TBMD.core.forecasting.MLPForecaster import MLPModel, MLPForecaster

class TestMLPForecaster(unittest.TestCase):

    def setUp(self):
        """Set up a simple testing environment."""
        self.in_dim = 5
        self.out_dim = 5
        self.t_steps = 30

        # Generate synthetic data
        # We will create a simple pattern so the model can learn it somewhat
        self.x_history = np.zeros((self.t_steps, self.in_dim))
        self.x_history[0] = np.random.rand(self.in_dim)
        for i in range(1, self.t_steps):
            self.x_history[i] = self.x_history[i-1] * 0.9 + 0.1

    def test_mlpmodel_initialization(self):
        """Test the underlying PyTorch MLPModel initialization."""
        model = MLPModel(in_dim=self.in_dim, out_dim=self.out_dim, hidden_dim=64, num_layers=3)
        self.assertIsInstance(model, torch.nn.Module)

        # Test forward pass shape
        x = torch.randn(10, self.in_dim)
        out = model(x)
        self.assertEqual(out.shape, (10, self.out_dim))

    def test_forecaster_initialization(self):
        """Test MLPForecaster initialization and device selection."""
        # Auto device
        forecaster_auto = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim)
        self.assertIsNotNone(forecaster_auto.device)
        self.assertEqual(forecaster_auto.in_dim, self.in_dim)
        self.assertEqual(forecaster_auto.out_dim, self.out_dim)

        # Explicit CPU device
        forecaster_cpu = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
        self.assertEqual(forecaster_cpu.device.type, 'cpu')

    def test_prepare_data(self):
        """Test data preparation output shapes and types."""
        forecaster = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
        train_loader, val_loader = forecaster.prepare_data(
            self.x_history, val_split=0.2, batch_size=4, shuffle=False
        )

        # Check shapes in train_loader
        x_batch, y_batch = next(iter(train_loader))
        self.assertEqual(x_batch.shape, (4, self.in_dim))
        self.assertEqual(y_batch.shape, (4, self.out_dim))
        self.assertTrue(isinstance(x_batch, torch.Tensor))

        # With val_split=0.0
        train_loader_only, val_loader_none = forecaster.prepare_data(
            self.x_history, val_split=0.0, batch_size=4
        )
        self.assertIsNone(val_loader_none)

    def test_train(self):
        """Test the training loop metrics."""
        forecaster = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
        history = forecaster.train(
            self.x_history, num_epochs=5, batch_size=4, val_split=0.2, verbose=False
        )

        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 5)
        self.assertEqual(len(history['val_loss']), 5)

    def test_predict_next(self):
        """Test predicting the next time step."""
        forecaster = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
        forecaster.train(self.x_history, num_epochs=2, verbose=False)

        x_curr = self.x_history[-1]
        x_next = forecaster.predict_next(x_curr)

        self.assertEqual(x_next.shape, (self.out_dim,))
        self.assertTrue(isinstance(x_next, np.ndarray))

    def test_predict_sequence(self):
        """Test sequence prediction logic."""
        forecaster = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
        forecaster.train(self.x_history, num_epochs=2, verbose=False)

        n_steps = 7
        x_start = self.x_history[0]
        seq = forecaster.predict_sequence(x_start, n_steps)

        self.assertEqual(seq.shape, (n_steps, self.out_dim))
        self.assertTrue(isinstance(seq, np.ndarray))

    def test_evaluate(self):
        """Test evaluation metrics dictionary structure and types."""
        forecaster = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
        forecaster.train(self.x_history, num_epochs=2, verbose=False)

        metrics = forecaster.evaluate(self.x_history)

        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('rel_frob_err', metrics)

        self.assertIsInstance(metrics['mse'], float)
        self.assertIsInstance(metrics['rmse'], float)
        self.assertIsInstance(metrics['r2'], float)
        self.assertIsInstance(metrics['rel_frob_err'], float)

    def test_save_load_model(self):
        """Test saving and loading the model weights and state."""
        forecaster = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
        forecaster.train(self.x_history, num_epochs=3, verbose=False)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            forecaster.save_model(tmp_path)

            loaded_forecaster = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
            loaded_forecaster.load_model(tmp_path)

            # Predict with both to ensure outputs match
            x_test = self.x_history[0]
            pred_orig = forecaster.predict_next(x_test)
            pred_loaded = loaded_forecaster.predict_next(x_test)

            np.testing.assert_allclose(pred_orig, pred_loaded, atol=1e-5)
            self.assertEqual(forecaster.best_val_loss, loaded_forecaster.best_val_loss)
            self.assertEqual(len(forecaster.training_history['train_loss']),
                             len(loaded_forecaster.training_history['train_loss']))

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @patch('matplotlib.pyplot.show')
    def test_plot_training_history(self, mock_show):
        """Test plotting training history does not crash."""
        forecaster = MLPForecaster(in_dim=self.in_dim, out_dim=self.out_dim, device='cpu')
        forecaster.train(self.x_history, num_epochs=2, verbose=False)

        try:
            forecaster.plot_training_history()
        except Exception as e:
            self.fail(f"plot_training_history raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()
