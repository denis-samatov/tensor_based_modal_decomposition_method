import unittest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import patch

from algorithm.TBMD.models.LSTMForecaster import LSTMModel, LSTMForecaster

class TestLSTMModel(unittest.TestCase):
    def test_init_and_forward(self):
        # Test initialization and forward pass
        in_dim, hidden_dim, out_dim = 3, 10, 2
        model = LSTMModel(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=2)

        # Batch of 4, sequence length 5
        batch_size, seq_len = 4, 5
        x = torch.randn(batch_size, seq_len, in_dim)

        out = model(x)
        self.assertEqual(out.shape, (batch_size, out_dim))

class TestLSTMForecaster(unittest.TestCase):
    def setUp(self):
        self.in_dim = 3
        self.out_dim = 3
        self.seq_length = 5
        self.t_steps = 50
        # Generate some synthetic data (T, W) -> (t_steps, in_dim)
        self.x_history = np.random.randn(self.t_steps, self.in_dim)

        self.forecaster = LSTMForecaster(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            seq_length=self.seq_length,
            hidden_dim=8,
            num_layers=1,
            device='cpu' # Force CPU for deterministic test environments without assuming GPU
        )

    def test_init(self):
        self.assertEqual(self.forecaster.in_dim, self.in_dim)
        self.assertEqual(self.forecaster.out_dim, self.out_dim)
        self.assertEqual(self.forecaster.seq_length, self.seq_length)
        self.assertTrue(isinstance(self.forecaster.model, LSTMModel))

    def test_make_lagged_dataset(self):
        X, Y = self.forecaster.make_lagged_dataset(self.x_history, self.seq_length)

        expected_samples = self.t_steps - self.seq_length
        self.assertEqual(X.shape, (expected_samples, self.seq_length, self.in_dim))
        self.assertEqual(Y.shape, (expected_samples, self.out_dim))

    def test_prepare_data(self):
        # With validation split
        train_loader, val_loader = self.forecaster.prepare_data(self.x_history, val_split=0.2, batch_size=8)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)

        # Without validation split
        train_loader_only, val_loader_none = self.forecaster.prepare_data(self.x_history, val_split=0.0, batch_size=8)
        self.assertIsNotNone(train_loader_only)
        self.assertIsNone(val_loader_none)

    def test_train_epoch_and_validate(self):
        train_loader, val_loader = self.forecaster.prepare_data(self.x_history, val_split=0.2, batch_size=8)

        train_loss = self.forecaster.train_epoch(train_loader)
        self.assertIsInstance(train_loss, float)
        self.assertTrue(train_loss >= 0)

        val_loss = self.forecaster.validate(val_loader)
        self.assertIsInstance(val_loss, float)
        self.assertTrue(val_loss >= 0)

    def test_train(self):
        history = self.forecaster.train(
            self.x_history,
            num_epochs=2,
            batch_size=8,
            val_split=0.2,
            verbose=False,
            save_best=False
        )
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)

    def test_predict_next(self):
        # Input shape (seq_length, W)
        x_window = self.x_history[:self.seq_length]
        pred = self.forecaster.predict_next(x_window)
        self.assertEqual(pred.shape, (self.out_dim,))

        # Test incorrect sequence length
        with self.assertRaises(ValueError):
            self.forecaster.predict_next(self.x_history[:self.seq_length - 1])

    def test_predict_sequence(self):
        x_start = self.x_history[:self.seq_length]
        n_steps = 4
        seq_pred = self.forecaster.predict_sequence(x_start, n_steps=n_steps)
        self.assertEqual(seq_pred.shape, (n_steps, self.out_dim))

        # Test incorrect sequence length
        with self.assertRaises(ValueError):
            self.forecaster.predict_sequence(self.x_history[:self.seq_length - 1], n_steps=n_steps)

    def test_save_and_load_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_lstm_model.pt")

            # Change a parameter to verify it gets loaded
            self.forecaster.best_val_loss = 0.42

            self.forecaster.save_model(model_path)
            self.assertTrue(os.path.exists(model_path))

            # Create a new instance and load
            new_forecaster = LSTMForecaster(in_dim=self.in_dim, out_dim=self.out_dim, seq_length=self.seq_length, hidden_dim=8, num_layers=1, device='cpu')
            new_forecaster.load_model(model_path)

            self.assertAlmostEqual(new_forecaster.best_val_loss, 0.42)

            # Check model weights are somewhat identical
            # by comparing prediction on identical input
            x_window = self.x_history[:self.seq_length]
            pred_orig = self.forecaster.predict_next(x_window)
            pred_loaded = new_forecaster.predict_next(x_window)
            np.testing.assert_array_almost_equal(pred_orig, pred_loaded)

    def test_evaluate(self):
        metrics = self.forecaster.evaluate(self.x_history)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertIn('rel_frob_err', metrics)

    @patch('matplotlib.pyplot.show')
    def test_plot_training_history(self, mock_show):
        self.forecaster.training_history = {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4]
        }
        self.forecaster.plot_training_history()
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
