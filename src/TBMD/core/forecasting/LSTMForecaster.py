import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Union, List, Any

# Import config
try:
    from TBMD.config import LSTMForecasterConfig
except ImportError:
    # Fallback if running standalone
    LSTMForecasterConfig = None


class LSTMModel(nn.Module):
    """An LSTM model for time series forecasting.

    This model consists of an LSTM layer followed by a fully connected layer.

    Args:
        in_dim (int): The input dimension.
        hidden_dim (int): The hidden state dimension.
        out_dim (int): The output dimension.
        num_layers (int, optional): The number of LSTM layers. Defaults to 1.
        dropout_rate (float, optional): The dropout rate for regularization.
            Applied only if `num_layers` > 1. Defaults to 0.0.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 1,
                 dropout_rate: float = 0.0):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x_seq (torch.Tensor): A tensor of shape (batch, seq_len, in_dim).

        Returns:
            torch.Tensor: The model's predictions, with shape (batch, out_dim).
        """
        # LSTM layer
        # out shape: (batch, seq_len, hidden_dim)
        out, (h_n, c_n) = self.lstm(x_seq)

        # Take the last time step output
        # shape: (batch, hidden_dim)
        out_last = out[:, -1, :]

        # Fully connected layer to get predictions
        # shape: (batch, out_dim)
        out_fc = self.fc(out_last)

        return out_fc


class LSTMForecaster:
    """A pipeline for training and using LSTM forecasting models.

    This class encapsulates the process of training, validating, and using an
    LSTM model for time series forecasting.

    Args:
        in_dim (int): The input dimension (number of features).
        out_dim (int): The output dimension (number of features to predict).
        seq_length (int, optional): The length of the input sequence. Defaults
            to 5.
        hidden_dim (int, optional): The hidden state dimension. Defaults to 64.
        num_layers (int, optional): The number of LSTM layers. Defaults to 1.
        dropout_rate (float, optional): The dropout rate for regularization.
            Defaults to 0.0.
        lr (float, optional): The learning rate. Defaults to 1e-3.
        weight_decay (float, optional): The L2 regularization parameter.
            Defaults to 1e-5.
        device (str, optional): The device to run computations on ('cpu',
            'cuda', or 'mps'). If None, the device is automatically selected.
    """

    def __init__(self,
                 in_dim: Optional[int] = None,
                 out_dim: Optional[int] = None,
                 seq_length: int = 5,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout_rate: float = 0.0,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: str = None,
                 config: Optional[LSTMForecasterConfig] = None):
        """Initializes the LSTMForecaster.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            seq_length (int, optional): The length of the input sequence.
                Defaults to 5.
            hidden_dim (int, optional): The hidden state dimension. Defaults to
                64.
            num_layers (int, optional): The number of LSTM layers. Defaults to
                1.
            dropout_rate (float, optional): The dropout rate for
                regularization. Defaults to 0.0.
            lr (float, optional): The learning rate. Defaults to 1e-3.
            weight_decay (float, optional): The L2 regularization parameter.
                Defaults to 1e-5.
            device (str, optional): The device to run computations on. If
                `None`, the device is automatically selected.
        """
        # Handle config vs individual parameters
        if config is not None:
            # New API: use config
            self.config = config
            # Store dimensions
            if in_dim is not None:
                self.config.in_dim = in_dim
            if out_dim is not None:
                self.config.out_dim = out_dim
        else:
            # Old API: create config from parameters
            if LSTMForecasterConfig is None:
                raise ImportError("LSTMForecasterConfig not available. Please update imports.")

            self.config = LSTMForecasterConfig(
                in_dim=in_dim,
                out_dim=out_dim,
                seq_length=seq_length if seq_length is not None else 5,
                hidden_size=hidden_dim if hidden_dim is not None else 64,
                num_layers=num_layers if num_layers is not None else 1,
                dropout=dropout_rate if dropout_rate is not None else 0.0,
                learning_rate=lr if lr is not None else 1e-3,
                weight_decay=weight_decay if weight_decay is not None else 1e-5,
                device=device
            )

        # Set device from config
        if self.config.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else
                                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(self.config.device)

        if self.config.verbose:
            print(f"Using device: {self.device}")

        # Store configuration
        self.seq_length = self.config.seq_length
        self.in_dim = self.config.in_dim if self.config.in_dim else in_dim
        self.out_dim = self.config.out_dim if self.config.out_dim else out_dim

        # Initialize model
        self.model = LSTMModel(
            in_dim=self.in_dim,
            hidden_dim=self.config.hidden_size,
            out_dim=self.out_dim,
            num_layers=self.config.num_layers,
            dropout_rate=self.config.dropout
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Initialize loss function
        self.loss_fn = nn.MSELoss()

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')

    def make_lagged_dataset(self,
                           x_history: np.ndarray,
                           seq_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a lagged dataset from a time series.

        Args:
            x_history (np.ndarray): The time series data, with shape (T, W),
                where T is the number of time steps and W is the number of
                features.
            seq_length (Optional[int]): The length of the input sequences. If
                None, `self.seq_length` is used.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the input
            sequences `X_seq` (shape (N, seq_length, W)) and the target
            outputs `Y_seq` (shape (N, W)), where N = T - seq_length.
        """
        if seq_length is None:
            seq_length = self.seq_length

        T = x_history.shape[0]
        W = x_history.shape[1]

        # Check if we have enough data
        if T <= seq_length:
            raise ValueError(f"Not enough time steps ({T}) to create sequences of length {seq_length}")

        # Create sequences and targets efficiently using sliding window views
        from numpy.lib.stride_tricks import sliding_window_view

        # Input sequences: all but the last time step for the windows
        # sliding_window_view gives shape (N, W, seq_length), so we transpose to (N, seq_length, W)
        X_seq = sliding_window_view(x_history[:-1], window_shape=seq_length, axis=0).transpose(0, 2, 1)

        # Target x(t): from t=seq_length to T-1
        Y_seq = x_history[seq_length:]

        return X_seq, Y_seq

    def prepare_data(self,
                    x_history: np.ndarray,
                    val_split: float = 0.2,
                    batch_size: int = 32,
                    shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepares the training and validation data loaders.

        Args:
            x_history (np.ndarray): The time series data, with shape (T, W).
            val_split (float, optional): The validation split ratio. Defaults
                to 0.2.
            batch_size (int, optional): The batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the training data.
                Defaults to True.

        Returns:
            Tuple[DataLoader, Optional[DataLoader]]: A tuple containing the
            training and validation data loaders. The validation loader is
            `None` if `val_split` is 0.
        """
        # Create lagged sequences
        X_seq, Y_seq = self.make_lagged_dataset(x_history, self.seq_length)

        if val_split > 0:
            num_samples = len(X_seq)
            split_idx = int((1 - val_split) * num_samples)
            # Ensure we keep at least one sample in each split; otherwise fall back to train-only
            if split_idx <= 0 or split_idx >= num_samples:
                val_split = 0

        if val_split > 0:
            # Split into training and validation sets (time ordered)
            split_idx = int((1 - val_split) * len(X_seq))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = Y_seq[:split_idx], Y_seq[split_idx:]

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

            # Create datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            return train_loader, val_loader
        else:
            # Use all data for training
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            y_tensor = torch.tensor(Y_seq, dtype=torch.float32)

            # Create dataset and loader
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )

            return train_loader, None

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Trains the model for one epoch.

        Args:
            train_loader (DataLoader): The training data loader.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            # Move batch to device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """Validates the model.

        Args:
            val_loader (DataLoader): The validation data loader.

        Returns:
            float: The average validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move batch to device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)

                total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss

    def train(self,
             x_history: np.ndarray,
             num_epochs: Optional[int] = None,
             batch_size: Optional[int] = None,
             val_split: Optional[float] = None,
             early_stopping_patience: Optional[int] = None,
             verbose: Optional[bool] = None,
             save_best: bool = True,
             model_path: str = None) -> Dict[str, List[float]]:
        """Trains the model.

        Args:
            x_history (np.ndarray): The time series data, with shape (T, W).
            num_epochs (int, optional): The number of training epochs. Defaults
                to 300.
            batch_size (int, optional): The batch size. Defaults to 32.
            val_split (float, optional): The validation split ratio. Defaults
                to 0.2.
            early_stopping_patience (int, optional): The patience for early
                stopping. Defaults to 20.
            verbose (bool, optional): Whether to print progress. Defaults to
                True.
            save_best (bool, optional): Whether to save the best model.
                Defaults to True.
            model_path (str, optional): The path to save the best model.
                Defaults to None.

        Returns:
            Dict[str, List[float]]: The training history.
        """
        # Use config defaults if not provided
        num_epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        batch_size = batch_size if batch_size is not None else self.config.batch_size
        val_split = val_split if val_split is not None else self.config.val_split
        early_stopping_patience = early_stopping_patience if early_stopping_patience is not None else self.config.early_stopping_patience
        verbose = verbose if verbose is not None else self.config.verbose

        # Prepare data
        train_loader, val_loader = self.prepare_data(
            x_history,
            val_split=val_split,
            batch_size=batch_size,
            shuffle=self.config.shuffle
        )

        # Initialize early stopping counter
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)

            # Validate if validation data is available
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.training_history['val_loss'].append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if save_best and model_path is not None:
                        self.save_model(model_path)
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.6f} - Val loss: {val_loss:.6f}")

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            else:
                if verbose and (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.6f}")

        if verbose:
            print("Training complete!")

        return self.training_history

    def predict_next(self, x_window: np.ndarray) -> np.ndarray:
        """Predicts the next state from a given window.

        Args:
            x_window (np.ndarray): The input window, with shape (seq_length, W)
                or (1, seq_length, W).

        Returns:
            np.ndarray: The predicted next state, with shape (W,).
        """
        self.model.eval()

        # Check input shape and adjust if needed
        if len(x_window.shape) == 2:  # (seq_length, W)
            # Add batch dimension
            x_window = x_window[np.newaxis, :, :]  # (1, seq_length, W)

        # Check sequence length
        if x_window.shape[1] != self.seq_length:
            raise ValueError(f"Input window has sequence length {x_window.shape[1]}, expected {self.seq_length}")

        # Convert to tensor and move to device
        x_tensor = torch.tensor(x_window, dtype=torch.float32).to(self.device)

        # Make prediction
        with torch.no_grad():
            x_next = self.model(x_tensor)

        # Convert back to numpy
        y_pred = x_next.detach().cpu().numpy()[0]  # Remove batch dimension

        # If delta mode, prediction is Δc; true c_{t+1} = c_t + Δc
        if getattr(self.config, 'delta_forecast', False):
            y_pred = x_window[0, -1, :] + y_pred

        return y_pred

    def predict_sequence(self, x_start_window: np.ndarray, n_steps: int) -> np.ndarray:
        """Predicts a sequence of future states.

        Args:
            x_start_window (np.ndarray): The starting window, with shape
                (seq_length, W).
            n_steps (int): The number of steps to predict.

        Returns:
            np.ndarray: The predicted sequence, with shape (n_steps, W).
        """
        self.model.eval()

        # Check input shape
        if x_start_window.shape[0] != self.seq_length:
            raise ValueError(f"Input window has sequence length {x_start_window.shape[0]}, expected {self.seq_length}")

        # Initialize input window on device
        # (seq_length, W) -> (1, seq_length, W)
        input_window_tensor = torch.tensor(x_start_window, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Pre-allocate sequence tensor on device
        sequence_tensor = torch.zeros((n_steps, self.out_dim), device=self.device)

        # Generate predictions iteratively
        with torch.no_grad():
            for i in range(n_steps):
                # Predict next step
                # input_window_tensor shape: (1, seq_length, W)
                next_step_tensor = self.model(input_window_tensor) # (1, out_dim)

                # Store prediction
                sequence_tensor[i] = next_step_tensor.squeeze(0)

                # Update window by removing oldest step and adding prediction
                # input_window_tensor[:, 1:, :] is (1, seq_length-1, W)
                # next_step_tensor.unsqueeze(1) is (1, 1, out_dim)
                input_window_tensor = torch.cat([
                    input_window_tensor[:, 1:, :],
                    next_step_tensor.unsqueeze(1)
                ], dim=1)

        return sequence_tensor.cpu().numpy()





    def save_model(self, path: str) -> None:
        """Saves the model.

        Args:
            path (str): The path to save the model to.
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Save model and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
            'seq_length': self.seq_length,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }, path)

    def load_model(self, path: str) -> None:
        """Loads the model.

        Args:
            path (str): The path to load the model from.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load metadata
        self.in_dim = checkpoint.get('in_dim', self.in_dim)
        self.out_dim = checkpoint.get('out_dim', self.out_dim)
        self.seq_length = checkpoint.get('seq_length', self.seq_length)
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)

    def plot_training_history(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plots the training history.

        Args:
            figsize (Tuple[int, int], optional): The figure size. Defaults to
                (10, 6).
        """
        plt.figure(figsize=figsize)
        epochs = range(1, len(self.training_history['train_loss']) + 1)

        plt.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')

        if self.training_history['val_loss']:
            plt.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')

        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def evaluate(self, x_history: np.ndarray) -> Dict[str, float]:
        """Evaluates the model on historical data.

        Args:
            x_history (np.ndarray): The time series data, with shape (T, W).

        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        self.model.eval()

        # Create lagged dataset
        X_seq, Y_seq = self.make_lagged_dataset(x_history, self.seq_length)

        # Convert to tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(Y_seq, dtype=torch.float32).to(self.device)

        # Make predictions
        with torch.no_grad():
            y_pred = self.model(X_tensor)

        # Calculate metrics
        mse_loss = self.loss_fn(y_pred, y_tensor).item()
        rmse = torch.sqrt(torch.mean((y_pred - y_tensor) ** 2)).item()

        # Calculate R-squared
        y_mean = torch.mean(y_tensor)
        ss_tot = torch.sum((y_tensor - y_mean) ** 2)
        ss_res = torch.sum((y_tensor - y_pred) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Relative Frobenius error
        rel_frob_err = torch.norm(y_pred - y_tensor, p='fro') / torch.norm(y_tensor, p='fro')

        return {
            'mse': mse_loss,
            'rmse': rmse,
            'r2': r2.item(),
            'rel_frob_err': rel_frob_err.item()
        }
