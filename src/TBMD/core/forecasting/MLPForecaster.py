import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import config
try:
    from TBMD.config import MLPForecasterConfig
except ImportError:
    # Fallback if running standalone
    MLPForecasterConfig = None


class MLPModel(nn.Module):
    """A multi-layer perceptron model for time series forecasting.

    Args:
        in_dim (int): The input dimension.
        out_dim (int): The output dimension.
        hidden_dim (int, optional): The hidden layer dimension. Defaults to 256.
        dropout_rate (float, optional): The dropout rate for regularization.
            Defaults to 0.3.
        num_layers (int, optional): The number of hidden layers. Defaults to 2.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        num_layers: int = 2,
    ):
        super().__init__()

        # Input layer, Hidden layers, Output layer
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *[
                layer
                for _ in range(num_layers - 1)
                for layer in (
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                )
            ],
            nn.Linear(hidden_dim, out_dim),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.net(x)


class MLPForecaster:
    """A pipeline for training and using MLP forecasting models.

    Args:
        in_dim (int): The input dimension.
        out_dim (int): The output dimension.
        hidden_dim (int, optional): The hidden layer dimension. Defaults to 256.
        dropout_rate (float, optional): The dropout rate for regularization.
            Defaults to 0.3.
        num_layers (int, optional): The number of hidden layers. Defaults to 2.
        lr (float, optional): The learning rate. Defaults to 1e-3.
        weight_decay (float, optional): The L2 regularization parameter.
            Defaults to 1e-5.
        device (str, optional): The device to run computations on ('cpu',
            'cuda', or 'mps'). If `None`, the device is automatically selected.
    """

    def __init__(
        self,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        num_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = None,
        config: Optional[MLPForecasterConfig] = None,
    ):
        """Initializes the MLPForecaster.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            hidden_dim (int, optional): The hidden layer dimension. Defaults
                to 256.
            dropout_rate (float, optional): The dropout rate for
                regularization. Defaults to 0.3.
            num_layers (int, optional): The number of hidden layers. Defaults
                to 2.
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
            if in_dim is not None:
                self.config.in_dim = in_dim
            if out_dim is not None:
                self.config.out_dim = out_dim
        else:
            # Old API: create config from parameters
            if MLPForecasterConfig is None:
                raise ImportError("MLPForecasterConfig not available. Please update imports.")

            self.config = MLPForecasterConfig(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_size=hidden_dim if hidden_dim is not None else 256,
                dropout=dropout_rate if dropout_rate is not None else 0.3,
                num_layers=num_layers if num_layers is not None else 2,
                learning_rate=lr if lr is not None else 1e-3,
                weight_decay=weight_decay if weight_decay is not None else 1e-5,
                device=device,
            )

        # Set device from config
        if self.config.device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        else:
            self.device = torch.device(self.config.device)

        if self.config.verbose:
            print(f"Using device: {self.device}")

        # Store dimensions
        self.in_dim = self.config.in_dim if self.config.in_dim else in_dim
        self.out_dim = self.config.out_dim if self.config.out_dim else out_dim

        # Initialize model and hyperparameters
        self.model = MLPModel(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_dim=self.config.hidden_size,
            dropout_rate=self.config.dropout,
            num_layers=self.config.num_layers,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.loss_fn = nn.MSELoss()
        self.training_history = {"train_loss": [], "val_loss": []}
        self.best_val_loss = float("inf")

    def prepare_data(
        self,
        x_history: np.ndarray,
        val_split: float = 0.2,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepares the training and validation data loaders.

        Args:
            x_history (np.ndarray): The historical data, with shape (T, W).
            val_split (float, optional): The validation split ratio. Defaults
                to 0.2.
            batch_size (int, optional): The batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to
                True.

        Returns:
            Tuple[DataLoader, Optional[DataLoader]]: A tuple containing the
            training and validation data loaders. The validation loader is
            `None` if `val_split` is 0.
        """
        # Create input-output pairs
        X_input = x_history[:-1, :]  # (T-1, W)

        if getattr(self.config, "delta_forecast", False):
            X_target = x_history[1:, :] - X_input
        else:
            X_target = x_history[1:, :]  # (T-1, W)

        if val_split > 0:
            num_samples = len(X_input)
            split_idx = int((1 - val_split) * num_samples)
            if split_idx <= 0 or split_idx >= num_samples:
                val_split = 0

        if val_split > 0:
            # Split into training and validation sets
            split_idx = int((1 - val_split) * len(X_input))
            X_train, X_val = X_input[:split_idx], X_input[split_idx:]
            y_train, y_val = X_target[:split_idx], X_target[split_idx:]

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

            # Create datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader
        else:
            # Use all data for training
            X_tensor = torch.tensor(X_input, dtype=torch.float32)
            y_tensor = torch.tensor(X_target, dtype=torch.float32)

            # Create dataset and loader
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

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

    def train(
        self,
        x_history: np.ndarray,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        val_split: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        verbose: Optional[bool] = None,
        save_best: bool = True,
        model_path: str = None,
    ) -> Dict[str, List[float]]:
        """Trains the model.

        Args:
            x_history (np.ndarray): The historical data, with shape (T, W).
            num_epochs (int, optional): The number of training epochs. Defaults
                to 500.
            batch_size (int, optional): The batch size. Defaults to 32.
            val_split (float, optional): The validation split ratio. Defaults
                to 0.2.
            early_stopping_patience (int, optional): The patience for early
                stopping. Defaults to 20.
            verbose (bool, optional): Whether to print progress. Defaults to
                True.
            save_best (bool, optional): Whether to save the best model.
                Defaults to True.
            model_path (str, optional): The path to save the best model to.
                Defaults to None.

        Returns:
            Dict[str, List[float]]: The training history.
        """
        # Use config defaults if not provided
        num_epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        batch_size = batch_size if batch_size is not None else self.config.batch_size
        val_split = val_split if val_split is not None else self.config.val_split
        early_stopping_patience = (
            early_stopping_patience
            if early_stopping_patience is not None
            else self.config.early_stopping_patience
        )
        verbose = verbose if verbose is not None else self.config.verbose

        # Prepare data
        train_loader, val_loader = self.prepare_data(
            x_history, val_split=val_split, batch_size=batch_size, shuffle=self.config.shuffle
        )

        # Initialize early stopping counter
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.training_history["train_loss"].append(train_loss)

            # Validate if validation data is available
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.training_history["val_loss"].append(val_loss)

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if save_best and model_path is not None:
                        self.save_model(model_path)
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 50 == 0:
                    print(
                        f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss:.6f} - Val loss: {val_loss:.6f}"
                    )

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
            else:
                if verbose and (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss:.6f}")

        if verbose:
            print("Training complete!")

        return self.training_history

    def predict_next(self, x_current: np.ndarray) -> np.ndarray:
        """Predicts the next state.

        Args:
            x_current (np.ndarray): The current state vector, with shape (W,).

        Returns:
            np.ndarray: The predicted next state vector, with shape (W,).
        """
        self.model.eval()

        # Convert input to tensor and move to device
        x_tensor = torch.tensor(x_current, dtype=torch.float32).to(self.device)

        # Make prediction
        with torch.no_grad():
            x_next = self.model(x_tensor)

        # Convert back to numpy
        y_pred = x_next.detach().cpu().numpy()

        if getattr(self.config, "delta_forecast", False):
            y_pred = x_current + y_pred

        return y_pred

    def predict_sequence(self, x_start: np.ndarray, n_steps: int) -> np.ndarray:
        """Predicts a sequence of future states.

        Args:
            x_start (np.ndarray): The starting state vector, with shape (W,).
            n_steps (int): The number of steps to predict.

        Returns:
            np.ndarray: The predicted sequence, with shape (n_steps, W).
        """
        self.model.eval()

        # Initialize sequence with starting state
        sequence = np.zeros((n_steps + 1, self.in_dim))
        sequence[0] = x_start

        # Generate predictions iteratively
        x_current = x_start
        for i in range(n_steps):
            x_next = self.predict_next(x_current)
            sequence[i + 1] = x_next
            x_current = x_next

        return sequence[1:, :]  # Remove the starting state

    def save_model(self, path: str) -> None:
        """Saves the model.

        Args:
            path (str): The path to save the model to.
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Save model and metadata
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
                "training_history": self.training_history,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """Loads the model.

        Args:
            path (str): The path to load the model from.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Load model and optimizer states
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load metadata
        self.in_dim = checkpoint.get("in_dim", self.in_dim)
        self.out_dim = checkpoint.get("out_dim", self.out_dim)
        self.training_history = checkpoint.get("training_history", self.training_history)
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)

    def plot_training_history(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plots the training history.

        Args:
            figsize (Tuple[int, int], optional): The figure size. Defaults to
                (10, 6).
        """
        plt.figure(figsize=figsize)
        epochs = range(1, len(self.training_history["train_loss"]) + 1)

        plt.plot(epochs, self.training_history["train_loss"], "b-", label="Training Loss")

        if self.training_history["val_loss"]:
            plt.plot(epochs, self.training_history["val_loss"], "r-", label="Validation Loss")

        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def evaluate(self, x_history: np.ndarray) -> Dict[str, float]:
        """Evaluates the model on historical data.

        Args:
            x_history (np.ndarray): The historical data, with shape (T, W).

        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
        """
        self.model.eval()

        # Create input-output pairs
        X_input = x_history[:-1, :]  # (T-1, W)

        if getattr(self.config, "delta_forecast", False):
            X_target = x_history[1:, :] - X_input
        else:
            X_target = x_history[1:, :]  # (T-1, W)

        # Convert to tensors
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(X_target, dtype=torch.float32).to(self.device)

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
        rel_frob_err = torch.norm(y_pred - y_tensor, p="fro") / torch.norm(y_tensor, p="fro")

        return {"mse": mse_loss, "rmse": rmse, "r2": r2.item(), "rel_frob_err": rel_frob_err.item()}
