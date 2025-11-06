import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Union, List, Any


class MLPModel(nn.Module):
    """A multi-layer perceptron model for time series forecasting."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, 
                 dropout_rate: float = 0.3, num_layers: int = 2):
        """
        Initialize the MLP model.
        
        Parameters
        ----------
        in_dim : int
            The input dimension.
        out_dim : int
            The output dimension.
        hidden_dim : int, optional
            The hidden layer dimension, by default 256.
        dropout_rate : float, optional
            The dropout rate for regularization, by default 0.3.
        num_layers : int, optional
            The number of hidden layers, by default 2.
        """
        super().__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.net(x)


class MLPForecaster:
    """A complete pipeline for training and using MLP forecasting models.

    Parameters
    ----------
    in_dim : int
        The input dimension.
    out_dim : int
        The output dimension.
    hidden_dim : int, optional
        The hidden layer dimension, by default 256.
    dropout_rate : float, optional
        The dropout rate for regularization, by default 0.3.
    num_layers : int, optional
        The number of hidden layers, by default 2.
    lr : float, optional
        The learning rate, by default 1e-3.
    weight_decay : float, optional
        The L2 regularization parameter, by default 1e-5.
    device : str, optional
        The device to run computations on ('cpu', 'cuda', or 'mps'), by
        default None.
    """
    
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 256, 
                 dropout_rate: float = 0.3,
                 num_layers: int = 2,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: str = None):
        """
        Initialize the MLP forecaster.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate for regularization
            num_layers: Number of hidden layers
            lr: Learning rate
            weight_decay: L2 regularization parameter
            device: Device to run computations on ('cpu', 'cuda', or 'mps')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Initialize model and hyperparameters
        self.model = MLPModel(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            num_layers=num_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.loss_fn = nn.MSELoss()
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }
        self.best_val_loss = float('inf')
        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def prepare_data(self, 
                     x_history: np.ndarray, 
                     val_split: float = 0.2,
                     batch_size: int = 32,
                     shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare the training and validation data loaders.
        
        Parameters
        ----------
        x_history : np.ndarray
            The historical data array of shape (T, W).
        val_split : float, optional
            The validation split ratio, by default 0.2.
        batch_size : int, optional
            The batch size, by default 32.
        shuffle : bool, optional
            Whether to shuffle the data, by default True.
            
        Returns
        -------
        Tuple[DataLoader, Optional[DataLoader]]
            A tuple containing the training data loader and the validation data
            loader, which is `None` if `val_split` is 0.
        """
        # Create input-output pairs
        X_input = x_history[:-1, :]   # (T-1, W)
        X_target = x_history[1:, :]   # (T-1, W)
        
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
            X_tensor = torch.tensor(X_input, dtype=torch.float32)
            y_tensor = torch.tensor(X_target, dtype=torch.float32)
            
            # Create dataset and loader
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=shuffle
            )
            
            return train_loader, None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train the model for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            The training data loader.
            
        Returns
        -------
        float
            The average training loss for this epoch.
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
        """Validate the model.
        
        Parameters
        ----------
        val_loader : DataLoader
            The validation data loader.
            
        Returns
        -------
        float
            The average validation loss.
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
              num_epochs: int = 500,
              batch_size: int = 32,
              val_split: float = 0.2,
              early_stopping_patience: int = 20,
              verbose: bool = True,
              save_best: bool = True,
              model_path: str = None) -> Dict[str, List[float]]:
        """Train the model.
        
        Parameters
        ----------
        x_history : np.ndarray
            The historical data array of shape (T, W).
        num_epochs : int, optional
            The number of training epochs, by default 500.
        batch_size : int, optional
            The batch size, by default 32.
        val_split : float, optional
            The validation split ratio, by default 0.2.
        early_stopping_patience : int, optional
            The patience for early stopping, by default 20.
        verbose : bool, optional
            Whether to print progress, by default True.
        save_best : bool, optional
            Whether to save the best model, by default True.
        model_path : str, optional
            The path to save the best model to, by default None.
            
        Returns
        -------
        Dict[str, List[float]]
            The training history.
        """
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            x_history, 
            val_split=val_split,
            batch_size=batch_size
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
    
    def predict_next(self, x_current: np.ndarray) -> np.ndarray:
        """Predict the next state.
        
        Parameters
        ----------
        x_current : np.ndarray
            The current state vector of shape (W,).
            
        Returns
        -------
        np.ndarray
            The predicted next state vector of shape (W,).
        """
        self.model.eval()
        
        # Convert input to tensor and move to device
        x_tensor = torch.tensor(x_current, dtype=torch.float32).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            x_next = self.model(x_tensor)
        
        # Convert back to numpy
        return x_next.detach().cpu().numpy()
    
    def predict_sequence(self, x_start: np.ndarray, n_steps: int) -> np.ndarray:
        """Predict a sequence of future states.
        
        Parameters
        ----------
        x_start : np.ndarray
            The starting state vector of shape (W,).
        n_steps : int
            The number of steps to predict.
            
        Returns
        -------
        np.ndarray
            The predicted sequence of shape (n_steps, W).
        """
        self.model.eval()
        
        # Initialize sequence with starting state
        sequence = np.zeros((n_steps + 1, self.in_dim))
        sequence[0] = x_start
        
        # Generate predictions iteratively
        x_current = x_start
        for i in range(n_steps):
            x_next = self.predict_next(x_current)
            sequence[i+1] = x_next
            x_current = x_next
        
        return sequence[1:, :]  # Remove the starting state
    
    def save_model(self, path: str) -> None:
        """Save the model.
        
        Parameters
        ----------
        path : str
            The path to save the model to.
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
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load the model.
        
        Parameters
        ----------
        path : str
            The path to load the model from.
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load metadata
        self.in_dim = checkpoint.get('in_dim', self.in_dim)
        self.out_dim = checkpoint.get('out_dim', self.out_dim)
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.best_val_loss = checkpoint.get('best_val_loss', self.best_val_loss)
    
    def plot_training_history(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """Plot the training history.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            The figure size, by default (10, 6).
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
        """Evaluate the model on historical data.
        
        Parameters
        ----------
        x_history : np.ndarray
            The historical data array of shape (T, W).
            
        Returns
        -------
        Dict[str, float]
            A dictionary of evaluation metrics.
        """
        self.model.eval()
        
        # Create input-output pairs
        X_input = x_history[:-1, :]   # (T-1, W)
        X_target = x_history[1:, :]   # (T-1, W)
        
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
        rel_frob_err = torch.norm(y_pred - y_tensor, p='fro') / torch.norm(y_tensor, p='fro')
        
        return {
            'mse': mse_loss,
            'rmse': rmse,
            'r2': r2.item(),
            'rel_frob_err': rel_frob_err.item()
        }
