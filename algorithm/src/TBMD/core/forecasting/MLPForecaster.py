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
    from TBMD.config import MLPForecasterConfig
except ImportError:
    # Fallback if running standalone
    MLPForecasterConfig = None


class MLPModel(nn.Module):
    """Multi-layer perceptron model for time series forecasting."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, 
                 dropout_rate: float = 0.3, num_layers: int = 2):
        """
        Initialize the MLP model.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate for regularization
            num_layers: Number of hidden layers
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
        """Forward pass through the model."""
        return self.net(x)


class MLPForecaster:
    """Complete pipeline for training and using MLP forecasting models."""
    
    def __init__(self, 
                 in_dim: Optional[int] = None,
                 out_dim: Optional[int] = None,
                 config: Optional['MLPForecasterConfig'] = None,
                 # Старые параметры для обратной совместимости
                 hidden_dim: Optional[int] = None, 
                 dropout_rate: Optional[float] = None,
                 num_layers: Optional[int] = None,
                 lr: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 device: Optional[str] = None):
        """
        Initialize the MLP forecaster.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            config: MLPForecasterConfig instance (recommended)
            hidden_dim: Hidden layer dimension (deprecated, use config)
            dropout_rate: Dropout rate (deprecated, use config)
            num_layers: Number of hidden layers (deprecated, use config)
            lr: Learning rate (deprecated, use config)
            weight_decay: L2 regularization (deprecated, use config)
            device: Device to run on (deprecated, use config)
        
        Examples:
            >>> # New way (recommended):
            >>> from algorithm.TBMD.config import MLPForecasterConfig
            >>> config = MLPForecasterConfig(hidden_size=512, num_epochs=1000)
            >>> forecaster = MLPForecaster(in_dim=10, out_dim=10, config=config)
            >>> 
            >>> # Old way (still works):
            >>> forecaster = MLPForecaster(in_dim=10, out_dim=10, hidden_dim=256)
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
        
        # Store dimensions
        self.in_dim = self.config.in_dim if self.config.in_dim else in_dim
        self.out_dim = self.config.out_dim if self.config.out_dim else out_dim
        
        # Initialize model and hyperparameters
        self.model = MLPModel(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden_dim=self.config.hidden_size,
            dropout_rate=self.config.dropout,
            num_layers=self.config.num_layers
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        self.loss_fn = nn.MSELoss()
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }
        self.best_val_loss = float('inf')
    
    def prepare_data(self, 
                     x_history: np.ndarray, 
                     val_split: float = 0.2,
                     batch_size: int = 32,
                     shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare training and validation data loaders.
        
        Args:
            x_history: Historical data array of shape (T, W)
            val_split: Validation split ratio
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader (None if val_split=0)
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
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            avg_loss: Average training loss for this epoch
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
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss: Average validation loss
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
        """
        Train the model.
        
        Args:
            x_history: Historical data array of shape (T, W)
            num_epochs: Number of training epochs
            batch_size: Batch size
            val_split: Validation split ratio
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            save_best: Whether to save the best model
            model_path: Path to save the best model
            
        Returns:
            history: Training history
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
    
    def predict_next(self, x_current: np.ndarray) -> np.ndarray:
        """
        Predict the next state.
        
        Args:
            x_current: Current state vector of shape (W,)
            
        Returns:
            x_next: Predicted next state vector of shape (W,)
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
        """
        Predict a sequence of future states.
        
        Args:
            x_start: Starting state vector of shape (W,)
            n_steps: Number of steps to predict
            
        Returns:
            sequence: Predicted sequence of shape (n_steps, W)
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
        """
        Save the model.
        
        Args:
            path: Path to save the model
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
        """
        Load the model.
        
        Args:
            path: Path to load the model from
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
        """
        Plot the training history.
        
        Args:
            figsize: Figure size
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
        """
        Evaluate the model on historical data.
        
        Args:
            x_history: Historical data array of shape (T, W)
            
        Returns:
            metrics: Evaluation metrics
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
