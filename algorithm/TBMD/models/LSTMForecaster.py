import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Union, List, Any


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 1, 
                 dropout_rate: float = 0.0):
        """
        Initialize the LSTM model.
        
        Args:
            in_dim: Input dimension
            hidden_dim: Hidden state dimension
            out_dim: Output dimension
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization (only applied if num_layers > 1)
        """
        super().__init__()
        
        # Apply dropout between LSTM layers when num_layers > 1
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x_seq: Tensor of shape (batch, seq_len, in_dim)
            
        Returns:
            Predictions of shape (batch, out_dim)
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
    """Complete pipeline for training and using LSTM forecasting models."""
    
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 seq_length: int = 5,
                 hidden_dim: int = 64, 
                 num_layers: int = 1,
                 dropout_rate: float = 0.0,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-5,
                 device: str = None):
        """
        Initialize the LSTM forecaster.
        
        Args:
            in_dim: Input dimension (number of features)
            out_dim: Output dimension (number of features to predict)
            seq_length: Length of input sequence (time steps to look back)
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
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
        
        # Store configuration
        self.seq_length = seq_length
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize model
        self.model = LSTMModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
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
        """
        Create a lagged dataset from a time series.
        
        Args:
            x_history: Time series data of shape (T, W) where T is time steps and W is features
            seq_length: Length of input sequences, if None uses self.seq_length
            
        Returns:
            X_seq: Input sequences of shape (N, seq_length, W)
            Y_seq: Target outputs of shape (N, W)
            where N = T - seq_length
        """
        if seq_length is None:
            seq_length = self.seq_length
            
        T = x_history.shape[0]
        W = x_history.shape[1]
        
        # Check if we have enough data
        if T <= seq_length:
            raise ValueError(f"Not enough time steps ({T}) to create sequences of length {seq_length}")
        
        # Create sequences and targets
        X_seq = []
        Y_seq = []
        for t in range(seq_length, T):
            # Window [t-seq_length, ..., t-1]
            x_window = x_history[t-seq_length:t, :]  # (seq_length, W)
            X_seq.append(x_window)
            
            # Target x(t)
            y_value = x_history[t, :]  # (W,)
            Y_seq.append(y_value)
        
        X_seq = np.array(X_seq)  # (N, seq_length, W)
        Y_seq = np.array(Y_seq)  # (N, W)
        
        return X_seq, Y_seq
    
    def prepare_data(self, 
                    x_history: np.ndarray, 
                    val_split: float = 0.2,
                    batch_size: int = 32,
                    shuffle: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare training and validation data loaders.
        
        Args:
            x_history: Time series data of shape (T, W)
            val_split: Validation split ratio
            batch_size: Batch size
            shuffle: Whether to shuffle the training data
            
        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader (None if val_split=0)
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
             num_epochs: int = 300,
             batch_size: int = 32,
             val_split: float = 0.2,
             early_stopping_patience: int = 20,
             verbose: bool = True,
             save_best: bool = True,
             model_path: str = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            x_history: Time series data of shape (T, W)
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
    
    def predict_next(self, x_window: np.ndarray) -> np.ndarray:
        """
        Predict the next state from the given window.
        
        Args:
            x_window: Input window of shape (seq_length, W) or (1, seq_length, W)
            
        Returns:
            x_next: Predicted next state of shape (W,)
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
        return x_next.detach().cpu().numpy()[0]  # Remove batch dimension
    
    def predict_sequence(self, x_start_window: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Predict a sequence of future states.
        
        Args:
            x_start_window: Starting window of shape (seq_length, W)
            n_steps: Number of steps to predict
            
        Returns:
            sequence: Predicted sequence of shape (n_steps, W)
        """
        self.model.eval()
        
        # Check input shape
        if x_start_window.shape[0] != self.seq_length:
            raise ValueError(f"Input window has sequence length {x_start_window.shape[0]}, expected {self.seq_length}")
        
        # Initialize sequence with starting window
        input_window = x_start_window.copy()
        sequence = np.zeros((n_steps, self.out_dim))
        
        # Generate predictions iteratively
        for i in range(n_steps):
            # Predict next step
            next_step = self.predict_next(input_window)
            sequence[i] = next_step
            
            # Update window by removing oldest step and adding prediction
            input_window = np.vstack([input_window[1:], next_step])
        
        return sequence
    
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
            'seq_length': self.seq_length,
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
        self.seq_length = checkpoint.get('seq_length', self.seq_length)
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
            x_history: Time series data of shape (T, W)
            
        Returns:
            metrics: Evaluation metrics
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
