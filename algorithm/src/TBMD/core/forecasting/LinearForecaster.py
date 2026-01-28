import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import r2_score

# Import config
try:
    from TBMD.config import LinearForecasterConfig
except ImportError:
    LinearForecasterConfig = None


class LinearForecaster:
    """
    Linear forecasting model using a transformation matrix learned via pseudoinverse.
    
    This model learns a linear transformation matrix M such that x(t+1) = M @ x(t).
    The matrix is learned by solving for the pseudoinverse of the system:
    X_output = X_input @ M^T where X_input and X_output are matrices of consecutive states.
    """
    
    def __init__(self, 
                 config: Optional['LinearForecasterConfig'] = None,
                 use_torch: Optional[bool] = None, 
                 device: Optional[str] = None):
        """
        Initialize the linear forecaster.
        
        Args:
            config: LinearForecasterConfig instance (recommended)
            use_torch: Whether to use PyTorch (deprecated, use config)
            device: Device to use if using PyTorch (deprecated, use config)
        
        Examples:
            >>> # New way (recommended):
            >>> from algorithm.TBMD.config import LinearForecasterConfig
            >>> config = LinearForecasterConfig()
            >>> forecaster = LinearForecaster(config=config)
            >>> 
            >>> # Old way (still works):
            >>> forecaster = LinearForecaster(use_torch=True, device='cuda')
        """
        # Handle config vs individual parameters
        if config is not None:
            self.config = config
        else:
            # Old API: create config from parameters
            if LinearForecasterConfig is None:
                raise ImportError("LinearForecasterConfig not available.")
            
            self.config = LinearForecasterConfig(
                device=device
            )
            if use_torch is not None:
                # LinearForecasterConfig doesn't have use_torch, but we can infer backend
                self.config.backend = 'pytorch' if use_torch else 'numpy'
        
        self.M = None
        self.trained = False
        self.use_torch = (self.config.backend == 'pytorch')
        self.metrics = {}
        
        if self.use_torch:
            if self.config.device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
            else:
                self.device = torch.device(self.config.device)
            if self.config.verbose:
                print(f"Using device: {self.device}")
    
    def train(self, x_history: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """
        Train the linear forecaster by learning the transformation matrix M.
        
        Args:
            x_history: Time series data of shape (T, W) where T is time steps and W is features
            verbose: Whether to print training information
            
        Returns:
            metrics: Dictionary of training metrics
        """
        if verbose:
            print("Training linear forecaster...")
        
        # Create input-output pairs
        X_input = x_history[:-1, :]   # (T-1, W)
        X_output = x_history[1:, :]   # (T-1, W)
        
        if self.use_torch:
            # Convert to PyTorch tensors
            X_input_tensor = torch.tensor(X_input, dtype=torch.float32, device=self.device)
            X_output_tensor = torch.tensor(X_output, dtype=torch.float32, device=self.device)
            
            # Calculate pseudoinverse and transformation matrix with regularization
            if hasattr(torch.linalg, 'pinv'):
                X_input_pinv = torch.linalg.pinv(X_input_tensor, rcond=1e-3)
            else:
                X_input_pinv = torch.pinverse(X_input_tensor, rcond=1e-3)  # fallback
            self.M = X_input_pinv @ X_output_tensor
                
            # Calculate predictions for evaluation
            X_output_est = X_input_tensor @ torch.transpose(self.M, 0, 1)
            
            # Calculate metrics
            mse = torch.mean((X_output_tensor - X_output_est) ** 2).item()
            rmse = np.sqrt(mse)
            rel_frob_err = torch.norm(X_output_tensor - X_output_est, p='fro') / torch.norm(X_output_tensor, p='fro')
            rel_frob_err = rel_frob_err.item()
            
            # Convert back to numpy for R2 calculation
            X_output_np = X_output_tensor.detach().cpu().numpy()
            X_output_est_np = X_output_est.detach().cpu().numpy()
            r2 = r2_score(X_output_np, X_output_est_np, multioutput='uniform_average')
            
        else:
            # Calculate pseudoinverse and transformation matrix using numpy
            self.M = np.linalg.pinv(X_input, rcond=1e-3) @ X_output  # (W, W)
            
            # Calculate predictions for evaluation
            X_output_est = X_input @ self.M.T  # (T-1, W)
            
            # Calculate metrics
            mse = np.mean((X_output - X_output_est) ** 2)
            rmse = np.sqrt(mse)
            rel_frob_err = np.linalg.norm(X_output - X_output_est, 'fro') / np.linalg.norm(X_output, 'fro')
            r2 = r2_score(X_output, X_output_est, multioutput='uniform_average')
        
        # Store metrics
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'rel_frob_err': rel_frob_err,
            'r2': r2
        }
        
        self.trained = True
        
        if verbose:
            print(f"Training complete. RMSE: {rmse:.5f}, Rel. Frob. Err: {rel_frob_err:.5f}, R²: {r2:.5f}")
        
        return self.metrics
    
    def predict_next(self, x_current: np.ndarray) -> np.ndarray:
        """
        Predict the next state.
        
        Args:
            x_current: Current state vector of shape (W,)
            
        Returns:
            x_next: Predicted next state vector of shape (W,)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() before making predictions.")
        
        if self.use_torch:
            # Convert to tensor
            if not isinstance(x_current, torch.Tensor):
                x_current_tensor = torch.tensor(x_current, dtype=torch.float32, device=self.device)
            else:
                x_current_tensor = x_current
                
            # Predict next state
            x_next = x_current_tensor @ self.M
            
            # Convert back to numpy
            return x_next.detach().cpu().numpy()
        else:
            # Predict using numpy
            return x_current @ self.M
    
    def predict_sequence(self, x_start: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Predict a sequence of future states.
        
        Args:
            x_start: Starting state vector of shape (W,)
            n_steps: Number of steps to predict
            
        Returns:
            sequence: Predicted sequence of shape (n_steps, W)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() before making predictions.")
        
        # Initialize sequence with starting state
        sequence = np.zeros((n_steps + 1, x_start.shape[0]))
        sequence[0] = x_start
        
        # Generate predictions iteratively
        x_current = x_start
        for i in range(n_steps):
            x_next = self.predict_next(x_current)
            sequence[i+1] = x_next
            x_current = x_next
        
        return sequence[1:]  # Return without the starting state
    
    def evaluate(self, x_history: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on historical data.
        
        Args:
            x_history: Time series data of shape (T, W)
            
        Returns:
            metrics: Evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() before evaluating.")
        
        # Create input-output pairs
        X_input = x_history[:-1, :]   # (T-1, W)
        X_output = x_history[1:, :]   # (T-1, W)
        
        if self.use_torch:
            # Convert to tensors
            X_input_tensor = torch.tensor(X_input, dtype=torch.float32, device=self.device)
            X_output_tensor = torch.tensor(X_output, dtype=torch.float32, device=self.device)
            
            # Make predictions
            X_output_est = X_input_tensor @ torch.transpose(self.M, 0, 1)
            
            # Calculate metrics
            mse = torch.mean((X_output_tensor - X_output_est) ** 2).item()
            rmse = np.sqrt(mse)
            rel_frob_err = torch.norm(X_output_tensor - X_output_est, p='fro') / torch.norm(X_output_tensor, p='fro')
            rel_frob_err = rel_frob_err.item()
            
            # Convert back to numpy for R2 calculation
            X_output_np = X_output_tensor.detach().cpu().numpy()
            X_output_est_np = X_output_est.detach().cpu().numpy()
            r2 = r2_score(X_output_np, X_output_est_np, multioutput='uniform_average')
        else:
            # Make predictions using numpy
            X_output_est = X_input @ self.M.T
            
            # Calculate metrics
            mse = np.mean((X_output - X_output_est) ** 2)
            rmse = np.sqrt(mse)
            rel_frob_err = np.linalg.norm(X_output - X_output_est, 'fro') / np.linalg.norm(X_output, 'fro')
            r2 = r2_score(X_output, X_output_est, multioutput='uniform_average')
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'rel_frob_err': rel_frob_err,
            'r2': r2
        }
        
        return metrics
    
    def plot_prediction_comparison(self, 
                                  x_history: np.ndarray, 
                                  feature_indices: List[int] = None, 
                                  n_steps_ahead: int = 10,
                                  figsize: Tuple[int, int] = (15, 8)) -> None:
        """
        Plot comparison between actual and predicted values.
        
        Args:
            x_history: Time series data of shape (T, W)
            feature_indices: Indices of features to plot (default: first 3 features)
            n_steps_ahead: Number of steps to predict ahead
            figsize: Figure size
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() before plotting predictions.")
        
        # If no feature indices provided, use the first 3 (or all if fewer)
        if feature_indices is None:
            feature_indices = list(range(min(3, x_history.shape[1])))
        
        # Split data into training and testing portions
        train_data = x_history[:-n_steps_ahead]
        test_data = x_history[-n_steps_ahead:]
        
        # Predict the future steps
        last_train_point = train_data[-1]
        predicted_sequence = self.predict_sequence(last_train_point, n_steps_ahead)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Generate x-axis points
        x_train = np.arange(len(train_data))
        x_test = np.arange(len(train_data), len(x_history))
        x_pred = np.arange(len(train_data), len(x_history))
        
        for idx, feature_idx in enumerate(feature_indices):
            plt.subplot(len(feature_indices), 1, idx+1)
            
            # Plot training data
            plt.plot(x_train, train_data[:, feature_idx], 'b-', label='Training Data')
            
            # Plot test data
            plt.plot(x_test, test_data[:, feature_idx], 'g-', label='Actual Future')
            
            # Plot predictions
            plt.plot(x_pred, predicted_sequence[:, feature_idx], 'r--', label='Predicted Future')
            
            plt.title(f'Feature {feature_idx}')
            plt.ylabel('Value')
            if idx == len(feature_indices) - 1:
                plt.xlabel('Time Step')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Create save dictionary
        save_dict = {
            'M': self.M.detach().cpu().numpy() if self.use_torch else self.M,
            'trained': self.trained,
            'metrics': self.metrics,
            'use_torch': self.use_torch
        }
        
        # Save using numpy
        np.savez(path, **save_dict)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        # Load the saved model
        loaded = np.load(path, allow_pickle=True)
        
        # Restore attributes
        self.trained = loaded['trained'].item()
        self.metrics = loaded['metrics'].item()
        self.use_torch = loaded['use_torch'].item()
        
        # Restore M, converting to tensor if needed
        if self.use_torch:
            if not hasattr(self, 'device'):
                self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
            self.M = torch.tensor(loaded['M'], dtype=torch.float32, device=self.device)
        else:
            self.M = loaded['M']
        
        print(f"Model loaded from {path}")
