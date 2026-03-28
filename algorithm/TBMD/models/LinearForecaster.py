import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import r2_score


class LinearForecaster:
    """A linear forecasting model.

    This model learns a linear transformation `M` such that `x(t+1) = M @ x(t)`,
    using a pseudoinverse method to solve the system.

    Args:
        use_torch (bool, optional): If `True`, use PyTorch for calculations,
            which can leverage GPU acceleration. Defaults to `False`.
        device (str, optional): The device to use if `use_torch` is `True`
            (e.g., 'cpu', 'cuda', 'mps'). If `None`, the device is automatically
            selected.
    """
    
    def __init__(self, use_torch: bool = False, device: str = None):
        """Initializes the LinearForecaster.

        Args:
            use_torch (bool, optional): If `True`, use PyTorch for
                calculations. Defaults to `False`.
            device (str, optional): The device to use if `use_torch` is `True`.
                If `None`, the device is automatically selected.
        """
        self.M = None
        self.trained = False
        self.use_torch = use_torch
        self.metrics = {}
        
        if use_torch:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
            else:
                self.device = torch.device(device)
            print(f"Using device: {self.device}")
    
    def train(self, x_history: np.ndarray, verbose: bool = True) -> Dict[str, float]:
        """Trains the linear forecaster by learning the transformation matrix `M`.

        Args:
            x_history (np.ndarray): The time series data, with shape (T, W),
                where T is the number of time steps and W is the number of
                features.
            verbose (bool, optional): Whether to print training information.
                Defaults to `True`.

        Returns:
            Dict[str, float]: A dictionary of training metrics.
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
            
            # Calculate pseudoinverse and transformation matrix
            if hasattr(torch, 'pinverse'):  # For newer PyTorch versions
                X_input_pinv = torch.pinverse(X_input_tensor)
                self.M = X_input_pinv @ X_output_tensor
            else:  # Fallback to numpy
                X_input_pinv = np.linalg.pinv(X_input)
                self.M = np.matmul(X_input_pinv, X_output)
                self.M = torch.tensor(self.M, dtype=torch.float32, device=self.device)
                
            # Calculate predictions for evaluation
            X_output_est = X_input_tensor @ self.M
            
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
            self.M = np.linalg.pinv(X_input) @ X_output  # (W, W)
            
            # Calculate predictions for evaluation
            X_output_est = X_input @ self.M  # (T-1, W)
            
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
        """Predicts the next state.

        Args:
            x_current (np.ndarray): The current state vector, with shape (W,).

        Returns:
            np.ndarray: The predicted next state vector, with shape (W,).
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
        """Predicts a sequence of future states.

        Args:
            x_start (np.ndarray): The starting state vector, with shape (W,).
            n_steps (int): The number of steps to predict.

        Returns:
            np.ndarray: The predicted sequence, with shape (n_steps, W).
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() before making predictions.")
        
        if self.use_torch:
            # Optimized path for PyTorch: Keep data on device
            if not isinstance(x_start, torch.Tensor):
                x_current = torch.tensor(x_start, dtype=torch.float32, device=self.device)
            else:
                x_current = x_start.to(dtype=torch.float32, device=self.device)

            # Pre-allocate output tensor on device
            # x_current shape is (W,)
            sequence = torch.zeros((n_steps + 1, x_current.shape[0]), dtype=torch.float32, device=self.device)
            sequence[0] = x_current

            # Generate predictions iteratively completely on device
            for i in range(n_steps):
                x_next = x_current @ self.M
                sequence[i+1] = x_next
                x_current = x_next

            # Convert back to numpy once at the end
            return sequence[1:].detach().cpu().numpy()

        else:
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
        """Evaluates the model on historical data.

        Args:
            x_history (np.ndarray): The time series data, with shape (T, W).

        Returns:
            Dict[str, float]: A dictionary of evaluation metrics.
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
            X_output_est = X_input_tensor @ self.M
            
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
            X_output_est = X_input @ self.M
            
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
        """Plots a comparison between actual and predicted values.

        Args:
            x_history (np.ndarray): The time series data, with shape (T, W).
            feature_indices (List[int], optional): The indices of the features
                to plot. If `None`, the first 3 features are plotted.
            n_steps_ahead (int, optional): The number of steps to predict
                ahead. Defaults to 10.
            figsize (Tuple[int, int], optional): The figure size. Defaults to
                (15, 8).
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
        """Saves the model to a file.

        Args:
            path (str): The path to save the model to.

        Raises:
            RuntimeError: If the model is not trained.
        """
        if not self.trained:
            raise RuntimeError("Cannot save untrained model")

        dir_path = os.path.dirname(os.path.abspath(path))
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
        """Loads the model from a file.

        Args:
            path (str): The path to load the model from.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

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
