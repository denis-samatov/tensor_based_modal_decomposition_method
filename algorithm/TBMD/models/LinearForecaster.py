import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from functools import wraps
from typing import Dict, Tuple, List, Optional, Union, Any
from sklearn.metrics import r2_score


def with_torch_conversion(func):
    """
    Decorator to handle PyTorch tensor conversions for LinearForecaster.
    If `self.use_torch` is True, this decorator:
    1. Converts numpy array inputs to PyTorch tensors on the correct device.
    2. Runs the function.
    3. Converts returned PyTorch tensors back to numpy arrays.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, 'use_torch', False):
            return func(self, *args, **kwargs)

        import torch
        import numpy as np

        # Convert args to tensors
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                new_args.append(torch.tensor(arg, dtype=torch.float32, device=self.device))
            elif isinstance(arg, torch.Tensor):
                new_args.append(arg.to(dtype=torch.float32, device=self.device))
            else:
                new_args.append(arg)

        # Convert kwargs to tensors
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                new_kwargs[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
            elif isinstance(v, torch.Tensor):
                new_kwargs[k] = v.to(dtype=torch.float32, device=self.device)
            else:
                new_kwargs[k] = v

        # Execute function
        result = func(self, *new_args, **new_kwargs)

        # Convert results back to numpy arrays
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
        elif isinstance(result, tuple):
            return tuple(r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else r for r in result)
        elif isinstance(result, dict):
            return {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}
        return result

    return wrapper


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
    

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Helper method to calculate common evaluation metrics.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            Dict[str, float]: A dictionary containing mse, rmse, rel_frob_err, and r2.
        """
        import numpy as np
        if self.use_torch:
            import torch
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()

        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        rel_frob_err = np.linalg.norm(y_true - y_pred, 'fro') / np.linalg.norm(y_true, 'fro')
        r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'rel_frob_err': float(rel_frob_err),
            'r2': float(r2)
        }

    @with_torch_conversion
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
            import torch
            # Solve the linear system X_input * M = X_output for M
            self.M = torch.linalg.lstsq(X_input, X_output).solution
        else:
            import numpy as np
            # Solve the linear system X_input * M = X_output for M using numpy
            self.M, _, _, _ = np.linalg.lstsq(X_input, X_output, rcond=None)
            
        # Calculate predictions for evaluation
        X_output_est = X_input @ self.M

        # Calculate metrics
        metrics = self._calculate_metrics(X_output, X_output_est)
        
        # Store metrics
        self.metrics = metrics
        
        self.trained = True
        
        if verbose:
            print(f"Training complete. RMSE: {metrics['rmse']:.5f}, Rel. Frob. Err: {metrics['rel_frob_err']:.5f}, R²: {metrics['r2']:.5f}")
        
        return self.metrics
    
    @with_torch_conversion
    def predict_next(self, x_current: np.ndarray) -> np.ndarray:
        """Predicts the next state.

        Args:
            x_current (np.ndarray): The current state vector, with shape (W,).

        Returns:
            np.ndarray: The predicted next state vector, with shape (W,).
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() before making predictions.")
        
        # The decorator handles tensor conversion
        return x_current @ self.M
    
    @with_torch_conversion
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
            import torch
            sequence = torch.zeros((n_steps + 1, x_start.shape[0]), dtype=torch.float32, device=self.device)
        else:
            import numpy as np
            sequence = np.zeros((n_steps + 1, x_start.shape[0]), dtype=x_start.dtype)

        sequence[0] = x_start

        # Generate predictions iteratively completely in device-agnostic way
        # x_start is automatically converted to correct type by decorator
        x_current = x_start
        for i in range(n_steps):
            x_next = x_current @ self.M
            sequence[i+1] = x_next
            x_current = x_next

        return sequence[1:]
    
    @with_torch_conversion
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
        
        # Make predictions using device-agnostic operation
        X_output_est = X_input @ self.M

        # Calculate metrics
        metrics = self._calculate_metrics(X_output, X_output_est)
        
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
        
        # Create metadata
        metadata = {
            'trained': self.trained,
            'metrics': self.metrics,
            'use_torch': self.use_torch
        }
        
        # Save using numpy
        # Use np.savez to save tensors and JSON-serialized metadata
        # This avoids using pickles and improves security
        np.savez(
            path,
            M=self.M.detach().cpu().numpy() if self.use_torch else self.M,
            metadata=json.dumps(metadata)
        )
        
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

        # Load the saved model without allowing pickles for security
        with np.load(path, allow_pickle=False) as loaded:
            # Restore metadata
            metadata_raw = loaded['metadata']
            # Handle both 0-d and 1-d arrays for robustness
            if metadata_raw.ndim == 0:
                metadata_json = str(metadata_raw)
            else:
                metadata_json = str(metadata_raw[0])

            metadata = json.loads(metadata_json)

            self.trained = metadata['trained']
            self.metrics = metadata['metrics']
            self.use_torch = metadata['use_torch']

            # Restore M, converting to tensor if needed
            M_data = loaded['M']
            if self.use_torch:
                if not hasattr(self, 'device'):
                    self.device = torch.device('cuda' if torch.cuda.is_available() else
                                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
                self.M = torch.tensor(M_data, dtype=torch.float32, device=self.device)
            else:
                self.M = M_data
        
        print(f"Model loaded from {path}")
