import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import random

from TBMD.core.forecasting.LSTMForecaster import LSTMForecaster

class ScheduledSamplingLSTMForecaster(LSTMForecaster):
    """LSTM Forecaster with Scheduled Sampling for autoregressive training."""
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = 0) -> float:
        """Trains the model for one epoch using Scheduled Sampling.

        Args:
            train_loader (DataLoader): The training data loader returning unrolled sequences.
            epoch (int): Current epoch, used for computing scheduled sampling probability.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        unroll_steps = getattr(self.config, 'use_scheduled_sampling_unroll_steps', getattr(self.config, 'ss_unroll_steps', 1))
        ss_decay_rate = getattr(self.config, 'ss_decay_rate', 0.0)
        ss_min_prob = getattr(self.config, 'ss_min_prob', 0.0)
        is_delta = getattr(self.config, 'delta_forecast', False)
        
        # p_teacher is the probability of using the TRUE previous state (Teacher Forcing)
        # It decays exponentially from 1.0 down to ss_min_prob
        p_teacher = max(ss_min_prob, np.exp(-ss_decay_rate * epoch))
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            
            # Y_batch shape: (B, unroll_steps, W)
            batch_loss = 0.0
            current_window = X_batch
            
            self.optimizer.zero_grad()
            
            for step in range(unroll_steps):
                # Predict next step
                y_pred = self.model(current_window) # shape: (batch, W)
                y_true = Y_batch[:, step, :]        # shape: (batch, W)
                
                batch_loss += self.loss_fn(y_pred, y_true)
                
                # Prepare window for the next step
                if step < unroll_steps - 1:
                    use_teacher = random.random() < p_teacher
                    chosen_output = y_true if use_teacher else y_pred.detach()
                    
                    if is_delta:
                        # If delta, add chosen delta to the last absolute state
                        last_state = current_window[:, -1, :]
                        next_step_tensor = last_state + chosen_output
                    else:
                        next_step_tensor = chosen_output
                        
                    current_window = torch.cat([
                        current_window[:, 1:, :],
                        next_step_tensor.unsqueeze(1)
                    ], dim=1)
            
            # Average loss over the unroll steps
            batch_loss = batch_loss / unroll_steps
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += batch_loss.item() * X_batch.size(0)
            
            avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """Evaluates the model using autoregressive rollout across the unroll_steps.

        Args:
            val_loader (DataLoader): The validation data loader.

        Returns:
            float: The average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        
        unroll_steps = getattr(self.config, 'use_scheduled_sampling_unroll_steps', getattr(self.config, 'ss_unroll_steps', 1))
        is_delta = getattr(self.config, 'delta_forecast', False)
        
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                
                batch_loss = 0.0
                current_window = X_batch
                
                for step in range(unroll_steps):
                    y_pred = self.model(current_window)
                    y_true = Y_batch[:, step, :]
                    
                    batch_loss += self.loss_fn(y_pred, y_true)
                    
                    if step < unroll_steps - 1:
                        if is_delta:
                            last_state = current_window[:, -1, :]
                            next_step_tensor = last_state + y_pred
                        else:
                            next_step_tensor = y_pred
                            
                        current_window = torch.cat([
                            current_window[:, 1:, :],
                            next_step_tensor.unsqueeze(1)
                        ], dim=1)
                        
                batch_loss = batch_loss / unroll_steps
                total_loss += batch_loss.item() * X_batch.size(0)
                
        return total_loss / len(val_loader.dataset)

