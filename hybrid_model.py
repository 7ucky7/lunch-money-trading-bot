#!/usr/bin/env python3
"""
Hybrid ML Model for Cryptocurrency Trading.

This module implements a hybrid approach combining XGBoost and LSTM
for improved trading predictions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from datetime import datetime
import joblib
import os

# Get logger
logger = logging.getLogger("HybridModel")

@dataclass
class HybridModelConfig:
    """Configuration for hybrid model."""
    lookback_periods: int = 60
    prediction_periods: int = 12
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    batch_size: int = 32
    epochs: int = 100
    train_split: float = 0.8
    model_path: str = "models/hybrid_model"
    use_gpu: bool = torch.cuda.is_available()

class LSTMModule(nn.Module):
    """LSTM component of hybrid model."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Self-attention on LSTM output
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        # Use attention output for final prediction
        combined = attn_out.transpose(0, 1)[:, -1, :]
        return self.fc(combined)

class HybridModel:
    """
    Hybrid model combining XGBoost and LSTM with attention.
    
    Features:
    - LSTM with attention for temporal pattern recognition
    - XGBoost for feature importance and non-linear relationships
    - Ensemble learning for final predictions
    - Automated hyperparameter optimization
    - Time series cross-validation
    """
    
    def __init__(self, config: Optional[HybridModelConfig] = None):
        """Initialize hybrid model."""
        self.config = config or HybridModelConfig()
        
        # Initialize models
        self.lstm_model = None
        self.xgb_model = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        
        # Set device
        self.device = torch.device("cuda" if self.config.use_gpu else "cpu")
        
        logger.info(f"Initialized hybrid model (using device: {self.device})")
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Input features
            y: Target values
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best hyperparameters
        """
        def objective(trial):
            # LSTM hyperparameters
            lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 32, 256)
            lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 4)
            lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5)
            
            # XGBoost hyperparameters
            xgb_max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
            xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3)
            xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
            
            # Update config
            temp_config = HybridModelConfig(
                lstm_hidden_dim=lstm_hidden_dim,
                lstm_num_layers=lstm_num_layers,
                lstm_dropout=lstm_dropout,
                xgb_max_depth=xgb_max_depth,
                xgb_learning_rate=xgb_learning_rate,
                xgb_n_estimators=xgb_n_estimators
            )
            
            # Create temporary model
            temp_model = HybridModel(temp_config)
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                temp_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = temp_model.predict(X_val)
                mse = np.mean((y_val - y_pred) ** 2)
                scores.append(mse)
            
            return np.mean(scores)
        
        # Create study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the hybrid model.
        
        Args:
            X: Input features
            y: Target values
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare sequences for LSTM
        X_seq, y_seq = self._prepare_sequences(X_scaled, y)
        
        # Train LSTM
        self._train_lstm(X_seq, y_seq)
        
        # Train XGBoost
        self._train_xgboost(X_scaled, y)
        
        # Train ensemble
        self._train_ensemble(X_scaled, y)
    
    def _train_lstm(self, X_seq: np.ndarray, y_seq: np.ndarray) -> None:
        """Train LSTM model."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # Create model
        input_dim = X_seq.shape[2]
        self.lstm_model = LSTMModule(
            input_dim=input_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            output_dim=self.config.prediction_periods,
            dropout=self.config.lstm_dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters())
        
        # Training loop
        self.lstm_model.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            outputs = self.lstm_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch [{epoch+1}/{self.config.epochs}], Loss: {loss.item():.4f}")
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost model."""
        self.xgb_model = xgb.XGBRegressor(
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            n_estimators=self.config.xgb_n_estimators,
            objective='reg:squarederror',
            tree_method='gpu_hist' if self.config.use_gpu else 'hist'
        )
        
        self.xgb_model.fit(
            X, y,
            eval_set=[(X, y)],
            early_stopping_rounds=20,
            verbose=False
        )
    
    def _train_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train ensemble model to combine LSTM and XGBoost predictions."""
        # Get predictions from both models
        lstm_preds = self._lstm_predict(X)
        xgb_preds = self.xgb_model.predict(X)
        
        # Combine predictions as features
        ensemble_features = np.column_stack([lstm_preds, xgb_preds])
        
        # Train a final XGBoost model to combine predictions
        self.ensemble_model = xgb.XGBRegressor(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            objective='reg:squarederror'
        )
        
        self.ensemble_model.fit(ensemble_features, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the hybrid model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        lstm_preds = self._lstm_predict(X_scaled)
        xgb_preds = self.xgb_model.predict(X_scaled)
        
        # Combine predictions
        ensemble_features = np.column_stack([lstm_preds, xgb_preds])
        
        # Make final prediction
        return self.ensemble_model.predict(ensemble_features)
    
    def _lstm_predict(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from LSTM model."""
        # Prepare sequences
        X_seq = self._prepare_sequences(X, None)[0]
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Make prediction
        self.lstm_model.eval()
        with torch.no_grad():
            predictions = self.lstm_model(X_tensor)
            
        return predictions.cpu().numpy()
    
    def _prepare_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare sequences for LSTM."""
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - self.config.lookback_periods):
            X_seq.append(X[i:(i + self.config.lookback_periods)])
            if y is not None:
                y_seq.append(y[i + self.config.lookback_periods])
        
        return np.array(X_seq), np.array(y_seq) if y_seq else None
    
    def save(self, path: Optional[str] = None) -> None:
        """Save model to disk."""
        path = path or self.config.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save LSTM model
        torch.save(self.lstm_model.state_dict(), f"{path}_lstm.pt")
        
        # Save XGBoost model
        self.xgb_model.save_model(f"{path}_xgb.json")
        
        # Save ensemble model
        self.ensemble_model.save_model(f"{path}_ensemble.json")
        
        # Save scaler
        joblib.dump(self.scaler, f"{path}_scaler.joblib")
        
        # Save config
        joblib.dump(self.config, f"{path}_config.joblib")
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Optional[str] = None) -> None:
        """Load model from disk."""
        path = path or self.config.model_path
        
        # Load config
        self.config = joblib.load(f"{path}_config.joblib")
        
        # Load LSTM model
        input_dim = self.config.lookback_periods  # This should be determined from the saved model
        self.lstm_model = LSTMModule(
            input_dim=input_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            output_dim=self.config.prediction_periods,
            dropout=self.config.lstm_dropout
        ).to(self.device)
        self.lstm_model.load_state_dict(torch.load(f"{path}_lstm.pt"))
        
        # Load XGBoost model
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model(f"{path}_xgb.json")
        
        # Load ensemble model
        self.ensemble_model = xgb.XGBRegressor()
        self.ensemble_model.load_model(f"{path}_ensemble.json")
        
        # Load scaler
        self.scaler = joblib.load(f"{path}_scaler.joblib")
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost model."""
        importance = self.xgb_model.feature_importances_
        features = [f"feature_{i}" for i in range(len(importance))]
        
        return pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False) 