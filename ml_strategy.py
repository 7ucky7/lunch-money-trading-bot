#!/usr/bin/env python3
"""
ML Strategy Module using Hybrid Model for Cryptocurrency Trading Bot.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os

# Local imports
from strategy import Strategy, Signal, SignalType
from data_manager import DataManager, TimeFrame, Candle
from hybrid_model import HybridModel, HybridModelConfig

# Get logger
logger = logging.getLogger("MLStrategy")

class MLStrategy(Strategy):
    """
    Machine Learning based trading strategy using hybrid model.
    """
    
    def __init__(self,
                data_manager: DataManager,
                config: Dict[str, Any] = None):
        """
        Initialize ML strategy.
        
        Args:
            data_manager: Data manager instance
            config: Strategy configuration
        """
        super().__init__(data_manager, config)
        
        # Set default parameters
        self.config = config or {}
        self.lookback_periods = self.config.get("lookback_periods", 60)
        self.prediction_periods = self.config.get("prediction_periods", 12)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.model_path = self.config.get("model_path", "models/trading_model")
        
        # Initialize hybrid model
        model_config = HybridModelConfig(
            lookback_periods=self.lookback_periods,
            prediction_periods=self.prediction_periods,
            model_path=self.model_path
        )
        self.model = HybridModel(model_config)
        
        # Try to load pre-trained model
        try:
            self.model.load(self.model_path)
            logger.info("Loaded pre-trained model")
        except Exception as e:
            logger.warning(f"No pre-trained model found: {e}")
        
        logger.info("ML Strategy initialized")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators and engineered features
        """
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Technical indicators
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()
        df['ma_diff'] = df['ma_10'] - df['ma_30']
        df['ma_ratio'] = df['ma_10'] / df['ma_30']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_ma'] = df['volatility'].rolling(window=10).mean()
        
        # Price channels
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['channel_width'] = (df['high_20'] - df['low_20']) / df['close']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Clean up and return
        return df.fillna(method='ffill').fillna(0)
    
    def generate_signals(self, product_id: str) -> List[Signal]:
        """
        Generate trading signals using hybrid ML model.
        
        Args:
            product_id: Product identifier
            
        Returns:
            List of Signal objects
        """
        signals = []
        
        try:
            # Get historical candles
            candles = self.data_manager.get_latest_candles(
                product_id=product_id,
                timeframe=TimeFrame.MINUTE_1,
                limit=self.lookback_periods + 100  # Extra data for feature calculation
            )
            
            if not candles:
                return []
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': c.timestamp,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            } for c in candles])
            
            # Prepare features
            df = self._prepare_features(df)
            
            # Select feature columns for prediction
            feature_columns = [
                'returns', 'log_returns', 
                'volume_ratio', 'volatility',
                'ma_diff', 'ma_ratio',
                'channel_width', 'rsi',
                'macd_hist', 'bb_width',
                'bb_position'
            ]
            
            # Get features for prediction
            X = df[feature_columns].values[-self.lookback_periods:]
            
            # Make prediction
            predictions = self.model.predict(X.reshape(1, -1))
            
            # Calculate prediction metrics
            current_price = df['close'].iloc[-1]
            predicted_price = predictions[0]
            price_change = (predicted_price - current_price) / current_price
            
            # Generate signal based on prediction
            timestamp = int(df['timestamp'].iloc[-1])
            
            if abs(price_change) > 0.01:  # 1% price change threshold
                signal_type = SignalType.BUY if price_change > 0 else SignalType.SELL
                confidence = min(abs(price_change) * 10, 1.0)  # Scale confidence
                
                if confidence >= self.confidence_threshold:
                    signals.append(Signal(
                        type=signal_type,
                        product_id=product_id,
                        timestamp=timestamp,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'predicted_price': predicted_price,
                            'price_change': price_change,
                            'features': dict(zip(feature_columns, X[-1]))
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}", exc_info=True)
        
        return signals
    
    async def train(self,
                   product_id: str,
                   start_date: str,
                   end_date: str,
                   optimize: bool = True) -> Dict[str, Any]:
        """
        Train the hybrid model on historical data.
        
        Args:
            product_id: Product identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            
            # Fetch historical data
            candles = await self.data_manager.fetch_historical_candles(
                product_id=product_id,
                timeframe=TimeFrame.MINUTE_1,
                start_time=start_ts,
                end_time=end_ts
            )
            
            if not candles:
                raise ValueError("No historical data available for training")
            
            # Convert to DataFrame and prepare features
            df = pd.DataFrame([{
                'timestamp': c.timestamp,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            } for c in candles])
            
            df = self._prepare_features(df)
            
            # Prepare training data
            feature_columns = [
                'returns', 'log_returns', 
                'volume_ratio', 'volatility',
                'ma_diff', 'ma_ratio',
                'channel_width', 'rsi',
                'macd_hist', 'bb_width',
                'bb_position'
            ]
            
            X = df[feature_columns].values
            y = df['close'].shift(-self.prediction_periods).values[:-self.prediction_periods]
            X = X[:-self.prediction_periods]
            
            # Optimize hyperparameters if requested
            if optimize:
                logger.info("Optimizing hyperparameters...")
                best_params = self.model.optimize_hyperparameters(X, y)
                
                # Create new model with optimized parameters
                model_config = HybridModelConfig(**best_params)
                self.model = HybridModel(model_config)
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X, y)
            
            # Save model
            self.model.save(self.model_path)
            
            logger.info("Model training completed")
            return {"status": "success", "message": "Model trained successfully"}
            
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            return {"status": "error", "message": str(e)} 