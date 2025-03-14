#!/usr/bin/env python3
"""
Strategy Framework Module.

This provides the foundation for all trading strategies in the system,
with common interfaces and utilities for signal generation,
backtesting compatibility, and strategy registration.
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StrategyFramework")

class SignalType(Enum):
    """Trade signal types."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    NEUTRAL = "neutral"

class TimeFrame(Enum):
    """Candle timeframes."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

@dataclass
class Signal:
    """Trading signal produced by strategies."""
    type: SignalType
    product_id: str
    price: float
    timestamp: int
    timeframe: TimeFrame
    strategy_name: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "type": self.type.value,
            "product_id": self.product_id,
            "price": self.price,
            "timestamp": self.timestamp,
            "timeframe": self.timeframe.value,
            "strategy_name": self.strategy_name,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create signal from dictionary."""
        return cls(
            type=SignalType(data["type"]),
            product_id=data["product_id"],
            price=data["price"],
            timestamp=data["timestamp"],
            timeframe=TimeFrame(data["timeframe"]),
            strategy_name=data["strategy_name"],
            confidence=data.get("confidence", 0.0),
            metadata=data.get("metadata", {})
        )

class Strategy(ABC):
    """
    Base strategy class that all strategies must inherit from.
    
    This abstract class defines the common interface for all trading strategies,
    ensuring consistency in how they generate signals and interact with the system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize strategy with optional configuration.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)
        self.timeframes = [TimeFrame(tf) for tf in self.config.get("timeframes", ["1h"])]
        self.products = self.config.get("products", [])
        self.enabled = self.config.get("enabled", True)
        
        # Performance metrics
        self.signals_generated = 0
        self.last_signal_time: Dict[str, int] = {}  # product_id -> timestamp
        
        logger.info(f"Initialized strategy: {self.name}")
    
    @abstractmethod
    async def generate_signals(self, 
                               data: Dict[str, pd.DataFrame], 
                               product_id: str) -> List[Signal]:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Dictionary of DataFrames with market data, keyed by timeframe
            product_id: Product identifier
            
        Returns:
            List of generated signals
        """
        pass
    
    async def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess market data before signal generation.
        
        Args:
            data: Dictionary of DataFrames with market data, keyed by timeframe
            
        Returns:
            Preprocessed data
        """
        processed_data = {}
        
        for timeframe, df in data.items():
            # Make a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Ensure index is datetime
            if not isinstance(processed_df.index, pd.DatetimeIndex):
                processed_df.index = pd.to_datetime(processed_df.index)
                
            # Ensure all required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in processed_df.columns:
                    logger.warning(f"Missing required column {col} in {timeframe} data")
            
            # Remove NaN values
            processed_df = processed_df.dropna()
            
            processed_data[timeframe] = processed_df
        
        return processed_data
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators on market data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added indicators
        """
        # Copy the dataframe to avoid modifying the original
        result = df.copy()
        
        # Simple Moving Averages
        for period in [20, 50, 200]:
            result[f'sma_{period}'] = result['close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        result['ema_12'] = result['close'].ewm(span=12, adjust=False).mean()
        result['ema_26'] = result['close'].ewm(span=26, adjust=False).mean()
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result['sma_20'] = result['close'].rolling(window=20).mean()
        result['std_20'] = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['sma_20'] + (result['std_20'] * 2)
        result['bb_lower'] = result['sma_20'] - (result['std_20'] * 2)
        
        # Average True Range (ATR)
        tr1 = abs(result['high'] - result['low'])
        tr2 = abs(result['high'] - result['close'].shift())
        tr3 = abs(result['low'] - result['close'].shift())
        result['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr_14'] = result['tr'].rolling(window=14).mean()
        
        return result
    
    def log_signal(self, signal: Signal) -> None:
        """Log a generated signal."""
        logger.info(f"Signal: {signal.type.value} {signal.product_id} at {signal.price} " +
                    f"[{signal.strategy_name}, conf: {signal.confidence:.2f}]")
        
        # Update metrics
        self.signals_generated += 1
        self.last_signal_time[signal.product_id] = signal.timestamp
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get strategy performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return {
            "strategy_name": self.name,
            "signals_generated": self.signals_generated,
            "last_signal_time": self.last_signal_time,
            "enabled": self.enabled,
            "products": self.products,
            "timeframes": [tf.value for tf in self.timeframes]
        }

class StrategyRegistry:
    """
    Registry for managing and accessing trading strategies.
    
    This class serves as a central registry for all strategies in the system,
    allowing them to be easily accessed by name and managed as a group.
    """
    
    def __init__(self):
        """Initialize strategy registry."""
        self.strategies: Dict[str, Strategy] = {}
        self.enabled_strategies: Dict[str, bool] = {}
        logger.info("Strategy registry initialized")
    
    def register_strategy(self, strategy: Strategy) -> None:
        """
        Register a strategy with the registry.
        
        Args:
            strategy: Strategy instance to register
        """
        strategy_name = strategy.name
        self.strategies[strategy_name] = strategy
        self.enabled_strategies[strategy_name] = strategy.enabled
        logger.info(f"Registered strategy: {strategy_name}")
    
    def get_strategy(self, strategy_name: str) -> Optional[Strategy]:
        """
        Get a strategy by name.
        
        Args:
            strategy_name: Name of the strategy to retrieve
            
        Returns:
            Strategy instance if found, None otherwise
        """
        return self.strategies.get(strategy_name)
    
    def get_all_strategies(self) -> List[Strategy]:
        """
        Get all registered strategies.
        
        Returns:
            List of all strategy instances
        """
        return list(self.strategies.values())
    
    def get_enabled_strategies(self) -> List[Strategy]:
        """
        Get all enabled strategies.
        
        Returns:
            List of enabled strategy instances
        """
        return [s for s in self.strategies.values() if s.enabled]
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """
        Enable a strategy.
        
        Args:
            strategy_name: Name of the strategy to enable
            
        Returns:
            True if strategy was found and enabled, False otherwise
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            self.enabled_strategies[strategy_name] = True
            logger.info(f"Enabled strategy: {strategy_name}")
            return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """
        Disable a strategy.
        
        Args:
            strategy_name: Name of the strategy to disable
            
        Returns:
            True if strategy was found and disabled, False otherwise
        """
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            self.enabled_strategies[strategy_name] = False
            logger.info(f"Disabled strategy: {strategy_name}")
            return True
        return False
    
    def get_strategies_for_product(self, product_id: str) -> List[Strategy]:
        """
        Get strategies that support a specific product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            List of compatible strategy instances
        """
        return [
            s for s in self.strategies.values() 
            if s.enabled and (not s.products or product_id in s.products)
        ]

class SignalCombiner:
    """
    Utility for combining signals from multiple strategies.
    
    This class provides methods for aggregating and filtering signals
    from different strategies to produce a consolidated trading decision.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize signal combiner.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.consensus_threshold = self.config.get("consensus_threshold", 0.6)
        self.strategy_weights = self.config.get("strategy_weights", {})
        logger.info("Signal combiner initialized")
    
    def combine_signals(self, signals: List[Signal]) -> Optional[Signal]:
        """
        Combine multiple signals into a single trading decision.
        
        Args:
            signals: List of signals to combine
            
        Returns:
            Combined signal or None if no consensus
        """
        if not signals:
            return None
            
        # Group signals by type
        signal_groups: Dict[SignalType, List[Signal]] = {}
        for signal in signals:
            if signal.type not in signal_groups:
                signal_groups[signal.type] = []
            signal_groups[signal.type].append(signal)
        
        # Find the most common signal type
        if not signal_groups:
            return None
            
        # Calculate weighted count for each signal type
        weighted_counts = {}
        total_weight = 0
        
        for signal_type, signals_list in signal_groups.items():
            weighted_count = 0
            for signal in signals_list:
                # Get strategy weight (default to 1.0)
                strategy_weight = self.strategy_weights.get(signal.strategy_name, 1.0)
                weighted_count += signal.confidence * strategy_weight
                total_weight += strategy_weight
                
            weighted_counts[signal_type] = weighted_count
        
        # Normalize by total weight
        if total_weight > 0:
            for signal_type in weighted_counts:
                weighted_counts[signal_type] /= total_weight
        
        # Find signal type with highest weight
        if not weighted_counts:
            return None
            
        best_type = max(weighted_counts.items(), key=lambda x: x[1])
        
        # Check if consensus threshold is met
        if best_type[1] < self.consensus_threshold:
            logger.debug(f"No consensus reached. Best: {best_type[0].value} with {best_type[1]:.2f}")
            return None
        
        # Create combined signal using the first signal of the winning type as template
        template_signal = signal_groups[best_type[0]][0]
        
        combined = Signal(
            type=best_type[0],
            product_id=template_signal.product_id,
            price=template_signal.price,
            timestamp=template_signal.timestamp,
            timeframe=template_signal.timeframe,
            strategy_name="combined",
            confidence=best_type[1],
            metadata={
                "contributing_strategies": [s.strategy_name for s in signal_groups[best_type[0]]],
                "consensus_level": best_type[1],
                "signals_count": len(signals)
            }
        )
        
        logger.info(f"Combined signal: {combined.type.value} for {combined.product_id} " +
                    f"with {combined.confidence:.2f} confidence")
        
        return combined
    
    def filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter signals based on confidence and other criteria.
        
        Args:
            signals: List of signals to filter
            
        Returns:
            Filtered list of signals
        """
        # Filter based on minimum confidence
        filtered = [s for s in signals if s.confidence >= self.min_confidence]
        
        # If timeframe filtering is desired, it would be applied here
        
        return filtered 