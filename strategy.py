#!/usr/bin/env python3
"""
Trading Strategy Module for Cryptocurrency Trading Bot.

This module provides a framework for implementing various trading strategies,
including technical analysis, machine learning, and order book-based approaches.
"""

import logging
import time
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum
import ta  # Technical Analysis library
from dataclasses import dataclass

# Local imports
from data_manager import Candle, Trade, OrderBook, TimeFrame, DataManager

# Get logger
logger = logging.getLogger("Strategy")

class SignalType(Enum):
    """Type of trading signal."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"  # Close existing position

@dataclass
class Signal:
    """Trading signal with metadata."""
    type: SignalType
    product_id: str
    timestamp: int
    price: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "product_id": self.product_id,
            "timestamp": self.timestamp,
            "price": self.price,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"Signal({self.type.value.upper()}, {self.product_id}, price={self.price:.2f}, confidence={self.confidence:.2f})"

class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategy implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, 
                data_manager: DataManager,
                config: Dict[str, Any] = None):
        """
        Initialize strategy.
        
        Args:
            data_manager: Data manager instance for market data
            config: Strategy configuration parameters
        """
        self.data_manager = data_manager
        self.config = config or {}
        self.name = self.__class__.__name__
        
        # Strategy state
        self.is_running = False
        self.last_update_time = 0
        self.current_signals: Dict[str, Signal] = {}  # Current active signals by product_id
        self.performance_metrics: Dict[str, float] = {}  # Performance metrics
        
        logger.info(f"Initialized strategy: {self.name}")
    
    @abstractmethod
    def generate_signals(self, product_id: str) -> List[Signal]:
        """
        Generate trading signals for a product.
        
        This is the main method to implement in strategy subclasses.
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            List of Signal objects
        """
        pass
    
    def update(self, product_id: str) -> List[Signal]:
        """
        Update strategy and generate signals.
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            List of Signal objects
        """
        try:
            # Record update time
            self.last_update_time = time.time()
            
            # Get signals from strategy implementation
            signals = self.generate_signals(product_id)
            
            # Store current signals
            for signal in signals:
                self.current_signals[product_id] = signal
                
            # Log signals
            for signal in signals:
                logger.info(f"Generated signal: {signal}")
                
            return signals
        except Exception as e:
            logger.error(f"Error updating strategy {self.name} for {product_id}: {e}", exc_info=True)
            return []
    
    def get_current_signal(self, product_id: str) -> Optional[Signal]:
        """Get current active signal for a product."""
        return self.current_signals.get(product_id)
    
    def start(self) -> None:
        """Start the strategy."""
        self.is_running = True
        logger.info(f"Started strategy: {self.name}")
    
    def stop(self) -> None:
        """Stop the strategy."""
        self.is_running = False
        logger.info(f"Stopped strategy: {self.name}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics."""
        return self.performance_metrics
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update strategy performance metrics."""
        self.performance_metrics.update(metrics)
        logger.debug(f"Updated performance metrics for {self.name}: {metrics}")
    
    def get_required_timeframes(self) -> List[TimeFrame]:
        """
        Get timeframes required by this strategy.
        
        Override this in subclasses if the strategy needs specific timeframes.
        
        Returns:
            List of TimeFrame enums
        """
        return [TimeFrame.MINUTE_1]  # Default to 1-minute candles
    
    def get_required_indicators(self) -> List[str]:
        """
        Get technical indicators required by this strategy.
        
        Override this in subclasses to specify required indicators.
        
        Returns:
            List of indicator names
        """
        return []  # No indicators by default

class MovingAverageCrossStrategy(Strategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Generates buy signals when fast MA crosses above slow MA,
    and sell signals when fast MA crosses below slow MA.
    """
    
    def __init__(self, 
                data_manager: DataManager,
                config: Dict[str, Any] = None):
        """
        Initialize strategy.
        
        Args:
            data_manager: Data manager instance
            config: Strategy configuration with parameters:
                - fast_ma_period: Fast moving average period
                - slow_ma_period: Slow moving average period
                - timeframe: Candle timeframe
        """
        super().__init__(data_manager, config)
        
        # Set default parameters if not provided
        self.fast_ma_period = self.config.get("fast_ma_period", 10)
        self.slow_ma_period = self.config.get("slow_ma_period", 30)
        self.timeframe = self.config.get("timeframe", TimeFrame.MINUTE_5)
        
        # Validate parameters
        if self.fast_ma_period >= self.slow_ma_period:
            logger.warning(f"Fast MA period {self.fast_ma_period} should be less than slow MA period {self.slow_ma_period}")
            
        logger.info(f"Initialized {self.name} with fast_ma={self.fast_ma_period}, slow_ma={self.slow_ma_period}, timeframe={self.timeframe}")
    
    def generate_signals(self, product_id: str) -> List[Signal]:
        """
        Generate signals based on moving average crossover.
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            List of Signal objects
        """
        # Get candles
        candles = self.data_manager.load_candles(
            product_id=product_id,
            timeframe=self.timeframe
        )
        
        if len(candles) < self.slow_ma_period + 2:
            logger.warning(f"Not enough candles for {product_id} to generate signals. Need at least {self.slow_ma_period + 2}")
            return []
        
        # Convert to DataFrame
        df = self.data_manager.candles_to_dataframe(candles)
        
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
        
        # Get the last two complete rows for calculating crossover
        if len(df) < 2:
            return []
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        signals = []
        
        # Check for crossover
        if pd.notna(last_row['fast_ma']) and pd.notna(last_row['slow_ma']) and \
           pd.notna(prev_row['fast_ma']) and pd.notna(prev_row['slow_ma']):
            
            # Bullish crossover: fast MA crosses above slow MA
            if prev_row['fast_ma'] <= prev_row['slow_ma'] and last_row['fast_ma'] > last_row['slow_ma']:
                signals.append(Signal(
                    type=SignalType.BUY,
                    product_id=product_id,
                    timestamp=int(time.time()),
                    price=last_row['close'],
                    confidence=0.7,
                    metadata={
                        "fast_ma": last_row['fast_ma'],
                        "slow_ma": last_row['slow_ma']
                    }
                ))
                
            # Bearish crossover: fast MA crosses below slow MA
            elif prev_row['fast_ma'] >= prev_row['slow_ma'] and last_row['fast_ma'] < last_row['slow_ma']:
                signals.append(Signal(
                    type=SignalType.SELL,
                    product_id=product_id,
                    timestamp=int(time.time()),
                    price=last_row['close'],
                    confidence=0.7,
                    metadata={
                        "fast_ma": last_row['fast_ma'],
                        "slow_ma": last_row['slow_ma']
                    }
                ))
        
        return signals
    
    def get_required_timeframes(self) -> List[TimeFrame]:
        """Get required timeframes."""
        return [self.timeframe]

class RSIStrategy(Strategy):
    """
    Relative Strength Index Strategy.
    
    Generates buy signals when RSI is below oversold threshold,
    and sell signals when RSI is above overbought threshold.
    """
    
    def __init__(self, 
                data_manager: DataManager,
                config: Dict[str, Any] = None):
        """
        Initialize strategy.
        
        Args:
            data_manager: Data manager instance
            config: Strategy configuration with parameters:
                - rsi_period: RSI calculation period
                - oversold_threshold: Threshold for oversold condition
                - overbought_threshold: Threshold for overbought condition
                - timeframe: Candle timeframe
        """
        super().__init__(data_manager, config)
        
        # Set default parameters if not provided
        self.rsi_period = self.config.get("rsi_period", 14)
        self.oversold_threshold = self.config.get("oversold_threshold", 30)
        self.overbought_threshold = self.config.get("overbought_threshold", 70)
        self.timeframe = self.config.get("timeframe", TimeFrame.MINUTE_15)
        
        logger.info(f"Initialized {self.name} with rsi_period={self.rsi_period}, " +
                   f"oversold={self.oversold_threshold}, overbought={self.overbought_threshold}, " +
                   f"timeframe={self.timeframe}")
    
    def generate_signals(self, product_id: str) -> List[Signal]:
        """
        Generate signals based on RSI.
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            List of Signal objects
        """
        # Get candles
        candles = self.data_manager.load_candles(
            product_id=product_id,
            timeframe=self.timeframe
        )
        
        if len(candles) < self.rsi_period + 2:
            logger.warning(f"Not enough candles for {product_id} to generate RSI signals. Need at least {self.rsi_period + 2}")
            return []
        
        # Convert to DataFrame
        df = self.data_manager.candles_to_dataframe(candles)
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(close=df['close'], window=self.rsi_period)
        df['rsi'] = rsi.rsi()
        
        # Get the last two complete rows for calculating crossover
        if len(df) < 2:
            return []
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        signals = []
        
        # Check RSI conditions
        if pd.notna(last_row['rsi']) and pd.notna(prev_row['rsi']):
            
            # RSI crosses below oversold threshold
            if prev_row['rsi'] >= self.oversold_threshold and last_row['rsi'] < self.oversold_threshold:
                signals.append(Signal(
                    type=SignalType.BUY,
                    product_id=product_id,
                    timestamp=int(time.time()),
                    price=last_row['close'],
                    confidence=0.6,
                    metadata={
                        "rsi": last_row['rsi'],
                        "threshold": self.oversold_threshold
                    }
                ))
                
            # RSI crosses above overbought threshold
            elif prev_row['rsi'] <= self.overbought_threshold and last_row['rsi'] > self.overbought_threshold:
                signals.append(Signal(
                    type=SignalType.SELL,
                    product_id=product_id,
                    timestamp=int(time.time()),
                    price=last_row['close'],
                    confidence=0.6,
                    metadata={
                        "rsi": last_row['rsi'],
                        "threshold": self.overbought_threshold
                    }
                ))
        
        return signals
    
    def get_required_timeframes(self) -> List[TimeFrame]:
        """Get required timeframes."""
        return [self.timeframe]
    
    def get_required_indicators(self) -> List[str]:
        """Get required indicators."""
        return ["rsi"]

class VolumePriceStrategy(Strategy):
    """
    Volume-Price Strategy.
    
    Looks for price increases accompanied by high volume.
    """
    
    def __init__(self, 
                data_manager: DataManager,
                config: Dict[str, Any] = None):
        """
        Initialize strategy.
        
        Args:
            data_manager: Data manager instance
            config: Strategy configuration with parameters:
                - volume_threshold: Volume threshold multiplier
                - price_change_threshold: Minimum price change percentage
                - lookback_periods: Number of periods to look back
                - timeframe: Candle timeframe
        """
        super().__init__(data_manager, config)
        
        # Set default parameters if not provided
        self.volume_threshold = self.config.get("volume_threshold", 2.0)  # 2x average volume
        self.price_change_threshold = self.config.get("price_change_threshold", 0.01)  # 1% price change
        self.lookback_periods = self.config.get("lookback_periods", 24)
        self.timeframe = self.config.get("timeframe", TimeFrame.HOUR_1)
        
        logger.info(f"Initialized {self.name} with volume_threshold={self.volume_threshold}, " +
                   f"price_change_threshold={self.price_change_threshold}, lookback={self.lookback_periods}")
    
    def generate_signals(self, product_id: str) -> List[Signal]:
        """
        Generate signals based on volume-price analysis.
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            List of Signal objects
        """
        # Get candles
        candles = self.data_manager.load_candles(
            product_id=product_id,
            timeframe=self.timeframe
        )
        
        if len(candles) < self.lookback_periods + 2:
            logger.warning(f"Not enough candles for {product_id} to generate volume-price signals. Need at least {self.lookback_periods + 2}")
            return []
        
        # Convert to DataFrame
        df = self.data_manager.candles_to_dataframe(candles)
        
        # Calculate average volume over lookback period
        df['avg_volume'] = df['volume'].rolling(window=self.lookback_periods).mean()
        
        # Calculate price change percentage
        df['price_change_pct'] = df['close'].pct_change()
        
        # Get the last row
        last_row = df.iloc[-1]
        
        signals = []
        
        # Check volume-price conditions
        if pd.notna(last_row['avg_volume']) and pd.notna(last_row['price_change_pct']):
            
            # Volume spike with positive price change
            if (last_row['volume'] > last_row['avg_volume'] * self.volume_threshold and 
                last_row['price_change_pct'] > self.price_change_threshold):
                
                signals.append(Signal(
                    type=SignalType.BUY,
                    product_id=product_id,
                    timestamp=int(time.time()),
                    price=last_row['close'],
                    confidence=0.65,
                    metadata={
                        "volume": last_row['volume'],
                        "avg_volume": last_row['avg_volume'],
                        "volume_ratio": last_row['volume'] / last_row['avg_volume'],
                        "price_change_pct": last_row['price_change_pct']
                    }
                ))
            
            # Volume spike with negative price change
            elif (last_row['volume'] > last_row['avg_volume'] * self.volume_threshold and 
                  last_row['price_change_pct'] < -self.price_change_threshold):
                
                signals.append(Signal(
                    type=SignalType.SELL,
                    product_id=product_id,
                    timestamp=int(time.time()),
                    price=last_row['close'],
                    confidence=0.65,
                    metadata={
                        "volume": last_row['volume'],
                        "avg_volume": last_row['avg_volume'],
                        "volume_ratio": last_row['volume'] / last_row['avg_volume'],
                        "price_change_pct": last_row['price_change_pct']
                    }
                ))
        
        return signals
    
    def get_required_timeframes(self) -> List[TimeFrame]:
        """Get required timeframes."""
        return [self.timeframe]

class OrderBookStrategy(Strategy):
    """
    Order Book Analysis Strategy.
    
    Analyzes order book to detect imbalances between buy and sell orders.
    """
    
    def __init__(self, 
                data_manager: DataManager,
                config: Dict[str, Any] = None):
        """
        Initialize strategy.
        
        Args:
            data_manager: Data manager instance
            config: Strategy configuration with parameters:
                - imbalance_threshold: Order book imbalance threshold
                - depth: Order book depth to analyze
                - price_levels: Number of price levels to consider
        """
        super().__init__(data_manager, config)
        
        # Set default parameters if not provided
        self.imbalance_threshold = self.config.get("imbalance_threshold", 2.0)  # 2x imbalance between bids and asks
        self.depth = self.config.get("depth", 2)  # Order book depth level
        self.price_levels = self.config.get("price_levels", 10)  # Number of price levels to consider
        
        logger.info(f"Initialized {self.name} with imbalance_threshold={self.imbalance_threshold}, " +
                   f"depth={self.depth}, price_levels={self.price_levels}")
    
    def generate_signals(self, product_id: str) -> List[Signal]:
        """
        Generate signals based on order book analysis.
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            List of Signal objects
        """
        # Fetch current order book
        orderbook = self.data_manager.fetch_current_orderbook(
            product_id=product_id,
            level=self.depth
        )
        
        if not orderbook:
            logger.warning(f"Could not fetch order book for {product_id}")
            return []
        
        # Calculate bid and ask volume for top price levels
        bid_volume = sum(size for _, size in orderbook.bids[:self.price_levels])
        ask_volume = sum(size for _, size in orderbook.asks[:self.price_levels])
        
        # Check for volume imbalance
        signals = []
        
        # Current market price (mid price)
        mid_price = (orderbook.bids[0][0] + orderbook.asks[0][0]) / 2
        
        # More buy orders than sell orders
        if bid_volume > ask_volume * self.imbalance_threshold:
            signals.append(Signal(
                type=SignalType.BUY,
                product_id=product_id,
                timestamp=int(time.time()),
                price=mid_price,
                confidence=0.7,
                metadata={
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "imbalance_ratio": bid_volume / ask_volume if ask_volume > 0 else float('inf')
                }
            ))
        
        # More sell orders than buy orders
        elif ask_volume > bid_volume * self.imbalance_threshold:
            signals.append(Signal(
                type=SignalType.SELL,
                product_id=product_id,
                timestamp=int(time.time()),
                price=mid_price,
                confidence=0.7,
                metadata={
                    "bid_volume": bid_volume,
                    "ask_volume": ask_volume,
                    "imbalance_ratio": ask_volume / bid_volume if bid_volume > 0 else float('inf')
                }
            ))
        
        return signals

class CompositeStrategy(Strategy):
    """
    Composite Strategy combining multiple sub-strategies with weighted signals.
    
    This strategy combines signals from multiple strategies and weights them
    based on their confidence and historical performance.
    """
    
    def __init__(self, 
                data_manager: DataManager,
                strategies: List[Strategy],
                config: Dict[str, Any] = None):
        """
        Initialize composite strategy.
        
        Args:
            data_manager: Data manager instance
            strategies: List of strategy instances to combine
            config: Strategy configuration with parameters:
                - weights: Dictionary of strategy name to weight
                - confidence_threshold: Minimum confidence threshold
        """
        super().__init__(data_manager, config)
        
        self.strategies = strategies
        self.weights = self.config.get("weights", {})
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        
        # Set default weights if not specified
        for strategy in self.strategies:
            if strategy.name not in self.weights:
                self.weights[strategy.name] = 1.0
                
        logger.info(f"Initialized {self.name} with {len(strategies)} sub-strategies")
        logger.debug(f"Strategy weights: {self.weights}")
    
    def generate_signals(self, product_id: str) -> List[Signal]:
        """
        Generate signals by combining sub-strategy signals.
        
        Args:
            product_id: Product identifier (e.g., 'BTC-USD')
            
        Returns:
            List of Signal objects
        """
        all_signals = []
        
        # Collect signals from all sub-strategies
        for strategy in self.strategies:
            signals = strategy.update(product_id)
            all_signals.extend(signals)
        
        if not all_signals:
            return []
        
        # Group signals by type
        buy_signals = [s for s in all_signals if s.type == SignalType.BUY]
        sell_signals = [s for s in all_signals if s.type == SignalType.SELL]
        
        # Calculate weighted confidence for each signal type
        buy_confidence = 0
        sell_confidence = 0
        
        for signal in buy_signals:
            strategy_name = signal.metadata.get("strategy") if signal.metadata else None
            weight = self.weights.get(strategy_name, 1.0) if strategy_name else 1.0
            buy_confidence += signal.confidence * weight
            
        for signal in sell_signals:
            strategy_name = signal.metadata.get("strategy") if signal.metadata else None
            weight = self.weights.get(strategy_name, 1.0) if strategy_name else 1.0
            sell_confidence += signal.confidence * weight
        
        # Normalize confidences
        if buy_signals:
            buy_confidence /= len(buy_signals)
        if sell_signals:
            sell_confidence /= len(sell_signals)
        
        # Generate composite signal
        signals = []
        
        # Get current price
        current_price = None
        if all_signals:
            current_price = all_signals[0].price
        
        # If we have confident signals, generate a composite signal
        if buy_confidence > self.confidence_threshold and buy_confidence > sell_confidence:
            signals.append(Signal(
                type=SignalType.BUY,
                product_id=product_id,
                timestamp=int(time.time()),
                price=current_price,
                confidence=buy_confidence,
                metadata={
                    "buy_confidence": buy_confidence,
                    "sell_confidence": sell_confidence,
                    "num_buy_signals": len(buy_signals),
                    "num_sell_signals": len(sell_signals)
                }
            ))
        elif sell_confidence > self.confidence_threshold and sell_confidence > buy_confidence:
            signals.append(Signal(
                type=SignalType.SELL,
                product_id=product_id,
                timestamp=int(time.time()),
                price=current_price,
                confidence=sell_confidence,
                metadata={
                    "buy_confidence": buy_confidence,
                    "sell_confidence": sell_confidence,
                    "num_buy_signals": len(buy_signals),
                    "num_sell_signals": len(sell_signals)
                }
            ))
        
        return signals
    
    def get_required_timeframes(self) -> List[TimeFrame]:
        """Get all required timeframes from sub-strategies."""
        timeframes = []
        for strategy in self.strategies:
            timeframes.extend(strategy.get_required_timeframes())
        # Remove duplicates
        return list(set(timeframes))
    
    def get_required_indicators(self) -> List[str]:
        """Get all required indicators from sub-strategies."""
        indicators = []
        for strategy in self.strategies:
            indicators.extend(strategy.get_required_indicators())
        # Remove duplicates
        return list(set(indicators))
    
    def start(self) -> None:
        """Start all sub-strategies."""
        super().start()
        for strategy in self.strategies:
            strategy.start()
    
    def stop(self) -> None:
        """Stop all sub-strategies."""
        super().stop()
        for strategy in self.strategies:
            strategy.stop()

class StrategyFactory:
    """Factory for creating strategy instances."""
    
    @staticmethod
    def create_strategy(strategy_name: str, 
                       data_manager: DataManager,
                       config: Dict[str, Any] = None) -> Strategy:
        """
        Create a strategy instance by name.
        
        Args:
            strategy_name: Name of the strategy to create
            data_manager: Data manager instance
            config: Strategy configuration
            
        Returns:
            Strategy instance
        """
        strategy_map = {
            "MovingAverageCross": MovingAverageCrossStrategy,
            "RSI": RSIStrategy,
            "VolumePrice": VolumePriceStrategy,
            "OrderBook": OrderBookStrategy
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unknown strategy: {strategy_name}")
            
        return strategy_map[strategy_name](data_manager, config)
    
    @staticmethod
    def create_composite_strategy(data_manager: DataManager,
                                 strategy_configs: List[Dict[str, Any]],
                                 composite_config: Dict[str, Any] = None) -> CompositeStrategy:
        """
        Create a composite strategy with multiple sub-strategies.
        
        Args:
            data_manager: Data manager instance
            strategy_configs: List of strategy configurations, each with 'name' and 'config'
            composite_config: Configuration for the composite strategy
            
        Returns:
            CompositeStrategy instance
        """
        strategies = []
        
        for strategy_config in strategy_configs:
            strategy_name = strategy_config.get("name")
            config = strategy_config.get("config", {})
            
            if not strategy_name:
                logger.warning("Missing strategy name in config")
                continue
                
            try:
                strategy = StrategyFactory.create_strategy(strategy_name, data_manager, config)
                strategies.append(strategy)
            except ValueError as e:
                logger.error(f"Error creating strategy: {e}")
        
        return CompositeStrategy(data_manager, strategies, composite_config) 