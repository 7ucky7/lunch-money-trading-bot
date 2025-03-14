#!/usr/bin/env python3
"""
Example Trading Strategies.

This module implements several example trading strategies
using the Strategy Framework.
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from strategy_framework import (
    Strategy,
    Signal,
    SignalType,
    TimeFrame
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ExampleStrategies")

class MACrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy.
    
    This strategy generates signals when a shorter-term moving average
    crosses over a longer-term moving average.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Moving Average Crossover Strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Default configuration
        self.short_period = self.config.get("short_period", 20)
        self.long_period = self.config.get("long_period", 50)
        self.signal_threshold = self.config.get("signal_threshold", 0.0)
        
        logger.info(f"Initialized MA Crossover Strategy: {self.short_period}/{self.long_period}")
    
    async def generate_signals(self, 
                             data: Dict[str, pd.DataFrame], 
                             product_id: str) -> List[Signal]:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: Dictionary of DataFrames with market data, keyed by timeframe
            product_id: Product identifier
            
        Returns:
            List of generated signals
        """
        signals = []
        
        # Process each timeframe
        for timeframe in self.timeframes:
            tf_str = timeframe.value
            
            # Skip if we don't have data for this timeframe
            if tf_str not in data:
                continue
                
            # Get DataFrame for this timeframe
            df = data[tf_str]
            
            # Make sure we have enough data points
            if len(df) < self.long_period + 2:
                logger.warning(f"Not enough data for {product_id} {tf_str}")
                continue
            
            # Calculate indicators
            df = self.calculate_ma_indicators(df)
            
            # Check for crossover signals
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Required data
            current_price = last_row['close']
            timestamp = int(time.time())
            
            # Determine signal confidence based on slope and distance
            slope_short = (last_row[f'sma_{self.short_period}'] - prev_row[f'sma_{self.short_period}']) / prev_row[f'sma_{self.short_period}']
            slope_long = (last_row[f'sma_{self.long_period}'] - prev_row[f'sma_{self.long_period}']) / prev_row[f'sma_{self.long_period}']
            
            # Buy signal: Short MA crosses above Long MA
            if (last_row[f'sma_{self.short_period}'] > last_row[f'sma_{self.long_period}'] and
                prev_row[f'sma_{self.short_period}'] <= prev_row[f'sma_{self.long_period}']):
                
                # Calculate confidence (0.5 to 1.0 based on slope difference)
                confidence = 0.5 + min(0.5, max(0, (slope_short - slope_long) * 10))
                
                # Create buy signal if confidence exceeds threshold
                if confidence > self.signal_threshold:
                    signal = Signal(
                        type=SignalType.BUY,
                        product_id=product_id,
                        price=current_price,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        strategy_name=self.name,
                        confidence=confidence,
                        metadata={
                            "short_ma": last_row[f'sma_{self.short_period}'],
                            "long_ma": last_row[f'sma_{self.long_period}'],
                            "slope_short": slope_short,
                            "slope_long": slope_long
                        }
                    )
                    self.log_signal(signal)
                    signals.append(signal)
            
            # Sell signal: Short MA crosses below Long MA
            elif (last_row[f'sma_{self.short_period}'] < last_row[f'sma_{self.long_period}'] and
                  prev_row[f'sma_{self.short_period}'] >= prev_row[f'sma_{self.long_period}']):
                
                # Calculate confidence (0.5 to 1.0 based on slope difference)
                confidence = 0.5 + min(0.5, max(0, (slope_long - slope_short) * 10))
                
                # Create sell signal if confidence exceeds threshold
                if confidence > self.signal_threshold:
                    signal = Signal(
                        type=SignalType.SELL,
                        product_id=product_id,
                        price=current_price,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        strategy_name=self.name,
                        confidence=confidence,
                        metadata={
                            "short_ma": last_row[f'sma_{self.short_period}'],
                            "long_ma": last_row[f'sma_{self.long_period}'],
                            "slope_short": slope_short,
                            "slope_long": slope_long
                        }
                    )
                    self.log_signal(signal)
                    signals.append(signal)
        
        return signals
    
    def calculate_ma_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving average indicators.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added indicators
        """
        # Copy the dataframe to avoid modifying the original
        result = df.copy()
        
        # Calculate simple moving averages
        result[f'sma_{self.short_period}'] = result['close'].rolling(window=self.short_period).mean()
        result[f'sma_{self.long_period}'] = result['close'].rolling(window=self.long_period).mean()
        
        return result

class RSIStrategy(Strategy):
    """
    Relative Strength Index (RSI) Strategy.
    
    This strategy generates signals based on RSI overbought/oversold conditions
    and divergence between price and RSI.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize RSI Strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Default configuration
        self.rsi_period = self.config.get("rsi_period", 14)
        self.overbought_threshold = self.config.get("overbought_threshold", 70)
        self.oversold_threshold = self.config.get("oversold_threshold", 30)
        self.signal_threshold = self.config.get("signal_threshold", 0.5)
        
        logger.info(f"Initialized RSI Strategy: period={self.rsi_period}, " +
                  f"thresholds={self.oversold_threshold}/{self.overbought_threshold}")
    
    async def generate_signals(self, 
                             data: Dict[str, pd.DataFrame], 
                             product_id: str) -> List[Signal]:
        """
        Generate trading signals based on RSI conditions.
        
        Args:
            data: Dictionary of DataFrames with market data, keyed by timeframe
            product_id: Product identifier
            
        Returns:
            List of generated signals
        """
        signals = []
        
        # Process each timeframe
        for timeframe in self.timeframes:
            tf_str = timeframe.value
            
            # Skip if we don't have data for this timeframe
            if tf_str not in data:
                continue
                
            # Get DataFrame for this timeframe
            df = data[tf_str]
            
            # Make sure we have enough data points
            if len(df) < self.rsi_period + 5:
                logger.warning(f"Not enough data for {product_id} {tf_str}")
                continue
            
            # Calculate indicators
            df = self.calculate_rsi_indicators(df)
            
            # Get last few rows for signal detection
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Required data
            current_price = last_row['close']
            timestamp = int(time.time())
            
            # Check for RSI oversold and starting to rise (buy signal)
            if (prev_row['rsi'] < self.oversold_threshold and 
                last_row['rsi'] > prev_row['rsi']):
                
                # Calculate confidence based on how oversold and strength of reversal
                oversold_factor = (self.oversold_threshold - prev_row['rsi']) / self.oversold_threshold
                reversal_strength = (last_row['rsi'] - prev_row['rsi']) / 5.0  # Normalize to ~0.2 per point
                
                confidence = 0.5 + (oversold_factor * 0.25) + (reversal_strength * 0.25)
                confidence = min(0.95, max(0.4, confidence))  # Clamp between 0.4 and 0.95
                
                # Create buy signal if confidence exceeds threshold
                if confidence > self.signal_threshold:
                    signal = Signal(
                        type=SignalType.BUY,
                        product_id=product_id,
                        price=current_price,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        strategy_name=self.name,
                        confidence=confidence,
                        metadata={
                            "rsi": last_row['rsi'],
                            "prev_rsi": prev_row['rsi'],
                            "oversold_factor": oversold_factor,
                            "reversal_strength": reversal_strength
                        }
                    )
                    self.log_signal(signal)
                    signals.append(signal)
            
            # Check for RSI overbought and starting to fall (sell signal)
            elif (prev_row['rsi'] > self.overbought_threshold and 
                  last_row['rsi'] < prev_row['rsi']):
                
                # Calculate confidence based on how overbought and strength of reversal
                overbought_factor = (prev_row['rsi'] - self.overbought_threshold) / (100 - self.overbought_threshold)
                reversal_strength = (prev_row['rsi'] - last_row['rsi']) / 5.0  # Normalize to ~0.2 per point
                
                confidence = 0.5 + (overbought_factor * 0.25) + (reversal_strength * 0.25)
                confidence = min(0.95, max(0.4, confidence))  # Clamp between 0.4 and 0.95
                
                # Create sell signal if confidence exceeds threshold
                if confidence > self.signal_threshold:
                    signal = Signal(
                        type=SignalType.SELL,
                        product_id=product_id,
                        price=current_price,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        strategy_name=self.name,
                        confidence=confidence,
                        metadata={
                            "rsi": last_row['rsi'],
                            "prev_rsi": prev_row['rsi'],
                            "overbought_factor": overbought_factor,
                            "reversal_strength": reversal_strength
                        }
                    )
                    self.log_signal(signal)
                    signals.append(signal)
        
        return signals
    
    def calculate_rsi_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI indicator.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added RSI indicator
        """
        # Copy the dataframe to avoid modifying the original
        result = df.copy()
        
        # Calculate RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        return result

class BollingerBandsStrategy(Strategy):
    """
    Bollinger Bands Strategy.
    
    This strategy generates signals based on price movements relative to
    Bollinger Bands.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Bollinger Bands Strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Default configuration
        self.period = self.config.get("period", 20)
        self.std_dev = self.config.get("std_dev", 2.0)
        self.signal_threshold = self.config.get("signal_threshold", 0.5)
        
        logger.info(f"Initialized Bollinger Bands Strategy: period={self.period}, " +
                  f"std_dev={self.std_dev}")
    
    async def generate_signals(self, 
                             data: Dict[str, pd.DataFrame], 
                             product_id: str) -> List[Signal]:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: Dictionary of DataFrames with market data, keyed by timeframe
            product_id: Product identifier
            
        Returns:
            List of generated signals
        """
        signals = []
        
        # Process each timeframe
        for timeframe in self.timeframes:
            tf_str = timeframe.value
            
            # Skip if we don't have data for this timeframe
            if tf_str not in data:
                continue
                
            # Get DataFrame for this timeframe
            df = data[tf_str]
            
            # Make sure we have enough data points
            if len(df) < self.period + 5:
                logger.warning(f"Not enough data for {product_id} {tf_str}")
                continue
            
            # Calculate indicators
            df = self.calculate_bb_indicators(df)
            
            # Get last few rows for signal detection
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            
            # Required data
            current_price = last_row['close']
            timestamp = int(time.time())
            
            # Buy signal: Price moves below lower band and starts to rise
            if (prev_row['close'] < prev_row['bb_lower'] and 
                last_row['close'] > prev_row['close'] and
                last_row['close'] < last_row['bb_lower']):
                
                # Calculate confidence based on how far below band and strength of bounce
                band_distance = (prev_row['bb_lower'] - prev_row['close']) / prev_row['bb_lower']
                bounce_strength = (last_row['close'] - prev_row['close']) / prev_row['close']
                
                confidence = 0.5 + (band_distance * 0.25) + (bounce_strength * 0.25)
                confidence = min(0.95, max(0.4, confidence))
                
                # Create buy signal if confidence exceeds threshold
                if confidence > self.signal_threshold:
                    signal = Signal(
                        type=SignalType.BUY,
                        product_id=product_id,
                        price=current_price,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        strategy_name=self.name,
                        confidence=confidence,
                        metadata={
                            "bb_width": last_row['bb_width'],
                            "bb_lower": last_row['bb_lower'],
                            "band_distance": band_distance,
                            "bounce_strength": bounce_strength
                        }
                    )
                    self.log_signal(signal)
                    signals.append(signal)
            
            # Sell signal: Price moves above upper band and starts to fall
            elif (prev_row['close'] > prev_row['bb_upper'] and 
                  last_row['close'] < prev_row['close'] and
                  last_row['close'] > last_row['bb_upper']):
                
                # Calculate confidence based on how far above band and strength of reversal
                band_distance = (prev_row['close'] - prev_row['bb_upper']) / prev_row['bb_upper']
                reversal_strength = (prev_row['close'] - last_row['close']) / prev_row['close']
                
                confidence = 0.5 + (band_distance * 0.25) + (reversal_strength * 0.25)
                confidence = min(0.95, max(0.4, confidence))
                
                # Create sell signal if confidence exceeds threshold
                if confidence > self.signal_threshold:
                    signal = Signal(
                        type=SignalType.SELL,
                        product_id=product_id,
                        price=current_price,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        strategy_name=self.name,
                        confidence=confidence,
                        metadata={
                            "bb_width": last_row['bb_width'],
                            "bb_upper": last_row['bb_upper'],
                            "band_distance": band_distance,
                            "reversal_strength": reversal_strength
                        }
                    )
                    self.log_signal(signal)
                    signals.append(signal)
        
        return signals
    
    def calculate_bb_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands indicators.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added Bollinger Bands indicators
        """
        # Copy the dataframe to avoid modifying the original
        result = df.copy()
        
        # Calculate Bollinger Bands
        result['sma'] = result['close'].rolling(window=self.period).mean()
        result['std'] = result['close'].rolling(window=self.period).std()
        
        result['bb_upper'] = result['sma'] + (result['std'] * self.std_dev)
        result['bb_lower'] = result['sma'] - (result['std'] * self.std_dev)
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['sma']
        
        return result

class MACDStrategy(Strategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy.
    
    This strategy generates signals based on MACD histogram reversals
    and zero-line crossovers.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MACD Strategy.
        
        Args:
            config: Strategy configuration
        """
        super().__init__(config)
        
        # Default configuration
        self.fast_period = self.config.get("fast_period", 12)
        self.slow_period = self.config.get("slow_period", 26)
        self.signal_period = self.config.get("signal_period", 9)
        self.signal_threshold = self.config.get("signal_threshold", 0.5)
        
        logger.info(f"Initialized MACD Strategy: {self.fast_period}/{self.slow_period}/{self.signal_period}")
    
    async def generate_signals(self, 
                             data: Dict[str, pd.DataFrame], 
                             product_id: str) -> List[Signal]:
        """
        Generate trading signals based on MACD.
        
        Args:
            data: Dictionary of DataFrames with market data, keyed by timeframe
            product_id: Product identifier
            
        Returns:
            List of generated signals
        """
        signals = []
        
        # Process each timeframe
        for timeframe in self.timeframes:
            tf_str = timeframe.value
            
            # Skip if we don't have data for this timeframe
            if tf_str not in data:
                continue
                
            # Get DataFrame for this timeframe
            df = data[tf_str]
            
            # Make sure we have enough data points
            if len(df) < self.slow_period + self.signal_period + 5:
                logger.warning(f"Not enough data for {product_id} {tf_str}")
                continue
            
            # Calculate indicators
            df = self.calculate_macd_indicators(df)
            
            # Get last few rows for signal detection
            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]
            two_ago = df.iloc[-3]
            
            # Required data
            current_price = last_row['close']
            timestamp = int(time.time())
            
            # Buy signal 1: MACD Histogram reversal from negative to positive
            if (prev_row['macd_hist'] < 0 and 
                two_ago['macd_hist'] < prev_row['macd_hist'] and
                last_row['macd_hist'] > 0):
                
                # Calculate confidence based on strength of reversal
                reversal_strength = last_row['macd_hist'] - prev_row['macd_hist']
                normalized_strength = min(1.0, reversal_strength / (0.01 * current_price))
                
                confidence = 0.5 + (normalized_strength * 0.4)
                
                # Create buy signal if confidence exceeds threshold
                if confidence > self.signal_threshold:
                    signal = Signal(
                        type=SignalType.BUY,
                        product_id=product_id,
                        price=current_price,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        strategy_name=self.name,
                        confidence=confidence,
                        metadata={
                            "macd": last_row['macd'],
                            "macd_signal": last_row['macd_signal'],
                            "macd_hist": last_row['macd_hist'],
                            "reversal_strength": reversal_strength
                        }
                    )
                    self.log_signal(signal)
                    signals.append(signal)
            
            # Buy signal 2: MACD line crosses above zero
            elif (prev_row['macd'] < 0 and last_row['macd'] > 0):
                confidence = 0.6 + min(0.3, last_row['macd'] / (0.01 * current_price))
                
                signal = Signal(
                    type=SignalType.BUY,
                    product_id=product_id,
                    price=current_price,
                    timestamp=timestamp,
                    timeframe=timeframe,
                    strategy_name=self.name,
                    confidence=confidence,
                    metadata={
                        "macd": last_row['macd'],
                        "macd_signal": last_row['macd_signal'],
                        "macd_hist": last_row['macd_hist'],
                        "zero_cross": True
                    }
                )
                self.log_signal(signal)
                signals.append(signal)
            
            # Sell signal 1: MACD Histogram reversal from positive to negative
            if (prev_row['macd_hist'] > 0 and 
                two_ago['macd_hist'] > prev_row['macd_hist'] and
                last_row['macd_hist'] < 0):
                
                # Calculate confidence based on strength of reversal
                reversal_strength = prev_row['macd_hist'] - last_row['macd_hist']
                normalized_strength = min(1.0, reversal_strength / (0.01 * current_price))
                
                confidence = 0.5 + (normalized_strength * 0.4)
                
                # Create sell signal if confidence exceeds threshold
                if confidence > self.signal_threshold:
                    signal = Signal(
                        type=SignalType.SELL,
                        product_id=product_id,
                        price=current_price,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        strategy_name=self.name,
                        confidence=confidence,
                        metadata={
                            "macd": last_row['macd'],
                            "macd_signal": last_row['macd_signal'],
                            "macd_hist": last_row['macd_hist'],
                            "reversal_strength": reversal_strength
                        }
                    )
                    self.log_signal(signal)
                    signals.append(signal)
            
            # Sell signal 2: MACD line crosses below zero
            elif (prev_row['macd'] > 0 and last_row['macd'] < 0):
                confidence = 0.6 + min(0.3, -last_row['macd'] / (0.01 * current_price))
                
                signal = Signal(
                    type=SignalType.SELL,
                    product_id=product_id,
                    price=current_price,
                    timestamp=timestamp,
                    timeframe=timeframe,
                    strategy_name=self.name,
                    confidence=confidence,
                    metadata={
                        "macd": last_row['macd'],
                        "macd_signal": last_row['macd_signal'],
                        "macd_hist": last_row['macd_hist'],
                        "zero_cross": True
                    }
                )
                self.log_signal(signal)
                signals.append(signal)
        
        return signals
    
    def calculate_macd_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD indicators.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added MACD indicators
        """
        # Copy the dataframe to avoid modifying the original
        result = df.copy()
        
        # Calculate MACD
        result['ema_fast'] = result['close'].ewm(span=self.fast_period, adjust=False).mean()
        result['ema_slow'] = result['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        result['macd'] = result['ema_fast'] - result['ema_slow']
        result['macd_signal'] = result['macd'].ewm(span=self.signal_period, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        return result 