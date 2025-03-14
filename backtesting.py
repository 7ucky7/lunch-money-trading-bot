#!/usr/bin/env python3
"""
Backtesting Engine for Espero Trading Bot.

This module provides functionality to backtest trading strategies
using historical data.
"""

import os
import sys
import json
import logging
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from copy import deepcopy
import argparse

# Import components
from data_manager import DataManager, TimeFrame
from position_manager import PositionManager, Position, PositionStatus, PositionType
from strategy_framework import StrategyRegistry, SignalCombiner, Signal, SignalType

# Import strategies
from example_strategies import (
    MACrossoverStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    MACDStrategy
)

# Import ML strategy if available
try:
    from ml_strategy import MLStrategy
    HAS_ML_STRATEGY = True
except ImportError:
    HAS_ML_STRATEGY = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest.log")
    ]
)
logger = logging.getLogger("Backtesting")

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(self):
        """Initialize backtest results."""
        self.initial_balance = 0.0
        self.final_balance = 0.0
        self.total_return = 0.0
        self.total_return_pct = 0.0
        self.annualized_return = 0.0
        self.trades = []
        self.positions = []
        self.daily_returns = []
        self.equity_curve = []
        self.drawdowns = []
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.calmar_ratio = 0.0
        self.trades_per_day = 0.0
        self.avg_trade_duration = timedelta(0)
        self.avg_profit_per_trade = 0.0
        self.avg_loss_per_trade = 0.0
        self.test_duration = timedelta(0)
        self.strategy_metrics = {}
        self.benchmark_return = 0.0
        self.excess_return = 0.0
        
    def calculate_metrics(self) -> None:
        """Calculate performance metrics from trade data."""
        if not self.trades:
            logger.warning("No trades to calculate metrics from")
            return
        
        # Basic metrics
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        self.win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in losing_trades))
        self.profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Calculate average profit and loss
        if winning_trades:
            self.avg_profit_per_trade = total_profit / len(winning_trades)
        if losing_trades:
            self.avg_loss_per_trade = total_loss / len(losing_trades)
        
        # Calculate trade duration
        trade_durations = []
        for trade in self.trades:
            if 'exit_time' in trade and 'entry_time' in trade:
                exit_time = datetime.fromisoformat(trade['exit_time'])
                entry_time = datetime.fromisoformat(trade['entry_time'])
                duration = exit_time - entry_time
                trade_durations.append(duration)
        
        if trade_durations:
            self.avg_trade_duration = sum(trade_durations, timedelta(0)) / len(trade_durations)
        
        # Calculate daily returns and equity curve
        if self.daily_returns:
            # Sharpe ratio (annualized)
            returns_series = pd.Series(self.daily_returns)
            self.sharpe_ratio = np.sqrt(252) * returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0
            
            # Sortino ratio (annualized)
            downside_returns = returns_series[returns_series < 0]
            downside_std = downside_returns.std()
            self.sortino_ratio = np.sqrt(252) * returns_series.mean() / downside_std if downside_std > 0 else 0
            
            # Calmar ratio
            if self.max_drawdown > 0:
                self.calmar_ratio = self.annualized_return / self.max_drawdown
        
        # Calculate trades per day
        if self.test_duration.days > 0:
            self.trades_per_day = len(self.trades) / self.test_duration.days
    
    def print_summary(self) -> None:
        """Print a summary of backtest results."""
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Test Duration: {self.test_duration}")
        logger.info(f"Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"Final Balance: ${self.final_balance:.2f}")
        logger.info(f"Total Return: ${self.total_return:.2f} ({self.total_return_pct:.2f}%)")
        logger.info(f"Annualized Return: {self.annualized_return:.2f}%")
        logger.info(f"Benchmark Return: {self.benchmark_return:.2f}%")
        logger.info(f"Excess Return: {self.excess_return:.2f}%")
        logger.info(f"Maximum Drawdown: {self.max_drawdown:.2f}%")
        logger.info("-" * 60)
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Win Rate: {self.win_rate:.2f}%")
        logger.info(f"Profit Factor: {self.profit_factor:.2f}")
        logger.info(f"Average Profit per Winning Trade: ${self.avg_profit_per_trade:.2f}")
        logger.info(f"Average Loss per Losing Trade: ${self.avg_loss_per_trade:.2f}")
        logger.info(f"Average Trade Duration: {self.avg_trade_duration}")
        logger.info(f"Trades per Day: {self.trades_per_day:.2f}")
        logger.info("-" * 60)
        logger.info(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {self.sortino_ratio:.2f}")
        logger.info(f"Calmar Ratio: {self.calmar_ratio:.2f}")
        logger.info("=" * 60)
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot backtest results.
        
        Args:
            save_path: Path to save the plot to (if None, display instead)
        """
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Convert dates to datetime objects if they are strings
        dates = [datetime.fromisoformat(d) if isinstance(d, str) else d for d in self.equity_curve[0]]
        
        # Plot equity curve
        axs[0].plot(dates, self.equity_curve[1], label='Portfolio Value')
        
        # If we have benchmark data, plot it as well
        if len(self.equity_curve) > 2:
            axs[0].plot(dates, self.equity_curve[2], label='Benchmark', linestyle='--')
        
        axs[0].set_title('Equity Curve')
        axs[0].set_ylabel('Portfolio Value ($)')
        axs[0].grid(True)
        axs[0].legend()
        
        # Plot drawdowns
        if self.drawdowns:
            dd_dates = [datetime.fromisoformat(d) if isinstance(d, str) else d for d in self.drawdowns[0]]
            axs[1].fill_between(dd_dates, 0, self.drawdowns[1], color='red', alpha=0.3)
            axs[1].set_title('Drawdowns')
            axs[1].set_ylabel('Drawdown (%)')
            axs[1].grid(True)
        
        # Plot trade markers on equity curve
        for trade in self.trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                entry_time = datetime.fromisoformat(trade['entry_time'])
                exit_time = datetime.fromisoformat(trade['exit_time'])
                
                # Find closest points in equity curve
                entry_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - entry_time))
                exit_idx = min(range(len(dates)), key=lambda i: abs(dates[i] - exit_time))
                
                # Plot markers
                if trade['pnl'] > 0:
                    axs[0].plot([dates[entry_idx]], [self.equity_curve[1][entry_idx]], 'g^', markersize=8)
                    axs[0].plot([dates[exit_idx]], [self.equity_curve[1][exit_idx]], 'gv', markersize=8)
                else:
                    axs[0].plot([dates[entry_idx]], [self.equity_curve[1][entry_idx]], 'r^', markersize=8)
                    axs[0].plot([dates[exit_idx]], [self.equity_curve[1][exit_idx]], 'rv', markersize=8)
        
        # Plot daily returns
        if self.daily_returns:
            daily_return_dates = dates[1:]  # assuming daily returns start from day 2
            axs[2].bar(daily_return_dates, self.daily_returns, color=['g' if r > 0 else 'r' for r in self.daily_returns])
            axs[2].set_title('Daily Returns')
            axs[2].set_ylabel('Return (%)')
            axs[2].grid(True)
        
        # Format the x-axis to show dates nicely
        fig.autofmt_xdate()
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save backtest results to a file.
        
        Args:
            file_path: Path to save results to
        """
        # Convert data to serializable format
        result_dict = self.__dict__.copy()
        
        # Convert timedelta objects to strings
        result_dict['test_duration'] = str(self.test_duration)
        result_dict['avg_trade_duration'] = str(self.avg_trade_duration)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Backtest results saved to {file_path}")


class BacktestEngine:
    """
    Engine for backtesting trading strategies using historical data.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the backtest engine.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_manager = None
        self.position_manager = PositionManager()
        self.strategy_registry = StrategyRegistry()
        self.signal_combiner = SignalCombiner(self.config.get("signal_weights", {}))
        
        # Backtest parameters
        self.initial_balance = self.config.get("backtest", {}).get("initial_balance", 10000.0)
        self.fee_rate = self.config.get("backtest", {}).get("fee_rate", 0.001)  # 0.1% fee by default
        self.slippage = self.config.get("backtest", {}).get("slippage", 0.0005)  # 0.05% slippage by default
        
        # Backtest state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.benchmark_equity = []
        
        logger.info(f"Backtest Engine initialized with config from {config_path}")
        logger.info(f"Initial balance: ${self.initial_balance:.2f}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary with configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    async def initialize(self):
        """
        Initialize all components of the backtest engine.
        """
        try:
            # Initialize Data Manager
            self.data_manager = DataManager(
                api_key=self.config['api_key'],
                api_secret=self.config['api_secret'],
                environment=self.config.get('environment', 'sandbox')
            )
            
            # Register strategies
            self._register_strategies()
            
            logger.info("Backtest Engine components initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _register_strategies(self):
        """
        Register trading strategies based on configuration.
        """
        strategy_configs = self.config.get('strategies', {})
        
        # Register standard strategies
        if 'ma_crossover' in strategy_configs:
            self.strategy_registry.register(
                MACrossoverStrategy(strategy_configs['ma_crossover'])
            )
        
        if 'rsi' in strategy_configs:
            self.strategy_registry.register(
                RSIStrategy(strategy_configs['rsi'])
            )
        
        if 'bollinger_bands' in strategy_configs:
            self.strategy_registry.register(
                BollingerBandsStrategy(strategy_configs['bollinger_bands'])
            )
        
        if 'macd' in strategy_configs:
            self.strategy_registry.register(
                MACDStrategy(strategy_configs['macd'])
            )
        
        # Register ML strategy if available
        if HAS_ML_STRATEGY and 'ml_strategy' in strategy_configs:
            self.strategy_registry.register(
                MLStrategy(self.data_manager, strategy_configs['ml_strategy'])
            )
        
        # Log registered strategies
        strategies = self.strategy_registry.get_all_strategies()
        logger.info(f"Registered {len(strategies)} strategies: {', '.join([s.name for s in strategies])}")
    
    def _get_all_timeframes(self) -> Set[str]:
        """
        Get all timeframes needed by registered strategies.
        
        Returns:
            Set of timeframe strings
        """
        timeframes = set()
        for strategy in self.strategy_registry.get_all_strategies():
            for tf in strategy.timeframes:
                timeframes.add(tf.value)
        return timeframes
    
    async def load_historical_data(self, 
                                  product_id: str, 
                                  start_date: datetime, 
                                  end_date: datetime) -> bool:
        """
        Load historical data for backtesting.
        
        Args:
            product_id: Product identifier
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            True if data was loaded successfully
        """
        logger.info(f"Loading historical data for {product_id} from {start_date} to {end_date}")
        
        # Get all required timeframes
        timeframes = self._get_all_timeframes()
        
        # Load data for each timeframe
        success = True
        for tf_str in timeframes:
            tf = TimeFrame(tf_str)
            
            try:
                data = await self.data_manager.fetch_historical_data(
                    product_id=product_id, 
                    timeframe=tf,
                    start_time=start_date,
                    end_time=end_date
                )
                
                if data is None or len(data) == 0:
                    logger.warning(f"No historical data available for {product_id} on {tf_str} timeframe")
                    success = False
                else:
                    logger.info(f"Loaded {len(data)} candles for {product_id} on {tf_str} timeframe")
            
            except Exception as e:
                logger.error(f"Error loading historical data for {product_id} on {tf_str}: {e}")
                success = False
        
        return success
    
    async def run_backtest(self, 
                          product_id: str, 
                          start_date: datetime, 
                          end_date: datetime) -> BacktestResult:
        """
        Run a backtest over a specified date range.
        
        Args:
            product_id: Product identifier
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult object with backtest results
        """
        logger.info(f"Starting backtest for {product_id} from {start_date} to {end_date}")
        
        # Initialize components if needed
        if self.data_manager is None:
            await self.initialize()
        
        # Load historical data
        success = await self.load_historical_data(product_id, start_date, end_date)
        if not success:
            logger.error("Failed to load all required historical data")
            return BacktestResult()
        
        # Reset backtest state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.trades = []
        self.equity_curve = [[], []]  # [dates, equity]
        self.benchmark_equity = []
        
        # Get main price DataFrame (1d timeframe)
        df_1d = self.data_manager.get_candles_df(product_id, "1d")
        if df_1d is None or df_1d.empty:
            logger.error("No daily candles available for backtesting")
            return BacktestResult()
        
        # Sort by time
        df_1d = df_1d.sort_values('time')
        
        # Calculate benchmark (buy and hold)
        initial_price = df_1d.iloc[0]['close']
        final_price = df_1d.iloc[-1]['close']
        benchmark_return_pct = (final_price / initial_price - 1) * 100
        
        # Calculate benchmark equity curve
        self.benchmark_equity = [self.initial_balance * (1 + (price / initial_price - 1)) 
                                for price in df_1d['close']]
        
        # Add dates and benchmark to equity curve
        self.equity_curve[0] = df_1d['time'].tolist()
        self.equity_curve.append(self.benchmark_equity)
        
        # Run simulation for each day
        previous_date = None
        daily_returns = []
        previous_equity = self.equity
        drawdowns = [[], []]  # [dates, drawdown_pct]
        max_equity = self.equity
        
        logger.info("Starting day-by-day simulation...")
        
        for i, row in df_1d.iterrows():
            current_date = pd.to_datetime(row['time']).to_pydatetime()
            
            # Skip if before start date
            if current_date < start_date:
                continue
            
            # Stop if after end date
            if current_date > end_date:
                break
            
            # Process this day
            logger.debug(f"Processing date: {current_date.date()}")
            await self._process_day(product_id, current_date)
            
            # Update equity curve
            self.equity_curve[1].append(self.equity)
            
            # Calculate daily return
            if previous_date is not None:
                daily_return = (self.equity / previous_equity - 1) * 100
                daily_returns.append(daily_return)
            
            # Calculate drawdown
            if self.equity > max_equity:
                max_equity = self.equity
            
            if max_equity > 0:
                drawdown_pct = (max_equity - self.equity) / max_equity * 100
                drawdowns[0].append(current_date)
                drawdowns[1].append(drawdown_pct)
            
            # Update previous values
            previous_date = current_date
            previous_equity = self.equity
        
        # Calculate max drawdown
        max_drawdown = max(drawdowns[1]) if drawdowns[1] else 0
        
        # Prepare final result
        result = BacktestResult()
        result.initial_balance = self.initial_balance
        result.final_balance = self.equity
        result.total_return = self.equity - self.initial_balance
        result.total_return_pct = (self.equity / self.initial_balance - 1) * 100
        
        # Calculate annualized return
        days = (end_date - start_date).days
        years = days / 365
        if years > 0:
            result.annualized_return = ((1 + result.total_return_pct / 100) ** (1 / years) - 1) * 100
        
        result.trades = self.trades
        result.positions = self.position_manager.positions
        result.daily_returns = daily_returns
        result.equity_curve = self.equity_curve
        result.drawdowns = drawdowns
        result.max_drawdown = max_drawdown
        result.test_duration = end_date - start_date
        result.benchmark_return = benchmark_return_pct
        result.excess_return = result.total_return_pct - benchmark_return_pct
        
        # Calculate additional metrics
        result.calculate_metrics()
        
        # Print summary
        result.print_summary()
        
        logger.info("Backtest completed successfully")
        return result
    
    async def _process_day(self, product_id: str, date: datetime) -> None:
        """
        Process a single day in the backtest.
        
        Args:
            product_id: Product identifier
            date: Date to process
        """
        # Get market data for all timeframes up to this date
        market_data = {}
        
        for tf_str in self._get_all_timeframes():
            df = self.data_manager.get_candles_df(product_id, tf_str)
            
            # Filter data up to this date
            df_filtered = df[pd.to_datetime(df['time']) <= date].copy()
            
            if not df_filtered.empty:
                market_data[tf_str] = df_filtered
        
        # Skip if we don't have enough data
        if not market_data:
            logger.warning(f"No market data available for {date}, skipping")
            return
        
        # Update positions with today's prices
        daily_candle = self.data_manager.get_candle_for_date(product_id, date, "1d")
        
        if daily_candle is not None:
            await self._update_positions(daily_candle)
        
        # Generate signals from all strategies
        signals = []
        for strategy in self.strategy_registry.get_all_strategies():
            strategy_signals = await strategy.generate_signals(market_data, product_id)
            signals.extend(strategy_signals)
        
        # Process signals if we have any
        if signals:
            combined_signal = self.signal_combiner.combine_signals(signals)
            
            if combined_signal:
                await self._process_signal(combined_signal, daily_candle)
    
    async def _update_positions(self, candle: Dict[str, Any]) -> None:
        """
        Update positions with current prices and check for exits.
        
        Args:
            candle: Candle data for the current day
        """
        # Get prices from candle
        open_price = float(candle['open'])
        high_price = float(candle['high'])
        low_price = float(candle['low'])
        close_price = float(candle['close'])
        
        # Update all open positions
        positions = self.position_manager.get_positions_by_status(PositionStatus.OPEN)
        
        for position in positions:
            # Update with close price
            position.update_price(close_price)
            
            # Check for stop loss hit during the day
            if position.stop_loss is not None:
                if position.stop_loss >= low_price:
                    # Stop loss was hit
                    logger.info(f"Stop loss triggered for position {position.id} at {position.stop_loss}")
                    position.close_price = position.stop_loss
                    position.close()
                    
                    # Update balance
                    self.balance += position.realized_pnl
                    
                    # Record trade
                    self._record_trade(position)
                    continue
            
            # Check for take profit hit during the day
            if position.take_profit is not None:
                if position.take_profit <= high_price:
                    # Take profit was hit
                    logger.info(f"Take profit triggered for position {position.id} at {position.take_profit}")
                    position.close_price = position.take_profit
                    position.close()
                    
                    # Update balance
                    self.balance += position.realized_pnl
                    
                    # Record trade
                    self._record_trade(position)
                    continue
        
        # Update equity with current position values
        self.equity = self.balance
        for position in self.position_manager.get_positions_by_status(PositionStatus.OPEN):
            self.equity += position.unrealized_pnl
    
    async def _process_signal(self, signal: Signal, candle: Dict[str, Any]) -> None:
        """
        Process a trading signal and execute the trade in the backtest.
        
        Args:
            signal: Trading signal to process
            candle: Current day's candle data
        """
        # Skip if confidence is too low
        min_confidence = self.config.get('min_trade_confidence', 0.65)
        if signal.confidence < min_confidence:
            logger.debug(f"Signal confidence {signal.confidence:.2f} below threshold {min_confidence}")
            return
        
        # Get prices from candle
        open_price = float(candle['open'])
        close_price = float(candle['close'])
        
        # Use open price for trade execution (assuming we trade at the open of the day)
        execution_price = open_price
        
        # Apply slippage (use higher price for buys, lower for sells)
        if signal.type == SignalType.BUY:
            execution_price *= (1 + self.slippage)
        else:
            execution_price *= (1 - self.slippage)
        
        # Handle BUY signals
        if signal.type == SignalType.BUY:
            # Calculate position size
            risk_per_trade = self.config.get('risk', {}).get('risk_per_trade', 0.02)  # Default 2% risk per trade
            max_trade_size = self.balance * risk_per_trade
            
            # Scale size by signal confidence
            confidence_factor = (signal.confidence - min_confidence) / (1 - min_confidence)
            trade_value = max_trade_size * confidence_factor
            
            # Skip if trade value is too small
            if trade_value < 10:  # Minimum trade size of $10
                logger.debug(f"Trade value too small: ${trade_value:.2f}")
                return
            
            # Calculate position size in units
            position_size = trade_value / execution_price
            
            # Apply trading fee
            fee = trade_value * self.fee_rate
            
            # Skip if we don't have enough balance
            if self.balance < trade_value + fee:
                logger.debug(f"Insufficient balance for trade: ${self.balance:.2f} < ${trade_value + fee:.2f}")
                return
            
            # Create position
            position = self.position_manager.create_position(
                product_id=signal.product_id,
                entry_price=execution_price,
                size=position_size,
                stop_loss=execution_price * (1 - self.config.get('risk', {}).get('stop_loss_pct', 0.05)),
                take_profit=execution_price * (1 + self.config.get('risk', {}).get('take_profit_pct', 0.1)),
                metadata={
                    "signal_confidence": signal.confidence,
                    "signal_timeframe": signal.timeframe.value if signal.timeframe else None,
                    "signal_strategies": signal.metadata.get("contributing_strategies", []),
                    "fee": fee
                }
            )
            
            # Update balance
            self.balance -= trade_value + fee
            
            logger.info(f"BUY position created: {position.id} at {execution_price:.2f} with size {position_size:.6f}")
        
        # Handle SELL signals
        elif signal.type == SignalType.SELL:
            # Find open positions for this product
            positions = self.position_manager.get_positions_by_product(signal.product_id)
            open_positions = [p for p in positions if p.status == PositionStatus.OPEN]
            
            if open_positions:
                for position in open_positions:
                    # Only close if we're in profit or the signal is very strong
                    in_profit = position.unrealized_pnl > 0
                    strong_signal = signal.confidence > 0.8
                    
                    if in_profit or strong_signal:
                        logger.info(f"Closing position {position.id} due to SELL signal " +
                                  f"(in profit: {in_profit}, strong signal: {strong_signal})")
                        
                        # Apply trading fee
                        fee = position.size * execution_price * self.fee_rate
                        
                        # Close position with current price
                        position.close_price = execution_price
                        position.close()
                        
                        # Update balance
                        self.balance += position.realized_pnl - fee
                        
                        # Record trade
                        self._record_trade(position, fee)
            else:
                logger.debug(f"No open positions to close for {signal.product_id}")
    
    def _record_trade(self, position: Position, exit_fee: float = 0.0) -> None:
        """
        Record a completed trade.
        
        Args:
            position: Closed position
            exit_fee: Fee paid on exit
        """
        # Get entry fee from metadata if available
        entry_fee = position.metadata.get("fee", 0.0)
        total_fee = entry_fee + exit_fee
        
        # Calculate actual PnL after fees
        pnl = position.realized_pnl - total_fee
        
        # Record trade
        trade = {
            "position_id": position.id,
            "product_id": position.product_id,
            "entry_price": position.entry_price,
            "exit_price": position.close_price,
            "size": position.size,
            "entry_time": position.entry_time.isoformat() if position.entry_time else None,
            "exit_time": position.close_time.isoformat() if position.close_time else None,
            "pnl": pnl,
            "pnl_percent": (pnl / (position.size * position.entry_price)) * 100 if position.size * position.entry_price > 0 else 0,
            "fees": total_fee,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "duration": (position.close_time - position.entry_time).total_seconds() if position.close_time and position.entry_time else 0,
            "signal_confidence": position.metadata.get("signal_confidence", 0.0),
            "signal_timeframe": position.metadata.get("signal_timeframe", None),
            "signal_strategies": position.metadata.get("signal_strategies", [])
        }
        
        self.trades.append(trade)
        
        logger.info(f"Recorded trade: {position.product_id} " +
                  f"Entry: {position.entry_price:.2f}, Exit: {position.close_price:.2f}, " +
                  f"PnL: ${pnl:.2f} ({trade['pnl_percent']:.2f}%)")


async def main():
    """
    Main entry point for the backtesting engine.
    """
    parser = argparse.ArgumentParser(description='Espero Trading Bot Backtesting Engine')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--product', type=str, required=True, help='Product to backtest (e.g., BTC-USD)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Generate performance plot')
    parser.add_argument('--save', type=str, help='Save results to file')
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD.")
        return
    
    # Create and run backtest
    engine = BacktestEngine(args.config)
    
    try:
        await engine.initialize()
        result = await engine.run_backtest(args.product, start_date, end_date)
        
        # Generate plot if requested
        if args.plot:
            plot_path = f"backtest_{args.product.replace('-', '_')}_{args.start}_to_{args.end}.png"
            result.plot_results(plot_path)
        
        # Save results if requested
        if args.save:
            result.save_to_file(args.save)
    
    except KeyboardInterrupt:
        logger.info("Backtest stopped by user.")
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main()) 