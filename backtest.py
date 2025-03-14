#!/usr/bin/env python3
"""
Backtesting Module for Cryptocurrency Trading Bot.

This module provides functionality for testing trading strategies using
historical data retrieved via API, calculating performance metrics, and visualizing results.
"""

import logging
import time
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Local imports
from strategy import Strategy, Signal, SignalType
from data_manager import DataManager, TimeFrame, Candle
from risk_manager import RiskManager, PositionSizing

# Get logger
logger = logging.getLogger("Backtest")

@dataclass
class BacktestTrade:
    """Represents a trade in backtesting."""
    entry_time: int
    exit_time: Optional[int]
    product_id: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    fees: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_reason: Optional[str] = None

@dataclass
class BacktestResult:
    """Results of a backtest run."""
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    total_pnl: float
    total_fees: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    avg_win_pnl: float
    avg_loss_pnl: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    start_time: int
    end_time: int
    initial_balance: float
    final_balance: float

class Backtester:
    """
    Backtesting engine for cryptocurrency trading strategies.
    
    Features:
    - Historical data simulation via API
    - Position sizing and risk management
    - Performance metrics calculation
    - Results visualization
    """
    
    def __init__(self,
                data_manager: DataManager,
                strategy: Strategy,
                risk_manager: RiskManager,
                config: Dict[str, Any] = None):
        """
        Initialize backtester.
        
        Args:
            data_manager: Data manager instance
            strategy: Strategy instance to test
            risk_manager: Risk manager instance
            config: Backtesting configuration
        """
        self.data_manager = data_manager
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.config = config or {}
        
        # Set default parameters
        self.initial_balance = self.config.get("initial_balance", 10000.0)  # $10,000 USD
        self.maker_fee = self.config.get("maker_fee", 0.004)  # 0.4%
        self.taker_fee = self.config.get("taker_fee", 0.006)  # 0.6%
        self.use_position_sizing = self.config.get("use_position_sizing", True)
        self.enable_fractional_sizing = self.config.get("enable_fractional_sizing", True)
        self.max_positions = self.config.get("max_positions", 5)
        self.require_profit_factor = self.config.get("require_profit_factor", 0.0)
        self.enable_stop_loss = self.config.get("enable_stop_loss", True)
        self.enable_take_profit = self.config.get("enable_take_profit", True)
        
        # State variables
        self.current_balance = self.initial_balance
        self.open_positions: Dict[str, BacktestTrade] = {}  # product_id -> BacktestTrade
        self.closed_trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[int] = []
        
        # Async loop
        self._loop = None
        
        logger.info(f"Backtester initialized with {strategy.name}, initial balance: ${self.initial_balance:.2f}")
    
    async def run_async(self,
                      product_ids: List[str],
                      start_date: str,
                      end_date: str,
                      timeframe: TimeFrame) -> BacktestResult:
        """
        Run backtest over specified period asynchronously.
        
        Args:
            product_ids: List of product IDs to trade
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            timeframe: Candle timeframe to use
            
        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Convert dates to timestamps
        start_timestamp = int(pd.Timestamp(start_date).timestamp())
        end_timestamp = int(pd.Timestamp(end_date).timestamp())
        
        # Load data for all products
        all_candles: Dict[str, List[Candle]] = {}
        for product_id in product_ids:
            candles = await self.data_manager.fetch_historical_candles(
                product_id=product_id,
                timeframe=timeframe,
                start_time=start_timestamp,
                end_time=end_timestamp
            )
            
            if not candles:
                logger.warning(f"No data found for {product_id}")
                continue
                
            all_candles[product_id] = candles
            logger.info(f"Loaded {len(candles)} candles for {product_id}")
        
        if not all_candles:
            raise ValueError("No data available for backtesting")
        
        # Reset state
        self.current_balance = self.initial_balance
        self.open_positions.clear()
        self.closed_trades.clear()
        self.equity_curve = [self.initial_balance]
        self.timestamps = [start_timestamp]
        
        # Find common timestamps across all products
        common_timestamps = set(c.timestamp for c in all_candles[next(iter(all_candles))])
        for candles in all_candles.values():
            common_timestamps &= set(c.timestamp for c in candles)
        common_timestamps = sorted(common_timestamps)
        
        # Create DataFrames for each product
        dfs = {}
        for product_id, candles in all_candles.items():
            df = self._candles_to_dataframe(candles)
            dfs[product_id] = df
        
        # Main simulation loop
        for timestamp in common_timestamps:
            # Update current candle for each product
            current_candles = {}
            for product_id, df in dfs.items():
                if timestamp in df.index:
                    current_candle = df.loc[timestamp]
                    current_candles[product_id] = current_candle
            
            if not current_candles:
                continue
            
            # Check stop losses and take profits
            self._check_exits(current_candles)
            
            # Generate signals
            for product_id, current_candle in current_candles.items():
                if product_id in self.open_positions:
                    continue  # Skip if we already have a position
                    
                # Get signals from strategy
                signals = self.strategy.generate_signals(product_id)
                
                for signal in signals:
                    if len(self.open_positions) >= self.max_positions:
                        break
                        
                    # Calculate position size
                    if self.use_position_sizing:
                        position_sizing = self.risk_manager.calculate_position_size(
                            signal=signal,
                            balance=self.current_balance,
                            available_balance=self.current_balance
                        )
                    else:
                        # Use fixed position size of 10% of balance
                        position_value = self.current_balance * 0.1
                        position_size = position_value / current_candle['close']
                        position_sizing = PositionSizing(
                            position_size=position_size,
                            position_value=position_value,
                            risk_amount=position_value * 0.02,  # 2% risk
                            risk_percent=0.02,
                            stop_loss_price=None,
                            take_profit_price=None
                        )
                    
                    # Open trade
                    if position_sizing.position_size > 0:
                        self._open_trade(
                            timestamp=timestamp,
                            product_id=product_id,
                            signal=signal,
                            position_sizing=position_sizing,
                            current_price=current_candle['close']
                        )
            
            # Update equity curve
            total_value = self.current_balance
            for position in self.open_positions.values():
                current_price = current_candles[position.product_id]['close']
                position_value = position.size * current_price
                if position.side == "buy":
                    total_value += position_value - (position.size * position.entry_price)
                else:
                    total_value += (position.size * position.entry_price) - position_value
            
            self.equity_curve.append(total_value)
            self.timestamps.append(timestamp)
        
        # Close any remaining positions at the end
        final_candles = {product_id: df.iloc[-1] for product_id, df in dfs.items()}
        self._close_all_positions(final_candles, "backtest_end")
        
        # Calculate performance metrics
        result = self._calculate_performance_metrics(
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )
        
        logger.info(f"Backtest completed: {len(self.closed_trades)} trades, " +
                   f"PnL: ${result.total_pnl:.2f} ({(result.final_balance/self.initial_balance - 1)*100:.1f}%)")
        
        return result
    
    def run(self,
           product_ids: List[str],
           start_date: str,
           end_date: str,
           timeframe: TimeFrame) -> BacktestResult:
        """
        Run backtest over specified period (synchronous wrapper).
        
        Args:
            product_ids: List of product IDs to trade
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            timeframe: Candle timeframe to use
            
        Returns:
            BacktestResult with performance metrics
        """
        # Create an event loop or use the existing one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        self._loop = loop
        
        # Run the async method in the event loop
        return loop.run_until_complete(self.run_async(
            product_ids=product_ids,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe
        ))
    
    def _candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """Convert list of candles to pandas DataFrame."""
        data = {
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": []
        }
        
        timestamps = []
        
        for candle in candles:
            timestamps.append(candle.timestamp)
            data["open"].append(candle.open)
            data["high"].append(candle.high)
            data["low"].append(candle.low)
            data["close"].append(candle.close)
            data["volume"].append(candle.volume)
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    def _open_trade(self,
                   timestamp: int,
                   product_id: str,
                   signal: Signal,
                   position_sizing: PositionSizing,
                   current_price: float) -> None:
        """Open a new trade."""
        # Calculate fees
        fee_rate = self.taker_fee  # Assume taker fee for backtesting
        fee_amount = position_sizing.position_value * fee_rate
        
        # Create trade
        trade = BacktestTrade(
            entry_time=timestamp,
            exit_time=None,
            product_id=product_id,
            side=signal.type.value,
            entry_price=current_price,
            exit_price=None,
            size=position_sizing.position_size,
            fees=fee_amount,
            stop_loss=position_sizing.stop_loss_price,
            take_profit=position_sizing.take_profit_price
        )
        
        # Update balance
        self.current_balance -= fee_amount
        
        # Add to open positions
        self.open_positions[product_id] = trade
        
        logger.debug(f"Opened {trade.side} trade: {product_id} {trade.size:.6f} @ {current_price:.2f}")
    
    def _close_trade(self,
                    product_id: str,
                    current_price: float,
                    reason: str) -> None:
        """Close an open trade."""
        if product_id not in self.open_positions:
            return
            
        trade = self.open_positions[product_id]
        
        # Calculate fees
        fee_rate = self.taker_fee  # Assume taker fee for backtesting
        exit_value = trade.size * current_price
        fee_amount = exit_value * fee_rate
        
        # Calculate P&L
        if trade.side == "buy":
            trade.pnl = (current_price - trade.entry_price) * trade.size - trade.fees - fee_amount
            trade.pnl_percent = (current_price / trade.entry_price - 1) * 100
        else:
            trade.pnl = (trade.entry_price - current_price) * trade.size - trade.fees - fee_amount
            trade.pnl_percent = (trade.entry_price / current_price - 1) * 100
        
        # Update trade
        trade.exit_time = int(time.time())
        trade.exit_price = current_price
        trade.fees += fee_amount
        trade.exit_reason = reason
        
        # Update balance
        self.current_balance += exit_value - fee_amount
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_positions[product_id]
        
        logger.debug(f"Closed {trade.side} trade: {product_id} @ {current_price:.2f}, " +
                    f"PnL: ${trade.pnl:.2f} ({trade.pnl_percent:.1f}%), reason: {reason}")
    
    def _close_all_positions(self, current_candles: Dict[str, pd.Series], reason: str) -> None:
        """Close all open positions."""
        for product_id in list(self.open_positions.keys()):
            if product_id in current_candles:
                self._close_trade(
                    product_id=product_id,
                    current_price=current_candles[product_id]['close'],
                    reason=reason
                )
    
    def _check_exits(self, current_candles: Dict[str, pd.Series]) -> None:
        """Check for stop loss and take profit exits."""
        for product_id in list(self.open_positions.keys()):
            if product_id not in current_candles:
                continue
                
            trade = self.open_positions[product_id]
            current_price = current_candles[product_id]['close']
            
            # Check stop loss
            if self.enable_stop_loss and trade.stop_loss is not None:
                if trade.side == "buy" and current_price <= trade.stop_loss:
                    self._close_trade(product_id, current_price, "stop_loss")
                    continue
                elif trade.side == "sell" and current_price >= trade.stop_loss:
                    self._close_trade(product_id, current_price, "stop_loss")
                    continue
            
            # Check take profit
            if self.enable_take_profit and trade.take_profit is not None:
                if trade.side == "buy" and current_price >= trade.take_profit:
                    self._close_trade(product_id, current_price, "take_profit")
                    continue
                elif trade.side == "sell" and current_price <= trade.take_profit:
                    self._close_trade(product_id, current_price, "take_profit")
                    continue
    
    def _calculate_performance_metrics(self,
                                    start_timestamp: int,
                                    end_timestamp: int) -> BacktestResult:
        """Calculate performance metrics from backtest results."""
        if not self.closed_trades:
            return BacktestResult(
                trades=[],
                equity_curve=pd.Series(self.equity_curve, index=self.timestamps),
                drawdown_curve=pd.Series(0, index=self.timestamps),
                total_pnl=0.0,
                total_fees=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_trade_pnl=0.0,
                avg_win_pnl=0.0,
                avg_loss_pnl=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                avg_trade_duration=0.0,
                start_time=start_timestamp,
                end_time=end_timestamp,
                initial_balance=self.initial_balance,
                final_balance=self.current_balance
            )
        
        # Basic metrics
        total_pnl = sum(t.pnl for t in self.closed_trades)
        total_fees = sum(t.fees for t in self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.closed_trades) if self.closed_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade metrics
        avg_trade_pnl = total_pnl / len(self.closed_trades)
        avg_win_pnl = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss_pnl = -gross_loss / len(losing_trades) if losing_trades else 0
        largest_win = max((t.pnl for t in winning_trades), default=0)
        largest_loss = min((t.pnl for t in losing_trades), default=0)
        
        # Trade duration
        durations = [(t.exit_time - t.entry_time) for t in self.closed_trades]
        avg_trade_duration = sum(durations) / len(durations) if durations else 0
        
        # Convert equity curve to pandas Series
        equity_series = pd.Series(self.equity_curve, index=self.timestamps)
        
        # Calculate drawdown
        rolling_max = equity_series.expanding().max()
        drawdown_series = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown_series.min())
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        # Calculate Sortino ratio (using 0 as minimum acceptable return)
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 1 else 0
        
        return BacktestResult(
            trades=self.closed_trades,
            equity_curve=equity_series,
            drawdown_curve=drawdown_series,
            total_pnl=total_pnl,
            total_fees=total_fees,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_trades=len(self.closed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_pnl=avg_trade_pnl,
            avg_win_pnl=avg_win_pnl,
            avg_loss_pnl=avg_loss_pnl,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            start_time=start_timestamp,
            end_time=end_timestamp,
            initial_balance=self.initial_balance,
            final_balance=self.current_balance
        )
    
    def plot_results(self, result: BacktestResult, show_drawdown: bool = True) -> None:
        """
        Plot backtest results.
        
        Args:
            result: BacktestResult to plot
            show_drawdown: Whether to show drawdown subplot
        """
        if show_drawdown:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot equity curve
        ax1.plot(result.equity_curve.index, result.equity_curve.values, label='Equity')
        ax1.set_title('Backtest Results')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        if show_drawdown:
            # Plot drawdown
            ax2.fill_between(result.drawdown_curve.index, 
                           result.drawdown_curve.values * 100, 
                           0, 
                           color='red', 
                           alpha=0.3, 
                           label='Drawdown')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True)
            ax2.legend()
            
            # Format x-axis dates
            ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self, result: BacktestResult) -> None:
        """
        Print backtest results in a formatted way.
        
        Args:
            result: BacktestResult to print
        """
        print("\n=== Backtest Results ===")
        print(f"Period: {datetime.datetime.fromtimestamp(result.start_time)} to {datetime.datetime.fromtimestamp(result.end_time)}")
        print(f"Initial Balance: ${result.initial_balance:,.2f}")
        print(f"Final Balance: ${result.final_balance:,.2f}")
        print(f"Total Return: {((result.final_balance/result.initial_balance - 1) * 100):,.2f}%")
        print(f"Total PnL: ${result.total_pnl:,.2f}")
        print(f"Total Fees: ${result.total_fees:,.2f}")
        print(f"\nTrade Statistics:")
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate*100:.1f}%")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Average Trade PnL: ${result.avg_trade_pnl:,.2f}")
        print(f"Average Winner: ${result.avg_win_pnl:,.2f}")
        print(f"Average Loser: ${result.avg_loss_pnl:,.2f}")
        print(f"Largest Winner: ${result.largest_win:,.2f}")
        print(f"Largest Loser: ${result.largest_loss:,.2f}")
        print(f"Average Trade Duration: {datetime.timedelta(seconds=int(result.avg_trade_duration))}")
        print(f"\nRisk Metrics:")
        print(f"Maximum Drawdown: {result.max_drawdown*100:.1f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        
    def get_trade_analysis(self, result: BacktestResult) -> pd.DataFrame:
        """
        Get detailed trade analysis as DataFrame.
        
        Args:
            result: BacktestResult to analyze
            
        Returns:
            DataFrame with trade analysis
        """
        if not result.trades:
            return pd.DataFrame()
            
        # Create DataFrame from trades
        trades_data = []
        for trade in result.trades:
            trades_data.append({
                'entry_time': pd.Timestamp(trade.entry_time, unit='s'),
                'exit_time': pd.Timestamp(trade.exit_time, unit='s'),
                'product_id': trade.product_id,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'fees': trade.fees,
                'duration': pd.Timedelta(seconds=trade.exit_time - trade.entry_time),
                'exit_reason': trade.exit_reason
            })
        
        df = pd.DataFrame(trades_data)
        
        # Add derived metrics
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['drawdown'] = df['cumulative_pnl'] - df['cumulative_pnl'].expanding().max()
        
        return df 