#!/usr/bin/env python3
"""
Espero Trading Bot - Main Application.

This is the main entry point for the trading bot that brings together
all components of the system.
"""

import os
import sys
import json
import logging
import asyncio
import argparse
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set

# Import all components
from data_manager import DataManager
from position_manager import PositionManager, Position, PositionStatus
from order_validator import OrderValidator, OrderValidationResult
from strategy_framework import StrategyRegistry, SignalCombiner, Signal, SignalType, TimeFrame

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
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger("TradingBot")

class TradingBot:
    """
    Main Trading Bot class that orchestrates all components.
    """
    
    def __init__(self, config_path: str, paper_trading: bool = True):
        """
        Initialize the Trading Bot.
        
        Args:
            config_path: Path to the configuration file
            paper_trading: Whether to run in paper trading mode
        """
        self.paper_trading = paper_trading
        self.is_running = False
        self.initialized = False
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_manager = None
        self.position_manager = None
        self.order_validator = None
        self.strategy_registry = StrategyRegistry()
        self.signal_combiner = SignalCombiner(self.config.get("signal_weights", {}))
        
        # Track active subscriptions
        self.active_products = set()
        
        # Performance tracking
        self.start_time = None
        self.trade_count = 0
        self.signal_count = 0
        
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Trading Bot initialized with config from {config_path}")
        logger.info(f"Paper Trading Mode: {paper_trading}")
    
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
            
            # Validate required config fields
            required_fields = ['api_key', 'api_secret', 'products']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in config")
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    async def initialize(self):
        """
        Initialize all components of the trading bot.
        """
        try:
            # Initialize Data Manager
            self.data_manager = DataManager(
                api_key=self.config['api_key'],
                api_secret=self.config['api_secret'],
                environment=self.config.get('environment', 'sandbox')
            )
            
            # Initialize Position Manager
            self.position_manager = PositionManager()
            
            # Load existing positions if any
            positions_file = 'positions.json'
            if os.path.exists(positions_file):
                self.position_manager.load_positions(positions_file)
                logger.info(f"Loaded {len(self.position_manager.positions)} existing positions")
            
            # Initialize Order Validator
            product_rules = await self.data_manager.get_product_rules()
            self.order_validator = OrderValidator(
                product_rules=product_rules,
                risk_params=self.config.get('risk', {})
            )
            
            # Register strategies
            self._register_strategies()
            
            # Set up data callbacks
            self.data_manager.register_candle_callback(self._handle_candle_update)
            self.data_manager.register_trade_callback(self._handle_trade_update)
            
            self.initialized = True
            logger.info("Trading Bot components initialized successfully")
        
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
    
    async def start(self):
        """
        Start the trading bot.
        """
        if not self.initialized:
            await self.initialize()
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info("Starting Trading Bot...")
        
        try:
            # Get products to monitor from config
            products = self.config['products']
            self.active_products = set(products)
            
            # Subscribe to data feeds
            await self.data_manager.subscribe_to_products(products)
            
            # Download historical data for all products
            timeframes = self._get_all_timeframes()
            
            for product_id in products:
                for timeframe in timeframes:
                    await self.data_manager.fetch_historical_data(
                        product_id=product_id, 
                        timeframe=timeframe,
                        start_time=datetime.now() - timedelta(days=30),
                        end_time=datetime.now()
                    )
            
            # Main loop - handle periodic tasks
            while self.is_running:
                await self._run_trading_cycle()
                await asyncio.sleep(self.config.get('cycle_interval', 60))  # Run cycle every minute by default
        
        except Exception as e:
            logger.error(f"Error in trading bot main loop: {e}")
            self.stop()
        
        finally:
            # Clean up
            if self.is_running:
                self.stop()
    
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
    
    async def _run_trading_cycle(self):
        """
        Run a single trading cycle.
        
        This method is called periodically to:
        1. Update positions with current prices
        2. Check for position exits (stop loss, take profit)
        3. Generate and process trading signals
        4. Execute trades based on signals
        """
        # Skip if no products are active
        if not self.active_products:
            return
        
        # Update all positions with latest prices
        for product_id in self.active_products:
            current_price = await self.data_manager.get_current_price(product_id)
            
            if current_price:
                # Update all positions for this product
                positions = self.position_manager.get_positions_by_product(product_id)
                for position in positions:
                    if position.status == PositionStatus.OPEN:
                        position.update_price(current_price)
        
        # Check for position exits
        await self._check_position_exits()
        
        # Generate and process signals
        await self._generate_and_process_signals()
        
        # Save positions
        self.position_manager.save_positions('positions.json')
        
        # Log performance metrics
        running_time = datetime.now() - self.start_time
        hours = running_time.total_seconds() / 3600
        
        if hours > 0:
            signals_per_hour = self.signal_count / hours
            trades_per_hour = self.trade_count / hours
            
            logger.info(f"Performance: Running for {running_time}, " +
                      f"Signals: {self.signal_count} ({signals_per_hour:.1f}/h), " +
                      f"Trades: {self.trade_count} ({trades_per_hour:.1f}/h)")
    
    async def _check_position_exits(self):
        """
        Check if any positions should be closed based on stop loss or take profit.
        """
        positions = self.position_manager.get_positions_by_status(PositionStatus.OPEN)
        
        for position in positions:
            if position.should_close():
                reason = "take profit" if position.current_price >= position.take_profit else "stop loss"
                logger.info(f"Closing position {position.id} due to {reason} trigger")
                
                # Close the position
                position.close()
                self.trade_count += 1
    
    async def _generate_and_process_signals(self):
        """
        Generate trading signals from all strategies and process them.
        """
        all_signals = []
        
        # Process each product
        for product_id in self.active_products:
            # Get market data for all timeframes
            market_data = {}
            
            for timeframe in self._get_all_timeframes():
                df = self.data_manager.get_candles_df(product_id, timeframe)
                if df is not None and not df.empty:
                    market_data[timeframe] = df
            
            # Skip if we don't have enough data
            if not market_data:
                logger.warning(f"Not enough market data for {product_id}, skipping signal generation")
                continue
            
            # Generate signals from all strategies
            product_signals = []
            for strategy in self.strategy_registry.get_all_strategies():
                signals = await strategy.generate_signals(market_data, product_id)
                product_signals.extend(signals)
                self.signal_count += len(signals)
            
            # Add to overall signals
            all_signals.extend(product_signals)
        
        # Process signals if we have any
        if all_signals:
            await self._process_signals(all_signals)
    
    async def _process_signals(self, signals: List[Signal]):
        """
        Process trading signals and execute trades.
        
        Args:
            signals: List of trading signals to process
        """
        # Group signals by product
        signals_by_product = {}
        for signal in signals:
            if signal.product_id not in signals_by_product:
                signals_by_product[signal.product_id] = []
            signals_by_product[signal.product_id].append(signal)
        
        # Process each product's signals
        for product_id, product_signals in signals_by_product.items():
            # Combine signals
            combined_signal = self.signal_combiner.combine_signals(product_signals)
            
            if combined_signal:
                logger.info(f"Combined signal for {product_id}: {combined_signal.type.name} " +
                          f"with confidence {combined_signal.confidence:.2f}")
                
                # Execute trade based on signal
                await self._execute_trade(combined_signal)
    
    async def _execute_trade(self, signal: Signal):
        """
        Execute a trade based on a trading signal.
        
        Args:
            signal: Trading signal to execute
        """
        # Skip if confidence is too low
        min_confidence = self.config.get('min_trade_confidence', 0.65)
        if signal.confidence < min_confidence:
            logger.info(f"Signal confidence {signal.confidence:.2f} below threshold {min_confidence}")
            return
        
        product_id = signal.product_id
        
        # Determine trade size based on account balance and risk settings
        account_balance = await self.data_manager.get_account_balance() if not self.paper_trading else 10000.0
        risk_per_trade = self.config.get('risk', {}).get('risk_per_trade', 0.02)  # Default 2% risk per trade
        max_trade_size = account_balance * risk_per_trade
        
        # Scale size by signal confidence
        confidence_factor = (signal.confidence - min_confidence) / (1 - min_confidence)
        trade_size = max_trade_size * confidence_factor
        
        # Get current price
        current_price = await self.data_manager.get_current_price(product_id)
        if not current_price:
            logger.warning(f"Cannot execute trade: unable to get current price for {product_id}")
            return
        
        try:
            # Prepare order parameters
            if signal.type == SignalType.BUY:
                # Check if we already have an open position
                existing_positions = self.position_manager.get_positions_by_product(product_id)
                open_positions = [p for p in existing_positions if p.status == PositionStatus.OPEN]
                
                # Skip if we already have an open position for this product
                if open_positions and not self.config.get('allow_multiple_positions', False):
                    logger.info(f"Skipping BUY signal: already have an open position for {product_id}")
                    return
                
                # Calculate position size
                position_size = trade_size / current_price
                
                # Validate order
                validation_result = self.order_validator.validate_new_order(
                    product_id=product_id,
                    side='BUY',
                    size=position_size,
                    price=current_price
                )
                
                if not validation_result.is_valid:
                    logger.warning(f"Order validation failed: {validation_result.message}")
                    return
                
                # Use normalized size from validation
                position_size = validation_result.modified_params.get('size', position_size)
                
                # Create position
                position = self.position_manager.create_position(
                    product_id=product_id,
                    entry_price=current_price,
                    size=position_size,
                    stop_loss=current_price * (1 - self.config.get('risk', {}).get('stop_loss_pct', 0.05)),
                    take_profit=current_price * (1 + self.config.get('risk', {}).get('take_profit_pct', 0.1)),
                    metadata={
                        "signal_confidence": signal.confidence,
                        "signal_timeframe": signal.timeframe.value if signal.timeframe else None,
                        "signal_strategies": signal.metadata.get("contributing_strategies", [])
                    }
                )
                
                logger.info(f"Created LONG position: {position.id} for {product_id} at {current_price} " +
                          f"with size {position_size:.6f}")
                self.trade_count += 1
            
            elif signal.type == SignalType.SELL:
                # For now, we only use SELL signals to close existing positions
                # In the future, we could support short selling here
                
                # Find open positions for this product
                positions = self.position_manager.get_positions_by_product(product_id)
                open_positions = [p for p in positions if p.status == PositionStatus.OPEN]
                
                if open_positions:
                    for position in open_positions:
                        # Only close if we're in profit or the signal is very strong
                        in_profit = position.unrealized_pnl > 0
                        strong_signal = signal.confidence > 0.8
                        
                        if in_profit or strong_signal:
                            logger.info(f"Closing position {position.id} due to SELL signal " +
                                      f"(in profit: {in_profit}, strong signal: {strong_signal})")
                            position.close()
                            self.trade_count += 1
                else:
                    logger.info(f"No open positions to close for {product_id}")
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _handle_candle_update(self, product_id: str, candle: Dict[str, Any]):
        """
        Handle candle updates from the data manager.
        
        Args:
            product_id: Product identifier
            candle: Candle data
        """
        # This method is called whenever a new candle is received
        # We don't need to do anything here as data is already stored in the DataManager
        pass
    
    async def _handle_trade_update(self, product_id: str, trade: Dict[str, Any]):
        """
        Handle trade updates from the data manager.
        
        Args:
            product_id: Product identifier
            trade: Trade data
        """
        # Update position prices on each trade
        price = float(trade.get('price', 0))
        if price > 0:
            positions = self.position_manager.get_positions_by_product(product_id)
            for position in positions:
                if position.status == PositionStatus.OPEN:
                    position.update_price(price)
                    
                    # Check for stop loss or take profit triggers
                    if position.should_close():
                        reason = "take profit" if price >= position.take_profit else "stop loss"
                        logger.info(f"Closing position {position.id} due to {reason} trigger at price {price}")
                        position.close()
                        self.trade_count += 1
    
    def stop(self):
        """
        Stop the trading bot.
        """
        logger.info("Stopping Trading Bot...")
        self.is_running = False
        
        # Save positions
        if self.position_manager:
            self.position_manager.save_positions('positions.json')
        
        # Close data manager connections
        if self.data_manager:
            asyncio.create_task(self.data_manager.close())
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """
        Print a summary of the trading bot session.
        """
        if not self.start_time:
            return
        
        running_time = datetime.now() - self.start_time
        
        logger.info("=" * 50)
        logger.info("Trading Bot Session Summary")
        logger.info("=" * 50)
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"End time: {datetime.now()}")
        logger.info(f"Duration: {running_time}")
        logger.info(f"Total signals: {self.signal_count}")
        logger.info(f"Total trades: {self.trade_count}")
        
        if self.position_manager:
            positions = self.position_manager.positions
            open_positions = [p for p in positions if p.status == PositionStatus.OPEN]
            closed_positions = [p for p in positions if p.status == PositionStatus.CLOSED]
            
            logger.info(f"Open positions: {len(open_positions)}")
            logger.info(f"Closed positions: {len(closed_positions)}")
            
            if closed_positions:
                total_pnl = sum(p.realized_pnl for p in closed_positions)
                win_count = sum(1 for p in closed_positions if p.realized_pnl > 0)
                loss_count = sum(1 for p in closed_positions if p.realized_pnl < 0)
                
                win_rate = win_count / len(closed_positions) if closed_positions else 0
                
                logger.info(f"Total P&L: {total_pnl:.2f}")
                logger.info(f"Win rate: {win_rate:.2%} ({win_count}/{len(closed_positions)})")
        
        logger.info("=" * 50)
    
    def _signal_handler(self, sig, frame):
        """
        Handle termination signals.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received termination signal {sig}")
        self.stop()
        sys.exit(0)

async def main():
    """
    Main entry point for the trading bot.
    """
    parser = argparse.ArgumentParser(description='Espero Trading Bot')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file path')
    parser.add_argument('--paper', action='store_true', help='Use paper trading mode')
    parser.add_argument('--live', action='store_true', help='Use live trading mode (not paper)')
    args = parser.parse_args()
    
    # Determine trading mode (paper trading by default)
    paper_trading = not args.live
    
    if not paper_trading:
        logger.warning("LIVE TRADING MODE ENABLED! Real funds will be used.")
        # Add a confirmation prompt for live trading
        confirm = input("Are you sure you want to trade with real funds? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Live trading cancelled. Exiting.")
            return
    
    # Create and start the trading bot
    bot = TradingBot(args.config, paper_trading)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user.")
    finally:
        if bot.is_running:
            bot.stop()

if __name__ == "__main__":
    asyncio.run(main()) 