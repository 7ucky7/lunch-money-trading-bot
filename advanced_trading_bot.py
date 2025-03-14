#!/usr/bin/env python3
"""
Advanced Trading Bot

This script implements a more sophisticated trading strategy using machine learning,
order book analysis, and dynamic position sizing based on market conditions.

DISCLAIMER: Trading cryptocurrencies involves substantial risk of loss
and is not suitable for every investor. Never trade with money you 
cannot afford to lose. This code is provided AS-IS with no guarantees.
"""

import os
import time
import json
import logging
from decimal import Decimal, getcontext
from datetime import datetime
import cbpro
import numpy as np
import asyncio
import requests
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Try to import coinbase_advanced_py, but make it optional
try:
    import coinbase_advanced_py as capy
    ADVANCED_API_AVAILABLE = True
except ImportError:
    ADVANCED_API_AVAILABLE = False
    print("Warning: coinbase_advanced_py not installed. Some features will be disabled.")
    print("To install, run: pip install coinbase_advanced_py")

# Configure precision for decimal calculations
getcontext().prec = 8

# Configure logging
logging.basicConfig(
    filename='advanced_trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AdvancedTradingBot")

# Load environment variables
load_dotenv()

# API credentials should be stored in environment variables for security
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")
API_PASSPHRASE = os.getenv("COINBASE_API_PASSPHRASE")

# Coinbase fee rates
MAKER_FEE = 0.004  # 0.4% maker fee
TAKER_FEE = 0.006  # 0.6% taker fee

# Risk management and strategy settings
STOP_LOSS_PERCENT = 0.005  # 0.5% stop loss
TRAILING_STOP_PERCENT = 0.002  # 0.2% trailing stop
TRADE_FREQUENCY = 1  # Trade frequency in seconds
MIN_SPREAD = 0.008  # Minimum spread
VOLATILITY_WINDOW = 5  # Window for volatility detection
RISK_PERCENT = 0.01  # 1% of balance per trade
KELLY_CRITERION_MULTIPLIER = 0.5  # Conservative Kelly Criterion sizing

# Cooldown settings after multiple losses
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_PERIOD = 60  # Cooldown period in seconds

# Trading pairs
PRODUCT_IDS = ['BTC-USD', 'ETH-USD', 'LTC-USD']

class AdvancedTradingBot:
    """Advanced trading bot with ML and sophisticated strategies."""
    
    def __init__(self):
        """Initialize the trading bot with Coinbase API credentials."""
        # Validate configuration parameters
        self._validate_configuration()
        
        # Initialize production client
        self.auth_client = cbpro.AuthenticatedClient(
            API_KEY, API_SECRET, API_PASSPHRASE
        )
        logger.info("Trading client initialized")
        
        # Initialize advanced client if available
        if ADVANCED_API_AVAILABLE:
            self.advanced_client = capy.Client(API_KEY, API_SECRET, API_PASSPHRASE)
            logger.info("Advanced API client initialized")
        
        # Performance metrics
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = 1000.0  # Initial assumed balance
        self.historical_profits = []
        
        # Time-based profit tracking
        self.daily_profit = 0.0
        self.weekly_profit = 0.0
        self.monthly_profit = 0.0
        self.daily_trades = 0
        self.weekly_trades = 0
        self.monthly_trades = 0
        self.hourly_profit = {}
        
        # Advanced performance metrics
        self.win_loss_ratio = 0.0
        self.average_profit_per_trade = 0.0
        
        # Model and price history storage
        self.price_history = {product_id: [] for product_id in PRODUCT_IDS}
        self.ml_models = {product_id: None for product_id in PRODUCT_IDS}
        
        # State tracking
        self.consecutive_losses = 0
        self.cooldown_until = 0
        self.current_day = datetime.now().day
        self.current_week = datetime.now().isocalendar()[1]
        self.current_month = datetime.now().month
        
        # API rate limiting
        self.api_calls = []
        self.max_calls_per_second = 3  # Adjust based on Coinbase limits
        
        logger.info(f"Bot initialized for trading {', '.join(PRODUCT_IDS)}")
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        # Check risk parameters
        if STOP_LOSS_PERCENT <= 0 or STOP_LOSS_PERCENT > 0.2:
            logger.warning(f"Unusual STOP_LOSS_PERCENT value: {STOP_LOSS_PERCENT}. Should be between 0 and 0.2 (20%)")
        
        if TRAILING_STOP_PERCENT <= 0 or TRAILING_STOP_PERCENT > 0.1:
            logger.warning(f"Unusual TRAILING_STOP_PERCENT value: {TRAILING_STOP_PERCENT}. Should be between 0 and 0.1 (10%)")
        
        if RISK_PERCENT <= 0 or RISK_PERCENT > 0.05:
            logger.warning(f"Unusual RISK_PERCENT value: {RISK_PERCENT}. Should be between 0 and 0.05 (5%)")
        
        if KELLY_CRITERION_MULTIPLIER <= 0 or KELLY_CRITERION_MULTIPLIER > 1:
            logger.warning(f"Unusual KELLY_CRITERION_MULTIPLIER value: {KELLY_CRITERION_MULTIPLIER}. Should be between 0 and 1")
        
        # Check API credentials
        if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
            logger.error("Missing API credentials. Bot will not function correctly.")
    
    def _respect_rate_limit(self):
        """Respect API rate limits to avoid being throttled."""
        now = time.time()
        # Keep only calls from the last second
        self.api_calls = [t for t in self.api_calls if now - t < 1.0]
        
        # If we've made too many calls in the last second, sleep
        if len(self.api_calls) >= self.max_calls_per_second:
            sleep_time = 1.0 - (now - self.api_calls[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit approaching. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Record this call
        self.api_calls.append(time.time())
    
    def check_account_balance(self):
        """Check available balance in the account for each currency."""
        try:
            self._respect_rate_limit()
            accounts = self.auth_client.get_accounts()
            balances = {}
            for account in accounts:
                currency = account['currency']
                balance = float(account['balance'])
                available = float(account['available'])
                balances[currency] = {'balance': balance, 'available': available}
                logger.info(f"Currency: {currency}, Balance: {balance}, Available: {available}")
            return balances
        except Exception as e:
            logger.error(f"Error checking account balance: {e}")
            return None
    
    def get_current_price(self, product_id):
        """Get the current market price for the product."""
        try:
            self._respect_rate_limit()
            ticker = self.auth_client.get_product_ticker(product_id=product_id)
            current_price = float(ticker['price'])
            logger.info(f"Current {product_id} price: ${current_price}")
            
            # Update price history
            self.price_history[product_id].append(current_price)
            if len(self.price_history[product_id]) > 100:  # Keep last 100 price points
                self.price_history[product_id].pop(0)
                
            return current_price
        except Exception as e:
            logger.error(f"Error getting current price for {product_id}: {e}")
            return None
    
    def get_recent_volume(self, product_id):
        """Get recent trading volume for a product."""
        try:
            self._respect_rate_limit()
            response = self.auth_client.get_product_ticker(product_id)
            volume = float(response['volume'])
            logger.info(f"Recent volume for {product_id}: {volume}")
            return volume
        except Exception as e:
            logger.error(f"Failed to fetch volume for {product_id}: {e}")
            return 0
    
    def get_order_book_imbalance(self, product_id):
        """Calculate the imbalance between bids and asks in the order book."""
        try:
            self._respect_rate_limit()
            order_book = self.auth_client.get_product_order_book(product_id, level=2)
            bids = sum(float(bid[1]) for bid in order_book['bids'])
            asks = sum(float(ask[1]) for ask in order_book['asks'])
            imbalance = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
            logger.info(f"Order book imbalance for {product_id}: {imbalance:.4f}")
            return imbalance
        except Exception as e:
            logger.error(f"Failed to fetch order book for {product_id}: {e}")
            return 0
    
    def calculate_position_size(self, balance, volatility):
        """Calculate position size based on account balance and market volatility."""
        # Ensure volatility is not zero to avoid division by zero
        volatility = max(volatility, 0.001)
        position_size = (balance * RISK_PERCENT) / (volatility * KELLY_CRITERION_MULTIPLIER)
        logger.info(f"Calculated position size: ${position_size:.2f}")
        return position_size
    
    def market_is_favorable(self, product_id):
        """Determine if market conditions are favorable for trading."""
        # Need at least VOLATILITY_WINDOW price points to calculate volatility
        if len(self.price_history[product_id]) < VOLATILITY_WINDOW:
            logger.info(f"Not enough price history for {product_id}")
            return False
        
        # Get current price and calculate metrics
        current_price = self.get_current_price(product_id)
        if current_price is None:
            return False
            
        recent_prices = self.price_history[product_id][-VOLATILITY_WINDOW:]
        volatility = np.std(recent_prices)
        moving_average = np.mean(recent_prices)
        price_change = abs(current_price - moving_average) / moving_average
        volume = self.get_recent_volume(product_id)
        order_book_imbalance = self.get_order_book_imbalance(product_id)
        
        # Calculate dynamic threshold based on price history
        if len(recent_prices) >= 4:  # Need at least 4 points for percentiles
            dynamic_imbalance_threshold = np.percentile(recent_prices, 75) - np.percentile(recent_prices, 25)
        else:
            dynamic_imbalance_threshold = MIN_SPREAD
        
        # Check if conditions meet our criteria
        is_favorable = (
            volatility > MIN_SPREAD and
            price_change > 0.0005 and
            volume > 5000 and
            abs(order_book_imbalance) > dynamic_imbalance_threshold
        )
        
        logger.info(f"Market conditions for {product_id}: volatility={volatility:.6f}, "
                   f"price_change={price_change:.6f}, volume={volume}, "
                   f"imbalance={order_book_imbalance:.6f}, favorable={is_favorable}")
        
        return is_favorable
    
    def train_ml_model(self, product_id):
        """Train an XGBoost model on historical price data."""
        if len(self.price_history[product_id]) < 50:
            logger.info(f"Not enough data to train model for {product_id}")
            return False
            
        try:
            # Prepare features and target
            prices = np.array(self.price_history[product_id])
            X = np.array([prices[i:i+5] for i in range(len(prices)-10)])
            y = np.array([prices[i+10] for i in range(len(prices)-10)])
            
            if len(X) < 10:
                logger.info(f"Not enough samples to train model for {product_id}")
                return False
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model.fit(X_train, y_train)
            
            # Store model
            self.ml_models[product_id] = model
            logger.info(f"ML model trained for {product_id}")
            return True
        except Exception as e:
            logger.error(f"Error training ML model for {product_id}: {e}")
            return False
    
    def predict_price_movement(self, product_id):
        """Predict future price movement using the trained ML model."""
        if self.ml_models[product_id] is None:
            logger.info(f"No trained model available for {product_id}")
            return None
            
        if len(self.price_history[product_id]) < 5:
            logger.info(f"Not enough recent price data for {product_id}")
            return None
            
        try:
            # Prepare feature
            recent_prices = np.array(self.price_history[product_id][-5:])
            X_pred = recent_prices.reshape(1, -1)
            
            # Make prediction
            predicted_price = self.ml_models[product_id].predict(X_pred)[0]
            current_price = self.price_history[product_id][-1]
            predicted_change = (predicted_price - current_price) / current_price
            
            logger.info(f"Predicted price for {product_id}: ${predicted_price:.2f} "
                       f"(change: {predicted_change:.4%})")
            
            return predicted_change
        except Exception as e:
            logger.error(f"Error predicting price for {product_id}: {e}")
            return None
    
    def update_performance_metrics(self, profit, product_id):
        """Update all performance metrics after a trade."""
        self.trade_count += 1
        self.total_profit += profit
        self.historical_profits.append(profit)
        
        if profit > 0:
            self.win_count += 1
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.consecutive_losses += 1
            
            # Implement cooldown after multiple consecutive losses
            if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self.cooldown_until = time.time() + COOLDOWN_PERIOD
                logger.warning(f"Entering cooldown period for {COOLDOWN_PERIOD} seconds after "
                              f"{self.consecutive_losses} consecutive losses")
        
        # Update peak balance and drawdown
        self.peak_balance = max(self.peak_balance, self.total_profit)
        current_drawdown = self.peak_balance - self.total_profit
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update time-based metrics
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d %H")
        
        # Reset daily metrics if needed
        if now.day != self.current_day:
            logger.info(f"New day detected. Resetting daily profit from ${self.daily_profit:.2f}")
            self.daily_profit = 0.0
            self.daily_trades = 0
            self.current_day = now.day
            
        # Reset weekly metrics if needed
        current_week = now.isocalendar()[1]
        if current_week != self.current_week:
            logger.info(f"New week detected. Resetting weekly profit from ${self.weekly_profit:.2f}")
            self.weekly_profit = 0.0
            self.weekly_trades = 0
            self.current_week = current_week
            
        # Reset monthly metrics if needed
        if now.month != self.current_month:
            logger.info(f"New month detected. Resetting monthly profit from ${self.monthly_profit:.2f}")
            self.monthly_profit = 0.0
            self.monthly_trades = 0
            self.current_month = now.month
        
        # Update current period metrics
        self.daily_profit += profit
        self.weekly_profit += profit
        self.monthly_profit += profit
        self.daily_trades += 1
        self.weekly_trades += 1
        self.monthly_trades += 1
        
        # Update hourly tracking
        if hour_key in self.hourly_profit:
            self.hourly_profit[hour_key] += profit
        else:
            self.hourly_profit[hour_key] = profit
        
        # Update advanced metrics
        self.win_loss_ratio = self.win_count / max(1, self.loss_count)
        self.average_profit_per_trade = self.total_profit / max(1, self.trade_count)
        
        logger.info(f"Trade completed for {product_id}. Profit: ${profit:.2f}, "
                   f"Total profit: ${self.total_profit:.2f}, Win rate: {self.win_count}/{self.trade_count} "
                   f"({(self.win_count/max(1, self.trade_count))*100:.1f}%)")
    
    def calculate_dynamic_profit_target(self):
        """Calculate a dynamic profit target based on recent performance."""
        if len(self.historical_profits) < 10:
            return 30.0  # Default target until enough data is gathered
            
        avg_profit = np.mean(self.historical_profits[-10:])
        target = max(avg_profit * 1.5, 30.0)
        logger.info(f"Dynamic profit target: ${target:.2f}")
        return target
    
    def execute_trade(self, product_id, side, price, size):
        """Execute a trade on Coinbase Pro."""
        try:
            # Validate trade parameters
            if float(price) <= 0 or float(size) <= 0:
                logger.error(f"Invalid trade parameters: price=${price}, size={size}")
                return None
                
            self._respect_rate_limit()
            # Place a limit order
            order = self.auth_client.place_limit_order(
                product_id=product_id,
                side=side,
                price=str(price),
                size=str(size)
            )
            
            if 'id' in order:
                logger.info(f"Placed {side} order for {product_id}: {size} at ${price} (ID: {order['id']})")
                return order
            else:
                logger.error(f"Failed to place {side} order: {order}")
                return None
        except Exception as e:
            logger.error(f"Error placing {side} order for {product_id}: {e}")
            return None
    
    def monitor_order(self, order_id, product_id, side, entry_price, stop_loss, take_profit):
        """Monitor an open order and implement stop loss and take profit logic."""
        try:
            self._respect_rate_limit()
            # Check if the order is filled
            order_status = self.auth_client.get_order(order_id)
            
            if order_status['status'] == 'done' and order_status['done_reason'] == 'filled':
                logger.info(f"Order {order_id} filled at {order_status['price']}")
                
                # If this was a buy order, place a corresponding sell order with stop loss and take profit
                if side == 'buy':
                    # Get the fill details
                    filled_price = float(order_status['price'])
                    size = float(order_status['size'])
                    
                    # Calculate initial stop loss and take profit prices
                    initial_stop_loss = filled_price * (1 - STOP_LOSS_PERCENT)
                    trailing_stop = initial_stop_loss
                    take_profit_price = filled_price * (1 + STOP_LOSS_PERCENT * 2)
                    
                    logger.info(f"Monitoring position for stop loss at ${initial_stop_loss:.2f} or take profit at ${take_profit_price:.2f}")
                    logger.info(f"Using trailing stop with {TRAILING_STOP_PERCENT*100:.2f}% distance")
                    
                    # Place actual take profit order
                    take_profit_order = None
                    try:
                        self._respect_rate_limit()
                        take_profit_order = self.auth_client.place_limit_order(
                            product_id=product_id,
                            side='sell',
                            price=str(take_profit_price),
                            size=str(size)
                        )
                        if 'id' in take_profit_order:
                            logger.info(f"Placed take profit order: {take_profit_order['id']}")
                        else:
                            logger.error(f"Failed to place take profit order: {take_profit_order}")
                            take_profit_order = None
                    except Exception as e:
                        logger.error(f"Error placing take profit order: {e}")
                    
                    # Monitor position until exit
                    position_open = True
                    highest_price = filled_price
                    
                    while position_open:
                        try:
                            # Sleep to avoid excessive API calls
                            time.sleep(1)
                            
                            # Check current price
                            current_price = self.get_current_price(product_id)
                            if current_price is None:
                                continue
                            
                            # Update highest price and trailing stop if price has moved up
                            if current_price > highest_price:
                                highest_price = current_price
                                new_trailing_stop = highest_price * (1 - TRAILING_STOP_PERCENT)
                                if new_trailing_stop > trailing_stop:
                                    trailing_stop = new_trailing_stop
                                    logger.info(f"Trailing stop updated to ${trailing_stop:.2f}")
                            
                            # Check if take profit order was filled
                            if take_profit_order:
                                self._respect_rate_limit()
                                tp_status = self.auth_client.get_order(take_profit_order['id'])
                                if tp_status['status'] == 'done' and tp_status['done_reason'] == 'filled':
                                    filled_sell_price = float(tp_status['price'])
                                    profit = (filled_sell_price - filled_price) * size
                                    logger.info(f"Take profit order filled at ${filled_sell_price:.2f}. Profit: ${profit:.2f}")
                                    
                                    # Update metrics
                                    self.update_performance_metrics(profit, product_id)
                                    position_open = False
                                    return profit
                            
                            # Check if stop loss was hit
                            if current_price <= trailing_stop:
                                logger.info(f"Stop loss triggered at ${current_price:.2f} (trailing stop: ${trailing_stop:.2f})")
                                
                                # Cancel take profit order if it exists
                                if take_profit_order:
                                    try:
                                        self._respect_rate_limit()
                                        self.auth_client.cancel_order(take_profit_order['id'])
                                        logger.info(f"Cancelled take profit order: {take_profit_order['id']}")
                                    except Exception as e:
                                        logger.error(f"Error cancelling take profit order: {e}")
                                
                                # Place market sell order
                                try:
                                    self._respect_rate_limit()
                                    sell_order = self.auth_client.place_market_order(
                                        product_id=product_id,
                                        side='sell',
                                        size=str(size)
                                    )
                                    if 'id' in sell_order:
                                        logger.info(f"Placed market sell order: {sell_order['id']}")
                                        # Wait for fill
                                        for _ in range(10):  # Try for 10 seconds
                                            time.sleep(1)
                                            self._respect_rate_limit()
                                            sell_status = self.auth_client.get_order(sell_order['id'])
                                            if sell_status['status'] == 'done':
                                                filled_sell_price = float(sell_status.get('price', current_price))
                                                loss = (filled_sell_price - filled_price) * size
                                                logger.info(f"Stop loss order filled at ${filled_sell_price:.2f}. Loss: ${loss:.2f}")
                                                
                                                # Update metrics
                                                self.update_performance_metrics(loss, product_id)
                                                position_open = False
                                                return loss
                                    else:
                                        logger.error(f"Failed to place stop loss market order: {sell_order}")
                                except Exception as e:
                                    logger.error(f"Error placing stop loss market order: {e}")
                                    
                                # If we can't place or confirm the sell order, estimate the loss
                                loss = (current_price - filled_price) * size
                                logger.warning(f"Estimating loss based on current price: ${loss:.2f}")
                                
                                # Update metrics
                                self.update_performance_metrics(loss, product_id)
                                position_open = False
                                return loss
                        
                        except Exception as e:
                            logger.error(f"Error in position monitoring loop: {e}")
                            # Continue monitoring despite errors
                
                # If this was a sell order (short position), implement similar logic
                elif side == 'sell':
                    # Get the fill details
                    filled_price = float(order_status['price'])
                    size = float(order_status['size'])
                    
                    # For short positions, stop loss is above and take profit is below
                    initial_stop_loss = filled_price * (1 + STOP_LOSS_PERCENT)
                    trailing_stop = initial_stop_loss
                    take_profit_price = filled_price * (1 - STOP_LOSS_PERCENT * 2)
                    
                    logger.info(f"Short position: Monitoring for stop loss at ${initial_stop_loss:.2f} or take profit at ${take_profit_price:.2f}")
                    logger.info(f"Using trailing stop with {TRAILING_STOP_PERCENT*100:.2f}% distance")
                    
                    # Here we would implement the full monitoring loop for short positions
                    # For now using a simplified approach
                    
                    # Simplified for now - estimate profit/loss based on current price
                    current_price = self.get_current_price(product_id)
                    if current_price is None:
                        return 0
                        
                    profit = (filled_price - current_price) * size
                    logger.info(f"Simplified short position P/L calculation: ${profit:.2f}")
                    
                    # Update metrics
                    self.update_performance_metrics(profit, product_id)
                    return profit
                    
            return None  # Order not filled yet
        except Exception as e:
            logger.error(f"Error monitoring order {order_id}: {e}")
            return None
    
    def run_trading_cycle(self):
        """Run one complete trading cycle across all product IDs."""
        logger.info("Starting trading cycle")
        
        # Check if we're in cooldown period
        if time.time() < self.cooldown_until:
            logger.info(f"In cooldown period. {int(self.cooldown_until - time.time())} seconds remaining.")
            return
        
        # Check account balance first
        balances = self.check_account_balance()
        if not balances:
            logger.error("Failed to get account balances. Skipping trading cycle.")
            return
        
        # Process each product
        for product_id in PRODUCT_IDS:
            try:
                logger.info(f"Analyzing {product_id}")
                
                # Get current price and add to history
                current_price = self.get_current_price(product_id)
                if current_price is None:
                    logger.warning(f"Skipping {product_id} due to price fetch failure")
                    continue
                
                # Check if we have enough data for analysis
                if len(self.price_history[product_id]) < VOLATILITY_WINDOW:
                    logger.info(f"Collecting more price data for {product_id}")
                    continue
                
                # Train model if we have enough data and no model exists
                if self.ml_models[product_id] is None and len(self.price_history[product_id]) >= 50:
                    self.train_ml_model(product_id)
                
                # Check if market conditions are favorable
                if not self.market_is_favorable(product_id):
                    logger.info(f"Market conditions not favorable for {product_id}")
                    continue
                
                # Get predicted price movement if we have a model
                if self.ml_models[product_id] is not None:
                    predicted_change = self.predict_price_movement(product_id)
                    
                    if predicted_change is None:
                        logger.warning(f"Unable to predict price movement for {product_id}")
                        continue
                        
                    # Only trade if prediction confidence is high enough
                    if abs(predicted_change) < 0.001:  # Less than 0.1% predicted change
                        logger.info(f"Predicted price change too small for {product_id}: {predicted_change:.4%}")
                        continue
                        
                    # Determine side based on prediction
                    side = 'buy' if predicted_change > 0 else 'sell'
                else:
                    # Fallback if no ML model: use order book imbalance
                    imbalance = self.get_order_book_imbalance(product_id)
                    if abs(imbalance) < 0.05:  # Less than 5% imbalance
                        logger.info(f"Order book imbalance too small for {product_id}: {imbalance:.4f}")
                        continue
                        
                    side = 'buy' if imbalance > 0 else 'sell'
                
                # Calculate volatility for position sizing
                volatility = np.std(self.price_history[product_id][-VOLATILITY_WINDOW:])
                
                # Get available balance for the base or quote currency
                base_currency, quote_currency = product_id.split('-')
                available_balance = balances.get(quote_currency, {}).get('available', 0) if side == 'buy' else balances.get(base_currency, {}).get('available', 0)
                
                # Calculate position size
                position_size = self.calculate_position_size(available_balance, volatility)
                
                # Ensure position size doesn't exceed available balance
                if side == 'buy':
                    max_size = available_balance / current_price
                    position_size = min(position_size, max_size)
                else:
                    position_size = min(position_size, available_balance)
                
                # Ensure minimum position size
                if position_size * current_price < 10:  # Minimum order size $10
                    logger.info(f"Position size too small for {product_id}: ${position_size * current_price:.2f}")
                    continue
                
                # Calculate stop loss and take profit levels
                stop_loss = current_price * (1 - STOP_LOSS_PERCENT) if side == 'buy' else current_price * (1 + STOP_LOSS_PERCENT)
                take_profit = current_price * (1 + STOP_LOSS_PERCENT * 2) if side == 'buy' else current_price * (1 - STOP_LOSS_PERCENT * 2)
                
                # Execute the trade
                order = self.execute_trade(product_id, side, current_price, position_size)
                
                if order:
                    # Monitor the order
                    self.monitor_order(order['id'], product_id, side, current_price, stop_loss, take_profit)
            
            except Exception as e:
                logger.error(f"Error processing {product_id}: {e}")
                continue
        
        logger.info("Completed trading cycle")

    async def run(self):
        """Main entry point to run the bot asynchronously."""
        logger.info("Starting Advanced Trading Bot")
        
        # Check if API credentials are available
        if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
            logger.error("API credentials are missing. Please set them in your .env file.")
            return
        
        try:
            while True:
                self.run_trading_cycle()
                await asyncio.sleep(TRADE_FREQUENCY)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            logger.info("Bot shutdown complete")


def main():
    """Main function to start the trading bot."""
    bot = AdvancedTradingBot()
    
    # Run the bot using asyncio
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(bot.run())
    except KeyboardInterrupt:
        print("Bot stopped by user")
    finally:
        loop.close()

if __name__ == "__main__":
    main() 