#!/usr/bin/env python3
"""
Lunch Money Trading Bot

This script implements a simple grid trading strategy on Coinbase Pro
with the goal of making small daily profits. This is for educational
purposes only and comes with significant risk.

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
from dotenv import load_dotenv
import schedule

# Configure precision for decimal calculations
getcontext().prec = 8

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LunchMoneyBot")

# Load environment variables
load_dotenv()

# API credentials should be stored in environment variables for security
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")
API_PASSPHRASE = os.getenv("COINBASE_API_PASSPHRASE")

# Trading parameters - adjust these to your risk tolerance
TRADING_PAIR = "BTC-USD"  # The cryptocurrency pair to trade
TARGET_PROFIT_USD = 20.00  # Target profit in USD
GRID_LEVELS = 5  # Number of buy/sell grid levels
GRID_SPREAD_PERCENTAGE = 0.5  # Percentage between grid levels
POSITION_SIZE_USD = 100.00  # Size of each position in USD

class LunchMoneyBot:
    """A simple grid trading bot for Coinbase Pro."""
    
    def __init__(self):
        """Initialize the trading bot with Coinbase API credentials."""
        self.auth_client = cbpro.AuthenticatedClient(
            API_KEY, API_SECRET, API_PASSPHRASE
        )
        logger.info(f"Bot initialized for trading {TRADING_PAIR}")
        
        # Track orders and positions
        self.buy_orders = []
        self.sell_orders = []
        self.positions = {}
        
        # Track profits
        self.daily_profit = Decimal('0.00')
        self.total_profit = Decimal('0.00')
        
        # Date tracking for daily reset
        self.current_day = datetime.now().day
    
    def check_account_balance(self):
        """Check available balance in the account."""
        try:
            accounts = self.auth_client.get_accounts()
            for account in accounts:
                logger.info(f"Currency: {account['currency']}, Balance: {account['balance']}, Available: {account['available']}")
            return accounts
        except Exception as e:
            logger.error(f"Error checking account balance: {e}")
            return None
    
    def get_current_price(self):
        """Get the current market price for the trading pair."""
        try:
            ticker = self.auth_client.get_product_ticker(product_id=TRADING_PAIR)
            current_price = Decimal(ticker['price'])
            logger.info(f"Current {TRADING_PAIR} price: ${current_price}")
            return current_price
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def setup_grid(self):
        """Set up the grid trading strategy with buy and sell orders."""
        # Cancel any existing orders
        self.cancel_all_orders()
        
        current_price = self.get_current_price()
        if not current_price:
            logger.error("Cannot set up grid without current price")
            return False
        
        logger.info(f"Setting up grid around price ${current_price}")
        
        # Calculate grid levels
        for i in range(1, GRID_LEVELS + 1):
            # Calculate buy price below current price
            buy_percentage = Decimal(i * GRID_SPREAD_PERCENTAGE / 100)
            buy_price = current_price * (1 - buy_percentage)
            buy_size = Decimal(POSITION_SIZE_USD) / buy_price
            
            # Calculate sell price above current price
            sell_percentage = Decimal(i * GRID_SPREAD_PERCENTAGE / 100)
            sell_price = current_price * (1 + sell_percentage)
            sell_size = Decimal(POSITION_SIZE_USD) / current_price
            
            # Place buy order
            try:
                buy_order = self.auth_client.place_limit_order(
                    product_id=TRADING_PAIR,
                    side='buy',
                    price=str(buy_price.quantize(Decimal('0.01'))),
                    size=str(buy_size.quantize(Decimal('0.00001')))
                )
                if 'id' in buy_order:
                    self.buy_orders.append(buy_order)
                    logger.info(f"Placed buy order at ${buy_price}: {buy_order['id']}")
                else:
                    logger.error(f"Failed to place buy order: {buy_order}")
            except Exception as e:
                logger.error(f"Error placing buy order: {e}")
            
            # Place sell order
            try:
                sell_order = self.auth_client.place_limit_order(
                    product_id=TRADING_PAIR,
                    side='sell',
                    price=str(sell_price.quantize(Decimal('0.01'))),
                    size=str(sell_size.quantize(Decimal('0.00001')))
                )
                if 'id' in sell_order:
                    self.sell_orders.append(sell_order)
                    logger.info(f"Placed sell order at ${sell_price}: {sell_order['id']}")
                else:
                    logger.error(f"Failed to place sell order: {sell_order}")
            except Exception as e:
                logger.error(f"Error placing sell order: {e}")
        
        return True
    
    def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            result = self.auth_client.cancel_all(product_id=TRADING_PAIR)
            logger.info(f"Cancelled orders: {result}")
            self.buy_orders = []
            self.sell_orders = []
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False
    
    def check_filled_orders(self):
        """Check for filled orders and update positions."""
        try:
            # Check buy orders
            for order in self.buy_orders[:]:
                order_status = self.auth_client.get_order(order['id'])
                if order_status['status'] == 'done' and order_status['done_reason'] == 'filled':
                    logger.info(f"Buy order filled: {order['id']} at {order_status['price']}")
                    self.buy_orders.remove(order)
                    
                    # Update positions
                    size = Decimal(order_status['size'])
                    price = Decimal(order_status['price'])
                    if TRADING_PAIR in self.positions:
                        self.positions[TRADING_PAIR]['size'] += size
                        self.positions[TRADING_PAIR]['cost'] += size * price
                    else:
                        self.positions[TRADING_PAIR] = {
                            'size': size,
                            'cost': size * price
                        }
                    
                    # Place a new sell order above the buy price
                    sell_price = price * Decimal('1.02')  # 2% profit target
                    try:
                        sell_order = self.auth_client.place_limit_order(
                            product_id=TRADING_PAIR,
                            side='sell',
                            price=str(sell_price.quantize(Decimal('0.01'))),
                            size=str(size.quantize(Decimal('0.00001')))
                        )
                        if 'id' in sell_order:
                            self.sell_orders.append(sell_order)
                            logger.info(f"Placed new sell order at ${sell_price}: {sell_order['id']}")
                    except Exception as e:
                        logger.error(f"Error placing new sell order: {e}")
            
            # Check sell orders
            for order in self.sell_orders[:]:
                order_status = self.auth_client.get_order(order['id'])
                if order_status['status'] == 'done' and order_status['done_reason'] == 'filled':
                    logger.info(f"Sell order filled: {order['id']} at {order_status['price']}")
                    self.sell_orders.remove(order)
                    
                    # Calculate profit
                    size = Decimal(order_status['size'])
                    price = Decimal(order_status['price'])
                    revenue = size * price
                    
                    # Update positions
                    if TRADING_PAIR in self.positions and self.positions[TRADING_PAIR]['size'] >= size:
                        avg_cost = self.positions[TRADING_PAIR]['cost'] / self.positions[TRADING_PAIR]['size']
                        cost = size * avg_cost
                        profit = revenue - cost
                        
                        self.positions[TRADING_PAIR]['size'] -= size
                        self.positions[TRADING_PAIR]['cost'] -= cost
                        
                        # Update profit tracking
                        self.daily_profit += profit
                        self.total_profit += profit
                        
                        logger.info(f"Realized profit: ${profit} - Daily: ${self.daily_profit} - Total: ${self.total_profit}")
                        
                        # Place a new buy order
                        buy_price = price * Decimal('0.98')  # 2% below the sell price
                        buy_size = Decimal(POSITION_SIZE_USD) / buy_price
                        try:
                            buy_order = self.auth_client.place_limit_order(
                                product_id=TRADING_PAIR,
                                side='buy',
                                price=str(buy_price.quantize(Decimal('0.01'))),
                                size=str(buy_size.quantize(Decimal('0.00001')))
                            )
                            if 'id' in buy_order:
                                self.buy_orders.append(buy_order)
                                logger.info(f"Placed new buy order at ${buy_price}: {buy_order['id']}")
                        except Exception as e:
                            logger.error(f"Error placing new buy order: {e}")
        except Exception as e:
            logger.error(f"Error checking filled orders: {e}")
    
    def check_daily_profit(self):
        """Check if we've reached our daily profit target."""
        current_day = datetime.now().day
        
        # Reset daily profit counter if it's a new day
        if current_day != self.current_day:
            logger.info(f"New day detected. Resetting daily profit from ${self.daily_profit}")
            self.daily_profit = Decimal('0.00')
            self.current_day = current_day
        
        # Check if we've hit our target
        if self.daily_profit >= Decimal(TARGET_PROFIT_USD):
            logger.info(f"Daily profit target reached: ${self.daily_profit} >= ${TARGET_PROFIT_USD}")
            # Optionally, you could cancel all orders and stop trading for the day
            # self.cancel_all_orders()
            return True
        return False
    
    def run_trading_cycle(self):
        """Run one complete trading cycle."""
        logger.info("Starting trading cycle")
        
        # Check if we already hit our profit target for the day
        if self.check_daily_profit():
            logger.info("Daily profit target already reached")
            return
        
        # Check current account balance
        self.check_account_balance()
        
        # Check for filled orders and update positions
        self.check_filled_orders()
        
        # If we don't have enough orders, set up the grid again
        if len(self.buy_orders) < GRID_LEVELS / 2 or len(self.sell_orders) < GRID_LEVELS / 2:
            logger.info("Not enough active orders. Setting up grid again.")
            self.setup_grid()
        
        logger.info("Completed trading cycle")

def main():
    """Main function to start the trading bot."""
    logger.info("Starting Lunch Money Trading Bot")
    
    # Check if API credentials are available
    if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
        logger.error("API credentials are missing. Please set them in your .env file.")
        return
    
    # Initialize the bot
    bot = LunchMoneyBot()
    
    # Initial setup
    bot.setup_grid()
    
    # Schedule regular checks
    schedule.every(5).minutes.do(bot.run_trading_cycle)
    
    # Main loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        bot.cancel_all_orders()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        bot.cancel_all_orders()

if __name__ == "__main__":
    main() 