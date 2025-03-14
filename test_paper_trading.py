#!/usr/bin/env python3
"""
Test script for Paper Trading.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any

from paper_trading import PaperTradingExecutor, AccountBalance
from order_executor import OrderStatus, Order
from order_validator import OrderType, OrderSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PaperTradingTest")

# Callback function for order updates
def order_update_callback(order: Order):
    """Callback for order updates."""
    logger.info(f"Order update: {order.id} - {order.status.value}")
    logger.info(f"  Product: {order.product_id}, Side: {order.side.value}, Type: {order.type.value}")
    if order.filled_size > 0:
        logger.info(f"  Filled: {order.filled_size} @ {order.filled_price}")
        logger.info(f"  Fees: {order.fill_fees}")
    if order.is_done():
        logger.info(f"  Done reason: {order.done_reason}")

async def print_account_balance(executor: PaperTradingExecutor, currencies: List[str]) -> None:
    """Print account balance."""
    account = executor.get_account_balance()
    logger.info("=== Account Balance ===")
    for currency in currencies:
        logger.info(f"{currency}: Available={account.get_available(currency):.6f}, Hold={account.get_hold(currency):.6f}, Total={account.get_total(currency):.6f}")
    logger.info("=======================")

async def test_market_simulation():
    """Test market simulation functionality."""
    logger.info("Testing market simulation...")
    
    # Create paper trading executor with initial balance
    executor = PaperTradingExecutor(
        initial_balances={"USD": 10000.0, "BTC": 0.1, "ETH": 1.0},
        volatility_pct=0.2,  # Higher volatility for testing
        slippage_pct=0.01
    )
    
    # Register for order updates
    executor.register_order_update_callback(order_update_callback)
    
    # Start market simulation
    await executor.start_market_simulation(update_interval=0.5)
    
    # Print initial balances
    await print_account_balance(executor, ["USD", "BTC", "ETH"])
    
    # Wait a bit for the market to initialize
    logger.info("Initializing market simulation...")
    await asyncio.sleep(2)
    
    # Log current prices
    logger.info(f"BTC-USD price: ${executor.price_cache.get('BTC-USD', 0):.2f}")
    logger.info(f"ETH-USD price: ${executor.price_cache.get('ETH-USD', 0):.2f}")
    
    # Wait a bit to observe price changes
    logger.info("Observing price changes...")
    for i in range(5):
        await asyncio.sleep(1)
        logger.info(f"BTC-USD price: ${executor.price_cache.get('BTC-USD', 0):.2f}")
        logger.info(f"ETH-USD price: ${executor.price_cache.get('ETH-USD', 0):.2f}")
    
    # Stop market simulation
    await executor.stop_market_simulation()
    logger.info("Market simulation test completed")

async def test_paper_trading_order_lifecycle():
    """Test paper trading order lifecycle."""
    logger.info("Testing paper trading order lifecycle...")
    
    # Create paper trading executor with initial balance
    executor = PaperTradingExecutor(
        initial_balances={"USD": 10000.0, "BTC": 0.1, "ETH": 1.0},
        volatility_pct=0.1,
        slippage_pct=0.01
    )
    
    # Register for order updates
    executor.register_order_update_callback(order_update_callback)
    
    # Start market simulation
    await executor.start_market_simulation(update_interval=0.5)
    
    # Print initial balances
    await print_account_balance(executor, ["USD", "BTC", "ETH"])
    
    # Wait for market to initialize
    await asyncio.sleep(2)
    
    # Place a market buy order
    logger.info("Placing market buy order...")
    market_buy = await executor.place_order(
        product_id="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        funds=1000.0
    )
    
    if market_buy:
        logger.info(f"Market buy order placed: {market_buy.id}")
        # Let it execute
        await asyncio.sleep(1)
    else:
        logger.error("Failed to place market buy order")
    
    # Print balances after market buy
    await print_account_balance(executor, ["USD", "BTC", "ETH"])
    
    # Place a limit sell order for half the BTC just bought
    btc_balance = executor.get_account_balance().get_available("BTC")
    sell_size = btc_balance / 2
    
    current_price = executor.price_cache.get("BTC-USD", 0)
    limit_price = current_price * 1.05  # 5% above current price
    
    logger.info(f"Placing limit sell order for {sell_size:.8f} BTC at ${limit_price:.2f}...")
    limit_sell = await executor.place_order(
        product_id="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        size=sell_size,
        price=limit_price
    )
    
    if limit_sell:
        logger.info(f"Limit sell order placed: {limit_sell.id}")
    else:
        logger.error("Failed to place limit sell order")
    
    # Print balances after limit sell placement
    await print_account_balance(executor, ["USD", "BTC", "ETH"])
    
    # Wait for price to move and potentially trigger the limit order
    logger.info("Waiting for price movement...")
    for i in range(10):
        await asyncio.sleep(1)
        current_price = executor.price_cache.get("BTC-USD", 0)
        logger.info(f"BTC-USD price: ${current_price:.2f} (Target: ${limit_price:.2f})")
        
        # Check if order is still active
        if limit_sell and not limit_sell.is_active():
            logger.info(f"Limit sell order {limit_sell.id} is no longer active")
            break
    
    # Place a stop-limit order
    stop_price = current_price * 0.95  # 5% below current price
    limit_price = current_price * 0.94  # 6% below current price
    
    logger.info(f"Placing stop-limit sell order at stop=${stop_price:.2f}, limit=${limit_price:.2f}...")
    stop_limit = await executor.place_order(
        product_id="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LIMIT,
        size=sell_size / 2,  # Sell a quarter of our original BTC
        price=limit_price,
        stop_price=stop_price
    )
    
    if stop_limit:
        logger.info(f"Stop-limit order placed: {stop_limit.id}")
    else:
        logger.error("Failed to place stop-limit order")
    
    # Print final balances
    await print_account_balance(executor, ["USD", "BTC", "ETH"])
    
    # Wait for some time to see if stop-limit gets triggered
    logger.info("Waiting for potential stop-limit trigger...")
    for i in range(10):
        await asyncio.sleep(1)
        current_price = executor.price_cache.get("BTC-USD", 0)
        logger.info(f"BTC-USD price: ${current_price:.2f} (Stop: ${stop_price:.2f})")
        
        # Check if order is still active
        if stop_limit and not stop_limit.is_active():
            logger.info(f"Stop-limit order {stop_limit.id} is no longer active")
            break
    
    # Cancel any remaining orders
    cancelled_count = await executor.cancel_all_orders()
    logger.info(f"Cancelled {cancelled_count} remaining orders")
    
    # Print final balances
    await print_account_balance(executor, ["USD", "BTC", "ETH"])
    
    # Stop market simulation
    await executor.stop_market_simulation()
    logger.info("Paper trading order lifecycle test completed")

async def test_multiple_products():
    """Test trading multiple products simultaneously."""
    logger.info("Testing trading multiple products...")
    
    # Create paper trading executor with initial balance
    executor = PaperTradingExecutor(
        initial_balances={"USD": 20000.0, "BTC": 0.2, "ETH": 2.0, "SOL": 20.0},
        volatility_pct=0.15
    )
    
    # Register for order updates
    executor.register_order_update_callback(order_update_callback)
    
    # Start market simulation
    await executor.start_market_simulation(update_interval=0.5)
    
    # Print initial balances
    await print_account_balance(executor, ["USD", "BTC", "ETH", "SOL"])
    
    # Wait for market to initialize
    await asyncio.sleep(2)
    
    # Place orders for different products
    products = ["BTC-USD", "ETH-USD", "SOL-USD"]
    orders = []
    
    # Place market buy orders
    for product in products:
        logger.info(f"Placing market buy order for {product}...")
        order = await executor.place_order(
            product_id=product,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            funds=1000.0
        )
        if order:
            orders.append(order)
            logger.info(f"Market buy order placed for {product}: {order.id}")
        else:
            logger.error(f"Failed to place market buy order for {product}")
    
    # Wait for orders to execute
    await asyncio.sleep(2)
    
    # Print balances after market buys
    await print_account_balance(executor, ["USD", "BTC", "ETH", "SOL"])
    
    # Place limit sell orders
    for product in products:
        base_currency = product.split("-")[0]
        balance = executor.get_account_balance().get_available(base_currency)
        sell_size = balance * 0.25  # Sell 25% of holdings
        
        current_price = executor.price_cache.get(product, 0)
        limit_price = current_price * 1.1  # 10% above current price
        
        logger.info(f"Placing limit sell order for {sell_size:.8f} {base_currency} at ${limit_price:.2f}...")
        order = await executor.place_order(
            product_id=product,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            size=sell_size,
            price=limit_price
        )
        if order:
            orders.append(order)
            logger.info(f"Limit sell order placed for {product}: {order.id}")
        else:
            logger.error(f"Failed to place limit sell order for {product}")
    
    # Wait for potential limit order fills
    logger.info("Waiting for potential limit order fills...")
    for i in range(10):
        await asyncio.sleep(1)
        # Print current prices
        for product in products:
            current_price = executor.price_cache.get(product, 0)
            logger.info(f"{product} price: ${current_price:.2f}")
    
    # Get active orders
    active_orders = executor.get_active_orders()
    logger.info(f"Active orders: {len(active_orders)}")
    
    # Cancel all remaining orders
    cancelled_count = await executor.cancel_all_orders()
    logger.info(f"Cancelled {cancelled_count} remaining orders")
    
    # Print final balances
    await print_account_balance(executor, ["USD", "BTC", "ETH", "SOL"])
    
    # Stop market simulation
    await executor.stop_market_simulation()
    logger.info("Multiple products test completed")

async def main():
    """Run all tests."""
    try:
        await test_market_simulation()
        logger.info("\n")
        
        await test_paper_trading_order_lifecycle()
        logger.info("\n")
        
        await test_multiple_products()
        logger.info("\n")
        
        logger.info("All paper trading tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 