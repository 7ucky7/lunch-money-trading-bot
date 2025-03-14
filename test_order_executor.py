#!/usr/bin/env python3
"""
Test script for Order Executor.
"""

import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from order_executor import OrderExecutor, OrderStatus, Order
from order_validator import OrderType, OrderSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OrderExecutorTest")

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

async def test_order_lifecycle():
    """Test the lifecycle of an order from placement to completion."""
    # Get API credentials from environment
    api_key = os.environ.get("COINBASE_API_KEY", "dummy_key")
    api_secret = os.environ.get("COINBASE_API_SECRET", "dummy_secret")
    api_passphrase = os.environ.get("COINBASE_API_PASSPHRASE", "dummy_passphrase")
    api_url = os.environ.get("COINBASE_API_URL", "https://api.exchange.coinbase.com")
    
    # Create order executor
    executor = OrderExecutor(
        exchange_api_url=api_url,
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase
    )
    
    # Register callback
    executor.register_order_update_callback(order_update_callback)
    
    # Set to True to actually place orders (requires valid API credentials)
    execute_real_orders = False
    
    if execute_real_orders:
        logger.info("Placing a limit buy order...")
        order = await executor.place_order(
            product_id="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=0.001,  # Small size for testing
            price=15000.0,  # Well below market price
            client_id=f"test_buy_{int(datetime.now().timestamp())}"
        )
        
        if order:
            logger.info(f"Order placed: {order.to_dict()}")
            
            # Wait 2 seconds
            await asyncio.sleep(2)
            
            # Check order status
            updated_order = await executor.get_order(order.id)
            logger.info(f"Order status: {updated_order.status.value}")
            
            # Cancel the order
            if updated_order.is_active():
                logger.info(f"Cancelling order: {order.id}")
                success = await executor.cancel_order(order.id)
                logger.info(f"Cancellation {'successful' if success else 'failed'}")
                
                # Check order status again
                await asyncio.sleep(1)
                final_order = await executor.get_order(order.id)
                logger.info(f"Final order status: {final_order.status.value}")
    else:
        # Mock order execution for testing without API access
        logger.info("Mocking order lifecycle...")
        
        # Create a mock order
        mock_order = Order(
            id="mock-order-1",
            client_id="mock-client-1",
            product_id="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            status=OrderStatus.OPEN,
            size=0.1,
            price=20000.0,
            created_at=datetime.now()
        )
        
        # Add to executor's tracking
        executor.orders[mock_order.id] = mock_order
        
        # Simulate partial fill
        logger.info("Simulating partial fill...")
        mock_order.filled_size = 0.05
        mock_order.filled_price = 20000.0
        mock_order.fill_fees = 0.05 * 20000.0 * 0.005  # 0.5% fee
        executor._notify_order_update(mock_order)
        
        await asyncio.sleep(1)
        
        # Simulate complete fill
        logger.info("Simulating complete fill...")
        mock_order.filled_size = 0.1
        mock_order.fill_fees = 0.1 * 20000.0 * 0.005  # 0.5% fee
        mock_order.status = OrderStatus.FILLED
        mock_order.done_at = datetime.now()
        mock_order.done_reason = "filled"
        executor._notify_order_update(mock_order)

async def test_paper_trading():
    """Demonstrate paper trading functionality."""
    logger.info("Simulating paper trading...")
    
    # Create a paper trading executor (no real API calls)
    paper_executor = OrderExecutor(
        exchange_api_url="https://paper-trading-api",
        api_key="paper_key",
        api_secret="paper_secret"
    )
    
    # Register callback
    paper_executor.register_order_update_callback(order_update_callback)
    
    # Simulate market buy
    buy_client_id = f"paper_buy_{int(datetime.now().timestamp())}"
    buy_order = Order(
        id=f"paper-{buy_client_id}",
        client_id=buy_client_id,
        product_id="ETH-USD",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        status=OrderStatus.PENDING,
        funds=1000.0,
        created_at=datetime.now()
    )
    
    # Add to tracking
    paper_executor.orders[buy_order.id] = buy_order
    paper_executor.client_order_map[buy_client_id] = buy_order.id
    
    # Simulate order acceptance
    logger.info("Paper trading: Order accepted")
    buy_order.status = OrderStatus.OPEN
    paper_executor._notify_order_update(buy_order)
    
    await asyncio.sleep(1)
    
    # Simulate fill at current market price
    market_price = 2000.0  # Simulated current ETH price
    filled_size = buy_order.funds / market_price
    buy_order.filled_size = filled_size
    buy_order.filled_price = market_price
    buy_order.fill_fees = buy_order.funds * 0.005  # 0.5% fee
    buy_order.status = OrderStatus.FILLED
    buy_order.done_at = datetime.now()
    buy_order.done_reason = "filled"
    
    logger.info("Paper trading: Order filled")
    paper_executor._notify_order_update(buy_order)
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Now place a limit sell for half the position
    sell_client_id = f"paper_sell_{int(datetime.now().timestamp())}"
    sell_size = filled_size / 2
    sell_price = market_price * 1.05  # 5% profit target
    
    sell_order = Order(
        id=f"paper-{sell_client_id}",
        client_id=sell_client_id,
        product_id="ETH-USD",
        side=OrderSide.SELL,
        type=OrderType.LIMIT,
        status=OrderStatus.PENDING,
        size=sell_size,
        price=sell_price,
        created_at=datetime.now()
    )
    
    # Add to tracking
    paper_executor.orders[sell_order.id] = sell_order
    paper_executor.client_order_map[sell_client_id] = sell_order.id
    
    # Simulate order acceptance
    logger.info("Paper trading: Sell order accepted")
    sell_order.status = OrderStatus.OPEN
    paper_executor._notify_order_update(sell_order)
    
    await asyncio.sleep(1)
    
    # Simulate price movement and fill
    logger.info("Paper trading: Price moved up, sell order filled")
    sell_order.filled_size = sell_size
    sell_order.filled_price = sell_price
    sell_order.fill_fees = sell_size * sell_price * 0.005  # 0.5% fee
    sell_order.status = OrderStatus.FILLED
    sell_order.done_at = datetime.now()
    sell_order.done_reason = "filled"
    paper_executor._notify_order_update(sell_order)
    
    # Calculate P&L
    buy_cost = buy_order.filled_price * (sell_size)
    sell_value = sell_order.filled_price * sell_size
    profit = sell_value - buy_cost
    fees = buy_order.fill_fees/2 + sell_order.fill_fees  # Half of buy fees + sell fees
    net_profit = profit - fees
    
    logger.info(f"Paper trading: Trade P&L calculation")
    logger.info(f"  Buy cost: ${buy_cost:.2f}")
    logger.info(f"  Sell value: ${sell_value:.2f}")
    logger.info(f"  Gross profit: ${profit:.2f}")
    logger.info(f"  Fees: ${fees:.2f}")
    logger.info(f"  Net profit: ${net_profit:.2f}")
    logger.info(f"  ROI: {(net_profit/buy_cost)*100:.2f}%")

async def main():
    """Run all tests."""
    try:
        logger.info("Testing order lifecycle...")
        await test_order_lifecycle()
        
        logger.info("\nTesting paper trading...")
        await test_paper_trading()
        
        logger.info("\nAll tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 