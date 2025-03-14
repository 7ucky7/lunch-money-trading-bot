#!/usr/bin/env python3
"""
Test script for Data Manager.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from data_manager import DataManager, TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataManagerTest")

async def print_trade(trade):
    """Callback for trade updates."""
    logger.info(f"Trade: {trade.product_id} - Price: {trade.price}, Size: {trade.size}, Side: {trade.side}")

async def print_candle(candle):
    """Callback for candle updates."""
    logger.info(f"Candle: Time: {datetime.fromtimestamp(candle.timestamp)}, Open: {candle.open}, Close: {candle.close}")

async def main():
    """Main test function."""
    # Initialize data manager
    data_manager = DataManager(
        api_key="YOUR_API_KEY",
        api_secret="YOUR_API_SECRET"
    )
    
    # Register callbacks
    data_manager.register_callback('trade', print_trade)
    data_manager.register_callback('candle', print_candle)
    
    try:
        # Start data manager with test products
        await data_manager.start(product_ids=["BTC-USD", "ETH-USD"])
        logger.info("Data manager started")
        
        # Wait for some data to accumulate
        await asyncio.sleep(30)
        
        # Test historical data fetching
        end_time = int(datetime.now().timestamp())
        start_time = end_time - 3600  # Last hour
        
        for product_id in ["BTC-USD", "ETH-USD"]:
            # Fetch historical candles
            candles = await data_manager.fetch_historical_candles(
                product_id=product_id,
                timeframe=TimeFrame.MINUTE_1,
                start_time=start_time,
                end_time=end_time
            )
            logger.info(f"Fetched {len(candles)} historical candles for {product_id}")
            
            # Get latest trades
            trades = data_manager.get_latest_trades(product_id, limit=10)
            logger.info(f"Latest {len(trades)} trades for {product_id}")
            
            # Get latest price
            price = data_manager.get_latest_price(product_id)
            logger.info(f"Latest price for {product_id}: {price}")
            
            # Get latest candles for different timeframes
            for timeframe in [TimeFrame.MINUTE_1, TimeFrame.MINUTE_5, TimeFrame.HOUR_1]:
                candles = data_manager.get_latest_candles(product_id, timeframe, limit=10)
                logger.info(f"Latest {len(candles)} {timeframe.value} candles for {product_id}")
            
            # Convert to DataFrame
            df = data_manager.to_dataframe(
                product_id=product_id,
                timeframe=TimeFrame.MINUTE_1,
                start_time=start_time,
                end_time=end_time
            )
            logger.info(f"DataFrame shape for {product_id}: {df.shape}")
        
        # Keep running to receive real-time updates
        logger.info("Receiving real-time updates for 5 minutes...")
        await asyncio.sleep(300)
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
    finally:
        # Clean up
        await data_manager.stop()
        logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(main()) 