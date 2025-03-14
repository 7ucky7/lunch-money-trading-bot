#!/usr/bin/env python3
"""
Data Manager Module.

Handles real-time market data streaming and historical data management
for the cryptocurrency trading bot.
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataManager")

class TimeFrame(Enum):
    """Supported timeframes for candle data."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"

@dataclass
class Candle:
    """Represents a single candlestick."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: int
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    product_id: str

class DataManager:
    """
    Manages market data including real-time streaming and historical data.
    
    Features:
    - Real-time WebSocket data streaming
    - Historical data retrieval and caching
    - Multiple timeframe support
    - Automatic reconnection
    - Data validation and cleaning
    """
    
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize data manager.
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
        """
        self.api_key = api_key
        self.api_secret = api_secret
        
        # WebSocket connection
        self.ws = None
        self.ws_connected = False
        self.subscribed_products = set()
        
        # Data storage
        self.candles: Dict[str, Dict[TimeFrame, List[Candle]]] = {}
        self.trades: Dict[str, List[Trade]] = {}
        self.latest_prices: Dict[str, float] = {}
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'trade': [],
            'candle': [],
            'error': []
        }
        
        # Cache settings
        self.max_candles = 1000
        self.max_trades = 5000
        
        logger.info("Data Manager initialized")
    
    async def start(self, product_ids: Optional[List[str]] = None):
        """
        Start data streaming.
        
        Args:
            product_ids: List of product IDs to subscribe to
        """
        if product_ids:
            self.subscribed_products.update(product_ids)
        
        # Start WebSocket connection
        await self._connect_websocket()
        
        # Initialize data structures for each product
        for product_id in self.subscribed_products:
            if product_id not in self.candles:
                self.candles[product_id] = {tf: [] for tf in TimeFrame}
            if product_id not in self.trades:
                self.trades[product_id] = []
    
    async def stop(self):
        """Stop data streaming and clean up resources."""
        if self.ws and not self.ws.closed:
            await self.ws.close()
        self.ws_connected = False
        logger.info("Data Manager stopped")
    
    async def _connect_websocket(self):
        """Establish WebSocket connection and subscribe to channels."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect('wss://ws-feed.pro.coinbase.com') as ws:
                    self.ws = ws
                    self.ws_connected = True
                    logger.info("WebSocket connected")
                    
                    # Subscribe to channels
                    await self._subscribe()
                    
                    # Start message handling loop
                    await self._handle_messages()
                    
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.ws_connected = False
            # Attempt to reconnect
            await asyncio.sleep(5)
            await self._connect_websocket()
    
    async def _subscribe(self):
        """Subscribe to WebSocket channels."""
        if not self.ws_connected:
            return
        
        subscribe_message = {
            "type": "subscribe",
            "product_ids": list(self.subscribed_products),
            "channels": ["matches", "heartbeat"]
        }
        
        await self.ws.send_json(subscribe_message)
        logger.info(f"Subscribed to {len(self.subscribed_products)} products")
    
    async def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data['type'] == 'match':
                        # Process trade
                        await self._process_trade(data)
                    elif data['type'] == 'heartbeat':
                        # Update connection status
                        self.ws_connected = True
                    elif data['type'] == 'error':
                        logger.error(f"WebSocket error: {data}")
                        await self._notify_callbacks('error', data)
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg}")
                    break
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket messages: {e}")
            self.ws_connected = False
            # Attempt to reconnect
            await self._connect_websocket()
    
    async def _process_trade(self, data: Dict[str, Any]):
        """Process incoming trade data."""
        try:
            trade = Trade(
                timestamp=int(datetime.fromisoformat(data['time'].replace('Z', '+00:00')).timestamp()),
                price=float(data['price']),
                size=float(data['size']),
                side=data['side'],
                product_id=data['product_id']
            )
            
            # Update latest price
            self.latest_prices[trade.product_id] = trade.price
            
            # Add to trade history
            self.trades[trade.product_id].append(trade)
            
            # Maintain trade history size
            if len(self.trades[trade.product_id]) > self.max_trades:
                self.trades[trade.product_id] = self.trades[trade.product_id][-self.max_trades:]
            
            # Update candles
            await self._update_candles(trade)
            
            # Notify callbacks
            await self._notify_callbacks('trade', trade)
            
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    async def _update_candles(self, trade: Trade):
        """Update candle data with new trade."""
        current_time = trade.timestamp
        
        for timeframe in TimeFrame:
            candle_timestamp = self._get_candle_timestamp(current_time, timeframe)
            candles = self.candles[trade.product_id][timeframe]
            
            if not candles or candles[-1].timestamp < candle_timestamp:
                # Create new candle
                new_candle = Candle(
                    timestamp=candle_timestamp,
                    open=trade.price,
                    high=trade.price,
                    low=trade.price,
                    close=trade.price,
                    volume=trade.size
                )
                candles.append(new_candle)
                
                # Maintain candle history size
                if len(candles) > self.max_candles:
                    candles = candles[-self.max_candles:]
                
                await self._notify_callbacks('candle', new_candle)
            else:
                # Update existing candle
                current_candle = candles[-1]
                current_candle.high = max(current_candle.high, trade.price)
                current_candle.low = min(current_candle.low, trade.price)
                current_candle.close = trade.price
                current_candle.volume += trade.size
    
    def _get_candle_timestamp(self, timestamp: int, timeframe: TimeFrame) -> int:
        """Get normalized timestamp for a candle based on timeframe."""
        dt = datetime.fromtimestamp(timestamp)
        
        if timeframe == TimeFrame.MINUTE_1:
            dt = dt.replace(second=0, microsecond=0)
        elif timeframe == TimeFrame.MINUTE_5:
            dt = dt.replace(minute=dt.minute - dt.minute % 5, second=0, microsecond=0)
        elif timeframe == TimeFrame.MINUTE_15:
            dt = dt.replace(minute=dt.minute - dt.minute % 15, second=0, microsecond=0)
        elif timeframe == TimeFrame.MINUTE_30:
            dt = dt.replace(minute=dt.minute - dt.minute % 30, second=0, microsecond=0)
        elif timeframe == TimeFrame.HOUR_1:
            dt = dt.replace(minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.HOUR_4:
            dt = dt.replace(hour=dt.hour - dt.hour % 4, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.HOUR_12:
            dt = dt.replace(hour=dt.hour - dt.hour % 12, minute=0, second=0, microsecond=0)
        elif timeframe == TimeFrame.DAY_1:
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return int(dt.timestamp())
    
    async def fetch_historical_candles(
        self,
        product_id: str,
        timeframe: TimeFrame,
        start_time: int,
        end_time: int
    ) -> List[Candle]:
        """
        Fetch historical candle data.
        
        Args:
            product_id: Product identifier
            timeframe: Candle timeframe
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of candles
        """
        try:
            # Convert timeframe to Coinbase format
            granularity = self._get_granularity(timeframe)
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.pro.coinbase.com/products/{product_id}/candles"
                params = {
                    "start": datetime.fromtimestamp(start_time).isoformat(),
                    "end": datetime.fromtimestamp(end_time).isoformat(),
                    "granularity": granularity
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Convert to Candle objects
                        candles = [
                            Candle(
                                timestamp=int(entry[0]),
                                open=float(entry[3]),
                                high=float(entry[2]),
                                low=float(entry[1]),
                                close=float(entry[4]),
                                volume=float(entry[5])
                            )
                            for entry in data
                        ]
                        
                        # Sort by timestamp
                        candles.sort(key=lambda x: x.timestamp)
                        return candles
                    else:
                        logger.error(f"Error fetching historical data: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return []
    
    def _get_granularity(self, timeframe: TimeFrame) -> int:
        """Convert timeframe to Coinbase granularity value."""
        granularity_map = {
            TimeFrame.MINUTE_1: 60,
            TimeFrame.MINUTE_5: 300,
            TimeFrame.MINUTE_15: 900,
            TimeFrame.MINUTE_30: 1800,
            TimeFrame.HOUR_1: 3600,
            TimeFrame.HOUR_4: 14400,
            TimeFrame.HOUR_12: 43200,
            TimeFrame.DAY_1: 86400
        }
        return granularity_map[timeframe]
    
    def get_latest_candles(
        self,
        product_id: str,
        timeframe: TimeFrame,
        limit: int = 100
    ) -> List[Candle]:
        """
        Get latest candles for a product and timeframe.
        
        Args:
            product_id: Product identifier
            timeframe: Candle timeframe
            limit: Maximum number of candles to return
            
        Returns:
            List of latest candles
        """
        if product_id not in self.candles or timeframe not in self.candles[product_id]:
            return []
            
        candles = self.candles[product_id][timeframe]
        return candles[-limit:]
    
    def get_latest_trades(
        self,
        product_id: str,
        limit: int = 100
    ) -> List[Trade]:
        """
        Get latest trades for a product.
        
        Args:
            product_id: Product identifier
            limit: Maximum number of trades to return
            
        Returns:
            List of latest trades
        """
        if product_id not in self.trades:
            return []
            
        trades = self.trades[product_id]
        return trades[-limit:]
    
    def get_latest_price(self, product_id: str) -> Optional[float]:
        """
        Get latest price for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Latest price or None if not available
        """
        return self.latest_prices.get(product_id)
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register callback for events.
        
        Args:
            event_type: Event type ('trade', 'candle', or 'error')
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Any):
        """Notify registered callbacks of events."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
    
    def to_dataframe(
        self,
        product_id: str,
        timeframe: TimeFrame,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Convert candle data to DataFrame.
        
        Args:
            product_id: Product identifier
            timeframe: Candle timeframe
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        if product_id not in self.candles or timeframe not in self.candles[product_id]:
            return pd.DataFrame()
            
        candles = self.candles[product_id][timeframe]
        
        # Filter by time range if specified
        if start_time is not None:
            candles = [c for c in candles if c.timestamp >= start_time]
        if end_time is not None:
            candles = [c for c in candles if c.timestamp <= end_time]
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': c.timestamp,
                'open': c.open,
                'high': c.high,
                'low': c.low,
                'close': c.close,
                'volume': c.volume
            }
            for c in candles
        ])
        
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
        
        return df 