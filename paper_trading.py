#!/usr/bin/env python3
"""
Paper Trading Module.

Simulates exchange interactions for testing and simulation without using real funds.
"""

import logging
import time
import json
import asyncio
import random
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

from order_executor import OrderExecutor, Order, OrderStatus
from order_validator import OrderType, OrderSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PaperTrading")

@dataclass
class AccountBalance:
    """Simulated account balance for paper trading."""
    available: Dict[str, float] = field(default_factory=dict)
    hold: Dict[str, float] = field(default_factory=dict)
    
    def get_available(self, currency: str) -> float:
        """Get available balance for a currency."""
        return self.available.get(currency, 0.0)
    
    def get_hold(self, currency: str) -> float:
        """Get hold amount for a currency."""
        return self.hold.get(currency, 0.0)
    
    def get_total(self, currency: str) -> float:
        """Get total balance for a currency."""
        return self.get_available(currency) + self.get_hold(currency)
    
    def add(self, currency: str, amount: float) -> None:
        """Add to available balance."""
        self.available[currency] = self.get_available(currency) + amount
    
    def hold_funds(self, currency: str, amount: float) -> bool:
        """Move funds from available to hold."""
        available = self.get_available(currency)
        if available < amount:
            return False
        
        self.available[currency] = available - amount
        self.hold[currency] = self.get_hold(currency) + amount
        return True
    
    def release_hold(self, currency: str, amount: float) -> None:
        """Release funds from hold back to available."""
        hold_amount = self.get_hold(currency)
        release_amount = min(amount, hold_amount)
        
        self.hold[currency] = hold_amount - release_amount
        self.available[currency] = self.get_available(currency) + release_amount
    
    def execute_trade(
        self, 
        base_currency: str, 
        quote_currency: str, 
        side: OrderSide, 
        size: float, 
        price: float, 
        fee_rate: float = 0.005
    ) -> float:
        """
        Execute a trade in the paper account.
        
        Args:
            base_currency: Base currency (e.g., BTC in BTC-USD)
            quote_currency: Quote currency (e.g., USD in BTC-USD)
            side: Order side (buy/sell)
            size: Size in base currency
            price: Execution price
            fee_rate: Fee rate (default 0.5%)
            
        Returns:
            Fee amount in quote currency
        """
        quote_amount = size * price
        fee = quote_amount * fee_rate
        
        if side == OrderSide.BUY:
            # For buy: release quote currency from hold, add base currency
            self.release_hold(quote_currency, quote_amount + fee)
            self.add(base_currency, size)
        else:
            # For sell: release base currency from hold, add quote currency minus fee
            self.release_hold(base_currency, size)
            self.add(quote_currency, quote_amount - fee)
        
        return fee


class PaperTradingExecutor(OrderExecutor):
    """
    Paper trading implementation of OrderExecutor.
    
    Simulates exchange interactions for testing without using real funds.
    """
    
    def __init__(
        self,
        initial_balances: Dict[str, float] = None,
        market_data_provider: Any = None,
        fee_rate: float = 0.005,
        latency_ms: Tuple[int, int] = (50, 200),
        fill_probability: Dict[str, float] = None,
        volatility_pct: float = 0.1,
        slippage_pct: float = 0.05
    ):
        """
        Initialize paper trading executor.
        
        Args:
            initial_balances: Initial account balances by currency
            market_data_provider: Optional market data provider for realistic simulation
            fee_rate: Trading fee rate
            latency_ms: Simulated API latency range (min, max) in milliseconds
            fill_probability: Probability of fill by order type, defaults to 100% for market, 20% for limit
            volatility_pct: Simulated market volatility percentage
            slippage_pct: Simulated slippage percentage
        """
        # Initialize with dummy API credentials
        super().__init__(
            exchange_api_url="https://paper-trading-api",
            api_key="paper_key",
            api_secret="paper_secret"
        )
        
        # Paper trading specific attributes
        self.account = AccountBalance()
        self.market_data = market_data_provider
        self.fee_rate = fee_rate
        self.latency_range = latency_ms
        self.fill_probability = fill_probability or {
            OrderType.MARKET.value: 1.0,
            OrderType.LIMIT.value: 0.2,
            OrderType.STOP.value: 0.3,
            OrderType.STOP_LIMIT.value: 0.2
        }
        self.volatility = volatility_pct
        self.slippage = slippage_pct
        
        # Price cache for simulation
        self.price_cache: Dict[str, float] = {}
        
        # Market simulation task
        self.market_simulation_task = None
        
        # Set initial balances
        if initial_balances:
            for currency, amount in initial_balances.items():
                self.account.add(currency, amount)
        
        logger.info("Paper Trading Executor initialized")
    
    async def start_market_simulation(self, update_interval: float = 1.0) -> None:
        """
        Start market simulation task.
        
        Args:
            update_interval: Price update interval in seconds
        """
        if self.market_simulation_task:
            return
        
        self.market_simulation_task = asyncio.create_task(
            self._run_market_simulation(update_interval)
        )
        logger.info("Market simulation started")
    
    async def stop_market_simulation(self) -> None:
        """Stop market simulation task."""
        if self.market_simulation_task:
            self.market_simulation_task.cancel()
            try:
                await self.market_simulation_task
            except asyncio.CancelledError:
                pass
            self.market_simulation_task = None
            logger.info("Market simulation stopped")
    
    async def _run_market_simulation(self, update_interval: float) -> None:
        """
        Run market simulation.
        
        Args:
            update_interval: Price update interval
        """
        try:
            while True:
                # Update prices for all products in the cache
                for product_id in list(self.price_cache.keys()):
                    await self._update_simulated_price(product_id)
                
                # Check for order fills
                await self._check_order_fills()
                
                # Wait for next update
                await asyncio.sleep(update_interval)
        except asyncio.CancelledError:
            logger.info("Market simulation task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in market simulation: {e}")
    
    async def _update_simulated_price(self, product_id: str) -> float:
        """
        Update simulated price for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Updated price
        """
        current_price = self.price_cache.get(product_id)
        
        # If we have a market data provider, use it
        if self.market_data and hasattr(self.market_data, "get_current_price"):
            try:
                real_price = await self.market_data.get_current_price(product_id)
                if real_price:
                    self.price_cache[product_id] = real_price
                    return real_price
            except Exception as e:
                logger.warning(f"Error getting price from market data provider: {e}")
        
        # Otherwise use simulation
        if not current_price:
            # Initialize with reasonable defaults if no price
            base, quote = product_id.split("-")
            if base == "BTC":
                current_price = 30000.0
            elif base == "ETH":
                current_price = 2000.0
            else:
                current_price = 100.0
        
        # Apply random walk with mean reversion
        price_change_pct = random.normalvariate(0, self.volatility / 3)
        new_price = current_price * (1 + price_change_pct)
        
        # Ensure price is positive
        new_price = max(new_price, 0.01)
        
        # Update cache
        self.price_cache[product_id] = new_price
        return new_price
    
    async def _check_order_fills(self) -> None:
        """Check for potential order fills based on current prices."""
        for order_id, order in list(self.orders.items()):
            if not order.is_active() or not order.id:
                continue
            
            # Get current price
            price = self.price_cache.get(order.product_id)
            if not price:
                price = await self._update_simulated_price(order.product_id)
            
            # Check if order should be filled
            if await self._should_fill_order(order, price):
                await self._fill_order(order, price)
    
    async def _should_fill_order(self, order: Order, current_price: float) -> bool:
        """
        Determine if an order should be filled.
        
        Args:
            order: Order to check
            current_price: Current simulated price
            
        Returns:
            True if order should be filled
        """
        # Market orders always fill
        if order.type == OrderType.MARKET:
            return True
        
        # Random fill based on probability
        if random.random() > self.fill_probability.get(order.type.value, 0.2):
            return False
        
        # For limit orders: buy fills if price <= limit price, sell fills if price >= limit price
        if order.type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and current_price <= order.price:
                return True
            if order.side == OrderSide.SELL and current_price >= order.price:
                return True
        
        # For stop orders: buy fills if price >= stop price, sell fills if price <= stop price
        if order.type == OrderType.STOP:
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                return True
            if order.side == OrderSide.SELL and current_price <= order.stop_price:
                return True
        
        # For stop-limit orders: first activate when stop price is hit, then behave like limit
        if order.type == OrderType.STOP_LIMIT:
            if "stop_triggered" not in order.metadata:
                # Check if stop price is hit
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    order.metadata["stop_triggered"] = True
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    order.metadata["stop_triggered"] = True
                else:
                    return False
            
            # Now act like a limit order
            if order.side == OrderSide.BUY and current_price <= order.price:
                return True
            if order.side == OrderSide.SELL and current_price >= order.price:
                return True
        
        return False
    
    async def _fill_order(self, order: Order, base_price: float) -> None:
        """
        Simulate filling an order.
        
        Args:
            order: Order to fill
            base_price: Base price for fill
        """
        # Calculate fill price with slippage
        direction = 1 if order.side == OrderSide.BUY else -1
        slippage_factor = 1 + (direction * self.slippage * random.random())
        
        if order.type == OrderType.MARKET:
            # Market orders get the base price with slippage
            fill_price = base_price * slippage_factor
        elif order.type == OrderType.LIMIT:
            # Limit orders get the limit price or better
            if order.side == OrderSide.BUY:
                fill_price = min(base_price * slippage_factor, order.price)
            else:
                fill_price = max(base_price * slippage_factor, order.price)
        elif order.type == OrderType.STOP:
            # Stop orders get the base price with slippage after triggering
            fill_price = base_price * slippage_factor
        elif order.type == OrderType.STOP_LIMIT:
            # Stop-limit orders get filled at limit price or better
            if order.side == OrderSide.BUY:
                fill_price = min(base_price * slippage_factor, order.price)
            else:
                fill_price = max(base_price * slippage_factor, order.price)
        else:
            # Default to base price
            fill_price = base_price
        
        # Determine fill size
        if order.size:
            fill_size = order.size
        elif order.funds:
            # For market buys with funds, calculate size based on fill price
            fill_size = order.funds / fill_price
        else:
            logger.error(f"Cannot fill order without size or funds: {order.id}")
            return
        
        # Extract product currencies
        try:
            base_currency, quote_currency = order.product_id.split("-")
        except ValueError:
            logger.error(f"Invalid product ID format: {order.product_id}")
            return
        
        # Execute the trade in the account
        quote_amount = fill_size * fill_price
        fee = quote_amount * self.fee_rate
        
        try:
            if order.side == OrderSide.BUY:
                # Check and hold funds
                if not self.account.hold_funds(quote_currency, quote_amount + fee):
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = "insufficient_funds"
                    self._notify_order_update(order)
                    return
            else:
                # Check and hold base currency
                if not self.account.hold_funds(base_currency, fill_size):
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = "insufficient_funds"
                    self._notify_order_update(order)
                    return
            
            # Execute the trade in the account
            fee_amount = self.account.execute_trade(
                base_currency=base_currency,
                quote_currency=quote_currency,
                side=order.side,
                size=fill_size,
                price=fill_price,
                fee_rate=self.fee_rate
            )
            
            # Update order
            order.filled_size = fill_size
            order.filled_price = fill_price
            order.fill_fees = fee_amount
            order.status = OrderStatus.FILLED
            order.done_at = datetime.now()
            order.done_reason = "filled"
            
            logger.info(f"Filled paper order: {order.id} {order.side.value} {fill_size} {order.product_id} @ {fill_price}")
            self._notify_order_update(order)
            
        except Exception as e:
            logger.error(f"Error filling paper order {order.id}: {e}")
            order.status = OrderStatus.FAILED
            order.reject_reason = str(e)
            self._notify_order_update(order)
    
    async def place_order(
        self,
        product_id: str,
        side: OrderSide,
        order_type: OrderType,
        size: Optional[float] = None,
        funds: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Order]:
        """
        Place a new paper order.
        
        Args:
            product_id: Product identifier
            side: Order side (buy/sell)
            order_type: Order type
            size: Order size in base currency
            funds: Order size in quote currency
            price: Limit price
            stop_price: Stop price
            client_id: Client-assigned order ID
            metadata: Additional order metadata
            
        Returns:
            Placed order if successful, None otherwise
        """
        # Extract currencies from product
        try:
            base_currency, quote_currency = product_id.split("-")
        except ValueError:
            logger.error(f"Invalid product ID format: {product_id}")
            return None
        
        # Generate client ID if not provided
        if not client_id:
            client_id = f"paper_{int(time.time() * 1000)}_{product_id}"
            
        # Create order object
        order = Order(
            id=f"paper-{client_id}",  # Assign a paper order ID
            client_id=client_id,
            product_id=product_id,
            side=side,
            type=order_type,
            status=OrderStatus.PENDING,
            size=size,
            funds=funds,
            price=price,
            stop_price=stop_price,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Check if we need to generate price for market orders
        if order_type == OrderType.MARKET and not self.price_cache.get(product_id):
            await self._update_simulated_price(product_id)
        
        # Store order in tracking dict
        self.orders[order.id] = order
        self.client_order_map[client_id] = order.id
        
        # Simulate API latency
        latency = random.randint(self.latency_range[0], self.latency_range[1]) / 1000.0
        await asyncio.sleep(latency)
        
        # Preliminary funds check
        if side == OrderSide.BUY:
            # For buy orders, check quote currency
            required_funds = 0.0
            
            if size and price:
                # Limit/stop orders with size
                required_funds = size * price * (1 + self.fee_rate)
            elif funds:
                # Market orders with funds
                required_funds = funds * (1 + self.fee_rate)
            elif size and self.price_cache.get(product_id):
                # Market orders with size
                required_funds = size * self.price_cache[product_id] * (1 + self.fee_rate)
            
            if required_funds > 0 and self.account.get_available(quote_currency) < required_funds:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "insufficient_funds"
                self._notify_order_update(order)
                return None
                
        else:  # SELL
            # For sell orders, check base currency
            if size and self.account.get_available(base_currency) < size:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "insufficient_funds"
                self._notify_order_update(order)
                return None
        
        # Update order status
        order.status = OrderStatus.OPEN
        self._notify_order_update(order)
        
        # For market orders, fill immediately
        if order_type == OrderType.MARKET:
            current_price = self.price_cache.get(product_id, 0)
            if current_price > 0:
                await self._fill_order(order, current_price)
            
        logger.info(f"Placed paper order: {order.to_dict()}")
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a paper order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if cancellation was successful
        """
        # Check if order exists
        order = self.orders.get(order_id)
        if not order:
            # Try to look up by client ID
            mapped_id = self.client_order_map.get(order_id)
            if mapped_id:
                order = self.orders.get(mapped_id)
            
            if not order:
                logger.warning(f"Order not found for cancellation: {order_id}")
                return False
                
        # Can only cancel active orders
        if not order.is_active():
            logger.warning(f"Cannot cancel non-active order: {order.to_dict()}")
            return False
            
        # Simulate API latency
        latency = random.randint(self.latency_range[0], self.latency_range[1]) / 1000.0
        await asyncio.sleep(latency)
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        order.done_at = datetime.now()
        order.done_reason = "canceled"
        
        # Release any held funds
        try:
            base_currency, quote_currency = order.product_id.split("-")
            
            if order.side == OrderSide.BUY and order.price:
                # Release quote currency
                amount = (order.size or 0) * order.price
                if amount > 0:
                    self.account.release_hold(quote_currency, amount * (1 + self.fee_rate))
            elif order.side == OrderSide.SELL:
                # Release base currency
                if order.size:
                    self.account.release_hold(base_currency, order.size)
        except Exception as e:
            logger.error(f"Error releasing funds on cancel: {e}")
        
        logger.info(f"Cancelled paper order: {order.to_dict()}")
        self._notify_order_update(order)
        return True
    
    async def cancel_all_orders(self, product_id: Optional[str] = None) -> int:
        """
        Cancel all open paper orders.
        
        Args:
            product_id: Optional product ID filter
            
        Returns:
            Number of orders cancelled
        """
        # Simulate API latency
        latency = random.randint(self.latency_range[0], self.latency_range[1]) / 1000.0
        await asyncio.sleep(latency)
        
        # Find orders to cancel
        orders_to_cancel = [
            order_id for order_id, order in self.orders.items()
            if order.is_active() and (not product_id or order.product_id == product_id)
        ]
        
        # Cancel each order
        cancelled_count = 0
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
                
        logger.info(f"Cancelled {cancelled_count} paper orders")
        return cancelled_count
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get paper order details.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order if found
        """
        # Look up order
        order = self.orders.get(order_id)
        if not order:
            # Try to look up by client ID
            mapped_id = self.client_order_map.get(order_id)
            if mapped_id:
                order = self.orders.get(mapped_id)
                
        # Simulate API latency
        latency = random.randint(self.latency_range[0], self.latency_range[1]) / 1000.0
        await asyncio.sleep(latency)
        
        return order
    
    def get_account_balance(self) -> AccountBalance:
        """Get current account balance."""
        return self.account
    
    def add_funds(self, currency: str, amount: float) -> None:
        """Add funds to paper trading account."""
        self.account.add(currency, amount)
        logger.info(f"Added {amount} {currency} to paper trading account")
    
    def reset_account(self, initial_balances: Dict[str, float] = None) -> None:
        """
        Reset paper trading account.
        
        Args:
            initial_balances: Optional initial balances
        """
        self.account = AccountBalance()
        
        if initial_balances:
            for currency, amount in initial_balances.items():
                self.account.add(currency, amount)
                
        logger.info("Reset paper trading account")

    async def _call_api(self, *args, **kwargs) -> Optional[Any]:
        """
        Override API call method to prevent real API calls.
        
        Returns:
            None to indicate API call not implemented in paper trading
        """
        logger.warning("Paper trading does not make real API calls")
        return None 