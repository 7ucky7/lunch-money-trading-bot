#!/usr/bin/env python3
"""
Order Management Module for Cryptocurrency Trading Bot.

This module handles order creation, submission, execution tracking,
and order state management.
"""

import logging
import time
import datetime
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import threading
import uuid
from dataclasses import dataclass, field

# Local imports
from api_client import CoinbaseClient
from strategy import Signal, SignalType
from risk_manager import PositionSizing

# Get logger
logger = logging.getLogger("OrderManager")

class OrderStatus(Enum):
    """Status of an order."""
    PENDING = "pending"       # Not yet submitted to exchange
    SUBMITTED = "submitted"   # Submitted to exchange
    OPEN = "open"             # Open on exchange
    FILLED = "filled"         # Completely filled
    CANCELED = "canceled"     # Canceled by user or system
    REJECTED = "rejected"     # Rejected by exchange
    EXPIRED = "expired"       # Expired on exchange
    PARTIALLY_FILLED = "partially_filled"  # Partially filled

class OrderType(Enum):
    """Type of order."""
    MARKET = "market"     # Market order
    LIMIT = "limit"       # Limit order
    STOP = "stop"         # Stop order
    STOP_LIMIT = "stop_limit"  # Stop limit order

class OrderSide(Enum):
    """Side of order."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Order data structure."""
    id: str                       # Local order ID
    exchange_id: Optional[str] = None  # Exchange order ID
    product_id: str = ""          # Product ID
    side: OrderSide = OrderSide.BUY  # Order side
    type: OrderType = OrderType.MARKET  # Order type
    size: float = 0.0             # Order size
    price: Optional[float] = None      # Limit price
    funds: Optional[float] = None      # Funds to use (for market orders)
    stop_price: Optional[float] = None  # Stop price
    time_in_force: str = "GTC"    # Time in force
    post_only: bool = False       # Post only flag
    status: OrderStatus = OrderStatus.PENDING  # Order status
    filled_size: float = 0.0      # Filled size
    executed_value: float = 0.0   # Executed value
    fill_fees: float = 0.0        # Fill fees
    created_at: float = field(default_factory=time.time)  # Creation time
    done_at: Optional[float] = None  # Completion time
    done_reason: Optional[str] = None  # Reason for completion
    reject_reason: Optional[str] = None  # Reason for rejection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "exchange_id": self.exchange_id,
            "product_id": self.product_id,
            "side": self.side.value,
            "type": self.type.value,
            "size": self.size,
            "price": self.price,
            "funds": self.funds,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force,
            "post_only": self.post_only,
            "status": self.status.value,
            "filled_size": self.filled_size,
            "executed_value": self.executed_value,
            "fill_fees": self.fill_fees,
            "created_at": self.created_at,
            "done_at": self.done_at,
            "done_reason": self.done_reason,
            "reject_reason": self.reject_reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create Order from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            exchange_id=data.get("exchange_id"),
            product_id=data.get("product_id", ""),
            side=OrderSide(data.get("side", "buy")),
            type=OrderType(data.get("type", "market")),
            size=float(data.get("size", 0.0)),
            price=float(data.get("price")) if data.get("price") is not None else None,
            funds=float(data.get("funds")) if data.get("funds") is not None else None,
            stop_price=float(data.get("stop_price")) if data.get("stop_price") is not None else None,
            time_in_force=data.get("time_in_force", "GTC"),
            post_only=data.get("post_only", False),
            status=OrderStatus(data.get("status", "pending")),
            filled_size=float(data.get("filled_size", 0.0)),
            executed_value=float(data.get("executed_value", 0.0)),
            fill_fees=float(data.get("fill_fees", 0.0)),
            created_at=float(data.get("created_at", time.time())),
            done_at=float(data.get("done_at")) if data.get("done_at") is not None else None,
            done_reason=data.get("done_reason"),
            reject_reason=data.get("reject_reason")
        )
    
    @classmethod
    def from_coinbase_response(cls, response: Dict[str, Any]) -> 'Order':
        """Create Order from Coinbase API response."""
        # Convert Coinbase timestamp to Unix timestamp
        created_at = None
        if "created_at" in response:
            try:
                created_at = datetime.datetime.fromisoformat(response["created_at"].replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                created_at = time.time()
        
        done_at = None
        if "done_at" in response and response["done_at"]:
            try:
                done_at = datetime.datetime.fromisoformat(response["done_at"].replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                done_at = None
        
        # Map Coinbase status to our OrderStatus
        status_map = {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.OPEN,
            "active": OrderStatus.OPEN,
            "done": OrderStatus.FILLED,
            "settled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELED,
            "canceled": OrderStatus.CANCELED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED
        }
        
        status = status_map.get(response.get("status", ""), OrderStatus.PENDING)
        
        # If order is done but not completely filled, mark as partially filled
        if status == OrderStatus.FILLED and response.get("filled_size", "0") != response.get("size", "0"):
            status = OrderStatus.PARTIALLY_FILLED
            
        # Map Coinbase order type to our OrderType
        type_map = {
            "limit": OrderType.LIMIT,
            "market": OrderType.MARKET,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT
        }
        
        order_type = type_map.get(response.get("type", ""), OrderType.MARKET)
        
        return cls(
            id=response.get("client_oid", str(uuid.uuid4())),
            exchange_id=response.get("id"),
            product_id=response.get("product_id", ""),
            side=OrderSide(response.get("side", "buy")),
            type=order_type,
            size=float(response.get("size", 0.0)),
            price=float(response.get("price")) if "price" in response and response["price"] else None,
            funds=float(response.get("funds")) if "funds" in response and response["funds"] else None,
            stop_price=float(response.get("stop_price")) if "stop_price" in response and response["stop_price"] else None,
            time_in_force=response.get("time_in_force", "GTC"),
            post_only=response.get("post_only", False),
            status=status,
            filled_size=float(response.get("filled_size", 0.0)),
            executed_value=float(response.get("executed_value", 0.0)),
            fill_fees=float(response.get("fill_fees", 0.0)),
            created_at=created_at or time.time(),
            done_at=done_at,
            done_reason=response.get("done_reason"),
            reject_reason=response.get("reject_reason")
        )

class OrderManager:
    """
    Order management for cryptocurrency trading.
    
    Responsibilities:
    - Create orders based on signals and position sizing
    - Submit orders to exchange
    - Track order status and fills
    - Cancel and replace orders as needed
    - Record order history
    """
    
    def __init__(self, 
                api_client: CoinbaseClient,
                config: Dict[str, Any] = None):
        """
        Initialize order manager.
        
        Args:
            api_client: API client for order execution
            config: Order management configuration
        """
        self.api_client = api_client
        self.config = config or {}
        
        # Set default parameters if not provided
        self.default_time_in_force = self.config.get("default_time_in_force", "GTC")
        self.default_order_type = self.config.get("default_order_type", OrderType.MARKET)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self.use_post_only = self.config.get("use_post_only", False)
        self.use_client_oid = self.config.get("use_client_oid", True)
        self.sync_interval = self.config.get("sync_interval", 15.0)  # Seconds
        self.auto_cancel_threshold = self.config.get("auto_cancel_threshold", 300.0)  # 5 minutes
        
        # Order tracking
        self.orders: Dict[str, Order] = {}  # Local order ID -> Order
        self.exchange_id_to_local: Dict[str, str] = {}  # Exchange order ID -> Local order ID
        self.product_orders: Dict[str, List[str]] = {}  # Product ID -> List of local order IDs
        
        # Order history
        self.completed_orders: List[Order] = []
        
        # Synchronization
        self._lock = threading.RLock()
        self._sync_thread = None
        self._stop_sync = False
        
        logger.info("Order manager initialized")
    
    def create_order_from_signal(self, 
                               signal: Signal, 
                               position_sizing: PositionSizing) -> Optional[Order]:
        """
        Create an order from a signal and position sizing.
        
        Args:
            signal: Trading signal
            position_sizing: Position sizing details
            
        Returns:
            Created Order or None if failed
        """
        # Validate inputs
        if not signal or not position_sizing:
            logger.error("Cannot create order: missing signal or position sizing")
            return None
            
        if position_sizing.position_size <= 0:
            logger.warning(f"Zero or negative position size: {position_sizing.position_size}")
            return None
            
        # Determine order side
        side = OrderSide.BUY if signal.type == SignalType.BUY else OrderSide.SELL
        
        # Determine order type
        order_type = self.default_order_type
        
        # Generate unique order ID
        order_id = str(uuid.uuid4())
        
        # Create order
        order = Order(
            id=order_id,
            product_id=signal.product_id,
            side=side,
            type=order_type,
            size=position_sizing.position_size,
            price=signal.price if order_type == OrderType.LIMIT else None,
            time_in_force=self.default_time_in_force,
            post_only=self.use_post_only if order_type == OrderType.LIMIT else False
        )
        
        # Set stop loss/take profit if using stop orders
        if position_sizing.stop_loss_price is not None and order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            order.stop_price = position_sizing.stop_loss_price
        
        logger.info(f"Created order: {order_id}, {side.value} {order.size} {signal.product_id} at {signal.price}")
        
        # Store order
        with self._lock:
            self.orders[order_id] = order
            
            if signal.product_id not in self.product_orders:
                self.product_orders[signal.product_id] = []
                
            self.product_orders[signal.product_id].append(order_id)
            
        return order
    
    def submit_order(self, order_id: str) -> Optional[Order]:
        """
        Submit an order to the exchange.
        
        Args:
            order_id: Local order ID
            
        Returns:
            Updated Order or None if failed
        """
        with self._lock:
            if order_id not in self.orders:
                logger.error(f"Cannot submit order: order ID {order_id} not found")
                return None
                
            order = self.orders[order_id]
            
            if order.status != OrderStatus.PENDING:
                logger.warning(f"Order {order_id} already submitted with status {order.status.value}")
                return order
        
        # Prepare order parameters
        params = {}
        
        # Add client_oid if enabled
        if self.use_client_oid:
            params['client_oid'] = order_id
            
        # Prepare order based on type
        if order.type == OrderType.MARKET:
            try:
                # Market order
                if order.size:
                    # Size-based market order
                    response = self.api_client.place_market_order(
                        product_id=order.product_id,
                        side=order.side.value,
                        size=str(order.size),
                        **params
                    )
                elif order.funds:
                    # Funds-based market order
                    response = self.api_client.place_market_order(
                        product_id=order.product_id,
                        side=order.side.value,
                        funds=str(order.funds),
                        **params
                    )
                else:
                    logger.error(f"Market order {order_id} has neither size nor funds")
                    order.status = OrderStatus.REJECTED
                    order.reject_reason = "Missing size or funds"
                    return order
            except Exception as e:
                logger.error(f"Error submitting market order {order_id}: {str(e)}")
                self._handle_submission_error(order, str(e))
                return order
                
        elif order.type == OrderType.LIMIT:
            # Limit order
            if not order.price:
                logger.error(f"Limit order {order_id} has no price")
                order.status = OrderStatus.REJECTED
                order.reject_reason = "Missing price"
                return order
                
            try:
                params.update({
                    'time_in_force': order.time_in_force,
                    'post_only': order.post_only
                })
                
                response = self.api_client.place_limit_order(
                    product_id=order.product_id,
                    side=order.side.value,
                    price=str(order.price),
                    size=str(order.size),
                    **params
                )
            except Exception as e:
                logger.error(f"Error submitting limit order {order_id}: {str(e)}")
                self._handle_submission_error(order, str(e))
                return order
                
        else:
            # Other order types not directly supported by simple client, use advanced if available
            logger.error(f"Order type {order.type.value} not supported by simple client")
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"Unsupported order type: {order.type.value}"
            return order
        
        # Process response
        try:
            if 'id' in response:
                # Update order with exchange response
                exchange_id = response['id']
                
                with self._lock:
                    # Update order data
                    order.exchange_id = exchange_id
                    order.status = OrderStatus.OPEN
                    
                    # Map exchange ID to local ID
                    self.exchange_id_to_local[exchange_id] = order_id
                    
                    # Update order with additional fields from response
                    if 'created_at' in response:
                        try:
                            created_at = datetime.datetime.fromisoformat(
                                response["created_at"].replace("Z", "+00:00")
                            ).timestamp()
                            order.created_at = created_at
                        except (ValueError, TypeError):
                            pass
                    
                    # Update sizes, fees, etc. if provided
                    if 'filled_size' in response:
                        order.filled_size = float(response.get('filled_size', 0))
                    if 'executed_value' in response:
                        order.executed_value = float(response.get('executed_value', 0))
                    if 'fill_fees' in response:
                        order.fill_fees = float(response.get('fill_fees', 0))
                    
                    # Check if already filled
                    if response.get('status') == 'done' and response.get('done_reason') == 'filled':
                        order.status = OrderStatus.FILLED
                        order.done_at = time.time()
                        order.done_reason = 'filled'
                
                logger.info(f"Order {order_id} submitted successfully, exchange ID: {exchange_id}")
                return order
            else:
                # No ID in response, likely an error
                error_message = response.get('message', 'Unknown error')
                logger.error(f"Error submitting order {order_id}: {error_message}")
                self._handle_submission_error(order, error_message)
                return order
                
        except Exception as e:
            # Error processing response
            logger.error(f"Error processing order response for {order_id}: {str(e)}")
            self._handle_submission_error(order, str(e))
            return order
    
    def _handle_submission_error(self, order: Order, error_message: str) -> None:
        """Handle order submission error."""
        order.status = OrderStatus.REJECTED
        order.reject_reason = error_message
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Local order ID
            
        Returns:
            True if cancellation was successful or attempted
        """
        with self._lock:
            if order_id not in self.orders:
                logger.error(f"Cannot cancel order: order ID {order_id} not found")
                return False
                
            order = self.orders[order_id]
            
            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
                logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
                return False
                
            if not order.exchange_id:
                # Order hasn't been submitted to exchange yet
                order.status = OrderStatus.CANCELED
                order.done_at = time.time()
                order.done_reason = "canceled_before_submission"
                logger.info(f"Order {order_id} canceled before submission")
                return True
        
        # Cancel on exchange
        try:
            response = self.api_client.cancel_order(order.exchange_id)
            
            # Coinbase returns the order ID on successful cancellation
            if response:
                with self._lock:
                    order.status = OrderStatus.CANCELED
                    order.done_at = time.time()
                    order.done_reason = "canceled_by_user"
                    
                logger.info(f"Order {order_id} canceled successfully")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}")
                return False
                
        except Exception as e:
            # Check if the error is because the order was already filled
            error_str = str(e).lower()
            if "not found" in error_str:
                # Order not found, probably already filled or canceled
                self.sync_order(order_id)
                logger.warning(f"Order {order_id} not found on exchange during cancellation")
                return True
            
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    def cancel_all_orders(self, product_id: Optional[str] = None) -> int:
        """
        Cancel all open orders, optionally for a specific product.
        
        Args:
            product_id: Optional product ID to filter orders
            
        Returns:
            Number of orders canceled
        """
        try:
            # Cancel on exchange
            canceled_ids = self.api_client.cancel_all(product_id=product_id)
            
            # Update local order states
            count = 0
            with self._lock:
                for exchange_id in canceled_ids:
                    if exchange_id in self.exchange_id_to_local:
                        local_id = self.exchange_id_to_local[exchange_id]
                        if local_id in self.orders:
                            order = self.orders[local_id]
                            order.status = OrderStatus.CANCELED
                            order.done_at = time.time()
                            order.done_reason = "canceled_by_user_bulk"
                            count += 1
            
            logger.info(f"Canceled {count} orders{' for ' + product_id if product_id else ''}")
            return count
            
        except Exception as e:
            logger.error(f"Error canceling all orders: {str(e)}")
            return 0
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by local ID.
        
        Args:
            order_id: Local order ID
            
        Returns:
            Order or None if not found
        """
        with self._lock:
            return self.orders.get(order_id)
    
    def get_orders_by_product(self, product_id: str) -> List[Order]:
        """
        Get all orders for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            List of Orders
        """
        with self._lock:
            if product_id not in self.product_orders:
                return []
                
            return [self.orders[order_id] for order_id in self.product_orders[product_id] 
                   if order_id in self.orders]
    
    def get_open_orders(self, product_id: Optional[str] = None) -> List[Order]:
        """
        Get all open orders, optionally filtered by product.
        
        Args:
            product_id: Optional product ID to filter orders
            
        Returns:
            List of open Orders
        """
        with self._lock:
            if product_id:
                if product_id not in self.product_orders:
                    return []
                    
                return [self.orders[order_id] for order_id in self.product_orders[product_id] 
                       if order_id in self.orders and 
                       self.orders[order_id].status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)]
            else:
                return [order for order in self.orders.values() 
                       if order.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)]
    
    def get_completed_orders(self, product_id: Optional[str] = None, limit: int = 100) -> List[Order]:
        """
        Get completed orders, optionally filtered by product.
        
        Args:
            product_id: Optional product ID to filter orders
            limit: Maximum number of orders to return
            
        Returns:
            List of completed Orders
        """
        with self._lock:
            if product_id:
                completed = [order for order in self.completed_orders 
                            if order.product_id == product_id]
            else:
                completed = self.completed_orders.copy()
                
            # Sort by completion time, newest first
            completed.sort(key=lambda o: o.done_at or 0, reverse=True)
            
            return completed[:limit]
    
    def sync_order(self, order_id: str) -> bool:
        """
        Sync an order's state with the exchange.
        
        Args:
            order_id: Local order ID
            
        Returns:
            True if sync was successful
        """
        with self._lock:
            if order_id not in self.orders:
                logger.error(f"Cannot sync order: order ID {order_id} not found")
                return False
                
            order = self.orders[order_id]
            
            if not order.exchange_id:
                logger.warning(f"Cannot sync order {order_id}: no exchange ID")
                return False
                
            # Don't sync already completed orders
            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
                return True
        
        # Fetch from exchange
        try:
            response = self.api_client.get_order(order.exchange_id)
            
            if not response:
                logger.error(f"No response when syncing order {order_id}")
                return False
                
            with self._lock:
                # Update order from response
                updated_order = Order.from_coinbase_response(response)
                
                # Update fields
                order.status = updated_order.status
                order.filled_size = updated_order.filled_size
                order.executed_value = updated_order.executed_value
                order.fill_fees = updated_order.fill_fees
                
                if updated_order.done_at:
                    order.done_at = updated_order.done_at
                    order.done_reason = updated_order.done_reason
                
                # If order is done, move to completed_orders
                if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
                    logger.info(f"Order {order_id} completed with status {order.status.value}")
                    
                    # Add to completed orders
                    self.completed_orders.append(order)
                    
                    # If we have too many completed orders, remove oldest
                    max_completed = self.config.get("max_completed_orders", 1000)
                    if len(self.completed_orders) > max_completed:
                        self.completed_orders = self.completed_orders[-max_completed:]
            
            return True
            
        except Exception as e:
            logger.error(f"Error syncing order {order_id}: {str(e)}")
            return False
    
    def sync_all_open_orders(self) -> int:
        """
        Sync all open orders with the exchange.
        
        Returns:
            Number of orders synced
        """
        open_orders = self.get_open_orders()
        if not open_orders:
            return 0
            
        count = 0
        for order in open_orders:
            if self.sync_order(order.id):
                count += 1
                
        logger.debug(f"Synced {count} open orders with exchange")
        return count
    
    def start_sync_thread(self) -> None:
        """Start the order sync thread."""
        if self._sync_thread and self._sync_thread.is_alive():
            logger.warning("Sync thread already running")
            return
            
        self._stop_sync = False
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            name="OrderSyncThread"
        )
        self._sync_thread.daemon = True
        self._sync_thread.start()
        
        logger.info("Started order sync thread")
    
    def stop_sync_thread(self) -> None:
        """Stop the order sync thread."""
        if not self._sync_thread or not self._sync_thread.is_alive():
            logger.debug("Sync thread not running")
            return
            
        self._stop_sync = True
        self._sync_thread.join(timeout=10.0)
        
        if self._sync_thread.is_alive():
            logger.warning("Sync thread did not stop cleanly")
        else:
            logger.info("Stopped order sync thread")
    
    def _sync_loop(self) -> None:
        """Order sync loop."""
        while not self._stop_sync:
            try:
                # Sync open orders
                self.sync_all_open_orders()
                
                # Check for stale orders to cancel
                self._check_stale_orders()
                
            except Exception as e:
                logger.error(f"Error in order sync loop: {str(e)}")
                
            # Sleep until next sync
            for _ in range(int(self.sync_interval * 2)):  # Check for stop twice per interval
                if self._stop_sync:
                    break
                time.sleep(0.5)
    
    def _check_stale_orders(self) -> None:
        """Check for stale orders that should be canceled."""
        now = time.time()
        auto_cancel_threshold = self.auto_cancel_threshold
        
        if auto_cancel_threshold <= 0:
            return  # Auto-cancel disabled
            
        with self._lock:
            for order in self.orders.values():
                if order.status not in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
                    continue
                    
                # Check if order is older than threshold
                if now - order.created_at > auto_cancel_threshold:
                    logger.info(f"Canceling stale order {order.id} (age: {now - order.created_at:.1f}s)")
                    
                    # Release lock before canceling
                    self._lock.release()
                    try:
                        self.cancel_order(order.id)
                    finally:
                        self._lock.acquire()
    
    def save_orders_to_file(self, filename: str) -> bool:
        """
        Save orders to a JSON file.
        
        Args:
            filename: Path to save file
            
        Returns:
            True if successful
        """
        try:
            data = {
                "current_orders": {order_id: order.to_dict() for order_id, order in self.orders.items()},
                "completed_orders": [order.to_dict() for order in self.completed_orders],
                "exchange_id_map": self.exchange_id_to_local,
                "product_orders": self.product_orders,
                "timestamp": time.time()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved orders to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving orders to {filename}: {str(e)}")
            return False
    
    def load_orders_from_file(self, filename: str) -> bool:
        """
        Load orders from a JSON file.
        
        Args:
            filename: Path to load file
            
        Returns:
            True if successful
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            with self._lock:
                # Load current orders
                self.orders = {
                    order_id: Order.from_dict(order_data) 
                    for order_id, order_data in data.get("current_orders", {}).items()
                }
                
                # Load completed orders
                self.completed_orders = [
                    Order.from_dict(order_data) 
                    for order_data in data.get("completed_orders", [])
                ]
                
                # Load exchange ID map
                self.exchange_id_to_local = data.get("exchange_id_map", {})
                
                # Load product orders
                self.product_orders = data.get("product_orders", {})
                
            logger.info(f"Loaded orders from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading orders from {filename}: {str(e)}")
            return False 