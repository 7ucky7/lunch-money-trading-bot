#!/usr/bin/env python3
"""
Order Executor Module.

Handles order execution and management for the cryptocurrency trading bot by
interfacing with exchange APIs.
"""

import logging
import time
import json
import hmac
import hashlib
import base64
import requests
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from order_validator import OrderType, OrderSide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OrderExecutor")

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"

@dataclass
class Order:
    """Order dataclass for tracking orders."""
    id: str
    client_id: str
    product_id: str
    side: OrderSide
    type: OrderType
    status: OrderStatus
    size: Optional[float] = None
    funds: Optional[float] = None
    price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_size: float = 0.0
    filled_price: Optional[float] = None
    fill_fees: float = 0.0
    created_at: datetime = None
    done_at: Optional[datetime] = None
    done_reason: Optional[str] = None
    reject_reason: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary for API requests or storage."""
        result = {
            "id": self.id,
            "client_id": self.client_id,
            "product_id": self.product_id,
            "side": self.side.value,
            "type": self.type.value,
            "status": self.status.value
        }
        
        if self.size is not None:
            result["size"] = str(self.size)
        if self.funds is not None:
            result["funds"] = str(self.funds)
        if self.price is not None:
            result["price"] = str(self.price)
        if self.stop_price is not None:
            result["stop_price"] = str(self.stop_price)
        
        result["filled_size"] = str(self.filled_size)
        if self.filled_price is not None:
            result["filled_price"] = str(self.filled_price)
        result["fill_fees"] = str(self.fill_fees)
        
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.done_at:
            result["done_at"] = self.done_at.isoformat()
        if self.done_reason:
            result["done_reason"] = self.done_reason
        if self.reject_reason:
            result["reject_reason"] = self.reject_reason
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create order from dictionary (API response or stored data)."""
        # Handle required fields
        order_id = data.get("id", "")
        client_id = data.get("client_id", "")
        product_id = data.get("product_id", "")
        side = OrderSide(data.get("side", "buy"))
        order_type = OrderType(data.get("type", "market"))
        status = OrderStatus(data.get("status", "pending"))
        
        # Handle optional fields
        size = float(data.get("size", 0)) if data.get("size") else None
        funds = float(data.get("funds", 0)) if data.get("funds") else None
        price = float(data.get("price", 0)) if data.get("price") else None
        stop_price = float(data.get("stop_price", 0)) if data.get("stop_price") else None
        
        filled_size = float(data.get("filled_size", 0))
        filled_price = float(data.get("filled_price", 0)) if data.get("filled_price") else None
        fill_fees = float(data.get("fill_fees", 0))
        
        created_at = datetime.fromisoformat(data.get("created_at")) if data.get("created_at") else None
        done_at = datetime.fromisoformat(data.get("done_at")) if data.get("done_at") else None
        
        done_reason = data.get("done_reason")
        reject_reason = data.get("reject_reason")
        metadata = data.get("metadata", {})
        
        return cls(
            id=order_id,
            client_id=client_id,
            product_id=product_id,
            side=side,
            type=order_type,
            status=status,
            size=size,
            funds=funds,
            price=price,
            stop_price=stop_price,
            filled_size=filled_size,
            filled_price=filled_price,
            fill_fees=fill_fees,
            created_at=created_at,
            done_at=done_at,
            done_reason=done_reason,
            reject_reason=reject_reason,
            metadata=metadata
        )

    def is_active(self) -> bool:
        """Check if order is active (pending or open)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN]
    
    def is_done(self) -> bool:
        """Check if order is done (filled, cancelled, rejected, expired, or failed)."""
        return not self.is_active()
    
    def update_from_response(self, response_data: Dict[str, Any]) -> None:
        """Update order from API response."""
        # Update status
        if "status" in response_data:
            try:
                self.status = OrderStatus(response_data["status"])
            except ValueError:
                logger.warning(f"Unknown order status: {response_data['status']}")
        
        # Update order ID if it was pending
        if self.id == "" and "id" in response_data:
            self.id = response_data["id"]
            
        # Update fill information
        if "filled_size" in response_data:
            self.filled_size = float(response_data["filled_size"])
        if "executed_value" in response_data and self.filled_size > 0:
            self.filled_price = float(response_data["executed_value"]) / self.filled_size
        if "fill_fees" in response_data:
            self.fill_fees = float(response_data["fill_fees"])
            
        # Update timestamps
        if "created_at" in response_data:
            self.created_at = datetime.fromisoformat(response_data["created_at"].replace('Z', '+00:00'))
        if "done_at" in response_data:
            self.done_at = datetime.fromisoformat(response_data["done_at"].replace('Z', '+00:00'))
            
        # Update done reason
        if "done_reason" in response_data:
            self.done_reason = response_data["done_reason"]
        
        # Update reject reason
        if "reject_reason" in response_data:
            self.reject_reason = response_data["reject_reason"]


class OrderExecutor:
    """
    Handles order execution by interfacing with exchange APIs.
    
    Features:
    - Order placement, cancellation, and status checking
    - Retry logic for API failures
    - Order tracking and management
    - Support for all order types
    - Event callbacks for order status changes
    """
    
    def __init__(
        self,
        exchange_api_url: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize order executor.
        
        Args:
            exchange_api_url: Exchange API URL
            api_key: API key
            api_secret: API secret
            api_passphrase: API passphrase (for some exchanges)
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        self.api_url = exchange_api_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Order tracking
        self.orders: Dict[str, Order] = {}  # Order ID -> Order
        self.client_order_map: Dict[str, str] = {}  # Client ID -> Order ID
        
        # Callbacks
        self.order_update_callbacks: List[Callable[[Order], None]] = []
        
        logger.info("Order Executor initialized")
    
    def register_order_update_callback(self, callback: Callable[[Order], None]) -> None:
        """Register callback for order updates."""
        self.order_update_callbacks.append(callback)
        logger.info(f"Registered order update callback: {callback.__name__}")
    
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
        Place a new order.
        
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
        # Generate client ID if not provided
        if not client_id:
            client_id = f"bot_{int(time.time() * 1000)}_{product_id}"
            
        # Create order object
        order = Order(
            id="",  # Will be assigned by exchange
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
        
        # Prepare request body
        request_data = {
            "product_id": product_id,
            "side": side.value,
            "type": order_type.value,
            "client_oid": client_id
        }
        
        if size is not None:
            request_data["size"] = str(size)
        if funds is not None:
            request_data["funds"] = str(funds)
        if price is not None:
            request_data["price"] = str(price)
        if stop_price is not None:
            request_data["stop_price"] = str(stop_price)
            request_data["stop"] = "loss" if side == OrderSide.SELL else "entry"
            
        # Store order in tracking dict
        self.orders[client_id] = order
        
        # Send API request
        response = await self._call_api("POST", "/orders", request_data)
        
        if not response or "id" not in response:
            order.status = OrderStatus.FAILED
            order.reject_reason = "API request failed"
            logger.error(f"Failed to place order: {order.to_dict()}")
            self._notify_order_update(order)
            return None
            
        # Update order with response data
        order.update_from_response(response)
        
        # Update order tracking
        if order.id:
            self.orders[order.id] = order
            self.client_order_map[client_id] = order.id
            
        logger.info(f"Placed order: {order.to_dict()}")
        self._notify_order_update(order)
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if cancellation was successful, False otherwise
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
            
        # Send API request
        response = await self._call_api("DELETE", f"/orders/{order.id}")
        
        if not response:
            logger.error(f"Failed to cancel order: {order.to_dict()}")
            return False
            
        # Update order status
        order.status = OrderStatus.CANCELLED
        order.done_at = datetime.now()
        order.done_reason = "canceled"
        
        logger.info(f"Cancelled order: {order.to_dict()}")
        self._notify_order_update(order)
        return True
    
    async def cancel_all_orders(self, product_id: Optional[str] = None) -> int:
        """
        Cancel all open orders, optionally filtered by product.
        
        Args:
            product_id: Optional product ID to filter orders
            
        Returns:
            Number of orders cancelled
        """
        # Prepare request parameters
        params = {}
        if product_id:
            params["product_id"] = product_id
            
        # Send API request
        response = await self._call_api("DELETE", "/orders", params=params)
        
        if not response or not isinstance(response, list):
            logger.error("Failed to cancel all orders")
            return 0
            
        # Update order statuses
        cancelled_count = 0
        for order_id in response:
            if order_id in self.orders:
                order = self.orders[order_id]
                order.status = OrderStatus.CANCELLED
                order.done_at = datetime.now()
                order.done_reason = "canceled"
                self._notify_order_update(order)
                cancelled_count += 1
                
        logger.info(f"Cancelled {cancelled_count} orders")
        return cancelled_count
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order details from the exchange.
        
        Args:
            order_id: Order ID
            
        Returns:
            Updated order if found, None otherwise
        """
        # Look up order
        order = self.orders.get(order_id)
        if not order:
            # Try to look up by client ID
            mapped_id = self.client_order_map.get(order_id)
            if mapped_id:
                order = self.orders.get(mapped_id)
                
            if not order:
                # Attempt to fetch unknown order from exchange
                response = await self._call_api("GET", f"/orders/{order_id}")
                if not response:
                    logger.warning(f"Order not found: {order_id}")
                    return None
                    
                # Create new order from response
                try:
                    order = Order.from_dict(response)
                    self.orders[order.id] = order
                    if order.client_id:
                        self.client_order_map[order.client_id] = order.id
                    logger.info(f"Retrieved new order: {order.to_dict()}")
                    return order
                except Exception as e:
                    logger.error(f"Failed to parse order response: {e}")
                    return None
        
        # Order exists in our tracking, update it from exchange
        response = await self._call_api("GET", f"/orders/{order.id}")
        if not response:
            logger.warning(f"Failed to get order details: {order.id}")
            return order
            
        # Update order with response data
        previous_status = order.status
        order.update_from_response(response)
        
        # Notify if status changed
        if previous_status != order.status:
            self._notify_order_update(order)
            
        logger.info(f"Updated order: {order.to_dict()}")
        return order
    
    async def list_orders(self, status_filter: Optional[List[OrderStatus]] = None, product_id: Optional[str] = None) -> List[Order]:
        """
        List orders matching the given criteria.
        
        Args:
            status_filter: List of order statuses to filter by
            product_id: Product ID to filter by
            
        Returns:
            List of matching orders
        """
        # Prepare request parameters
        params = {}
        if product_id:
            params["product_id"] = product_id
        
        # Convert status filter to string
        if status_filter:
            status_values = [status.value for status in status_filter]
            params["status"] = ",".join(status_values)
            
        # Send API request
        response = await self._call_api("GET", "/orders", params=params)
        
        if not response:
            logger.warning("Failed to list orders")
            return []
            
        # Process response
        orders = []
        for order_data in response:
            try:
                order_id = order_data.get("id")
                
                # Update existing order if we have it
                if order_id in self.orders:
                    order = self.orders[order_id]
                    previous_status = order.status
                    order.update_from_response(order_data)
                    
                    # Notify if status changed
                    if previous_status != order.status:
                        self._notify_order_update(order)
                        
                else:
                    # Create new order
                    order = Order.from_dict(order_data)
                    self.orders[order.id] = order
                    if order.client_id:
                        self.client_order_map[order.client_id] = order.id
                
                orders.append(order)
                
            except Exception as e:
                logger.error(f"Failed to parse order data: {e}")
                
        logger.info(f"Listed {len(orders)} orders")
        return orders
    
    async def sync_orders(self) -> Tuple[int, int, int]:
        """
        Synchronize local order cache with exchange.
        
        Returns:
            Tuple of (updated, added, removed) order counts
        """
        # Fetch all active orders from exchange
        active_orders = await self.list_orders(
            status_filter=[OrderStatus.PENDING, OrderStatus.OPEN]
        )
        
        # Track metrics
        updated_count = 0
        added_count = 0
        
        # Build set of active order IDs from exchange
        exchange_order_ids = set()
        for order in active_orders:
            exchange_order_ids.add(order.id)
            
            # Track if this was a new order
            if order.id not in self.orders:
                added_count += 1
            else:
                updated_count += 1
        
        # Find locally active orders that are no longer active on exchange
        removed_count = 0
        local_active_ids = [
            order_id for order_id, order in self.orders.items()
            if order.is_active() and order.id  # Exclude pending orders with no ID
        ]
        
        for order_id in local_active_ids:
            if order_id not in exchange_order_ids:
                # Order is no longer active on exchange
                order = self.orders[order_id]
                
                # Mark as failed if we don't know what happened
                if order.status in [OrderStatus.PENDING, OrderStatus.OPEN]:
                    order.status = OrderStatus.FAILED
                    order.done_at = datetime.now()
                    order.done_reason = "sync_not_found"
                    self._notify_order_update(order)
                    removed_count += 1
        
        logger.info(f"Synced orders: {updated_count} updated, {added_count} added, {removed_count} removed")
        return updated_count, added_count, removed_count
    
    def get_active_orders(self, product_id: Optional[str] = None) -> List[Order]:
        """
        Get all currently active orders.
        
        Args:
            product_id: Optional product ID to filter by
            
        Returns:
            List of active orders
        """
        active_orders = [
            order for order in self.orders.values()
            if order.is_active() and (not product_id or order.product_id == product_id)
        ]
        return active_orders
    
    def get_order_by_client_id(self, client_id: str) -> Optional[Order]:
        """Get order by client ID."""
        order_id = self.client_order_map.get(client_id)
        if order_id:
            return self.orders.get(order_id)
        return None
    
    async def _call_api(
        self,
        method: str,
        endpoint: str,
        data: Any = None,
        params: Dict[str, str] = None
    ) -> Optional[Any]:
        """
        Call exchange API with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body
            params: URL parameters
            
        Returns:
            Response data if successful, None otherwise
        """
        url = f"{self.api_url}{endpoint}"
        headers = self._get_auth_headers(method, endpoint, data)
        
        for attempt in range(1, self.max_retries + 1):
            try:
                if method == "GET":
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                elif method == "POST":
                    response = requests.post(url, headers=headers, json=data, params=params, timeout=10)
                elif method == "DELETE":
                    response = requests.delete(url, headers=headers, params=params, timeout=10)
                else:
                    logger.error(f"Unsupported HTTP method: {method}")
                    return None
                
                # Check for successful response
                if response.status_code in (200, 201):
                    return response.json()
                
                # Handle specific error codes
                if response.status_code == 400:
                    logger.error(f"Bad request: {response.text}")
                    return None
                if response.status_code == 401:
                    logger.error("Unauthorized, check API credentials")
                    return None
                if response.status_code == 404:
                    logger.warning(f"Resource not found: {endpoint}")
                    return None
                
                # Retry on 429 (rate limit) and 5xx (server errors)
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = int(response.headers.get("Retry-After", self.retry_delay * attempt))
                    logger.warning(f"API call failed with {response.status_code}, retrying in {retry_after}s (attempt {attempt}/{self.max_retries})")
                    await asyncio.sleep(retry_after)
                    continue
                
                # Other error
                logger.error(f"API call failed with status {response.status_code}: {response.text}")
                return None
                
            except requests.exceptions.Timeout:
                logger.warning(f"API call timed out, retrying (attempt {attempt}/{self.max_retries})")
                await asyncio.sleep(self.retry_delay * attempt)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                await asyncio.sleep(self.retry_delay * attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error in API call: {e}")
                return None
                
        logger.error(f"API call failed after {self.max_retries} attempts")
        return None
    
    def _get_auth_headers(self, method: str, endpoint: str, body: Any = None) -> Dict[str, str]:
        """
        Generate authentication headers for API request.
        This implementation is for Coinbase Pro/Exchange.
        Adjust according to your exchange's requirements.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            body: Request body
            
        Returns:
            Headers dict with authentication
        """
        timestamp = str(int(time.time()))
        body_str = json.dumps(body) if body else ""
        
        message = timestamp + method + endpoint + body_str
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('ascii'),
            hashlib.sha256
        )
        signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')
        
        headers = {
            'Content-Type': 'application/json',
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp
        }
        
        if self.api_passphrase:
            headers['CB-ACCESS-PASSPHRASE'] = self.api_passphrase
            
        return headers
    
    def _notify_order_update(self, order: Order) -> None:
        """Notify registered callbacks of order updates."""
        for callback in self.order_update_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order update callback: {e}")
                
    def clear_inactive_orders(self, max_age_days: int = 7) -> int:
        """
        Clear inactive orders older than the specified age.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of orders cleared
        """
        cutoff_time = datetime.now().timestamp() - (max_age_days * 86400)
        cleared_count = 0
        
        orders_to_remove = []
        for order_id, order in self.orders.items():
            if order.is_done() and order.done_at:
                if order.done_at.timestamp() < cutoff_time:
                    orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            self.orders.pop(order_id, None)
            # Remove from client ID map if present
            client_ids = [c_id for c_id, o_id in self.client_order_map.items() if o_id == order_id]
            for client_id in client_ids:
                self.client_order_map.pop(client_id, None)
            cleared_count += 1
            
        logger.info(f"Cleared {cleared_count} inactive orders")
        return cleared_count 