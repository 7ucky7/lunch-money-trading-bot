#!/usr/bin/env python3
"""
Order Validator Module.

Handles order validation and execution checks for the cryptocurrency trading bot.
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal, ROUND_DOWN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OrderValidator")

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

@dataclass
class OrderValidationResult:
    """Result of order validation."""
    is_valid: bool
    message: str
    modified_params: Optional[Dict[str, Any]] = None

class OrderValidator:
    """
    Validates and sanitizes orders before execution.
    
    Features:
    - Order parameter validation
    - Size and price normalization
    - Balance and position checks
    - Risk limit validation
    - Market rules compliance
    """
    
    def __init__(self, product_rules: Dict[str, Dict[str, Any]]):
        """
        Initialize order validator.
        
        Args:
            product_rules: Dictionary of product trading rules
                Example:
                {
                    "BTC-USD": {
                        "base_min_size": "0.0001",
                        "base_max_size": "100",
                        "quote_increment": "0.01",
                        "base_increment": "0.00000001",
                        "min_market_funds": "5",
                        "max_market_funds": "1000000",
                        "margin_enabled": false
                    }
                }
        """
        self.product_rules = product_rules
        logger.info("Order Validator initialized")
    
    def validate_new_order(
        self,
        product_id: str,
        side: OrderSide,
        order_type: OrderType,
        size: Optional[float] = None,
        funds: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        available_balance: float = 0.0,
        current_positions: Dict[str, float] = None
    ) -> OrderValidationResult:
        """
        Validate a new order.
        
        Args:
            product_id: Product identifier
            side: Order side (buy/sell)
            order_type: Order type
            size: Order size in base currency
            funds: Order size in quote currency
            price: Limit price
            stop_price: Stop price
            available_balance: Available balance
            current_positions: Current positions by product
            
        Returns:
            Validation result
        """
        try:
            # Check if product exists
            if product_id not in self.product_rules:
                return OrderValidationResult(
                    is_valid=False,
                    message=f"Invalid product: {product_id}"
                )
            
            rules = self.product_rules[product_id]
            
            # Validate basic parameters
            if not size and not funds:
                return OrderValidationResult(
                    is_valid=False,
                    message="Either size or funds must be specified"
                )
            
            if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not price:
                return OrderValidationResult(
                    is_valid=False,
                    message=f"Price required for {order_type.value} orders"
                )
            
            if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not stop_price:
                return OrderValidationResult(
                    is_valid=False,
                    message=f"Stop price required for {order_type.value} orders"
                )
            
            # Normalize size and price
            modified_params = {}
            
            if size:
                normalized_size = self._normalize_size(size, rules['base_increment'])
                if normalized_size != size:
                    modified_params['size'] = normalized_size
                size = normalized_size
            
            if price:
                normalized_price = self._normalize_price(price, rules['quote_increment'])
                if normalized_price != price:
                    modified_params['price'] = normalized_price
                price = normalized_price
            
            # Validate size limits
            if size:
                min_size = float(rules['base_min_size'])
                max_size = float(rules['base_max_size'])
                
                if size < min_size:
                    return OrderValidationResult(
                        is_valid=False,
                        message=f"Size {size} below minimum {min_size}"
                    )
                
                if size > max_size:
                    return OrderValidationResult(
                        is_valid=False,
                        message=f"Size {size} above maximum {max_size}"
                    )
            
            # Validate funds limits for market orders
            if order_type == OrderType.MARKET and funds:
                min_funds = float(rules['min_market_funds'])
                max_funds = float(rules['max_market_funds'])
                
                if funds < min_funds:
                    return OrderValidationResult(
                        is_valid=False,
                        message=f"Funds {funds} below minimum {min_funds}"
                    )
                
                if funds > max_funds:
                    return OrderValidationResult(
                        is_valid=False,
                        message=f"Funds {funds} above maximum {max_funds}"
                    )
            
            # Check available balance
            required_funds = self._calculate_required_funds(
                side=side,
                order_type=order_type,
                size=size,
                funds=funds,
                price=price
            )
            
            if required_funds > available_balance:
                return OrderValidationResult(
                    is_valid=False,
                    message=f"Insufficient funds: required {required_funds}, available {available_balance}"
                )
            
            # Check position limits
            if current_positions:
                current_position = current_positions.get(product_id, 0.0)
                if side == OrderSide.BUY:
                    new_position = current_position + (size or 0)
                    if new_position > float(rules['base_max_size']):
                        return OrderValidationResult(
                            is_valid=False,
                            message=f"Position size {new_position} would exceed maximum {rules['base_max_size']}"
                        )
                else:  # SELL
                    if not rules.get('margin_enabled', False):
                        new_position = current_position - (size or 0)
                        if new_position < 0:
                            return OrderValidationResult(
                                is_valid=False,
                                message="Insufficient position size for sell order"
                            )
            
            # Order is valid
            return OrderValidationResult(
                is_valid=True,
                message="Order validation successful",
                modified_params=modified_params if modified_params else None
            )
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return OrderValidationResult(
                is_valid=False,
                message=f"Validation error: {str(e)}"
            )
    
    def validate_order_update(
        self,
        product_id: str,
        order_id: str,
        new_size: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> OrderValidationResult:
        """
        Validate order update parameters.
        
        Args:
            product_id: Product identifier
            order_id: Order identifier
            new_size: New order size
            new_price: New order price
            
        Returns:
            Validation result
        """
        try:
            if product_id not in self.product_rules:
                return OrderValidationResult(
                    is_valid=False,
                    message=f"Invalid product: {product_id}"
                )
            
            rules = self.product_rules[product_id]
            modified_params = {}
            
            # Validate and normalize new size
            if new_size is not None:
                normalized_size = self._normalize_size(new_size, rules['base_increment'])
                if normalized_size != new_size:
                    modified_params['size'] = normalized_size
                new_size = normalized_size
                
                min_size = float(rules['base_min_size'])
                max_size = float(rules['base_max_size'])
                
                if new_size < min_size:
                    return OrderValidationResult(
                        is_valid=False,
                        message=f"Size {new_size} below minimum {min_size}"
                    )
                
                if new_size > max_size:
                    return OrderValidationResult(
                        is_valid=False,
                        message=f"Size {new_size} above maximum {max_size}"
                    )
            
            # Validate and normalize new price
            if new_price is not None:
                normalized_price = self._normalize_price(new_price, rules['quote_increment'])
                if normalized_price != new_price:
                    modified_params['price'] = normalized_price
            
            return OrderValidationResult(
                is_valid=True,
                message="Order update validation successful",
                modified_params=modified_params if modified_params else None
            )
            
        except Exception as e:
            logger.error(f"Error validating order update: {e}")
            return OrderValidationResult(
                is_valid=False,
                message=f"Validation error: {str(e)}"
            )
    
    def _normalize_size(self, size: float, increment: str) -> float:
        """Normalize order size to valid increment."""
        increment_decimal = Decimal(increment)
        size_decimal = Decimal(str(size))
        normalized = size_decimal.quantize(increment_decimal, rounding=ROUND_DOWN)
        return float(normalized)
    
    def _normalize_price(self, price: float, increment: str) -> float:
        """Normalize price to valid increment."""
        increment_decimal = Decimal(increment)
        price_decimal = Decimal(str(price))
        normalized = price_decimal.quantize(increment_decimal, rounding=ROUND_DOWN)
        return float(normalized)
    
    def _calculate_required_funds(
        self,
        side: OrderSide,
        order_type: OrderType,
        size: Optional[float] = None,
        funds: Optional[float] = None,
        price: Optional[float] = None
    ) -> float:
        """Calculate required funds for order."""
        if order_type == OrderType.MARKET:
            if funds:
                return funds * 1.01  # Add 1% buffer for market orders
            return 0.0  # Can't calculate without market price
        
        if size and price:
            return size * price * 1.01  # Add 1% buffer
        
        return 0.0  # Can't calculate without size and price 