#!/usr/bin/env python3
"""
Test script for Order Validator.
"""

import logging
import json
from order_validator import OrderValidator, OrderType, OrderSide, OrderValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OrderValidatorTest")

# Test product rules
PRODUCT_RULES = {
    "BTC-USD": {
        "base_min_size": "0.0001",
        "base_max_size": "100",
        "quote_increment": "0.01",
        "base_increment": "0.00000001",
        "min_market_funds": "5",
        "max_market_funds": "1000000",
        "margin_enabled": False
    },
    "ETH-USD": {
        "base_min_size": "0.001",
        "base_max_size": "1000",
        "quote_increment": "0.01",
        "base_increment": "0.00001",
        "min_market_funds": "5",
        "max_market_funds": "1000000",
        "margin_enabled": False
    }
}

def test_basic_validation():
    """Test basic order validation."""
    validator = OrderValidator(PRODUCT_RULES)
    
    # Test valid market buy
    result = validator.validate_new_order(
        product_id="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        funds=1000.0,
        available_balance=1500.0
    )
    
    logger.info(f"Market buy validation: {result}")
    assert result.is_valid
    
    # Test valid limit buy
    result = validator.validate_new_order(
        product_id="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        size=0.1,
        price=50000.0,
        available_balance=6000.0
    )
    
    logger.info(f"Limit buy validation: {result}")
    assert result.is_valid
    
    # Test invalid product
    result = validator.validate_new_order(
        product_id="INVALID-PAIR",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        funds=1000.0
    )
    
    logger.info(f"Invalid product validation: {result}")
    assert not result.is_valid

def test_size_normalization():
    """Test size and price normalization."""
    validator = OrderValidator(PRODUCT_RULES)
    
    # Test size normalization
    result = validator.validate_new_order(
        product_id="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        size=0.12345678912345,
        price=50000.0,
        available_balance=10000.0
    )
    
    logger.info(f"Size normalization: {result}")
    assert result.is_valid
    assert result.modified_params
    assert result.modified_params['size'] == 0.12345678
    
    # Test price normalization
    result = validator.validate_new_order(
        product_id="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        size=0.1,
        price=50000.123,
        available_balance=10000.0
    )
    
    logger.info(f"Price normalization: {result}")
    assert result.is_valid
    assert result.modified_params
    assert result.modified_params['price'] == 50000.12

def test_balance_checks():
    """Test balance validation."""
    validator = OrderValidator(PRODUCT_RULES)
    
    # Test insufficient balance
    result = validator.validate_new_order(
        product_id="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        size=1.0,
        price=50000.0,
        available_balance=1000.0
    )
    
    logger.info(f"Insufficient balance validation: {result}")
    assert not result.is_valid
    
    # Test sufficient balance with buffer
    result = validator.validate_new_order(
        product_id="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        size=1.0,
        price=50000.0,
        available_balance=51000.0  # Include 1% buffer
    )
    
    logger.info(f"Sufficient balance validation: {result}")
    assert result.is_valid

def test_position_limits():
    """Test position limit validation."""
    validator = OrderValidator(PRODUCT_RULES)
    
    # Test position size limit
    result = validator.validate_new_order(
        product_id="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        size=150.0,  # Above max size
        price=50000.0,
        available_balance=10000000.0
    )
    
    logger.info(f"Position size limit validation: {result}")
    assert not result.is_valid
    
    # Test selling more than current position
    result = validator.validate_new_order(
        product_id="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        size=1.0,
        current_positions={"BTC-USD": 0.5}  # Only 0.5 BTC available
    )
    
    logger.info(f"Oversell validation: {result}")
    assert not result.is_valid

def test_order_updates():
    """Test order update validation."""
    validator = OrderValidator(PRODUCT_RULES)
    
    # Test valid size update
    result = validator.validate_order_update(
        product_id="BTC-USD",
        order_id="test_order",
        new_size=0.5
    )
    
    logger.info(f"Size update validation: {result}")
    assert result.is_valid
    
    # Test invalid size update
    result = validator.validate_order_update(
        product_id="BTC-USD",
        order_id="test_order",
        new_size=0.00001  # Below min size
    )
    
    logger.info(f"Invalid size update validation: {result}")
    assert not result.is_valid

def main():
    """Run all tests."""
    try:
        logger.info("Testing basic validation...")
        test_basic_validation()
        
        logger.info("\nTesting size normalization...")
        test_size_normalization()
        
        logger.info("\nTesting balance checks...")
        test_balance_checks()
        
        logger.info("\nTesting position limits...")
        test_position_limits()
        
        logger.info("\nTesting order updates...")
        test_order_updates()
        
        logger.info("\nAll tests passed!")
        
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 