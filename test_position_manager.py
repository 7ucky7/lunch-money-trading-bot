#!/usr/bin/env python3
"""
Test script for Position Manager.
"""

import logging
import json
from datetime import datetime
from position_manager import PositionManager, PositionType, PositionStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PositionManagerTest")

def test_position_lifecycle():
    """Test basic position lifecycle."""
    manager = PositionManager()
    
    # Create a long position
    position = manager.create_position(
        product_id="BTC-USD",
        type=PositionType.LONG,
        entry_price=50000.0,
        size=1.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        metadata={"strategy": "test"}
    )
    
    logger.info(f"Created position: {position.to_dict()}")
    assert position.status == PositionStatus.OPEN
    
    # Update position with price movement
    position = manager.update_position(
        position_id=position.id,
        current_price=52000.0
    )
    
    logger.info(f"Updated position PnL: {position.unrealized_pnl}")
    assert position.unrealized_pnl == 2000.0
    
    # Test take profit trigger
    position = manager.update_position(
        position_id=position.id,
        current_price=55500.0
    )
    
    assert position.should_close()
    
    # Close position
    position = manager.close_position(
        position_id=position.id,
        price=55500.0
    )
    
    logger.info(f"Closed position PnL: {position.realized_pnl}")
    assert position.status == PositionStatus.CLOSED
    assert position.realized_pnl == 5500.0

def test_multiple_positions():
    """Test managing multiple positions."""
    manager = PositionManager()
    
    # Create multiple positions
    positions = [
        manager.create_position(
            product_id="BTC-USD",
            type=PositionType.LONG,
            entry_price=50000.0,
            size=1.0
        ),
        manager.create_position(
            product_id="ETH-USD",
            type=PositionType.SHORT,
            entry_price=3000.0,
            size=10.0
        )
    ]
    
    # Update prices
    for position in positions:
        if position.product_id == "BTC-USD":
            manager.update_position(position.id, 51000.0)
        else:
            manager.update_position(position.id, 2900.0)
    
    # Get metrics
    metrics = manager.get_position_metrics()
    logger.info(f"Position metrics: {json.dumps(metrics, indent=2)}")
    
    # Test filtering
    btc_positions = manager.get_positions(product_id="BTC-USD")
    assert len(btc_positions) == 1
    
    open_positions = manager.get_positions(status=PositionStatus.OPEN)
    assert len(open_positions) == 2

def test_persistence():
    """Test position persistence."""
    manager = PositionManager()
    
    # Create some positions
    manager.create_position(
        product_id="BTC-USD",
        type=PositionType.LONG,
        entry_price=50000.0,
        size=1.0
    )
    manager.create_position(
        product_id="ETH-USD",
        type=PositionType.SHORT,
        entry_price=3000.0,
        size=10.0
    )
    
    # Save positions
    manager.save_positions("test_positions.json")
    
    # Create new manager and load positions
    new_manager = PositionManager()
    new_manager.load_positions("test_positions.json")
    
    # Verify positions were loaded correctly
    assert len(new_manager.positions) == 2
    assert "BTC-USD" in new_manager.product_positions
    assert "ETH-USD" in new_manager.product_positions

def main():
    """Run all tests."""
    try:
        logger.info("Testing position lifecycle...")
        test_position_lifecycle()
        
        logger.info("\nTesting multiple positions...")
        test_multiple_positions()
        
        logger.info("\nTesting position persistence...")
        test_persistence()
        
        logger.info("\nAll tests passed!")
        
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 