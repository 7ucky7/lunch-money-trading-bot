#!/usr/bin/env python3
"""
Position Manager Module.

Handles position tracking and management for the cryptocurrency trading bot.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PositionManager")

class PositionStatus(Enum):
    """Position status enumeration."""
    PENDING = "pending"      # Order placed but not filled
    OPEN = "open"           # Position is active
    CLOSED = "closed"       # Position has been closed
    CANCELLED = "cancelled" # Order was cancelled

class PositionType(Enum):
    """Position type enumeration."""
    LONG = "long"
    SHORT = "short"

@dataclass
class Position:
    """
    Represents a trading position.
    
    Attributes:
        id: Unique position identifier
        product_id: Product being traded
        type: Position type (long/short)
        entry_price: Average entry price
        current_price: Current market price
        size: Position size in base currency
        unrealized_pnl: Unrealized profit/loss
        realized_pnl: Realized profit/loss
        status: Position status
        entry_time: Entry timestamp
        close_time: Close timestamp (if closed)
        stop_loss: Stop loss price
        take_profit: Take profit price
        metadata: Additional position data
    """
    id: str
    product_id: str
    type: PositionType
    entry_price: float
    current_price: float
    size: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: PositionStatus = PositionStatus.PENDING
    entry_time: Optional[int] = None
    close_time: Optional[int] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = None

    def update_price(self, price: float):
        """
        Update position with new price.
        
        Args:
            price: Current market price
        """
        self.current_price = price
        self._calculate_pnl()
    
    def _calculate_pnl(self):
        """Calculate unrealized PnL."""
        if self.status != PositionStatus.OPEN:
            return
            
        price_diff = self.current_price - self.entry_price
        if self.type == PositionType.SHORT:
            price_diff = -price_diff
            
        self.unrealized_pnl = price_diff * self.size
    
    def close(self, price: float):
        """
        Close the position.
        
        Args:
            price: Closing price
        """
        if self.status != PositionStatus.OPEN:
            return
            
        self.current_price = price
        self._calculate_pnl()
        self.realized_pnl = self.unrealized_pnl
        self.unrealized_pnl = 0
        self.status = PositionStatus.CLOSED
        self.close_time = int(datetime.now().timestamp())
    
    def should_close(self) -> bool:
        """
        Check if position should be closed based on stop loss or take profit.
        
        Returns:
            True if position should be closed
        """
        if self.status != PositionStatus.OPEN:
            return False
            
        if self.type == PositionType.LONG:
            if self.stop_loss and self.current_price <= self.stop_loss:
                return True
            if self.take_profit and self.current_price >= self.take_profit:
                return True
        else:  # SHORT
            if self.stop_loss and self.current_price >= self.stop_loss:
                return True
            if self.take_profit and self.current_price <= self.take_profit:
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'id': self.id,
            'product_id': self.product_id,
            'type': self.type.value,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'size': self.size,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'status': self.status.value,
            'entry_time': self.entry_time,
            'close_time': self.close_time,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create position from dictionary."""
        return cls(
            id=data['id'],
            product_id=data['product_id'],
            type=PositionType(data['type']),
            entry_price=data['entry_price'],
            current_price=data['current_price'],
            size=data['size'],
            unrealized_pnl=data['unrealized_pnl'],
            realized_pnl=data['realized_pnl'],
            status=PositionStatus(data['status']),
            entry_time=data['entry_time'],
            close_time=data['close_time'],
            stop_loss=data['stop_loss'],
            take_profit=data['take_profit'],
            metadata=data['metadata']
        )

class PositionManager:
    """
    Manages trading positions.
    
    Features:
    - Position tracking and updates
    - PnL calculation
    - Stop loss and take profit management
    - Position persistence
    - Risk metrics calculation
    """
    
    def __init__(self):
        """Initialize position manager."""
        self.positions: Dict[str, Position] = {}  # id -> Position
        self.product_positions: Dict[str, List[str]] = {}  # product_id -> [position_ids]
        logger.info("Position Manager initialized")
    
    def create_position(
        self,
        product_id: str,
        type: PositionType,
        entry_price: float,
        size: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Position:
        """
        Create a new position.
        
        Args:
            product_id: Product identifier
            type: Position type
            entry_price: Entry price
            size: Position size
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            metadata: Additional position data
            
        Returns:
            Created position
        """
        # Generate position ID
        position_id = f"pos_{int(datetime.now().timestamp())}_{product_id}"
        
        # Create position
        position = Position(
            id=position_id,
            product_id=product_id,
            type=type,
            entry_price=entry_price,
            current_price=entry_price,
            size=size,
            status=PositionStatus.OPEN,
            entry_time=int(datetime.now().timestamp()),
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {}
        )
        
        # Store position
        self.positions[position_id] = position
        if product_id not in self.product_positions:
            self.product_positions[product_id] = []
        self.product_positions[product_id].append(position_id)
        
        logger.info(f"Created position: {position_id}")
        return position
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get position by ID.
        
        Args:
            position_id: Position identifier
            
        Returns:
            Position or None if not found
        """
        return self.positions.get(position_id)
    
    def get_positions(
        self,
        product_id: Optional[str] = None,
        status: Optional[PositionStatus] = None
    ) -> List[Position]:
        """
        Get positions with optional filtering.
        
        Args:
            product_id: Filter by product ID
            status: Filter by position status
            
        Returns:
            List of positions
        """
        positions = []
        
        if product_id:
            position_ids = self.product_positions.get(product_id, [])
        else:
            position_ids = list(self.positions.keys())
        
        for position_id in position_ids:
            position = self.positions[position_id]
            if status and position.status != status:
                continue
            positions.append(position)
        
        return positions
    
    def update_position(
        self,
        position_id: str,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Position]:
        """
        Update position with new price and optional stops.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            Updated position or None if not found
        """
        position = self.get_position(position_id)
        if not position:
            return None
            
        position.update_price(current_price)
        
        if stop_loss is not None:
            position.stop_loss = stop_loss
        if take_profit is not None:
            position.take_profit = take_profit
        
        return position
    
    def close_position(
        self,
        position_id: str,
        price: float
    ) -> Optional[Position]:
        """
        Close a position.
        
        Args:
            position_id: Position identifier
            price: Closing price
            
        Returns:
            Closed position or None if not found
        """
        position = self.get_position(position_id)
        if not position:
            return None
            
        position.close(price)
        logger.info(f"Closed position {position_id} with PnL: {position.realized_pnl}")
        return position
    
    def get_total_pnl(self) -> Dict[str, float]:
        """
        Calculate total PnL across all positions.
        
        Returns:
            Dictionary with realized and unrealized PnL
        """
        total_realized = 0.0
        total_unrealized = 0.0
        
        for position in self.positions.values():
            total_realized += position.realized_pnl
            total_unrealized += position.unrealized_pnl
        
        return {
            'realized': total_realized,
            'unrealized': total_unrealized,
            'total': total_realized + total_unrealized
        }
    
    def get_product_exposure(self, product_id: str) -> float:
        """
        Calculate total exposure for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Total exposure (positive for long, negative for short)
        """
        exposure = 0.0
        
        for position_id in self.product_positions.get(product_id, []):
            position = self.positions[position_id]
            if position.status != PositionStatus.OPEN:
                continue
                
            if position.type == PositionType.LONG:
                exposure += position.size
            else:
                exposure -= position.size
        
        return exposure
    
    def save_positions(self, filepath: str):
        """
        Save positions to file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'positions': {
                pos_id: pos.to_dict()
                for pos_id, pos in self.positions.items()
            },
            'product_positions': self.product_positions
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved positions to {filepath}")
    
    def load_positions(self, filepath: str):
        """
        Load positions from file.
        
        Args:
            filepath: Path to load file
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.positions = {
                pos_id: Position.from_dict(pos_data)
                for pos_id, pos_data in data['positions'].items()
            }
            self.product_positions = data['product_positions']
            
            logger.info(f"Loaded {len(self.positions)} positions from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
    
    def get_position_metrics(self) -> Dict[str, Any]:
        """
        Calculate various position metrics.
        
        Returns:
            Dictionary with position metrics
        """
        total_positions = len(self.positions)
        open_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
        pnl = self.get_total_pnl()
        
        # Calculate win rate
        closed_positions = [p for p in self.positions.values() if p.status == PositionStatus.CLOSED]
        winning_positions = len([p for p in closed_positions if p.realized_pnl > 0])
        win_rate = winning_positions / len(closed_positions) if closed_positions else 0
        
        # Calculate average PnL
        avg_realized_pnl = pnl['realized'] / len(closed_positions) if closed_positions else 0
        
        return {
            'total_positions': total_positions,
            'open_positions': open_positions,
            'pnl': pnl,
            'win_rate': win_rate,
            'avg_realized_pnl': avg_realized_pnl,
            'product_exposure': {
                product_id: self.get_product_exposure(product_id)
                for product_id in self.product_positions.keys()
            }
        } 