#!/usr/bin/env python3
"""
Risk Management Module for Cryptocurrency Trading Bot.

This module handles position sizing, stop-loss management, take-profit levels,
risk metrics calculation, and trade safety checks.
"""

import logging
import time
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import math

# Local imports
from strategy import Signal, SignalType

# Get logger
logger = logging.getLogger("RiskManager")

@dataclass
class PositionSizing:
    """Position sizing calculation result."""
    position_size: float  # Size in base currency (e.g., BTC)
    position_value: float  # Value in quote currency (e.g., USD)
    risk_amount: float  # Amount at risk in quote currency
    risk_percent: float  # Risk percentage of account balance
    stop_loss_price: Optional[float] = None  # Stop loss price
    take_profit_price: Optional[float] = None  # Take profit price

class RiskManager:
    """
    Risk management for cryptocurrency trading.
    
    Responsibilities:
    - Calculate position sizes
    - Set stop-loss and take-profit levels
    - Enforce risk limits
    - Monitor overall portfolio risk
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
        """
        self.config = config or {}
        
        # Set default risk parameters if not provided
        self.risk_percent_per_trade = self.config.get("risk_percent", 0.01)  # 1% of balance per trade
        self.max_risk_percent_total = self.config.get("max_risk_percent_total", 0.05)  # 5% max total risk
        self.max_trades_per_product = self.config.get("max_trades_per_product", 3)
        self.default_stop_loss_percent = self.config.get("stop_loss_percent", 0.02)  # 2% stop loss
        self.default_take_profit_percent = self.config.get("take_profit_percent", 0.04)  # 4% take profit
        self.use_atr_for_stops = self.config.get("use_atr_for_stops", True)
        self.atr_multiplier = self.config.get("atr_multiplier", 2.5)
        self.kelly_criterion_multiplier = self.config.get("kelly_criterion_multiplier", 0.5)  # Half-Kelly
        self.position_sizing_method = self.config.get("position_sizing_method", "risk_percent")
        self.max_position_size_quote = self.config.get("max_position_size_quote", float('inf'))
        self.min_position_size_quote = self.config.get("min_position_size_quote", 10.0)  # $10 minimum
        
        # Track active positions
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.total_risk_amount = 0.0
        
        logger.info(f"Risk manager initialized with risk_percent={self.risk_percent_per_trade}, " +
                   f"stop_loss_percent={self.default_stop_loss_percent}")
    
    def calculate_position_size(self, 
                              signal: Signal, 
                              balance: float, 
                              available_balance: float,
                              atr: Optional[float] = None) -> PositionSizing:
        """
        Calculate position size based on risk parameters.
        
        Args:
            signal: Trade signal
            balance: Total account balance in quote currency
            available_balance: Available balance for trading
            atr: Optional Average True Range for volatility-based sizing
            
        Returns:
            PositionSizing object with calculated position details
        """
        # Ensure we have a valid price
        if not signal.price or signal.price <= 0:
            logger.error(f"Invalid price {signal.price} for position sizing")
            return PositionSizing(
                position_size=0.0,
                position_value=0.0,
                risk_amount=0.0,
                risk_percent=0.0
            )
        
        # Determine risk amount in quote currency (e.g., USD)
        risk_amount = balance * self.risk_percent_per_trade
        
        # Limit risk amount to available balance
        risk_amount = min(risk_amount, available_balance * 0.95)  # Use at most 95% of available
        
        # Calculate stop loss price
        stop_loss_price = None
        stop_loss_percent = self.default_stop_loss_percent
        
        if self.use_atr_for_stops and atr is not None:
            # Use ATR for more dynamic stop loss
            atr_stop_distance = atr * self.atr_multiplier
            stop_loss_percent = atr_stop_distance / signal.price
            
            logger.debug(f"Using ATR-based stop: ATR={atr}, distance={atr_stop_distance}, percent={stop_loss_percent:.4f}")
        
        if signal.type == SignalType.BUY:
            stop_loss_price = signal.price * (1 - stop_loss_percent)
        else:  # SELL
            stop_loss_price = signal.price * (1 + stop_loss_percent)
        
        # Calculate take profit price
        take_profit_percent = self.default_take_profit_percent
        if take_profit_percent < stop_loss_percent * 1.5:
            # Ensure risk:reward is at least 1:1.5
            take_profit_percent = stop_loss_percent * 1.5
        
        take_profit_price = None
        if signal.type == SignalType.BUY:
            take_profit_price = signal.price * (1 + take_profit_percent)
        else:  # SELL
            take_profit_price = signal.price * (1 - take_profit_percent)
        
        # Calculate position size based on risk
        price_distance = abs(signal.price - stop_loss_price)
        
        # Determine position size and value
        if price_distance > 0:
            # Risk-based position sizing
            position_value = risk_amount / (price_distance / signal.price)
        else:
            # Fallback if price distance is too small
            position_value = risk_amount * 10
            logger.warning(f"Price distance too small ({price_distance}), using fallback position value: {position_value}")
        
        # Apply maximum position size limit
        position_value = min(position_value, self.max_position_size_quote, available_balance * 0.95)
        
        # Ensure minimum position size
        if position_value < self.min_position_size_quote:
            if available_balance >= self.min_position_size_quote:
                position_value = self.min_position_size_quote
                logger.info(f"Increased position to minimum size: {position_value}")
            else:
                logger.warning(f"Insufficient balance ({available_balance}) for minimum position size ({self.min_position_size_quote})")
                position_value = 0
        
        # Calculate base currency amount (e.g., BTC)
        position_size = position_value / signal.price
        
        # Calculate actual risk amount and percent
        actual_risk_amount = position_value * stop_loss_percent
        actual_risk_percent = actual_risk_amount / balance if balance > 0 else 0
        
        logger.info(f"Position sizing: {position_size:.6f} ({position_value:.2f} USD), " +
                   f"risk: {actual_risk_amount:.2f} USD ({actual_risk_percent:.2%}), " +
                   f"stop: {stop_loss_price:.2f}, target: {take_profit_price:.2f}")
        
        return PositionSizing(
            position_size=position_size,
            position_value=position_value,
            risk_amount=actual_risk_amount,
            risk_percent=actual_risk_percent,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
    
    def calculate_kelly_position_size(self,
                                    signal: Signal,
                                    balance: float,
                                    win_rate: float,
                                    risk_reward_ratio: float) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            signal: Trade signal
            balance: Account balance
            win_rate: Historical win rate (0.0 to 1.0)
            risk_reward_ratio: Ratio of average win to average loss
            
        Returns:
            Position size as fraction of balance
        """
        # Kelly formula: f* = p - (1-p)/r
        # where p = win probability, r = win/loss ratio
        
        f_star = win_rate - (1 - win_rate) / risk_reward_ratio
        
        # Apply half-Kelly (or other multiplier) for more conservative sizing
        f_star = f_star * self.kelly_criterion_multiplier
        
        # Ensure positive value (avoid short positions if Kelly is negative)
        f_star = max(0, f_star)
        
        # Cap maximum position size
        f_star = min(f_star, self.risk_percent_per_trade * 5)
        
        kelly_position_value = balance * f_star
        
        logger.debug(f"Kelly position sizing: win_rate={win_rate:.2f}, RR={risk_reward_ratio:.2f}, " +
                    f"f*={f_star:.4f}, value={kelly_position_value:.2f}")
        
        return kelly_position_value
    
    def register_position(self, 
                        product_id: str, 
                        position_sizing: PositionSizing,
                        signal: Signal) -> bool:
        """
        Register a new position with the risk manager.
        
        Args:
            product_id: Product identifier
            position_sizing: Position sizing details
            signal: Trade signal
            
        Returns:
            True if position registered successfully
        """
        if product_id in self.active_positions:
            position_count = len(self.active_positions[product_id])
            if position_count >= self.max_trades_per_product:
                logger.warning(f"Maximum number of trades ({self.max_trades_per_product}) reached for {product_id}")
                return False
        else:
            self.active_positions[product_id] = {}
        
        # Generate a unique position ID
        position_id = f"{product_id}_{int(time.time())}_{len(self.active_positions[product_id])}"
        
        # Create position record
        position = {
            "id": position_id,
            "product_id": product_id,
            "signal_type": signal.type.value,
            "entry_price": signal.price,
            "entry_time": signal.timestamp,
            "size": position_sizing.position_size,
            "value": position_sizing.position_value,
            "stop_loss": position_sizing.stop_loss_price,
            "take_profit": position_sizing.take_profit_price,
            "risk_amount": position_sizing.risk_amount,
            "risk_percent": position_sizing.risk_percent,
            "trailing_stop": None,
            "trailing_activation_price": None,
            "status": "open"
        }
        
        # Register position
        self.active_positions[product_id][position_id] = position
        
        # Update total risk
        self.total_risk_amount += position_sizing.risk_amount
        
        logger.info(f"Registered position {position_id}: {product_id} {signal.type.value} at {signal.price}")
        return True
    
    def update_position(self, 
                      position_id: str, 
                      current_price: float,
                      update_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update position with current price and optional data.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            update_data: Optional data to update in the position
            
        Returns:
            Updated position data or empty dict if not found
        """
        # Find position by ID
        position = None
        for product_positions in self.active_positions.values():
            if position_id in product_positions:
                position = product_positions[position_id]
                break
        
        if not position:
            logger.warning(f"Position {position_id} not found for update")
            return {}
        
        # Update position data if provided
        if update_data:
            position.update(update_data)
        
        # Update position metrics based on current price
        entry_price = position["entry_price"]
        position_type = position["signal_type"]
        position_size = position["size"]
        
        if position_type == "buy":
            # Long position
            unrealized_pnl = (current_price - entry_price) * position_size
            unrealized_pnl_percent = (current_price - entry_price) / entry_price
        else:
            # Short position
            unrealized_pnl = (entry_price - current_price) * position_size
            unrealized_pnl_percent = (entry_price - current_price) / entry_price
        
        position["current_price"] = current_price
        position["unrealized_pnl"] = unrealized_pnl
        position["unrealized_pnl_percent"] = unrealized_pnl_percent
        position["current_value"] = position_size * current_price
        position["last_update_time"] = int(time.time())
        
        # Check for trailing stop update
        self._update_trailing_stop(position, current_price)
        
        return position
    
    def _update_trailing_stop(self, position: Dict[str, Any], current_price: float) -> None:
        """
        Update trailing stop if enabled for this position.
        
        Args:
            position: Position data
            current_price: Current market price
        """
        # Check if trailing stop is enabled
        if position.get("trailing_stop") is None:
            return
            
        position_type = position["signal_type"]
        activation_price = position.get("trailing_activation_price")
        
        # Check if trailing stop has been activated
        if activation_price is None:
            return
            
        trailing_stop = position["trailing_stop"]
        
        if position_type == "buy":
            # For long positions, trail upward
            if current_price > activation_price:
                # Calculate new stop price
                new_stop = current_price * (1 - trailing_stop)
                
                # Only update if new stop is higher than current stop
                if position.get("stop_loss") is None or new_stop > position["stop_loss"]:
                    position["stop_loss"] = new_stop
                    logger.info(f"Updated trailing stop for {position['id']} to {new_stop}")
        else:
            # For short positions, trail downward
            if current_price < activation_price:
                # Calculate new stop price
                new_stop = current_price * (1 + trailing_stop)
                
                # Only update if new stop is lower than current stop
                if position.get("stop_loss") is None or new_stop < position["stop_loss"]:
                    position["stop_loss"] = new_stop
                    logger.info(f"Updated trailing stop for {position['id']} to {new_stop}")
    
    def set_trailing_stop(self, 
                        position_id: str, 
                        trail_percent: float,
                        activation_percent: float = 0.0) -> bool:
        """
        Set a trailing stop for a position.
        
        Args:
            position_id: Position identifier
            trail_percent: Trailing distance as percent
            activation_percent: Price movement required to activate trailing
            
        Returns:
            True if trailing stop set successfully
        """
        # Find position by ID
        position = None
        for product_positions in self.active_positions.values():
            if position_id in product_positions:
                position = product_positions[position_id]
                break
        
        if not position:
            logger.warning(f"Position {position_id} not found for trailing stop")
            return False
        
        entry_price = position["entry_price"]
        position_type = position["signal_type"]
        
        # Calculate activation price
        if position_type == "buy":
            # For long positions, activate when price moves up by activation_percent
            activation_price = entry_price * (1 + activation_percent)
        else:
            # For short positions, activate when price moves down by activation_percent
            activation_price = entry_price * (1 - activation_percent)
        
        # Set trailing stop parameters
        position["trailing_stop"] = trail_percent
        position["trailing_activation_price"] = activation_price
        
        logger.info(f"Set trailing stop for {position_id}: trail={trail_percent:.2%}, " +
                   f"activation={activation_price}")
        
        return True
    
    def close_position(self, 
                     position_id: str, 
                     close_price: float,
                     close_reason: str = "manual") -> Dict[str, Any]:
        """
        Close a position and update risk metrics.
        
        Args:
            position_id: Position identifier
            close_price: Price at which position is closed
            close_reason: Reason for closing the position
            
        Returns:
            Closed position data or empty dict if not found
        """
        # Find position by ID
        position = None
        product_id = None
        
        for pid, product_positions in self.active_positions.items():
            if position_id in product_positions:
                position = product_positions[position_id]
                product_id = pid
                break
        
        if not position:
            logger.warning(f"Position {position_id} not found for closing")
            return {}
        
        # Calculate final P&L
        entry_price = position["entry_price"]
        position_type = position["signal_type"]
        position_size = position["size"]
        
        if position_type == "buy":
            # Long position
            realized_pnl = (close_price - entry_price) * position_size
            realized_pnl_percent = (close_price - entry_price) / entry_price
        else:
            # Short position
            realized_pnl = (entry_price - close_price) * position_size
            realized_pnl_percent = (entry_price - close_price) / entry_price
        
        # Update position data
        position["close_price"] = close_price
        position["close_time"] = int(time.time())
        position["realized_pnl"] = realized_pnl
        position["realized_pnl_percent"] = realized_pnl_percent
        position["close_reason"] = close_reason
        position["status"] = "closed"
        
        # Remove from active positions and update risk
        if product_id:
            del self.active_positions[product_id][position_id]
            if not self.active_positions[product_id]:
                del self.active_positions[product_id]
                
        # Update total risk
        self.total_risk_amount -= position.get("risk_amount", 0)
        
        logger.info(f"Closed position {position_id}: {product_id} at {close_price}, " +
                   f"PnL: {realized_pnl:.2f} ({realized_pnl_percent:.2%}), reason: {close_reason}")
        
        return position
    
    def check_stop_loss(self, position_id: str, current_price: float) -> bool:
        """
        Check if stop loss has been triggered.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            
        Returns:
            True if stop loss triggered
        """
        # Find position by ID
        position = None
        for product_positions in self.active_positions.values():
            if position_id in product_positions:
                position = product_positions[position_id]
                break
        
        if not position:
            return False
            
        stop_price = position.get("stop_loss")
        if stop_price is None:
            return False
            
        position_type = position["signal_type"]
        
        # Check if stop loss triggered
        if position_type == "buy" and current_price <= stop_price:
            logger.info(f"Stop loss triggered for {position_id}: price={current_price}, stop={stop_price}")
            return True
        elif position_type == "sell" and current_price >= stop_price:
            logger.info(f"Stop loss triggered for {position_id}: price={current_price}, stop={stop_price}")
            return True
            
        return False
    
    def check_take_profit(self, position_id: str, current_price: float) -> bool:
        """
        Check if take profit has been triggered.
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            
        Returns:
            True if take profit triggered
        """
        # Find position by ID
        position = None
        for product_positions in self.active_positions.values():
            if position_id in product_positions:
                position = product_positions[position_id]
                break
        
        if not position:
            return False
            
        take_profit_price = position.get("take_profit")
        if take_profit_price is None:
            return False
            
        position_type = position["signal_type"]
        
        # Check if take profit triggered
        if position_type == "buy" and current_price >= take_profit_price:
            logger.info(f"Take profit triggered for {position_id}: price={current_price}, target={take_profit_price}")
            return True
        elif position_type == "sell" and current_price <= take_profit_price:
            logger.info(f"Take profit triggered for {position_id}: price={current_price}, target={take_profit_price}")
            return True
            
        return False
    
    def get_active_positions(self, product_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active positions, optionally filtered by product.
        
        Args:
            product_id: Optional product identifier to filter positions
            
        Returns:
            List of active position data
        """
        positions = []
        
        if product_id:
            if product_id in self.active_positions:
                positions.extend(self.active_positions[product_id].values())
        else:
            for product_positions in self.active_positions.values():
                positions.extend(product_positions.values())
                
        return positions
    
    def get_portfolio_risk(self) -> Dict[str, Any]:
        """
        Get current portfolio risk metrics.
        
        Returns:
            Dictionary with risk metrics
        """
        total_positions = sum(len(positions) for positions in self.active_positions.values())
        products_traded = len(self.active_positions)
        
        # Sum up unrealized P&L
        total_unrealized_pnl = 0.0
        total_position_value = 0.0
        positions_in_profit = 0
        positions_in_loss = 0
        
        for product_positions in self.active_positions.values():
            for position in product_positions.values():
                total_position_value += position.get("current_value", position.get("value", 0))
                
                if "unrealized_pnl" in position:
                    total_unrealized_pnl += position["unrealized_pnl"]
                    
                    if position["unrealized_pnl"] > 0:
                        positions_in_profit += 1
                    elif position["unrealized_pnl"] < 0:
                        positions_in_loss += 1
        
        return {
            "total_positions": total_positions,
            "products_traded": products_traded,
            "total_position_value": total_position_value,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_risk_amount": self.total_risk_amount,
            "positions_in_profit": positions_in_profit,
            "positions_in_loss": positions_in_loss
        }
    
    def check_max_risk_exceeded(self, additional_risk: float = 0.0, balance: float = 0.0) -> bool:
        """
        Check if adding a new position would exceed maximum risk.
        
        Args:
            additional_risk: Additional risk amount to consider
            balance: Account balance for percentage calculation
            
        Returns:
            True if max risk would be exceeded
        """
        if balance <= 0:
            # Can't calculate risk percentage without balance
            return self.total_risk_amount + additional_risk > 0
            
        total_risk_percent = (self.total_risk_amount + additional_risk) / balance
        
        if total_risk_percent > self.max_risk_percent_total:
            logger.warning(f"Maximum risk exceeded: {total_risk_percent:.2%} > {self.max_risk_percent_total:.2%}")
            return True
            
        return False
    
    def adjust_for_correlation(self, 
                             position_sizing: PositionSizing, 
                             product_id: str,
                             correlations: Dict[str, float]) -> PositionSizing:
        """
        Adjust position size based on correlation with existing positions.
        
        Args:
            position_sizing: Original position sizing
            product_id: Product identifier
            correlations: Dictionary of correlations between products
            
        Returns:
            Adjusted position sizing
        """
        # Start with the original position sizing
        adjusted = position_sizing
        
        # Check for existing positions in correlated assets
        correlated_exposure = 0.0
        
        for existing_product in self.active_positions:
            if existing_product == product_id:
                continue
                
            correlation_key = f"{product_id}_{existing_product}"
            alt_correlation_key = f"{existing_product}_{product_id}"
            
            # Look up correlation coefficient
            correlation = correlations.get(correlation_key, correlations.get(alt_correlation_key, 0))
            
            # Skip if correlation is low
            if abs(correlation) < 0.5:
                continue
                
            # Calculate exposure from this product
            product_exposure = sum(p.get("value", 0) for p in self.active_positions[existing_product].values())
            
            # Add to correlated exposure, weighted by correlation coefficient
            correlated_exposure += product_exposure * abs(correlation)
        
        if correlated_exposure > 0:
            # Adjust position size based on correlated exposure
            # Reduce position more when correlation is higher
            reduction_factor = min(0.8, correlated_exposure / (correlated_exposure + adjusted.position_value))
            
            new_position_value = adjusted.position_value * (1 - reduction_factor)
            new_position_size = new_position_value / (adjusted.position_value / adjusted.position_size)
            
            # Update position sizing
            adjusted.position_size = new_position_size
            adjusted.position_value = new_position_value
            adjusted.risk_amount = adjusted.risk_amount * (1 - reduction_factor)
            
            logger.info(f"Adjusted position size for correlation: {product_id}, " +
                      f"original={position_sizing.position_value:.2f}, " +
                      f"adjusted={new_position_value:.2f}, " +
                      f"reduction={reduction_factor:.2%}")
        
        return adjusted 