#!/usr/bin/env python3
"""
Configuration and environment variable management for the Cryptocurrency Trading Bot.

This module handles secure loading of API credentials, trading parameters, and other
configuration settings. It implements proper validation and fallbacks.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
import json

# Configure logger
logger = logging.getLogger("Config")

class Config:
    """Configuration manager for trading bot."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        # API connection settings
        "api_url": "https://api.exchange.coinbase.com",
        "max_retries": 3,
        "retry_delay": 2.0,
        "connection_timeout": 30,
        "max_connections": 5,
        
        # Rate limiting settings
        "max_calls_per_second": 3,
        "max_calls_per_minute": 150,
        
        # Logging settings
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        
        # Risk management settings
        "stop_loss_percent": 0.005,        # 0.5% stop loss
        "trailing_stop_percent": 0.002,    # 0.2% trailing stop
        "trade_frequency": 1,              # Trade frequency in seconds
        "min_spread": 0.008,               # Minimum spread
        "volatility_window": 5,            # Window for volatility detection
        "risk_percent": 0.01,              # 1% of balance per trade
        "kelly_criterion_multiplier": 0.5, # Conservative Kelly Criterion sizing
        "max_consecutive_losses": 3,       # Number of losses before cooldown
        "cooldown_period": 60,             # Cooldown period in seconds
        
        # Trading settings
        "product_ids": ["BTC-USD", "ETH-USD", "LTC-USD"],
        "maker_fee": 0.004,                # 0.4% maker fee
        "taker_fee": 0.006,                # 0.6% taker fee
        "min_order_size_usd": 10.00,       # Minimum order size in USD
        
        # Advanced features
        "use_ml_prediction": True,
        "use_order_book_analysis": True,
        "use_enhanced_technical_indicators": True,
        "backtest_mode": False
    }
    
    def __init__(self, env_file: str = ".env", config_file: Optional[str] = None):
        """
        Initialize configuration from environment variables and optional config file.
        
        Args:
            env_file: Path to .env file
            config_file: Optional path to JSON config file that overrides defaults
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load environment variables
        self._load_env_variables(env_file)
        
        # Load config file if provided
        if config_file:
            self._load_config_file(config_file)
            
        # Validate critical configuration settings
        self._validate_configuration()
        
        # Log configuration (excluding sensitive data)
        self._log_configuration()
    
    def _load_env_variables(self, env_file: str) -> None:
        """Load and validate environment variables."""
        # Load .env file
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        else:
            logger.warning(f"Environment file {env_file} not found")
            
        # Required API credentials
        self.api_key = os.getenv("COINBASE_API_KEY")
        self.api_secret = os.getenv("COINBASE_API_SECRET")
        self.api_passphrase = os.getenv("COINBASE_API_PASSPHRASE")
        
        # Optional environment overrides for config values
        for key in self.config:
            env_key = f"TRADING_BOT_{key.upper()}"
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Convert environment variable to appropriate type
                if isinstance(self.config[key], bool):
                    self.config[key] = env_value.lower() in ("true", "t", "1", "yes", "y")
                elif isinstance(self.config[key], int):
                    try:
                        self.config[key] = int(env_value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_key}: {env_value}")
                elif isinstance(self.config[key], float):
                    try:
                        self.config[key] = float(env_value)
                    except ValueError:
                        logger.warning(f"Invalid float value for {env_key}: {env_value}")
                elif isinstance(self.config[key], list):
                    try:
                        self.config[key] = json.loads(env_value)
                    except json.JSONDecodeError:
                        # Assume comma-separated list for simple lists
                        self.config[key] = [item.strip() for item in env_value.split(',')]
                else:
                    self.config[key] = env_value
                    
                logger.debug(f"Overrode {key} with environment variable {env_key}")
    
    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            if not os.path.exists(config_file):
                logger.warning(f"Config file {config_file} not found")
                return
                
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                
            # Update configuration with file values
            for key, value in file_config.items():
                if key in self.config:
                    self.config[key] = value
                else:
                    logger.warning(f"Unknown configuration key in file: {key}")
                    
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    def _validate_configuration(self) -> None:
        """Validate critical configuration settings."""
        # Check API credentials
        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            logger.error("Missing API credentials. Set COINBASE_API_KEY, COINBASE_API_SECRET, and COINBASE_API_PASSPHRASE")
            if not self.config.get("backtest_mode", False):
                sys.exit(1)  # Exit if not in backtest mode
                
        # Validate risk parameters
        if self.config["stop_loss_percent"] <= 0 or self.config["stop_loss_percent"] > 0.2:
            logger.warning(f"Unusual stop_loss_percent value: {self.config['stop_loss_percent']}. Should be between 0 and 0.2 (20%)")
            
        if self.config["trailing_stop_percent"] <= 0 or self.config["trailing_stop_percent"] > 0.1:
            logger.warning(f"Unusual trailing_stop_percent value: {self.config['trailing_stop_percent']}. Should be between 0 and 0.1 (10%)")
            
        if self.config["risk_percent"] <= 0 or self.config["risk_percent"] > 0.05:
            logger.warning(f"Unusual risk_percent value: {self.config['risk_percent']}. Should be between 0 and 0.05 (5%)")
            
        if self.config["kelly_criterion_multiplier"] <= 0 or self.config["kelly_criterion_multiplier"] > 1:
            logger.warning(f"Unusual kelly_criterion_multiplier value: {self.config['kelly_criterion_multiplier']}. Should be between 0 and 1")
            
        # Validate product IDs
        if not self.config["product_ids"]:
            logger.error("No product_ids specified in configuration")
            sys.exit(1)
    
    def _log_configuration(self) -> None:
        """Log non-sensitive configuration settings."""
        # Create a copy of config without sensitive data
        safe_config = self.config.copy()
        
        # Log configuration
        logger.info("Configuration loaded successfully")
        logger.debug(f"Configuration: {safe_config}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        
    def get_api_credentials(self) -> Dict[str, str]:
        """Get API credentials as a dictionary."""
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "api_passphrase": self.api_passphrase
        }
        
    @property
    def product_ids(self) -> List[str]:
        """Get list of product IDs to trade."""
        return self.config["product_ids"]
        
    @property
    def is_backtest_mode(self) -> bool:
        """Check if running in backtest mode."""
        return self.config.get("backtest_mode", False)
        
    @property
    def log_level(self) -> str:
        """Get log level as string."""
        return self.config.get("log_level", "INFO")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = self.config.copy()
        # Don't include API credentials
        return config_dict 