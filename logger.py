#!/usr/bin/env python3
"""
Enhanced logging system for Cryptocurrency Trading Bot.

This module configures logging with rotating file handlers, console output,
and different log levels for different types of messages.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import datetime
from typing import Optional, Dict, Any

class Logger:
    """Enhanced logging system with multiple handlers and log rotation."""
    
    # Log levels
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    def __init__(
        self,
        name: str = "TradingBot",
        log_level: str = "INFO",
        log_dir: str = "logs",
        log_to_console: bool = True,
        log_format: Optional[str] = None,
        max_file_size_mb: int = 10,
        backup_count: int = 5,
    ):
        """
        Initialize the enhanced logging system.
        
        Args:
            name: Logger name
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            log_to_console: Whether to output logs to console
            log_format: Custom log format (or None for default)
            max_file_size_mb: Maximum log file size before rotation
            backup_count: Number of backup logs to keep
        """
        self.name = name
        self.log_level = self._get_log_level(log_level)
        self.log_dir = log_dir
        self.log_to_console = log_to_console
        self.log_format = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.backup_count = backup_count
        
        # Create logger instance
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Remove existing handlers if any
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Add handlers
        self._add_file_handler()
        if log_to_console:
            self._add_console_handler()
            
        # Log startup message
        self.logger.info(f"Logger initialized: {name}")
    
    def _get_log_level(self, level_name: str) -> int:
        """Convert string log level to numeric constant."""
        level_name = level_name.upper()
        if level_name not in self.LOG_LEVELS:
            print(f"Warning: Invalid log level '{level_name}'. Using INFO.")
            return logging.INFO
        return self.LOG_LEVELS[level_name]
    
    def _add_file_handler(self) -> None:
        """Add rotating file handler."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        log_filename = f"{self.log_dir}/{self.name.lower()}_{timestamp}.log"
        
        file_handler = RotatingFileHandler(
            log_filename,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Add separate error log handler for ERROR and CRITICAL
        error_log_filename = f"{self.log_dir}/{self.name.lower()}_{timestamp}_errors.log"
        error_handler = RotatingFileHandler(
            error_log_filename,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def _add_console_handler(self) -> None:
        """Add console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger

def setup_logging(
    config: Dict[str, Any],
    name: str = "TradingBot",
) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        config: Configuration dictionary
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    log_level = config.get("log_level", "INFO")
    log_dir = config.get("log_dir", "logs")
    log_format = config.get("log_format", None)
    log_to_console = config.get("log_to_console", True)
    max_file_size_mb = config.get("max_file_size_mb", 10)
    backup_count = config.get("backup_count", 5)
    
    # Initialize logger
    logger_instance = Logger(
        name=name,
        log_level=log_level,
        log_dir=log_dir,
        log_to_console=log_to_console,
        log_format=log_format,
        max_file_size_mb=max_file_size_mb,
        backup_count=backup_count
    )
    
    return logger_instance.get_logger() 