#!/usr/bin/env python3
"""
Enhanced API client for Coinbase Pro and Advanced API.

This module provides a wrapper around the Coinbase API clients with
improved error handling, rate limiting, connection management, 
and automatic retries.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
import cbpro
from functools import wraps
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
import datetime
import json

# Try to import coinbase_advanced_py, but make it optional
try:
    import coinbase_advanced_py as capy
    ADVANCED_API_AVAILABLE = True
except ImportError:
    ADVANCED_API_AVAILABLE = False

logger = logging.getLogger("APIClient")

class RateLimiter:
    """Rate limiter to prevent API throttling."""
    
    def __init__(self, 
                 calls_per_second: int = 3, 
                 calls_per_minute: int = 150,
                 burst_allowance: int = 5):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum API calls per second
            calls_per_minute: Maximum API calls per minute
            burst_allowance: Number of consecutive bursts allowed before throttling
        """
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.burst_allowance = burst_allowance
        
        # Tracking variables
        self.second_calls = []
        self.minute_calls = []
        self.consecutive_bursts = 0
        
        logger.debug(f"Rate limiter initialized: {calls_per_second}/sec, {calls_per_minute}/min")
    
    def _cleanup_old_calls(self) -> None:
        """Remove outdated call timestamps."""
        now = time.time()
        
        # Keep only calls from the last second
        self.second_calls = [t for t in self.second_calls if now - t < 1.0]
        
        # Keep only calls from the last minute
        self.minute_calls = [t for t in self.minute_calls if now - t < 60.0]
    
    def wait_if_needed(self) -> None:
        """
        Wait if necessary to comply with rate limits.
        This method is blocking and should be called before each API request.
        """
        now = time.time()
        self._cleanup_old_calls()
        
        # Check second limit
        second_count = len(self.second_calls)
        if second_count >= self.calls_per_second:
            # We've hit the per-second limit
            self.consecutive_bursts += 1
            
            # If we've had too many consecutive bursts, enforce a cooldown
            if self.consecutive_bursts > self.burst_allowance:
                sleep_time = 1.0
                logger.warning(f"Rate limit burst threshold exceeded. Cooling down for {sleep_time}s")
                time.sleep(sleep_time)
                self.consecutive_bursts = 0
                self._cleanup_old_calls()  # Refresh after waiting
            else:
                # Just wait until the oldest call is 1 second old
                oldest_call = self.second_calls[0]
                sleep_time = max(0, 1.0 - (now - oldest_call))
                
                if sleep_time > 0:
                    logger.debug(f"Rate limit approaching: {second_count}/{self.calls_per_second} per second. Waiting {sleep_time:.3f}s")
                    time.sleep(sleep_time)
        else:
            # Reset burst counter if we're not at the limit
            self.consecutive_bursts = 0
        
        # Check minute limit
        minute_count = len(self.minute_calls)
        if minute_count >= self.calls_per_minute:
            # Wait until oldest call is a minute old
            oldest_call = self.minute_calls[0]
            sleep_time = max(0, 60.0 - (now - oldest_call))
            
            if sleep_time > 0:
                logger.warning(f"Minute rate limit reached: {minute_count}/{self.calls_per_minute}. Waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        # Record this call
        self.second_calls.append(time.time())
        self.minute_calls.append(time.time())

class APIRetry:
    """Decorator for automatic retry of API calls."""
    
    def __init__(self, 
                 max_retries: int = 3, 
                 retry_delay: float = 2.0,
                 backoff_factor: float = 1.5,
                 retry_exceptions: List[Exception] = None):
        """
        Initialize retry decorator.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            backoff_factor: Multiplier for delay after each retry
            retry_exceptions: List of exception types to retry on
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.retry_exceptions = retry_exceptions or [
            ConnectionError, Timeout, RequestException, 
            cbpro.AuthenticationError, cbpro.PublicAPIError
        ]
    
    def __call__(self, func):
        """Apply retry logic to the decorated function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = self.retry_delay
            
            for attempt in range(self.max_retries + 1):
                try:
                    if attempt > 0:
                        logger.warning(f"Retry attempt {attempt}/{self.max_retries} for {func.__name__}...")
                    
                    return func(*args, **kwargs)
                
                except tuple(self.retry_exceptions) as e:
                    last_exception = e
                    
                    # Some APIs return error codes we should not retry
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = getattr(e.response, 'status_code', None)
                        
                        # Don't retry client errors (except for 429 Too Many Requests)
                        if status_code and 400 <= status_code < 500 and status_code != 429:
                            logger.error(f"API error {status_code} in {func.__name__}: {str(e)}")
                            raise
                    
                    if attempt < self.max_retries:
                        logger.warning(f"API call {func.__name__} failed: {str(e)}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= self.backoff_factor
                    else:
                        logger.error(f"API call {func.__name__} failed after {self.max_retries} retries: {str(e)}")
                        raise
                
                except Exception as e:
                    logger.error(f"Unhandled exception in {func.__name__}: {str(e)}")
                    raise
            
            # This shouldn't be reached, but just in case
            raise last_exception
        
        return wrapper

class CoinbaseClient:
    """Enhanced Coinbase API client with improved error handling and rate limiting."""
    
    def __init__(self, 
                 api_key: str, 
                 api_secret: str, 
                 api_passphrase: str,
                 api_url: Optional[str] = None,
                 max_retries: int = 3,
                 retry_delay: float = 2.0,
                 calls_per_second: int = 3,
                 calls_per_minute: int = 150):
        """
        Initialize enhanced Coinbase API client.
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            api_passphrase: Coinbase API passphrase
            api_url: Optional custom API URL (None for default)
            max_retries: Maximum retry attempts for failed API calls
            retry_delay: Initial delay between retries (seconds)
            calls_per_second: Maximum API calls per second
            calls_per_minute: Maximum API calls per minute
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.api_url = api_url
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            calls_per_second=calls_per_second,
            calls_per_minute=calls_per_minute
        )
        
        # Initialize retry decorator
        self.retry_decorator = APIRetry(
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Initialize clients
        self._init_clients()
        
        # Connection health tracking
        self.last_successful_call = None
        self.failed_calls_count = 0
        self.connection_healthy = True
        
        logger.info("Coinbase API client initialized")
    
    def _init_clients(self) -> None:
        """Initialize Coinbase API clients."""
        # Initialize standard cbpro client
        kwargs = {}
        if self.api_url:
            kwargs['api_url'] = self.api_url
            
        self.auth_client = cbpro.AuthenticatedClient(
            self.api_key, self.api_secret, self.api_passphrase, **kwargs
        )
        
        # Initialize advanced client if available
        self.advanced_client = None
        if ADVANCED_API_AVAILABLE:
            try:
                self.advanced_client = capy.Client(
                    self.api_key, self.api_secret, self.api_passphrase
                )
                logger.info("Advanced API client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced API client: {str(e)}")
    
    def _check_connection_health(self) -> None:
        """Check API connection health and reconnect if needed."""
        now = time.time()
        
        # If we've never made a successful call or it's been too long
        if (self.last_successful_call is None or 
            now - self.last_successful_call > 300 or  # 5 minutes
            self.failed_calls_count >= 5):
            
            if self.connection_healthy:
                logger.warning("Connection health check failed. Attempting to reconnect...")
                self.connection_healthy = False
            
            # Attempt to reconnect
            try:
                self._init_clients()
                self.ping()  # Test the connection
                self.connection_healthy = True
                self.failed_calls_count = 0
                logger.info("Successfully reconnected to Coinbase API")
            except Exception as e:
                logger.error(f"Failed to reconnect to Coinbase API: {str(e)}")
    
    @APIRetry()
    def ping(self) -> bool:
        """
        Test API connection by fetching time.
        
        Returns:
            True if connection is successful
        """
        self.rate_limiter.wait_if_needed()
        _ = self.auth_client.get_time()
        self.last_successful_call = time.time()
        self.failed_calls_count = 0
        return True
    
    def _api_call_wrapper(self, api_method: Callable, *args, **kwargs) -> Any:
        """
        Wrapper for API calls with rate limiting and error handling.
        
        Args:
            api_method: API method to call
            args: Positional arguments for the API method
            kwargs: Keyword arguments for the API method
            
        Returns:
            API call result
        """
        self._check_connection_health()
        self.rate_limiter.wait_if_needed()
        
        try:
            result = api_method(*args, **kwargs)
            self.last_successful_call = time.time()
            self.failed_calls_count = 0
            return result
        except Exception as e:
            self.failed_calls_count += 1
            raise
    
    @APIRetry()
    def get_accounts(self) -> List[Dict[str, Any]]:
        """Get all accounts."""
        return self._api_call_wrapper(self.auth_client.get_accounts)
    
    @APIRetry()
    def get_account(self, account_id: str) -> Dict[str, Any]:
        """Get a specific account by ID."""
        return self._api_call_wrapper(self.auth_client.get_account, account_id)
    
    @APIRetry()
    def get_product_ticker(self, product_id: str) -> Dict[str, Any]:
        """Get ticker for a product."""
        return self._api_call_wrapper(self.auth_client.get_product_ticker, product_id=product_id)
    
    @APIRetry()
    def get_product_order_book(self, product_id: str, level: int = 1) -> Dict[str, Any]:
        """Get order book for a product."""
        return self._api_call_wrapper(self.auth_client.get_product_order_book, product_id=product_id, level=level)
    
    @APIRetry()
    def get_product_trades(self, product_id: str) -> List[Dict[str, Any]]:
        """Get recent trades for a product."""
        return self._api_call_wrapper(self.auth_client.get_product_trades, product_id=product_id)
    
    @APIRetry()
    def get_product_24hr_stats(self, product_id: str) -> Dict[str, Any]:
        """Get 24-hour stats for a product."""
        return self._api_call_wrapper(self.auth_client.get_product_24hr_stats, product_id=product_id)
    
    @APIRetry()
    def get_products(self) -> List[Dict[str, Any]]:
        """Get list of available products."""
        return self._api_call_wrapper(self.auth_client.get_products)
    
    @APIRetry()
    def get_product_historic_rates(self, 
                                  product_id: str, 
                                  start=None, 
                                  end=None, 
                                  granularity=None) -> List[List[Any]]:
        """Get historic rates for a product."""
        return self._api_call_wrapper(
            self.auth_client.get_product_historic_rates,
            product_id=product_id,
            start=start,
            end=end,
            granularity=granularity
        )
    
    @APIRetry()
    def get_time(self) -> Dict[str, Any]:
        """Get API server time."""
        return self._api_call_wrapper(self.auth_client.get_time)
    
    @APIRetry()
    def place_limit_order(self, 
                         product_id: str, 
                         side: str, 
                         price: str, 
                         size: str,
                         **kwargs) -> Dict[str, Any]:
        """Place a limit order."""
        return self._api_call_wrapper(
            self.auth_client.place_limit_order,
            product_id=product_id,
            side=side,
            price=price,
            size=size,
            **kwargs
        )
    
    @APIRetry()
    def place_market_order(self, 
                          product_id: str, 
                          side: str, 
                          **kwargs) -> Dict[str, Any]:
        """Place a market order."""
        return self._api_call_wrapper(
            self.auth_client.place_market_order,
            product_id=product_id,
            side=side,
            **kwargs
        )
    
    @APIRetry()
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        return self._api_call_wrapper(self.auth_client.cancel_order, order_id)
    
    @APIRetry()
    def cancel_all(self, product_id: Optional[str] = None) -> List[str]:
        """Cancel all orders (or for a specific product)."""
        return self._api_call_wrapper(self.auth_client.cancel_all, product_id=product_id)
    
    @APIRetry()
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get information for a specific order."""
        return self._api_call_wrapper(self.auth_client.get_order, order_id)
    
    @APIRetry()
    def get_orders(self, product_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Get list of orders."""
        return self._api_call_wrapper(self.auth_client.get_orders, product_id=product_id, **kwargs)
    
    @APIRetry()
    def get_fills(self, product_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Get list of filled orders."""
        return self._api_call_wrapper(self.auth_client.get_fills, product_id=product_id, **kwargs)
    
    # Advanced API methods (if available)
    def has_advanced_api(self) -> bool:
        """Check if advanced API is available."""
        return self.advanced_client is not None
    
    def get_advanced_client(self) -> Any:
        """Get the advanced API client (if available)."""
        if not self.has_advanced_api():
            logger.warning("Advanced API client requested but not available")
            return None
        return self.advanced_client 