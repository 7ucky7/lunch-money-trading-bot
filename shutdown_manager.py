#!/usr/bin/env python3
"""
Shutdown manager for graceful application termination.

This module provides a robust shutdown mechanism with signal handlers,
resource cleanup, and proper termination notification.
"""

import signal
import logging
import sys
import time
import threading
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable, Set

logger = logging.getLogger("ShutdownManager")

class ShutdownManager:
    """
    Manages graceful application shutdown.
    
    This class sets up signal handlers for SIGINT, SIGTERM, etc. and
    provides a way to register cleanup functions that need to be called
    during shutdown (e.g., closing connections, saving state).
    """
    
    def __init__(self, 
                app_name: str = "TradingBot", 
                shutdown_timeout: float = 30.0,
                exit_code: int = 0):
        """
        Initialize shutdown manager.
        
        Args:
            app_name: Application name for logging
            shutdown_timeout: Maximum time in seconds to wait for cleanup
            exit_code: Default exit code to use when shutting down
        """
        self.app_name = app_name
        self.shutdown_timeout = shutdown_timeout
        self.exit_code = exit_code
        
        # Shutdown flag
        self._shutdown_requested = False
        self._shutdown_in_progress = False
        self._shutdown_completed = False
        
        # Cleanup tasks and locks
        self._cleanup_tasks: Dict[str, Callable] = {}
        self._lock = threading.RLock()
        
        # Registered event loops
        self._event_loops: Set[asyncio.AbstractEventLoop] = set()
        
        # Install signal handlers
        self._setup_signal_handlers()
        
        logger.info(f"Shutdown manager initialized for {app_name}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for SIGINT (Ctrl+C), SIGTERM, etc."""
        try:
            # Register signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Optional: Handle SIGHUP for configuration reload
            if hasattr(signal, 'SIGHUP'):  # Not available on Windows
                signal.signal(signal.SIGHUP, self._reload_handler)
                
            logger.debug("Signal handlers installed")
        except (ValueError, AttributeError) as e:
            # This can happen if running in a thread
            logger.warning(f"Failed to set up signal handlers: {e}")
    
    def _signal_handler(self, sig: int, frame) -> None:
        """
        Handle termination signals.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        sig_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else str(sig)
        logger.info(f"Received signal {sig_name}")
        
        if self._shutdown_in_progress:
            logger.warning("Shutdown already in progress. Forcing exit...")
            sys.exit(1)
        
        self.initiate_shutdown(reason=f"Received signal {sig_name}")
    
    def _reload_handler(self, sig: int, frame) -> None:
        """
        Handle reload signal (SIGHUP).
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        logger.info("Received reload signal (SIGHUP)")
        # You can add custom reload logic here
    
    def register_cleanup_task(self, name: str, task: Callable) -> None:
        """
        Register a cleanup task to run during shutdown.
        
        Args:
            name: Task name for logging
            task: Callable to execute during shutdown
        """
        with self._lock:
            if name in self._cleanup_tasks:
                logger.warning(f"Cleanup task '{name}' already registered, overwriting")
            self._cleanup_tasks[name] = task
            logger.debug(f"Registered cleanup task: {name}")
    
    def unregister_cleanup_task(self, name: str) -> None:
        """
        Unregister a cleanup task.
        
        Args:
            name: Task name to unregister
        """
        with self._lock:
            if name in self._cleanup_tasks:
                del self._cleanup_tasks[name]
                logger.debug(f"Unregistered cleanup task: {name}")
            else:
                logger.warning(f"Cleanup task '{name}' not found")
    
    def register_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Register an asyncio event loop to stop during shutdown.
        
        Args:
            loop: Asyncio event loop
        """
        with self._lock:
            self._event_loops.add(loop)
            logger.debug(f"Registered event loop: {id(loop)}")
    
    def unregister_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Unregister an asyncio event loop.
        
        Args:
            loop: Asyncio event loop to unregister
        """
        with self._lock:
            if loop in self._event_loops:
                self._event_loops.remove(loop)
                logger.debug(f"Unregistered event loop: {id(loop)}")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested
    
    def is_shutdown_in_progress(self) -> bool:
        """Check if shutdown is currently in progress."""
        return self._shutdown_in_progress
    
    def initiate_shutdown(self, 
                          reason: str = "Shutdown requested", 
                          exit_code: Optional[int] = None,
                          exit_immediately: bool = False) -> None:
        """
        Initiate application shutdown.
        
        Args:
            reason: Reason for shutdown (for logging)
            exit_code: Optional exit code to override default
            exit_immediately: If True, calls sys.exit after shutdown
        """
        with self._lock:
            if self._shutdown_requested:
                logger.warning("Shutdown already requested")
                return
                
            self._shutdown_requested = True
            
        logger.info(f"Initiating shutdown: {reason}")
        
        # Start shutdown in a separate thread to avoid blocking
        shutdown_thread = threading.Thread(
            target=self._shutdown_procedure,
            args=(exit_code or self.exit_code, exit_immediately),
            name="ShutdownThread"
        )
        shutdown_thread.daemon = True
        shutdown_thread.start()
        
        # For immediate shutdown, wait for the thread to complete
        if exit_immediately:
            shutdown_thread.join(timeout=self.shutdown_timeout)
    
    def _shutdown_procedure(self, exit_code: int, exit_immediately: bool) -> None:
        """
        Execute the shutdown procedure.
        
        Args:
            exit_code: Exit code to use
            exit_immediately: Whether to exit after shutdown
        """
        with self._lock:
            if self._shutdown_in_progress:
                return
            self._shutdown_in_progress = True
        
        logger.info(f"Starting shutdown procedure for {self.app_name}")
        start_time = time.time()
        
        try:
            # Stop asyncio event loops first
            self._stop_event_loops()
            
            # Run cleanup tasks
            self._run_cleanup_tasks()
            
            # Mark shutdown as completed
            self._shutdown_completed = True
            
            elapsed = time.time() - start_time
            logger.info(f"Shutdown procedure completed in {elapsed:.2f} seconds")
            
            # Exit if requested
            if exit_immediately:
                logger.info(f"Exiting with code {exit_code}")
                sys.exit(exit_code)
                
        except Exception as e:
            logger.error(f"Error during shutdown procedure: {e}", exc_info=True)
            if exit_immediately:
                logger.info(f"Forcing exit with code 1 due to shutdown error")
                sys.exit(1)
    
    def _stop_event_loops(self) -> None:
        """Stop all registered asyncio event loops."""
        if not self._event_loops:
            return
            
        logger.info(f"Stopping {len(self._event_loops)} event loops")
        
        for loop in self._event_loops:
            try:
                # Schedule loop.stop() to run in the event loop
                if loop.is_running():
                    logger.debug(f"Stopping event loop: {id(loop)}")
                    loop.call_soon_threadsafe(loop.stop)
            except Exception as e:
                logger.error(f"Error stopping event loop: {e}")
    
    def _run_cleanup_tasks(self) -> None:
        """Run all registered cleanup tasks with timeout monitoring."""
        if not self._cleanup_tasks:
            logger.info("No cleanup tasks registered")
            return
            
        logger.info(f"Running {len(self._cleanup_tasks)} cleanup tasks")
        tasks_copy = self._cleanup_tasks.copy()
        
        # Create a timeout event
        timeout_event = threading.Event()
        
        # Start a timer thread to set the event after timeout
        timer = threading.Timer(
            self.shutdown_timeout, 
            lambda: timeout_event.set()
        )
        timer.daemon = True
        timer.start()
        
        try:
            for name, task in tasks_copy.items():
                # Check if timeout has occurred
                if timeout_event.is_set():
                    logger.warning(f"Shutdown timeout ({self.shutdown_timeout}s) reached. Skipping remaining tasks.")
                    break
                    
                # Run the cleanup task
                try:
                    logger.debug(f"Running cleanup task: {name}")
                    task()
                    logger.debug(f"Completed cleanup task: {name}")
                except Exception as e:
                    logger.error(f"Error in cleanup task '{name}': {e}", exc_info=True)
        finally:
            # Cancel the timer if it's still running
            timer.cancel()

# Global shutdown manager instance
_shutdown_manager = None

def get_shutdown_manager(app_name: str = "TradingBot") -> ShutdownManager:
    """
    Get or create the global shutdown manager instance.
    
    Args:
        app_name: Application name for logging
        
    Returns:
        ShutdownManager instance
    """
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager(app_name=app_name)
    return _shutdown_manager 