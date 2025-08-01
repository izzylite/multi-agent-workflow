"""
Browserbase Agent Tools Module

Provides CrewAI tools that leverage Browserbase for navigation, element interaction, 
and data extraction. These tools are designed to work seamlessly with CrewAI agents
for web automation and scraping tasks.
"""

import asyncio
import logging
import time
import hashlib
import json
import threading
from typing import List, Optional, Dict, Any, Union, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .browserbase_manager import BrowserbaseManager, SessionInfo, SessionConfig
from .browser_operations import BrowserOperations, BrowserOperationResult, BrowserOperationType
from .exceptions import (
    SessionCreationError, SessionConnectionError, NavigationError,
    ElementNotFoundError, ElementInteractionError, ScreenshotError,
    TimeoutError, classify_error, is_retryable_error, get_error_severity
)


# Global registry for browser managers to solve CrewAI serialization issues
class BrowserManagerRegistry:
    """Global registry for browser managers to avoid Pydantic serialization issues."""
    
    _lock = threading.Lock()
    _managers: Dict[str, BrowserbaseManager] = {}
    _default_manager: Optional[BrowserbaseManager] = None
    
    @classmethod
    def register_manager(cls, manager: BrowserbaseManager, name: str = "default") -> str:
        """Register a browser manager with the registry."""
        with cls._lock:
            cls._managers[name] = manager
            if name == "default":
                cls._default_manager = manager
            return name
    
    @classmethod
    def get_manager(cls, name: str = "default") -> Optional[BrowserbaseManager]:
        """Get a browser manager from the registry."""
        with cls._lock:
            return cls._managers.get(name)
    
    @classmethod
    def get_default_manager(cls) -> Optional[BrowserbaseManager]:
        """Get the default browser manager."""
        with cls._lock:
            return cls._default_manager
    
    @classmethod
    def clear(cls):
        """Clear all registered managers."""
        with cls._lock:
            cls._managers.clear()
            cls._default_manager = None


# Global instance
browser_registry = BrowserManagerRegistry()


# Global session registry to share sessions across tools
class SessionRegistry:
    """Global registry for browser sessions to avoid creating too many sessions."""
    
    _lock = threading.Lock()
    _sessions: Dict[str, SessionInfo] = {}
    _default_session: Optional[SessionInfo] = None
    _default_operations: Optional[BrowserOperations] = None
    
    @classmethod
    def register_session(cls, session: SessionInfo, name: str = "default") -> str:
        """Register a session with the global registry."""
        with cls._lock:
            cls._sessions[name] = session
            if name == "default":
                cls._default_session = session
            return name
    
    @classmethod
    def get_session(cls, name: str = "default") -> Optional[SessionInfo]:
        """Get a session from the registry."""
        with cls._lock:
            return cls._sessions.get(name)
    
    @classmethod
    def get_default_session(cls) -> Optional[SessionInfo]:
        """Get the default session."""
        with cls._lock:
            return cls._default_session
    
    @classmethod
    def register_operations(cls, operations: BrowserOperations, name: str = "default") -> None:
        """Register browser operations with the registry."""
        with cls._lock:
            if name == "default":
                cls._default_operations = operations
    
    @classmethod
    def get_default_operations(cls) -> Optional[BrowserOperations]:
        """Get the default browser operations."""
        with cls._lock:
            return cls._default_operations
    
    @classmethod
    def clear(cls):
        """Clear all sessions from the registry."""
        with cls._lock:
            cls._sessions.clear()
            cls._default_session = None
            cls._default_operations = None


# Global session registry instance
session_registry = SessionRegistry()


class ToolResultStatus(Enum):
    """Status of tool execution results."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RETRY = "retry"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    status: ToolResultStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0
    retry_count: int = 0
    cache_hit: bool = False


class AntiBotConfig(BaseModel):
    """Configuration for anti-bot features."""
    enable_random_delays: bool = True
    min_delay: float = 0.5
    max_delay: float = 2.0
    enable_human_mouse: bool = True
    enable_user_agent_rotation: bool = True
    enable_proxy_rotation: bool = False
    enable_viewport_randomization: bool = True
    enable_cookie_management: bool = True
    enable_stealth_mode: bool = False
    mouse_movement_variance: float = 0.3
    typing_speed_variance: float = 0.2
    user_agents: List[str] = Field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0"
    ])
    proxy_list: Optional[List[str]] = None
    viewport_sizes: List[Dict[str, int]] = Field(default_factory=lambda: [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1280, "height": 720}
    ])


class BrowserbaseToolInput(BaseModel):
    """Base input schema for Browserbase tools."""
    session_config: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional session configuration for this operation"
    )
    timeout: int = Field(
        default=30000, 
        description="Timeout for the operation in milliseconds"
    )
    retry_attempts: int = Field(
        default=3, 
        description="Number of retry attempts for failed operations"
    )


class BrowserbaseTool(BaseTool):
    """
    Base class for Browserbase-powered agent tools.
    
    This class provides a unified interface for browser automation tools
    that integrate with CrewAI's agent system. It handles session management,
    error handling, caching, and anti-bot features.
    """
    
    name: str = "BrowserbaseTool"
    description: str = "Base class for Browserbase-powered agent tools"
    args_schema: Type[BaseModel] = BrowserbaseToolInput
    
    def __init__(self, 
                 browser_manager: BrowserbaseManager,
                 anti_bot_config: Optional[AntiBotConfig] = None,
                 enable_caching: bool = True,
                 cache_ttl: int = 300,  # 5 minutes
                 **kwargs):
        """
        Initialize the BrowserbaseTool.
        
        Args:
            browser_manager: BrowserbaseManager instance for session management
            anti_bot_config: Configuration for anti-bot features
            enable_caching: Whether to enable result caching
            cache_ttl: Cache time-to-live in seconds
            **kwargs: Additional arguments passed to BaseTool
        """
        # Store the browser manager and other attributes as instance variables
        # that are not Pydantic fields
        self._browser_manager = browser_manager
        self._browser_operations = None
        self._current_session = None
        self._anti_bot_config = anti_bot_config or AntiBotConfig()
        self._enable_caching = enable_caching
        self._cache_ttl = cache_ttl
        self._cache_dict = {}
        self._cache_timestamps_dict = {}
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'retry_attempts': 0,
            'total_duration': 0.0
        }
        
        super().__init__(**kwargs)
    
    @property
    def browser_manager(self) -> BrowserbaseManager:
        """Get the browser manager."""
        # First try to get from instance
        if hasattr(self, '_browser_manager') and self._browser_manager is not None:
            return self._browser_manager
        
        # Fall back to global registry
        manager = browser_registry.get_default_manager()
        if manager is not None:
            return manager
            
        raise RuntimeError("Browser manager not initialized. Tool was not properly created and no default manager registered.")
    
    @property
    def browser_operations(self) -> Optional[BrowserOperations]:
        """Get the browser operations."""
        if not hasattr(self, '_browser_operations'):
            self._browser_operations = None
        return self._browser_operations
    
    @browser_operations.setter
    def browser_operations(self, value: Optional[BrowserOperations]):
        """Set the browser operations."""
        self._browser_operations = value
    
    @property
    def current_session(self) -> Optional[SessionInfo]:
        """Get the current session."""
        if not hasattr(self, '_current_session'):
            self._current_session = None
        return self._current_session
    
    @current_session.setter
    def current_session(self, value: Optional[SessionInfo]):
        """Set the current session."""
        self._current_session = value
    
    @property
    def anti_bot_config(self) -> AntiBotConfig:
        """Get the anti-bot configuration."""
        if not hasattr(self, '_anti_bot_config'):
            self._anti_bot_config = AntiBotConfig()
        return self._anti_bot_config
    
    @property
    def enable_caching(self) -> bool:
        """Get the caching setting."""
        return self._enable_caching
    
    @property
    def cache_ttl(self) -> int:
        """Get the cache TTL."""
        return self._cache_ttl
    
    @property
    def _cache(self) -> Dict[str, ToolResult]:
        """Get the cache."""
        return self._cache_dict
    
    @_cache.setter
    def _cache(self, value: Dict[str, ToolResult]):
        """Set the cache."""
        self._cache_dict = value
    
    @property
    def _cache_timestamps(self) -> Dict[str, datetime]:
        """Get the cache timestamps."""
        return self._cache_timestamps_dict
    
    @_cache_timestamps.setter
    def _cache_timestamps(self, value: Dict[str, datetime]):
        """Set the cache timestamps."""
        self._cache_timestamps_dict = value
    
    @property
    def logger(self) -> logging.Logger:
        """Get the logger."""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        return self._logger
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get the stats."""
        if not hasattr(self, '_stats'):
            self._stats = {
                'total_operations': 0,
                'successful_operations': 0,
                'failed_operations': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'retry_attempts': 0,
                'total_duration': 0.0
            }
        return self._stats
    
    def _get_cache_key(self, operation: str, **kwargs) -> str:
        """
        Generate a cache key for the operation.
        
        Args:
            operation: Name of the operation
            **kwargs: Operation parameters
            
        Returns:
            Cache key string
        """
        # Create a deterministic string representation of the parameters
        param_str = json.dumps(kwargs, sort_keys=True, default=str)
        # Include session ID if available for session-specific caching
        session_id = getattr(self.current_session, 'session_id', 'no_session')
        return hashlib.md5(f"{operation}:{session_id}:{param_str}".encode()).hexdigest()
    
    def _should_cache_operation(self, operation: str, **kwargs) -> bool:
        """
        Determine if an operation should be cached.
        
        Args:
            operation: Name of the operation
            **kwargs: Operation parameters
            
        Returns:
            True if operation should be cached, False otherwise
        """
        # Don't cache operations that are likely to change frequently
        non_cacheable_operations = ['screenshot', 'wait']
        
        # Don't cache if caching is disabled
        if not self.enable_caching:
            return False
        
        # Don't cache certain operation types
        if operation in non_cacheable_operations:
            return False
        
        # Don't cache if force_refresh is specified
        if kwargs.get('force_refresh', False):
            return False
        
        return True
    
    def _get_cached_result(self, cache_key: str) -> Optional[ToolResult]:
        """
        Get a cached result if it exists and is still valid.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached result if valid, None otherwise
        """
        if not self.enable_caching:
            return None
        
        if cache_key not in self._cache:
            return None
        
        # Check if cache entry has expired
        timestamp = self._cache_timestamps.get(cache_key)
        if timestamp and datetime.now() - timestamp > timedelta(seconds=self.cache_ttl):
            # Remove expired cache entry
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            self.logger.debug(f"Cache entry expired: {cache_key}")
            return None
        
        result = self._cache[cache_key]
        result.cache_hit = True
        self.stats['cache_hits'] += 1
        self.logger.debug(f"Cache hit: {cache_key}")
        return result
    
    def _invalidate_cache(self, pattern: str = None) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Optional pattern to match cache keys (if None, clears all)
            
        Returns:
            Number of cache entries invalidated
        """
        if not self.enable_caching:
            return 0
        
        invalidated_count = 0
        
        if pattern is None:
            # Clear all cache
            invalidated_count = len(self._cache)
            self._cache.clear()
            self._cache_timestamps.clear()
            self.logger.info(f"Cleared all cache entries ({invalidated_count} entries)")
        else:
            # Clear matching cache entries
            keys_to_remove = [key for key in self._cache.keys() if pattern in key]
            for key in keys_to_remove:
                del self._cache[key]
                del self._cache_timestamps[key]
                invalidated_count += 1
            self.logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
        
        return invalidated_count
    
    def _cache_result(self, cache_key: str, result: ToolResult) -> None:
        """
        Cache a tool result.
        
        Args:
            cache_key: Cache key for the result
            result: Tool result to cache
        """
        if not self.enable_caching:
            return
        
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()
        self.stats['cache_misses'] += 1
        self.logger.debug(f"Cached result: {cache_key} (TTL: {self.cache_ttl}s)")
    
    async def _ensure_session(self, session_config: Optional[Dict[str, Any]] = None) -> SessionInfo:
        """
        Ensure a browser session is available for the tool.
        
        Args:
            session_config: Optional session configuration
            
        Returns:
            Active session info
        """
        # First try to get existing session from global registry
        existing_session = session_registry.get_default_session()
        existing_operations = session_registry.get_default_operations()
        
        if existing_session and existing_operations:
            # Use existing session and operations
            self.current_session = existing_session
            self.browser_operations = existing_operations
            self.logger.info(f"Reusing existing session: {existing_session.session_id}")
            return existing_session
        
        # Create new session if none exists
        if self.current_session is None:
            # Apply anti-bot configuration to session
            enhanced_config = await self._configure_session_with_anti_bot(session_config)
            
            # Create session configuration
            config = None
            if enhanced_config:
                config = SessionConfig(**enhanced_config)
            
            # Get or create a session
            self.current_session = self.browser_manager.get_session(config)
            
            # Initialize browser operations
            self.browser_operations = BrowserOperations(self.browser_manager)
            
            # Connect to the session
            await self.browser_operations.connect_session(self.current_session)
            
            # Register with global registry for sharing
            session_registry.register_session(self.current_session)
            session_registry.register_operations(self.browser_operations)
            
            self.logger.info(f"Created new session: {self.current_session.session_id} with anti-bot features")
        
        return self.current_session
    
    async def _release_session(self) -> None:
        """Release the current browser session."""
        if self.current_session:
            if self.browser_operations:
                await self.browser_operations.disconnect_session()
            
            self.browser_manager.release_session(self.current_session)
            self.current_session = None
            self.browser_operations = None
            
            self.logger.info("Released browser session")
    
    async def _apply_anti_bot_features(self) -> None:
        """
        Apply anti-bot features like random delays, user agent rotation, and viewport randomization.
        """
        if not self.anti_bot_config.enable_random_delays:
            return
        
        # Random delay with more sophisticated randomization
        base_delay = self.anti_bot_config.min_delay + (
            (self.anti_bot_config.max_delay - self.anti_bot_config.min_delay) * 
            (hash(str(time.time())) % 100) / 100
        )
        
        # Add additional variance based on operation type
        variance = (hash(str(time.time())) % 50) / 100  # 0-0.5 additional seconds
        delay = base_delay + variance
        
        self.logger.debug(f"Applying anti-bot delay: {delay:.2f}s (base: {base_delay:.2f}s, variance: {variance:.2f}s)")
        await asyncio.sleep(delay)
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent from the configured list."""
        if not self.anti_bot_config.enable_user_agent_rotation:
            return self.anti_bot_config.user_agents[0]
        
        import random
        return random.choice(self.anti_bot_config.user_agents)
    
    def _get_random_viewport(self) -> Dict[str, int]:
        """Get a random viewport size from the configured list."""
        if not self.anti_bot_config.enable_viewport_randomization:
            return self.anti_bot_config.viewport_sizes[0]
        
        import random
        return random.choice(self.anti_bot_config.viewport_sizes)
    
    def _get_random_proxy(self) -> Optional[str]:
        """Get a random proxy from the configured list."""
        if not self.anti_bot_config.enable_proxy_rotation or not self.anti_bot_config.proxy_list:
            return None
        
        import random
        return random.choice(self.anti_bot_config.proxy_list)
    
    async def _apply_human_like_behavior(self, operation_type: str) -> None:
        """
        Apply human-like behavior patterns based on operation type.
        
        Args:
            operation_type: Type of operation ('click', 'type', 'navigate', etc.)
        """
        if not self.anti_bot_config.enable_human_mouse:
            return
        
        # Different delays for different operation types
        operation_delays = {
            'click': (0.1, 0.3),
            'type': (0.05, 0.15),
            'navigate': (0.5, 1.5),
            'extract': (0.2, 0.5),
            'screenshot': (0.3, 0.8)
        }
        
        if operation_type in operation_delays:
            min_delay, max_delay = operation_delays[operation_type]
            delay = min_delay + ((max_delay - min_delay) * (hash(str(time.time())) % 100) / 100)
            self.logger.debug(f"Applying human-like delay for {operation_type}: {delay:.2f}s")
            await asyncio.sleep(delay)
    
    async def _configure_session_with_anti_bot(self, session_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Configure session with anti-bot features.
        
        Args:
            session_config: Optional existing session configuration
            
        Returns:
            Enhanced session configuration with anti-bot features
        """
        config = session_config or {}
        
        # Apply user agent rotation
        if self.anti_bot_config.enable_user_agent_rotation:
            config['user_agent'] = self._get_random_user_agent()
        
        # Apply viewport randomization
        if self.anti_bot_config.enable_viewport_randomization:
            viewport = self._get_random_viewport()
            config['viewport_width'] = viewport['width']
            config['viewport_height'] = viewport['height']
        
        # Apply proxy rotation
        if self.anti_bot_config.enable_proxy_rotation:
            proxy = self._get_random_proxy()
            if proxy:
                config['proxy_server'] = proxy
        
        # Apply stealth mode
        if self.anti_bot_config.enable_stealth_mode:
            config['stealth_mode'] = True
        
        return config
    
    async def _handle_error(self, error: Exception, operation: str) -> ToolResult:
        """
        Handle errors during tool execution.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            
        Returns:
            Tool result with error information
        """
        error_class = classify_error(error)
        severity = get_error_severity(error)
        
        self.logger.error(f"Error in {operation}: {error} (Class: {error_class}, Severity: {severity})")
        
        # Update statistics
        self.stats['failed_operations'] += 1
        
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error=str(error),
            duration=0.0
        )
    
    def _run(self, **kwargs) -> str:
        """
        Synchronous execution of the tool.
        
        This method is required by CrewAI. For browser tools, we run the
        async method in a new event loop.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool result as string
        """
        import asyncio
        import threading
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If there's already a running loop, run in a thread
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self._arun(**kwargs))
                    finally:
                        new_loop.close()
                
                # Use a thread to run the async method
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result()
            else:
                # No running loop, we can use the current one
                return loop.run_until_complete(self._arun(**kwargs))
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(**kwargs))
            finally:
                loop.close()
    
    async def _arun(self, **kwargs) -> str:
        """
        Asynchronous execution of the tool.
        
        This is the main entry point for tool execution. Subclasses
        should override this method to implement specific tool logic.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool result as string
        """
        raise NotImplementedError("Subclasses must implement _arun method")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get tool execution statistics.
        
        Returns:
            Dictionary containing tool statistics
        """
        return {
            **self.stats,
            'cache_size': len(self._cache),
            'cache_ttl': self.cache_ttl,
            'anti_bot_enabled': self.anti_bot_config.enable_random_delays
        }
    
    def clear_cache(self) -> None:
        """Clear the tool's result cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Tool cache cleared")
    
    def _log_operation_stats(self, operation: str, duration: float, success: bool) -> None:
        """
        Log operation statistics with enhanced metrics.
        
        Args:
            operation: Name of the operation
            duration: Operation duration in seconds
            success: Whether the operation was successful
        """
        self.stats['total_operations'] += 1
        self.stats['total_duration'] += duration
        
        if success:
            self.stats['successful_operations'] += 1
        else:
            self.stats['failed_operations'] += 1
        
        # Calculate performance metrics
        avg_duration = self.stats['total_duration'] / self.stats['total_operations']
        success_rate = (self.stats['successful_operations'] / self.stats['total_operations']) * 100
        cache_hit_rate = 0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])) * 100
        
        # Enhanced logging with more metrics
        self.logger.info(
            f"Operation: {operation} | "
            f"Duration: {duration:.2f}s | "
            f"Success: {success} | "
            f"Avg Duration: {avg_duration:.2f}s | "
            f"Success Rate: {success_rate:.1f}% | "
            f"Cache Hit Rate: {cache_hit_rate:.1f}% | "
            f"Cache Hits: {self.stats['cache_hits']} | "
            f"Cache Misses: {self.stats['cache_misses']}"
        )
    
    async def cleanup(self) -> None:
        """Clean up resources used by the tool."""
        await self._release_session()
        self.clear_cache()
        self.logger.info("Tool cleanup completed")


class NavigationTool(BrowserbaseTool):
    """
    Tool for browser navigation operations.
    
    Provides functionality for navigating to URLs, handling browser history,
    and managing page loads with configurable wait conditions.
    """
    
    name: str = "NavigationTool"
    description: str = "Navigate to URLs and manage browser history"
    
    class NavigationInput(BrowserbaseToolInput):
        url: str = Field(..., description="URL to navigate to")
        wait_until: str = Field(
            default="networkidle", 
            description="Wait condition: 'load', 'domcontentloaded', 'networkidle'"
        )
        wait_for_selector: Optional[str] = Field(
            default=None,
            description="Optional CSS selector to wait for after navigation"
        )
    
    args_schema: Type[BaseModel] = NavigationInput
    
    async def _arun(self, url: str, wait_until: str = "networkidle", 
                    wait_for_selector: Optional[str] = None, **kwargs) -> str:
        """
        Navigate to a URL with optional wait conditions.
        
        Args:
            url: URL to navigate to
            wait_until: Wait condition for page load
            wait_for_selector: Optional CSS selector to wait for
            **kwargs: Additional arguments
            
        Returns:
            Navigation result as JSON string
        """
        start_time = time.time()
        
        try:
            # Apply anti-bot features
            await self._apply_anti_bot_features()
            await self._apply_human_like_behavior('navigate')
            
            # Ensure session is available
            await self._ensure_session(kwargs.get('session_config'))
            
            # Perform navigation
            result = await self.browser_operations.navigate(
                url=url,
                wait_until=wait_until,
                timeout=kwargs.get('timeout', 30000)
            )
            
            # Wait for optional selector
            if wait_for_selector:
                await self.browser_operations.wait_for_element(
                    selector=wait_for_selector,
                    timeout=kwargs.get('timeout', 10000)
                )
            
            duration = time.time() - start_time
            
            # Log operation statistics
            self._log_operation_stats("navigation", duration, True)
            
            return json.dumps({
                'status': 'success',
                'url': url,
                'duration': duration,
                'wait_condition': wait_until,
                'wait_selector': wait_for_selector
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_operation_stats("navigation", duration, False)
            
            # Handle error and return proper JSON
            error_result = await self._handle_error(e, "navigation")
            return json.dumps({
                'status': 'error',
                'error': error_result.error,
                'duration': duration,
                'timestamp': error_result.timestamp.isoformat(),
                'url': url
            })


class InteractionTool(BrowserbaseTool):
    """
    Tool for browser element interaction operations.
    
    Provides functionality for clicking elements, typing text, selecting options,
    and other user interactions with web page elements.
    """
    
    name: str = "InteractionTool"
    description: str = "Interact with web page elements (click, type, select)"
    
    class InteractionInput(BrowserbaseToolInput):
        action: str = Field(..., description="Action to perform: 'click', 'type', 'select'")
        selector: str = Field(..., description="CSS selector for the target element")
        value: Optional[str] = Field(default=None, description="Value for type/select actions")
        timeout: int = Field(default=5000, description="Timeout for element interaction")
    
    args_schema: Type[BaseModel] = InteractionInput
    
    async def _arun(self, action: str, selector: str, value: Optional[str] = None, 
                    timeout: int = 5000, **kwargs) -> str:
        """
        Perform element interaction.
        
        Args:
            action: Action to perform ('click', 'type', 'select')
            selector: CSS selector for the target element
            value: Value for type/select actions
            timeout: Timeout for the interaction
            **kwargs: Additional arguments
            
        Returns:
            Interaction result as JSON string
        """
        start_time = time.time()
        
        try:
            # Apply anti-bot features
            await self._apply_anti_bot_features()
            await self._apply_human_like_behavior(action)
            
            # Ensure session is available
            await self._ensure_session(kwargs.get('session_config'))
            
            # Perform the interaction
            if action == 'click':
                result = await self.browser_operations.click(selector, timeout)
            elif action == 'type':
                if not value:
                    raise ValueError("Value is required for type action")
                result = await self.browser_operations.type_text(selector, value, timeout)
            elif action == 'select':
                if not value:
                    raise ValueError("Value is required for select action")
                result = await self.browser_operations.select_option(selector, value, timeout)
            else:
                raise ValueError(f"Unsupported action: {action}")
            
            duration = time.time() - start_time
            
            # Log operation statistics
            self._log_operation_stats("interaction", duration, True)
            
            return json.dumps({
                'status': 'success',
                'action': action,
                'selector': selector,
                'value': value,
                'duration': duration
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_operation_stats("interaction", duration, False)
            error_result = await self._handle_error(e, f"interaction_{action}")
            return json.dumps({
                'status': 'error',
                'error': error_result.error,
                'duration': duration,
                'timestamp': error_result.timestamp.isoformat(),
                'action': action,
                'selector': selector,
                'value': value
            })


class ExtractionTool(BrowserbaseTool):
    """
    Tool for data extraction from web pages.
    
    Provides functionality for extracting text, HTML, attributes, and structured
    data from web page elements.
    """
    
    name: str = "ExtractionTool"
    description: str = "Extract data from web page elements"
    
    class ExtractionInput(BrowserbaseToolInput):
        extraction_type: str = Field(..., description="Type of extraction: 'text', 'html', 'attributes'")
        selector: str = Field(..., description="CSS selector for the target element")
        attributes: Optional[List[str]] = Field(default=None, description="Attributes to extract (for attributes type)")
    
    args_schema: Type[BaseModel] = ExtractionInput
    
    async def _arun(self, extraction_type: str, selector: str, 
                    attributes: Optional[List[str]] = None, **kwargs) -> str:
        """
        Extract data from web page elements.
        
        Args:
            extraction_type: Type of extraction ('text', 'html', 'attributes')
            selector: CSS selector for the target element
            attributes: Attributes to extract (for attributes type)
            **kwargs: Additional arguments
            
        Returns:
            Extracted data as JSON string
        """
        start_time = time.time()
        
        try:
            # Apply anti-bot features
            await self._apply_anti_bot_features()
            await self._apply_human_like_behavior('extract')
            
            # Ensure session is available
            await self._ensure_session(kwargs.get('session_config'))
            
            # Perform the extraction
            if extraction_type == 'text':
                result = await self.browser_operations.extract_text(selector)
            elif extraction_type == 'html':
                result = await self.browser_operations.extract_html(selector)
            elif extraction_type == 'attributes':
                if not attributes:
                    raise ValueError("Attributes list is required for attributes extraction")
                result = await self.browser_operations.extract_attributes(selector, attributes)
            else:
                raise ValueError(f"Unsupported extraction type: {extraction_type}")
            
            duration = time.time() - start_time
            
            # Log operation statistics
            self._log_operation_stats("extraction", duration, True)
            
            return json.dumps({
                'status': 'success',
                'extraction_type': extraction_type,
                'selector': selector,
                'data': result.data,
                'duration': duration
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_operation_stats(f"extraction_{extraction_type}", duration, False)
            
            # Handle error and return proper JSON
            error_result = await self._handle_error(e, f"extraction_{extraction_type}")
            return json.dumps({
                'status': 'error',
                'error': error_result.error,
                'duration': duration,
                'timestamp': error_result.timestamp.isoformat(),
                'extraction_type': extraction_type,
                'selector': selector
            })


class ScreenshotTool(BrowserbaseTool):
    """
    Tool for capturing screenshots.
    
    Provides functionality for taking screenshots of web pages for
    verification, debugging, and documentation purposes.
    """
    
    name: str = "ScreenshotTool"
    description: str = "Capture screenshots of web pages"
    
    class ScreenshotInput(BrowserbaseToolInput):
        path: Optional[str] = Field(default=None, description="Path to save screenshot (optional)")
        full_page: bool = Field(default=False, description="Whether to capture full page")
    
    args_schema: Type[BaseModel] = ScreenshotInput
    
    async def _arun(self, path: Optional[str] = None, full_page: bool = False, **kwargs) -> str:
        """
        Capture a screenshot of the current page.
        
        Args:
            path: Path to save screenshot (optional)
            full_page: Whether to capture full page
            **kwargs: Additional arguments
            
        Returns:
            Screenshot result as JSON string
        """
        start_time = time.time()
        
        try:
            # Apply anti-bot features
            await self._apply_anti_bot_features()
            await self._apply_human_like_behavior('screenshot')
            
            # Ensure session is available
            await self._ensure_session(kwargs.get('session_config'))
            
            # Take screenshot
            result = await self.browser_operations.take_screenshot(path, full_page)
            
            duration = time.time() - start_time
            
            # Log operation statistics
            self._log_operation_stats("screenshot", duration, True)
            
            return json.dumps({
                'status': 'success',
                'path': result.data,
                'full_page': full_page,
                'duration': duration
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_operation_stats("screenshot", duration, False)
            error_result = await self._handle_error(e, "screenshot")
            return json.dumps({
                'status': 'error',
                'error': error_result.error,
                'duration': duration,
                'timestamp': error_result.timestamp.isoformat(),
                'path': path,
                'full_page': full_page
            })


class WaitingTool(BrowserbaseTool):
    """
    Tool for handling dynamic content loading and timing.
    
    Provides functionality for waiting for elements, navigation completion,
    and other timing-dependent operations.
    """
    
    name: str = "WaitingTool"
    description: str = "Wait for elements, navigation, or timing conditions"
    
    class WaitingInput(BrowserbaseToolInput):
        wait_type: str = Field(..., description="Type of wait: 'element', 'navigation', 'time'")
        selector: Optional[str] = Field(default=None, description="CSS selector for element wait")
        timeout: int = Field(default=10000, description="Timeout for the wait operation")
        duration: Optional[float] = Field(default=None, description="Duration for time wait (seconds)")
    
    args_schema: Type[BaseModel] = WaitingInput
    
    async def _arun(self, wait_type: str, selector: Optional[str] = None, 
                    timeout: int = 10000, duration: Optional[float] = None, **kwargs) -> str:
        """
        Wait for various conditions.
        
        Args:
            wait_type: Type of wait ('element', 'navigation', 'time')
            selector: CSS selector for element wait
            timeout: Timeout for the wait operation
            duration: Duration for time wait (seconds)
            **kwargs: Additional arguments
            
        Returns:
            Wait result as JSON string
        """
        start_time = time.time()
        
        try:
            # Apply anti-bot features
            await self._apply_anti_bot_features()
            await self._apply_human_like_behavior('wait')
            
            # Ensure session is available
            await self._ensure_session(kwargs.get('session_config'))
            
            # Perform the wait operation
            if wait_type == 'element':
                if not selector:
                    raise ValueError("Selector is required for element wait")
                result = await self.browser_operations.wait_for_element(selector, timeout)
            elif wait_type == 'navigation':
                result = await self.browser_operations.wait_for_navigation(timeout)
            elif wait_type == 'time':
                if not duration:
                    raise ValueError("Duration is required for time wait")
                await asyncio.sleep(duration)
                result = BrowserOperationResult(
                    success=True,
                    operation_type=BrowserOperationType.WAIT,
                    data=f"Waited for {duration} seconds"
                )
            else:
                raise ValueError(f"Unsupported wait type: {wait_type}")
            
            duration = time.time() - start_time
            
            # Log operation statistics
            self._log_operation_stats("wait", duration, True)
            
            return json.dumps({
                'status': 'success',
                'wait_type': wait_type,
                'selector': selector,
                'duration': duration,
                'data': result.data
            })
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_operation_stats("wait", duration, False)
            error_result = await self._handle_error(e, f"wait_{wait_type}")
            return json.dumps({
                'status': 'error',
                'error': error_result.error,
                'duration': duration,
                'timestamp': error_result.timestamp.isoformat(),
                'wait_type': wait_type,
                'selector': selector
            })


# Factory functions for creating tools
def create_navigation_tool(browser_manager: BrowserbaseManager, 
                          anti_bot_config: Optional[AntiBotConfig] = None) -> NavigationTool:
    """Create a NavigationTool instance."""
    return NavigationTool(browser_manager, anti_bot_config)


def create_interaction_tool(browser_manager: BrowserbaseManager,
                           anti_bot_config: Optional[AntiBotConfig] = None) -> InteractionTool:
    """Create an InteractionTool instance."""
    return InteractionTool(browser_manager, anti_bot_config)


def create_extraction_tool(browser_manager: BrowserbaseManager,
                          anti_bot_config: Optional[AntiBotConfig] = None) -> ExtractionTool:
    """Create an ExtractionTool instance."""
    return ExtractionTool(browser_manager, anti_bot_config)


def create_screenshot_tool(browser_manager: BrowserbaseManager,
                          anti_bot_config: Optional[AntiBotConfig] = None) -> ScreenshotTool:
    """Create a ScreenshotTool instance."""
    return ScreenshotTool(browser_manager, anti_bot_config)


def create_waiting_tool(browser_manager: BrowserbaseManager,
                       anti_bot_config: Optional[AntiBotConfig] = None) -> WaitingTool:
    """Create a WaitingTool instance."""
    return WaitingTool(browser_manager, anti_bot_config)


def create_all_browser_tools(browser_manager: BrowserbaseManager,
                            anti_bot_config: Optional[AntiBotConfig] = None) -> List[BrowserbaseTool]:
    """Create all browser tools with the same configuration."""
    return [
        create_navigation_tool(browser_manager, anti_bot_config),
        create_interaction_tool(browser_manager, anti_bot_config),
        create_extraction_tool(browser_manager, anti_bot_config),
        create_screenshot_tool(browser_manager, anti_bot_config),
        create_waiting_tool(browser_manager, anti_bot_config)
    ] 