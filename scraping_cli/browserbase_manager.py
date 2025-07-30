"""
Browserbase Integration Module

Provides Browserbase API integration for cloud browser session management
and web automation as the primary scraping tool.
"""

import os
import logging
import asyncio
import time
from typing import List, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager

from enum import Enum

from browserbase import Browserbase
from browserbase.types.session_create_params import BrowserSettings
from pydantic import TypeAdapter

from .exceptions import (
    SessionCreationError, SessionConnectionError, SessionTimeoutError,
    SessionHealthError, NetworkError, RetryExhaustedError, ConfigurationError,
    classify_error, is_retryable_error, get_error_severity
)
from .health_monitor import SessionHealthMonitor, HealthConfig, create_health_monitor


class SessionStatus(Enum):
    """Status of a browser session."""
    CREATING = "creating"
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class SessionConfig:
    """Configuration for Browserbase sessions."""
    user_agent: Optional[str] = None
    proxy_server: Optional[str] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    keep_alive: bool = True
    stealth_mode: bool = False
    captcha_image_selector: Optional[str] = None
    captcha_input_selector: Optional[str] = None
    context_id: Optional[str] = None
    context_persist: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    timeout: int = 30000  # milliseconds
    max_concurrent_sessions: int = 10
    session_ttl: int = 3600  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds


@dataclass
class SessionInfo:
    """Information about a browser session."""
    session_id: str
    connect_url: str
    status: SessionStatus
    created_at: datetime
    last_used: datetime
    config: SessionConfig
    error_count: int = 0
    last_error: Optional[str] = None


class SessionPool:
    """Advanced session pooling with configuration management."""
    
    def __init__(self, max_size: int = 10, min_size: int = 2, 
                 session_ttl: int = 3600, cleanup_interval: int = 300):
        """
        Initialize the session pool.
        
        Args:
            max_size: Maximum number of sessions in the pool
            min_size: Minimum number of sessions to maintain
            session_ttl: Time-to-live for sessions in seconds
            cleanup_interval: Interval for cleanup operations in seconds
        """
        self.max_size = max_size
        self.min_size = min_size
        self.session_ttl = session_ttl
        self.cleanup_interval = cleanup_interval
        
        self.available_sessions: List[SessionInfo] = []
        self.in_use_sessions: Dict[str, SessionInfo] = {}
        self.session_configs: Dict[str, SessionConfig] = {}
        
        self.stats = {
            'created': 0,
            'acquired': 0,
            'released': 0,
            'expired': 0,
            'errors': 0
        }
    
    def acquire(self, config: Optional[SessionConfig] = None) -> Optional[SessionInfo]:
        """
        Acquire a session from the pool.
        
        Args:
            config: Session configuration (optional)
            
        Returns:
            SessionInfo if available, None otherwise
        """
        # Try to get from available sessions
        if self.available_sessions:
            session_info = self.available_sessions.pop()
            
            # Check if session is still valid
            if self._is_session_valid(session_info):
                session_info.status = SessionStatus.ACTIVE
                session_info.last_used = datetime.now()
                self.in_use_sessions[session_info.session_id] = session_info
                self.stats['acquired'] += 1
                return session_info
            else:
                # Session expired, remove it
                self.stats['expired'] += 1
        
        # No available sessions, check if we can create more
        if len(self.in_use_sessions) < self.max_size:
            return None  # Let the manager create a new session
        
        return None
    
    def release(self, session_info: SessionInfo) -> None:
        """
        Release a session back to the pool.
        
        Args:
            session_info: Session to release
        """
        if session_info.session_id in self.in_use_sessions:
            del self.in_use_sessions[session_info.session_id]
            
            # Check if session should be kept in pool
            if (len(self.available_sessions) < self.max_size and 
                self._is_session_valid(session_info) and 
                session_info.error_count < 2):
                
                session_info.status = SessionStatus.IDLE
                self.available_sessions.append(session_info)
                self.stats['released'] += 1
            else:
                # Session is invalid or has too many errors
                self.stats['expired'] += 1
    
    def _is_session_valid(self, session_info: SessionInfo) -> bool:
        """Check if a session is still valid."""
        age = datetime.now() - session_info.created_at
        return age.total_seconds() < self.session_ttl
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions from the pool.
        
        Returns:
            Number of sessions cleaned up
        """
        expired_count = 0
        
        # Clean up available sessions
        valid_sessions = []
        for session_info in self.available_sessions:
            if self._is_session_valid(session_info):
                valid_sessions.append(session_info)
            else:
                expired_count += 1
        
        self.available_sessions = valid_sessions
        self.stats['expired'] += expired_count
        
        return expired_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self.stats,
            'available_sessions': len(self.available_sessions),
            'in_use_sessions': len(self.in_use_sessions),
            'total_sessions': len(self.available_sessions) + len(self.in_use_sessions),
            'pool_utilization': len(self.in_use_sessions) / self.max_size if self.max_size > 0 else 0
        }


class BrowserbaseManager:
    """Manages Browserbase cloud browser sessions for web automation."""
    
    def __init__(self, api_key: Optional[str] = None, project_id: Optional[str] = None, 
                 pool_size: int = 5, max_concurrent_sessions: Optional[int] = None,
                 max_retries: int = 3, session_timeout: int = 300,
                 health_config: Optional[HealthConfig] = None):
        """
        Initialize the Browserbase manager.
        
        Args:
            api_key: Browserbase API key (defaults to BROWSERBASE_API_KEY env var)
            project_id: Browserbase project ID (defaults to BROWSERBASE_PROJECT_ID env var)
            pool_size: Maximum number of sessions to keep in pool
            max_concurrent_sessions: Maximum total concurrent sessions (defaults to BROWSERBASE_MAX_SESSIONS env var or pool_size)
            max_retries: Maximum number of retries for failed operations
            session_timeout: Session timeout in seconds
            health_config: Configuration for health monitoring
        """
        self.api_key = api_key or os.getenv('BROWSERBASE_API_KEY')
        self.project_id = project_id or os.getenv('BROWSERBASE_PROJECT_ID')
        
        if not self.api_key:
            raise ConfigurationError("Browserbase API key is required. Set BROWSERBASE_API_KEY environment variable or pass api_key parameter.")
        
        if not self.project_id:
            raise ConfigurationError("Browserbase project ID is required. Set BROWSERBASE_PROJECT_ID environment variable or pass project_id parameter.")
        
        self.pool_size = pool_size
        
        # Configure max concurrent sessions
        if max_concurrent_sessions is None:
            # Try environment variable first, then fall back to pool_size
            env_max_sessions = os.getenv('BROWSERBASE_MAX_SESSIONS')
            if env_max_sessions:
                try:
                    self.max_concurrent_sessions = int(env_max_sessions)
                except ValueError:
                    # Use print instead of logger since logger isn't initialized yet
                    print(f"Warning: Invalid BROWSERBASE_MAX_SESSIONS value: {env_max_sessions}. Using pool_size.")
                    self.max_concurrent_sessions = pool_size
            else:
                self.max_concurrent_sessions = pool_size
        else:
            self.max_concurrent_sessions = max_concurrent_sessions
        
        # Validate max_concurrent_sessions
        if self.max_concurrent_sessions <= 0:
            raise ConfigurationError("max_concurrent_sessions must be greater than 0")
        
        self.max_retries = max_retries
        self.session_timeout = session_timeout
        
        # Initialize Browserbase client
        try:
            self.bb = Browserbase(api_key=self.api_key)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Browserbase client: {e}")
        
        # Session management
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.session_pool = SessionPool(
            max_size=pool_size,
            session_ttl=session_timeout,
            cleanup_interval=60
        )
        
        # Health monitoring
        self.health_monitor = create_health_monitor(health_config)
        self.health_monitor.add_recovery_callback(self._handle_session_recovery)
        
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'sessions_created': 0,
            'sessions_closed': 0,
            'errors': 0,
            'retries': 0,
            'recoveries': 0,
            'health_checks': 0
        }
    
    def _create_browser_settings(self, config: SessionConfig) -> BrowserSettings:
        """Create BrowserSettings from SessionConfig."""
        settings_dict = {}
        
        if config.user_agent:
            settings_dict['userAgent'] = config.user_agent
        
        if config.proxy_server:
            proxy_config = {'server': config.proxy_server}
            if config.proxy_username:
                proxy_config['username'] = config.proxy_username
            if config.proxy_password:
                proxy_config['password'] = config.proxy_password
            settings_dict['proxy'] = proxy_config
        
        if config.context_id:
            settings_dict['context'] = {
                'id': config.context_id,
                'persist': config.context_persist
            }
        
        if config.captcha_image_selector:
            settings_dict['captchaImageSelector'] = config.captcha_image_selector
        
        if config.captcha_input_selector:
            settings_dict['captchaInputSelector'] = config.captcha_input_selector
        
        if config.stealth_mode:
            settings_dict['stealth'] = True
        
        return TypeAdapter(BrowserSettings).validate_python(settings_dict)
    
    def _retry_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: Function to retry
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            RetryExhaustedError: If all retry attempts are exhausted
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.stats['errors'] += 1
                
                # Check if error is retryable
                if not is_retryable_error(e):
                    self.logger.error(f"Non-retryable error: {e}")
                    raise e
                
                # If this is the last attempt, raise RetryExhaustedError
                if attempt == self.max_retries:
                    self.logger.error(f"All retry attempts exhausted: {e}")
                    raise RetryExhaustedError(
                        f"Operation failed after {self.max_retries} retries",
                        max_retries=self.max_retries,
                        last_error=last_error
                    )
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = (2 ** attempt) * 1.0  # 1, 2, 4, 8 seconds
                    self.stats['retries'] += 1
                    self.logger.warning(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay}s: {e}")
                    time.sleep(delay)
    
    def _handle_session_recovery(self, session_id: str, session_info: Any, health_result: Any) -> None:
        """
        Handle session recovery when health check fails.
        
        Args:
            session_id: ID of the session to recover
            session_info: Session information
            health_result: Health check result
        """
        self.logger.info(f"Attempting to recover session {session_id}")
        self.stats['recoveries'] += 1
        
        try:
            # Close the problematic session
            if session_id in self.active_sessions:
                self.close_session(session_id)
            
            # Create a new session to replace it
            new_session = self.create_session(session_info.config if hasattr(session_info, 'config') else None)
            self.logger.info(f"Successfully recovered session {session_id} with new session {new_session.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to recover session {session_id}: {e}")
    
    def create_session(self, config: Optional[SessionConfig] = None) -> SessionInfo:
        """
        Create a new browser session with retry logic.
        
        Args:
            config: Session configuration (optional)
            
        Returns:
            SessionInfo object with session details
            
        Raises:
            SessionCreationError: If session creation fails after retries
        """
        config = config or SessionConfig()
        
        def _create_session_operation():
            """Inner function for session creation that can be retried."""
            self.logger.info(f"Creating new browser session with config: {config}")
            
            browser_settings = self._create_browser_settings(config)
            
            session = self.bb.sessions.create(
                project_id=self.project_id,
                browser_settings=browser_settings,
                keep_alive=config.keep_alive
            )
            
            session_info = SessionInfo(
                session_id=session.id,
                connect_url=session.connect_url,
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=config
            )
            
            self.active_sessions[session.id] = session_info
            self.stats['sessions_created'] += 1
            
            self.logger.info(f"Successfully created session {session.id}")
            return session_info
        
        try:
            return self._retry_operation(_create_session_operation)
        except RetryExhaustedError as e:
            raise SessionCreationError(
                f"Failed to create session after {self.max_retries} retries",
                details={'config': str(config), 'last_error': str(e.last_error)}
            )
        except Exception as e:
            raise SessionCreationError(f"Session creation failed: {e}")
    
    def get_session(self, config: Optional[SessionConfig] = None) -> SessionInfo:
        """
        Get a session from the pool or create a new one.
        
        Args:
            config: Session configuration (optional)
            
        Returns:
            SessionInfo object
            
        Raises:
            SessionCreationError: If max concurrent sessions limit is reached
        """
        # Check if we've reached the concurrent session limit
        total_active_sessions = len(self.active_sessions) + len(self.session_pool.in_use_sessions)
        if total_active_sessions >= self.max_concurrent_sessions:
            raise SessionCreationError(
                f"Maximum concurrent sessions limit reached ({self.max_concurrent_sessions}). "
                f"Currently have {total_active_sessions} active sessions. "
                "Please wait for some sessions to be released or increase the limit."
            )
        
        # Try to get from pool first
        session_info = self.session_pool.acquire(config)
        if session_info:
            self.active_sessions[session_info.session_id] = session_info
            self.logger.info(f"Reusing session {session_info.session_id} from pool")
            return session_info
        
        # Create new session if pool is empty
        return self.create_session(config)
    
    def release_session(self, session_info: SessionInfo) -> None:
        """
        Release a session back to the pool or close it.
        
        Args:
            session_info: Session to release
        """
        if session_info.session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_info.session_id} not found in active sessions")
            return
        
        # Remove from active sessions
        del self.active_sessions[session_info.session_id]
        
        # Add to pool's available sessions if it's not already there
        if session_info.session_id not in self.session_pool.in_use_sessions:
            # Session wasn't in pool, add it directly to available sessions
            if (len(self.session_pool.available_sessions) < self.session_pool.max_size and 
                self.session_pool._is_session_valid(session_info) and 
                session_info.error_count < 2):
                
                session_info.status = SessionStatus.IDLE
                self.session_pool.available_sessions.append(session_info)
                self.session_pool.stats['released'] += 1
            else:
                # Session is invalid or has too many errors
                self.session_pool.stats['expired'] += 1
        else:
            # Release to pool normally
            self.session_pool.release(session_info)
        
        self.logger.info(f"Released session {session_info.session_id} to pool")
    
    def close_session(self, session_id: str) -> None:
        """
        Close a specific session.
        
        Args:
            session_id: ID of the session to close
        """
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            try:
                # Note: Browserbase sessions are automatically closed when not in use
                # This is more for cleanup of our tracking
                session_info.status = SessionStatus.CLOSED
                del self.active_sessions[session_id]
                self.stats['sessions_closed'] += 1
                self.logger.info(f"Closed session {session_id}")
            except Exception as e:
                self.logger.error(f"Error closing session {session_id}: {e}")
        
        # Remove from pool if present
        self.session_pool.available_sessions = [s for s in self.session_pool.available_sessions if s.session_id != session_id]
        if session_id in self.session_pool.in_use_sessions:
            del self.session_pool.in_use_sessions[session_id]
    
    def close_all_sessions(self) -> None:
        """Close all active sessions and clear the pool."""
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            try:
                self.close_session(session_id)
            except Exception as e:
                self.logger.warning(f"Failed to close session {session_id}: {e}")
        
        # Clear the session pool
        self.session_pool.available_sessions.clear()
        self.session_pool.in_use_sessions.clear()
        
        # Also close any sessions that might be running on Browserbase
        try:
            sessions = self.bb.sessions.list()
            for session in sessions:
                if session.status == 'running':
                    try:
                        self.bb.sessions.delete(session.id)
                        self.logger.info(f"Closed orphaned session {session.id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to close orphaned session {session.id}: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to list sessions for cleanup: {e}")
        
        self.logger.info("Closed all sessions and cleared pool")
    
    def get_session_limits(self) -> Dict[str, Any]:
        """
        Get information about session limits and current usage.
        
        Returns:
            Dictionary with limit information
        """
        total_active_sessions = len(self.active_sessions) + len(self.session_pool.in_use_sessions)
        available_sessions = len(self.session_pool.available_sessions)
        
        return {
            'max_concurrent_sessions': self.max_concurrent_sessions,
            'current_active_sessions': total_active_sessions,
            'pool_available_sessions': available_sessions,
            'pool_in_use_sessions': len(self.session_pool.in_use_sessions),
            'manager_active_sessions': len(self.active_sessions),
            'sessions_remaining': max(0, self.max_concurrent_sessions - total_active_sessions),
            'utilization_percentage': (total_active_sessions / self.max_concurrent_sessions) * 100 if self.max_concurrent_sessions > 0 else 0
        }
    
    def get_session_health(self, session_id: str) -> Dict[str, Any]:
        """
        Get health information for a session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Dictionary with health information
        """
        if session_id not in self.active_sessions:
            return {'status': 'not_found'}
        
        session_info = self.active_sessions[session_id]
        age = datetime.now() - session_info.created_at
        idle_time = datetime.now() - session_info.last_used
        
        return {
            'status': session_info.status.value,
            'age_seconds': age.total_seconds(),
            'idle_seconds': idle_time.total_seconds(),
            'error_count': session_info.error_count,
            'last_error': session_info.last_error,
            'config': {
                'user_agent': session_info.config.user_agent,
                'proxy_server': session_info.config.proxy_server,
                'stealth_mode': session_info.config.stealth_mode
            }
        }
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up sessions that have exceeded the timeout.
        
        Returns:
            Number of sessions cleaned up
        """
        # Clean up active sessions
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session_info in self.active_sessions.items():
            if (now - session_info.last_used).total_seconds() > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.close_session(session_id)
        
        # Clean up pool sessions
        pool_cleaned = self.session_pool.cleanup_expired_sessions()
        
        total_cleaned = len(expired_sessions) + pool_cleaned
        if total_cleaned > 0:
            self.logger.info(f"Cleaned up {total_cleaned} expired sessions ({len(expired_sessions)} active, {pool_cleaned} pool)")
        
        return total_cleaned
    
    async def check_session_health(self, session_id: str) -> Dict[str, Any]:
        """
        Check the health of a specific session.
        
        Args:
            session_id: ID of the session to check
            
        Returns:
            Health check result
        """
        if session_id not in self.active_sessions:
            return {'status': 'not_found', 'error': 'Session not found'}
        
        session_info = self.active_sessions[session_id]
        self.stats['health_checks'] += 1
        
        try:
            result = await self.health_monitor.check_session_health(session_id, session_info)
            return {
                'status': result.status.value,
                'response_time': result.response_time,
                'details': result.details,
                'timestamp': result.timestamp.isoformat()
            }
        except Exception as e:
            self.logger.error(f"Health check failed for session {session_id}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def monitor_all_sessions(self) -> Dict[str, Any]:
        """
        Monitor health of all active sessions.
        
        Returns:
            Health monitoring results
        """
        try:
            results = await self.health_monitor.monitor_sessions(self.active_sessions)
            return {
                session_id: {
                    'status': result.status.value,
                    'response_time': result.response_time,
                    'details': result.details,
                    'timestamp': result.timestamp.isoformat()
                }
                for session_id, result in results.items()
            }
        except Exception as e:
            self.logger.error(f"Session monitoring failed: {e}")
            return {'error': str(e)}
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get overall health summary.
        
        Returns:
            Health summary information
        """
        try:
            overall_summary = self.health_monitor.get_overall_health_summary()
            return {
                **overall_summary,
                'health_checks_performed': self.stats['health_checks'],
                'recoveries_attempted': self.stats['recoveries']
            }
        except Exception as e:
            self.logger.error(f"Failed to get health summary: {e}")
            return {'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        pool_stats = self.session_pool.get_stats()
        session_limits = self.get_session_limits()
        return {
            **self.stats,
            'active_sessions': len(self.active_sessions),
            'pool_stats': pool_stats,
            'total_sessions': len(self.active_sessions) + pool_stats['total_sessions'],
            'session_limits': session_limits
        }
    
    @contextmanager
    def session_context(self, config: Optional[SessionConfig] = None):
        """
        Context manager for automatic session management.
        
        Args:
            config: Session configuration (optional)
            
        Yields:
            SessionInfo object
        """
        session_info = self.get_session(config)
        try:
            yield session_info
        finally:
            self.release_session(session_info)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup all sessions."""
        self.close_all_sessions()


# Factory function for creating BrowserbaseManager instances
def create_browserbase_manager(api_key: Optional[str] = None, project_id: Optional[str] = None,
                             pool_size: int = 5, max_concurrent_sessions: Optional[int] = None,
                             max_retries: int = 3, session_timeout: int = 300) -> BrowserbaseManager:
    """
    Create and return a new BrowserbaseManager instance.
    
    Args:
        api_key: Browserbase API key
        project_id: Browserbase project ID
        pool_size: Maximum number of sessions in pool
        max_retries: Maximum retries for failed operations
        session_timeout: Session timeout in seconds
        
    Returns:
        BrowserbaseManager instance
    """
    return BrowserbaseManager(
        api_key=api_key,
        project_id=project_id,
        pool_size=pool_size,
        max_concurrent_sessions=max_concurrent_sessions,
        max_retries=max_retries,
        session_timeout=session_timeout
    ) 