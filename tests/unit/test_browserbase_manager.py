"""
Unit tests for BrowserbaseManager.

Tests session management, pooling, error handling, and configuration
with mocked Browserbase SDK calls.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

from scraping_cli.browserbase_manager import (
    BrowserbaseManager, SessionPool, SessionConfig, SessionInfo, SessionStatus,
    create_browserbase_manager
)
from scraping_cli.exceptions import (
    SessionCreationError, SessionConnectionError, ConfigurationError,
    RetryExhaustedError, NetworkError
)


class TestSessionConfig:
    """Test SessionConfig dataclass."""
    
    def test_default_config(self):
        """Test SessionConfig with default values."""
        config = SessionConfig()
        assert config.user_agent is None
        assert config.proxy_server is None
        assert config.keep_alive is True
        assert config.stealth_mode is False
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080
        assert config.timeout == 30000
        assert config.max_concurrent_sessions == 10
        assert config.session_ttl == 3600
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
    
    def test_custom_config(self):
        """Test SessionConfig with custom values."""
        config = SessionConfig(
            user_agent="Custom Agent",
            proxy_server="proxy.example.com:8080",
            stealth_mode=True,
            viewport_width=1366,
            viewport_height=768,
            timeout=60000,
            max_concurrent_sessions=5,
            session_ttl=1800,
            retry_attempts=5,
            retry_delay=2.0
        )
        assert config.user_agent == "Custom Agent"
        assert config.proxy_server == "proxy.example.com:8080"
        assert config.stealth_mode is True
        assert config.viewport_width == 1366
        assert config.viewport_height == 768
        assert config.timeout == 60000
        assert config.max_concurrent_sessions == 5
        assert config.session_ttl == 1800
        assert config.retry_attempts == 5
        assert config.retry_delay == 2.0


class TestSessionPool:
    """Test SessionPool functionality."""
    
    @pytest.fixture
    def session_pool(self):
        """Create a SessionPool instance for testing."""
        return SessionPool(max_size=3, min_size=1, session_ttl=3600)
    
    @pytest.fixture
    def sample_session_info(self):
        """Create a sample SessionInfo for testing."""
        config = SessionConfig(user_agent="Test Agent")
        return SessionInfo(
            session_id="test-session-123",
            connect_url="wss://test.browserbase.com/session/123",
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(),
            last_used=datetime.now(),
            config=config
        )
    
    def test_pool_initialization(self, session_pool):
        """Test SessionPool initialization."""
        assert session_pool.max_size == 3
        assert session_pool.min_size == 1
        assert session_pool.session_ttl == 3600
        assert session_pool.cleanup_interval == 300
        assert len(session_pool.available_sessions) == 0
        assert len(session_pool.in_use_sessions) == 0
    
    def test_acquire_from_empty_pool(self, session_pool):
        """Test acquiring from empty pool returns None."""
        session_info = session_pool.acquire()
        assert session_info is None
    
    def test_acquire_and_release_session(self, session_pool, sample_session_info):
        """Test acquiring and releasing a session."""
        # Add session to available pool
        session_pool.available_sessions.append(sample_session_info)
        
        # Acquire session
        acquired = session_pool.acquire()
        assert acquired is not None
        assert acquired.session_id == "test-session-123"
        assert acquired.status == SessionStatus.ACTIVE
        assert acquired.session_id in session_pool.in_use_sessions
        
        # Release session
        session_pool.release(acquired)
        assert acquired.session_id not in session_pool.in_use_sessions
        assert len(session_pool.available_sessions) == 1
    
    def test_acquire_expired_session(self, session_pool):
        """Test that expired sessions are not acquired."""
        # Create an expired session
        expired_config = SessionConfig()
        expired_session = SessionInfo(
            session_id="expired-session",
            connect_url="wss://test.browserbase.com/session/expired",
            status=SessionStatus.IDLE,
            created_at=datetime.now() - timedelta(hours=2),  # 2 hours old
            last_used=datetime.now() - timedelta(hours=1),
            config=expired_config
        )
        session_pool.available_sessions.append(expired_session)
        
        # Try to acquire expired session
        acquired = session_pool.acquire()
        assert acquired is None
        assert len(session_pool.available_sessions) == 0  # Expired session removed
    
    def test_pool_size_limit(self, session_pool):
        """Test that pool respects size limits."""
        # Fill the pool
        for i in range(3):
            config = SessionConfig()
            session_info = SessionInfo(
                session_id=f"session-{i}",
                connect_url=f"wss://test.browserbase.com/session/{i}",
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=config
            )
            session_pool.in_use_sessions[session_info.session_id] = session_info
        
        # Try to acquire when pool is full
        acquired = session_pool.acquire()
        assert acquired is None
    
    def test_cleanup_expired_sessions(self, session_pool):
        """Test cleanup of expired sessions."""
        # Add some valid and expired sessions
        valid_config = SessionConfig()
        valid_session = SessionInfo(
            session_id="valid-session",
            connect_url="wss://test.browserbase.com/session/valid",
            status=SessionStatus.IDLE,
            created_at=datetime.now(),
            last_used=datetime.now(),
            config=valid_config
        )
        
        expired_config = SessionConfig()
        expired_session = SessionInfo(
            session_id="expired-session",
            connect_url="wss://test.browserbase.com/session/expired",
            status=SessionStatus.IDLE,
            created_at=datetime.now() - timedelta(hours=2),
            last_used=datetime.now() - timedelta(hours=1),
            config=expired_config
        )
        
        session_pool.available_sessions.extend([valid_session, expired_session])
        
        # Clean up expired sessions
        cleaned_count = session_pool.cleanup_expired_sessions()
        assert cleaned_count == 1
        assert len(session_pool.available_sessions) == 1
        assert session_pool.available_sessions[0].session_id == "valid-session"
    
    def test_pool_stats(self, session_pool, sample_session_info):
        """Test pool statistics tracking."""
        # Add session to pool
        session_pool.available_sessions.append(sample_session_info)
        
        # Acquire session
        acquired = session_pool.acquire()
        session_pool.release(acquired)
        
        stats = session_pool.get_stats()
        assert stats['created'] == 0  # No sessions created by pool
        assert stats['acquired'] == 1
        assert stats['released'] == 1
        assert stats['available_sessions'] == 1
        assert stats['in_use_sessions'] == 0
        assert stats['total_sessions'] == 1
        assert stats['pool_utilization'] == 0.0


class TestBrowserbaseManager:
    """Test BrowserbaseManager functionality."""
    
    @pytest.fixture
    def mock_browserbase(self):
        """Mock Browserbase SDK."""
        with patch('scraping_cli.browserbase_manager.Browserbase') as mock_bb:
            mock_bb.return_value = Mock()
            yield mock_bb
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables."""
        with patch.dict(os.environ, {
            'BROWSERBASE_API_KEY': 'test-api-key',
            'BROWSERBASE_PROJECT_ID': 'test-project-id'
        }):
            yield
    
    @pytest.fixture
    def manager(self, mock_browserbase, mock_env_vars):
        """Create a BrowserbaseManager instance for testing."""
        return BrowserbaseManager(pool_size=2, max_retries=2)
    
    def test_manager_initialization(self, mock_browserbase, mock_env_vars):
        """Test BrowserbaseManager initialization."""
        manager = BrowserbaseManager(pool_size=5, max_retries=3)
        
        assert manager.api_key == 'test-api-key'
        assert manager.project_id == 'test-project-id'
        assert manager.pool_size == 5
        assert manager.max_retries == 3
        assert manager.session_timeout == 300
        assert len(manager.active_sessions) == 0
        assert manager.session_pool.max_size == 5
    
    def test_manager_initialization_missing_api_key(self, mock_browserbase):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase API key is required"):
                BrowserbaseManager()
    
    def test_manager_initialization_missing_project_id(self, mock_browserbase):
        """Test initialization fails without project ID."""
        with patch.dict(os.environ, {'BROWSERBASE_API_KEY': 'test-key'}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase project ID is required"):
                BrowserbaseManager()
    
    def test_create_browser_settings(self, manager):
        """Test browser settings creation."""
        config = SessionConfig(
            user_agent="Custom Agent",
            proxy_server="proxy.example.com:8080",
            proxy_username="user",
            proxy_password="pass",
            stealth_mode=True,
            context_id="test-context",
            context_persist=True,
            captcha_image_selector="#captcha-img",
            captcha_input_selector="#captcha-input"
        )
        
        settings = manager._create_browser_settings(config)
        
        # Verify settings were created (actual validation depends on BrowserSettings)
        assert settings is not None
    
    def test_create_session_success(self, manager, mock_browserbase):
        """Test successful session creation."""
        # Mock the session creation
        mock_session = Mock()
        mock_session.id = "test-session-123"
        mock_session.connect_url = "wss://test.browserbase.com/session/123"
        
        manager.bb.sessions.create.return_value = mock_session
        
        # Create session
        session_info = manager.create_session()
        
        assert session_info.session_id == "test-session-123"
        assert session_info.connect_url == "wss://test.browserbase.com/session/123"
        assert session_info.status == SessionStatus.ACTIVE
        assert session_info.session_id in manager.active_sessions
        assert manager.stats['sessions_created'] == 1
    
    def test_create_session_with_config(self, manager, mock_browserbase):
        """Test session creation with custom config."""
        mock_session = Mock()
        mock_session.id = "test-session-456"
        mock_session.connect_url = "wss://test.browserbase.com/session/456"
        
        manager.bb.sessions.create.return_value = mock_session
        
        config = SessionConfig(
            user_agent="Custom Agent",
            stealth_mode=True
        )
        
        session_info = manager.create_session(config)
        
        assert session_info.config.user_agent == "Custom Agent"
        assert session_info.config.stealth_mode is True
        
        # Verify browser settings were created with config
        manager.bb.sessions.create.assert_called_once()
        call_args = manager.bb.sessions.create.call_args
        assert call_args[1]['project_id'] == 'test-project-id'
        assert call_args[1]['keep_alive'] is True
    
    def test_create_session_failure(self, manager, mock_browserbase):
        """Test session creation failure."""
        # Mock session creation to fail
        manager.bb.sessions.create.side_effect = Exception("API Error")
        
        with pytest.raises(SessionCreationError, match="Session creation failed"):
            manager.create_session()
    
    def test_create_session_retry_exhausted(self, manager, mock_browserbase):
        """Test session creation with retry exhaustion."""
        # Mock session creation to fail consistently
        manager.bb.sessions.create.side_effect = Exception("Network Error")
        
        with pytest.raises(SessionCreationError):
            manager.create_session()
        
        # Verify retry attempts
        assert manager.stats['retries'] == 2  # max_retries = 2
    
    def test_get_session_from_pool(self, manager):
        """Test getting session from pool."""
        # Add session to pool
        config = SessionConfig()
        session_info = SessionInfo(
            session_id="pool-session",
            connect_url="wss://test.browserbase.com/session/pool",
            status=SessionStatus.IDLE,
            created_at=datetime.now(),
            last_used=datetime.now(),
            config=config
        )
        manager.session_pool.available_sessions.append(session_info)
        
        # Get session from pool
        acquired = manager.get_session()
        
        assert acquired.session_id == "pool-session"
        assert acquired.session_id in manager.active_sessions
    
    def test_get_session_create_new(self, manager, mock_browserbase):
        """Test getting session creates new one when pool is empty."""
        mock_session = Mock()
        mock_session.id = "new-session"
        mock_session.connect_url = "wss://test.browserbase.com/session/new"
        
        manager.bb.sessions.create.return_value = mock_session
        
        # Get session (pool is empty)
        session_info = manager.get_session()
        
        assert session_info.session_id == "new-session"
        assert session_info.session_id in manager.active_sessions
    
    def test_release_session(self, manager):
        """Test releasing session back to pool."""
        # Create a session
        config = SessionConfig()
        session_info = SessionInfo(
            session_id="test-session",
            connect_url="wss://test.browserbase.com/session/test",
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(),
            last_used=datetime.now(),
            config=config
        )
        manager.active_sessions[session_info.session_id] = session_info
        
        # Release session
        manager.release_session(session_info)
        
        # Verify session is in pool
        assert session_info.session_id not in manager.active_sessions
        assert len(manager.session_pool.available_sessions) == 1
    
    def test_close_session(self, manager):
        """Test closing a specific session."""
        # Create a session
        config = SessionConfig()
        session_info = SessionInfo(
            session_id="test-session",
            connect_url="wss://test.browserbase.com/session/test",
            status=SessionStatus.ACTIVE,
            created_at=datetime.now(),
            last_used=datetime.now(),
            config=config
        )
        manager.active_sessions[session_info.session_id] = session_info
        
        # Close session
        manager.close_session("test-session")
        
        assert "test-session" not in manager.active_sessions
        assert manager.stats['sessions_closed'] == 1
    
    def test_close_all_sessions(self, manager):
        """Test closing all sessions."""
        # Create multiple sessions
        for i in range(3):
            config = SessionConfig()
            session_info = SessionInfo(
                session_id=f"session-{i}",
                connect_url=f"wss://test.browserbase.com/session/{i}",
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=config
            )
            manager.active_sessions[session_info.session_id] = session_info
        
        # Add sessions to pool
        pool_config = SessionConfig()
        pool_session = SessionInfo(
            session_id="pool-session",
            connect_url="wss://test.browserbase.com/session/pool",
            status=SessionStatus.IDLE,
            created_at=datetime.now(),
            last_used=datetime.now(),
            config=pool_config
        )
        manager.session_pool.available_sessions.append(pool_session)
        
        # Close all sessions
        manager.close_all_sessions()
        
        assert len(manager.active_sessions) == 0
        assert len(manager.session_pool.available_sessions) == 0
        assert len(manager.session_pool.in_use_sessions) == 0
    
    def test_session_context_manager(self, manager, mock_browserbase):
        """Test session context manager."""
        mock_session = Mock()
        mock_session.id = "context-session"
        mock_session.connect_url = "wss://test.browserbase.com/session/context"
        
        manager.bb.sessions.create.return_value = mock_session
        
        with manager.session_context() as session_info:
            assert session_info.session_id == "context-session"
            assert session_info.session_id in manager.active_sessions
        
        # Session should be released after context
        assert session_info.session_id not in manager.active_sessions
    
    def test_manager_context_manager(self, manager):
        """Test BrowserbaseManager as context manager."""
        # Create some sessions
        for i in range(2):
            config = SessionConfig()
            session_info = SessionInfo(
                session_id=f"session-{i}",
                connect_url=f"wss://test.browserbase.com/session/{i}",
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=config
            )
            manager.active_sessions[session_info.session_id] = session_info
        
        with manager:
            assert len(manager.active_sessions) == 2
        
        # All sessions should be closed after context
        assert len(manager.active_sessions) == 0
    
    def test_get_session_health(self, manager):
        """Test getting session health information."""
        # Create a session
        config = SessionConfig(user_agent="Test Agent")
        session_info = SessionInfo(
            session_id="test-session",
            connect_url="wss://test.browserbase.com/session/test",
            status=SessionStatus.ACTIVE,
            created_at=datetime.now() - timedelta(minutes=5),
            last_used=datetime.now() - timedelta(minutes=2),
            config=config,
            error_count=1,
            last_error="Test error"
        )
        manager.active_sessions[session_info.session_id] = session_info
        
        health = manager.get_session_health("test-session")
        
        assert health['status'] == 'active'
        assert health['error_count'] == 1
        assert health['last_error'] == "Test error"
        assert health['config']['user_agent'] == "Test Agent"
        assert health['age_seconds'] > 0
        assert health['idle_seconds'] > 0
    
    def test_get_session_health_not_found(self, manager):
        """Test getting health for non-existent session."""
        health = manager.get_session_health("non-existent")
        assert health['status'] == 'not_found'
    
    def test_cleanup_expired_sessions(self, manager):
        """Test cleanup of expired sessions."""
        # Create an expired session
        config = SessionConfig()
        expired_session = SessionInfo(
            session_id="expired-session",
            connect_url="wss://test.browserbase.com/session/expired",
            status=SessionStatus.ACTIVE,
            created_at=datetime.now() - timedelta(hours=2),
            last_used=datetime.now() - timedelta(hours=1),
            config=config
        )
        manager.active_sessions[expired_session.session_id] = expired_session
        
        # Clean up expired sessions
        cleaned_count = manager.cleanup_expired_sessions()
        
        assert cleaned_count > 0
        assert "expired-session" not in manager.active_sessions
    
    def test_get_stats(self, manager):
        """Test getting manager statistics."""
        stats = manager.get_stats()
        
        expected_keys = [
            'sessions_created', 'sessions_closed', 'errors', 'retries',
            'recoveries', 'health_checks', 'active_sessions', 'pool_stats',
            'total_sessions'
        ]
        
        for key in expected_keys:
            assert key in stats
    
    def test_retry_operation_success(self, manager):
        """Test retry operation with eventual success."""
        operation = Mock()
        operation.side_effect = [Exception("Temporary error"), "Success"]
        
        result = manager._retry_operation(operation, "arg1", kwarg1="value1")
        
        assert result == "Success"
        assert operation.call_count == 2
        assert manager.stats['retries'] == 1
    
    def test_retry_operation_exhausted(self, manager):
        """Test retry operation with exhausted retries."""
        operation = Mock()
        operation.side_effect = NetworkError("Connection timeout")
        
        with pytest.raises(RetryExhaustedError):
            manager._retry_operation(operation)
        
        assert operation.call_count == 3  # max_retries + 1
        assert manager.stats['retries'] == 2


class TestCreateBrowserbaseManager:
    """Test factory function for creating BrowserbaseManager."""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables."""
        with patch.dict(os.environ, {
            'BROWSERBASE_API_KEY': 'test-api-key',
            'BROWSERBASE_PROJECT_ID': 'test-project-id'
        }):
            yield
    
    def test_create_browserbase_manager(self, mock_env_vars):
        """Test factory function."""
        with patch('scraping_cli.browserbase_manager.BrowserbaseManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            manager = create_browserbase_manager(
                api_key="custom-key",
                project_id="custom-project",
                pool_size=10,
                max_retries=5,
                session_timeout=600
            )
            
            mock_manager_class.assert_called_once_with(
                api_key="custom-key",
                project_id="custom-project",
                pool_size=10,
                max_retries=5,
                session_timeout=600
            )
            assert manager == mock_manager
    
    def test_create_browserbase_manager_defaults(self, mock_env_vars):
        """Test factory function with default parameters."""
        with patch('scraping_cli.browserbase_manager.BrowserbaseManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager_class.return_value = mock_manager
            
            manager = create_browserbase_manager()
            
            mock_manager_class.assert_called_once_with(
                api_key=None,
                project_id=None,
                pool_size=5,
                max_retries=3,
                session_timeout=300
            )
            assert manager == mock_manager 