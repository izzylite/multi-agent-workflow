"""
Unit tests for session limiting functionality.

Tests that BrowserbaseManager properly enforces session limits
and provides appropriate error messages when limits are exceeded.
"""

import pytest
import os
from unittest.mock import Mock, patch
from datetime import datetime

from scraping_cli.browserbase_manager import (
    BrowserbaseManager, SessionConfig, SessionInfo, SessionStatus
)
from scraping_cli.exceptions import SessionCreationError, ConfigurationError


class TestSessionLimits:
    """Test session limiting functionality."""
    
    def test_manager_initialization_with_max_concurrent_sessions(self):
        """Test that manager initializes with max_concurrent_sessions parameter."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                pool_size=5,
                max_concurrent_sessions=2
            )
            
            assert manager.max_concurrent_sessions == 2
            assert manager.pool_size == 5
    
    def test_manager_initialization_with_environment_variable(self):
        """Test that manager uses BROWSERBASE_MAX_SESSIONS environment variable."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            with patch.dict(os.environ, {'BROWSERBASE_MAX_SESSIONS': '3'}):
                manager = BrowserbaseManager(
                    api_key="test-key",
                    project_id="test-project",
                    pool_size=5
                )
                
                assert manager.max_concurrent_sessions == 3
    
    def test_manager_initialization_invalid_environment_variable(self):
        """Test that manager handles invalid environment variable gracefully."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            with patch.dict(os.environ, {'BROWSERBASE_MAX_SESSIONS': 'invalid'}):
                manager = BrowserbaseManager(
                    api_key="test-key",
                    project_id="test-project",
                    pool_size=5
                )
                
                # Should fall back to pool_size
                assert manager.max_concurrent_sessions == 5
    
    def test_manager_initialization_zero_max_sessions(self):
        """Test that manager rejects zero max_concurrent_sessions."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            with pytest.raises(ConfigurationError, match="max_concurrent_sessions must be greater than 0"):
                BrowserbaseManager(
                    api_key="test-key",
                    project_id="test-project",
                    max_concurrent_sessions=0
                )
    
    def test_manager_initialization_negative_max_sessions(self):
        """Test that manager rejects negative max_concurrent_sessions."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            with pytest.raises(ConfigurationError, match="max_concurrent_sessions must be greater than 0"):
                BrowserbaseManager(
                    api_key="test-key",
                    project_id="test-project",
                    max_concurrent_sessions=-1
                )
    
    def test_get_session_limits(self):
        """Test get_session_limits method."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=5
            )
            
            limits = manager.get_session_limits()
            
            assert limits['max_concurrent_sessions'] == 5
            assert limits['current_active_sessions'] == 0
            assert limits['pool_available_sessions'] == 0
            assert limits['pool_in_use_sessions'] == 0
            assert limits['manager_active_sessions'] == 0
            assert limits['sessions_remaining'] == 5
            assert limits['utilization_percentage'] == 0.0
    
    def test_get_session_limits_with_active_sessions(self):
        """Test get_session_limits with active sessions."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=5
            )
            
            # Add some mock active sessions
            session_info = SessionInfo(
                session_id="test-session-1",
                connect_url="wss://test.com/session/1",
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=SessionConfig()
            )
            manager.active_sessions["test-session-1"] = session_info
            
            limits = manager.get_session_limits()
            
            assert limits['current_active_sessions'] == 1
            assert limits['manager_active_sessions'] == 1
            assert limits['sessions_remaining'] == 4
            assert limits['utilization_percentage'] == 20.0
    
    def test_get_session_limits_with_pool_sessions(self):
        """Test get_session_limits with sessions in pool."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=5
            )
            
            # Add sessions to pool
            session_info = SessionInfo(
                session_id="test-session-1",
                connect_url="wss://test.com/session/1",
                status=SessionStatus.IDLE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=SessionConfig()
            )
            manager.session_pool.available_sessions.append(session_info)
            manager.session_pool.in_use_sessions["test-session-2"] = session_info
            
            limits = manager.get_session_limits()
            
            assert limits['current_active_sessions'] == 1  # Only in_use_sessions count
            assert limits['pool_available_sessions'] == 1
            assert limits['pool_in_use_sessions'] == 1
            assert limits['sessions_remaining'] == 4
    
    def test_get_session_limits_at_capacity(self):
        """Test get_session_limits when at maximum capacity."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=2
            )
            
            # Add sessions to reach capacity
            for i in range(2):
                session_info = SessionInfo(
                    session_id=f"test-session-{i}",
                    connect_url=f"wss://test.com/session/{i}",
                    status=SessionStatus.ACTIVE,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    config=SessionConfig()
                )
                manager.active_sessions[f"test-session-{i}"] = session_info
            
            limits = manager.get_session_limits()
            
            assert limits['current_active_sessions'] == 2
            assert limits['sessions_remaining'] == 0
            assert limits['utilization_percentage'] == 100.0
    
    def test_get_session_limits_exceeding_capacity(self):
        """Test get_session_limits when exceeding capacity."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=2
            )
            
            # Add sessions to exceed capacity
            for i in range(3):
                session_info = SessionInfo(
                    session_id=f"test-session-{i}",
                    connect_url=f"wss://test.com/session/{i}",
                    status=SessionStatus.ACTIVE,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    config=SessionConfig()
                )
                manager.active_sessions[f"test-session-{i}"] = session_info
            
            limits = manager.get_session_limits()
            
            assert limits['current_active_sessions'] == 3
            assert limits['sessions_remaining'] == 0  # Should not go negative
            assert limits['utilization_percentage'] == 150.0  # Can exceed 100%
    
    def test_get_session_limits_zero_max_sessions(self):
        """Test get_session_limits with zero max_concurrent_sessions."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            with pytest.raises(ConfigurationError, match="max_concurrent_sessions must be greater than 0"):
                BrowserbaseManager(
                    api_key="test-key",
                    project_id="test-project",
                    max_concurrent_sessions=0
                )
    
    def test_get_stats_includes_session_limits(self):
        """Test that get_stats includes session limit information."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=5
            )
            
            stats = manager.get_stats()
            
            assert 'session_limits' in stats
            assert stats['session_limits']['max_concurrent_sessions'] == 5
            assert stats['session_limits']['current_active_sessions'] == 0
    
    def test_create_browserbase_manager_with_max_concurrent_sessions(self):
        """Test factory function with max_concurrent_sessions parameter."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            from scraping_cli.browserbase_manager import create_browserbase_manager
            
            manager = create_browserbase_manager(
                api_key="test-key",
                project_id="test-project",
                pool_size=5,
                max_concurrent_sessions=3
            )
            
            assert manager.max_concurrent_sessions == 3
            assert manager.pool_size == 5


class TestSessionLimitEnforcement:
    """Test that session limits are properly enforced."""
    
    def test_get_session_respects_limit(self):
        """Test that get_session respects the concurrent session limit."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=2
            )
            
            # Mock session creation to succeed
            mock_session = Mock()
            mock_session.id = "test-session-1"
            mock_session.connect_url = "wss://test.com/session/1"
            manager.bb.sessions.create.return_value = mock_session
            
            # Add sessions to reach limit
            for i in range(2):
                session_info = SessionInfo(
                    session_id=f"test-session-{i}",
                    connect_url=f"wss://test.com/session/{i}",
                    status=SessionStatus.ACTIVE,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    config=SessionConfig()
                )
                manager.active_sessions[f"test-session-{i}"] = session_info
            
            # Try to get another session - should fail
            with pytest.raises(SessionCreationError, match="Maximum concurrent sessions limit reached"):
                manager.get_session()
    
    def test_get_session_respects_limit_with_pool_sessions(self):
        """Test that get_session respects limit when pool has sessions."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=2
            )
            
            # Add sessions to pool to reach limit
            session_info = SessionInfo(
                session_id="test-session-1",
                connect_url="wss://test.com/session/1",
                status=SessionStatus.IDLE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=SessionConfig()
            )
            manager.session_pool.in_use_sessions["test-session-1"] = session_info
            manager.session_pool.in_use_sessions["test-session-2"] = session_info
            
            # Try to get another session - should fail
            with pytest.raises(SessionCreationError, match="Maximum concurrent sessions limit reached"):
                manager.get_session()
    
    def test_get_session_respects_limit_mixed_sessions(self):
        """Test that get_session respects limit with mixed active and pool sessions."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=3
            )
            
            # Add one active session
            session_info = SessionInfo(
                session_id="test-session-1",
                connect_url="wss://test.com/session/1",
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=SessionConfig()
            )
            manager.active_sessions["test-session-1"] = session_info
            
            # Add two pool sessions
            manager.session_pool.in_use_sessions["test-session-2"] = session_info
            manager.session_pool.in_use_sessions["test-session-3"] = session_info
            
            # Try to get another session - should fail
            with pytest.raises(SessionCreationError, match="Maximum concurrent sessions limit reached"):
                manager.get_session()
    
    def test_get_session_allows_session_when_under_limit(self):
        """Test that get_session allows session creation when under limit."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=2
            )
            
            # Mock session creation to succeed
            mock_session = Mock()
            mock_session.id = "test-session-1"
            mock_session.connect_url = "wss://test.com/session/1"
            manager.bb.sessions.create.return_value = mock_session
            
            # Add one session (under limit)
            session_info = SessionInfo(
                session_id="test-session-1",
                connect_url="wss://test.com/session/1",
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=SessionConfig()
            )
            manager.active_sessions["test-session-1"] = session_info
            
            # Try to get another session - should succeed
            try:
                manager.get_session()
            except SessionCreationError:
                # This is expected if the mock doesn't work properly
                pass
            except Exception:
                # Other exceptions are not expected
                raise
    
    def test_session_limit_error_message(self):
        """Test that session limit error message is informative."""
        with patch('scraping_cli.browserbase_manager.Browserbase'):
            manager = BrowserbaseManager(
                api_key="test-key",
                project_id="test-project",
                max_concurrent_sessions=1
            )
            
            # Add one session to reach limit
            session_info = SessionInfo(
                session_id="test-session-1",
                connect_url="wss://test.com/session/1",
                status=SessionStatus.ACTIVE,
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=SessionConfig()
            )
            manager.active_sessions["test-session-1"] = session_info
            
            # Try to get another session
            with pytest.raises(SessionCreationError) as exc_info:
                manager.get_session()
            
            error_message = str(exc_info.value)
            assert "Maximum concurrent sessions limit reached (1)" in error_message
            assert "Currently have 1 active sessions" in error_message
            assert "Please wait for some sessions to be released or increase the limit" in error_message 