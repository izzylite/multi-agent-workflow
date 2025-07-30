"""
Integration tests for BrowserbaseManager.

These tests require valid Browserbase API credentials to run.
If credentials are not available, tests will be skipped.
"""

import pytest
import os
from datetime import datetime
from unittest.mock import patch

from scraping_cli.browserbase_manager import (
    BrowserbaseManager, SessionConfig, SessionStatus
)
from scraping_cli.exceptions import (
    SessionCreationError, ConfigurationError
)


@pytest.fixture(scope="session")
def browserbase_credentials():
    """Get Browserbase credentials from environment."""
    api_key = os.getenv('BROWSERBASE_API_KEY')
    project_id = os.getenv('BROWSERBASE_PROJECT_ID')
    
    if not api_key or not project_id:
        pytest.skip("Browserbase credentials not available. Set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables.")
    
    return {
        'api_key': api_key,
        'project_id': project_id
    }


@pytest.fixture
def manager(browserbase_credentials):
    """Create a BrowserbaseManager instance for testing."""
    manager = BrowserbaseManager(
        api_key=browserbase_credentials['api_key'],
        project_id=browserbase_credentials['project_id'],
        pool_size=2,
        max_retries=2,
        session_timeout=60
    )
    
    yield manager
    
    # Cleanup: Close all sessions after each test
    try:
        manager.close_all_sessions()
    except Exception as e:
        print(f"Warning: Failed to cleanup sessions: {e}")


class TestBrowserbaseIntegration:
    """Integration tests for BrowserbaseManager with real API calls."""
    
    def test_manager_initialization(self, browserbase_credentials):
        """Test that manager can be initialized with valid credentials."""
        manager = BrowserbaseManager(
            api_key=browserbase_credentials['api_key'],
            project_id=browserbase_credentials['project_id']
        )
        
        assert manager.api_key == browserbase_credentials['api_key']
        assert manager.project_id == browserbase_credentials['project_id']
        assert manager.bb is not None
    
    def test_create_session_integration(self, manager):
        """Test creating a real browser session."""
        config = SessionConfig(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport_width=1280,
            viewport_height=720
        )
        
        session_info = manager.create_session(config)
        
        assert session_info.session_id is not None
        assert session_info.connect_url is not None
        assert session_info.status == SessionStatus.ACTIVE
        assert session_info.config == config
        assert session_info.session_id in manager.active_sessions
    
    def test_session_lifecycle_integration(self, manager):
        """Test complete session lifecycle."""
        # Create session
        session_info = manager.create_session()
        session_id = session_info.session_id
        
        # Verify session is active
        assert session_id in manager.active_sessions
        health = manager.get_session_health(session_id)
        assert health['status'] == 'active'
        
        # Release session
        manager.release_session(session_info)
        assert session_id not in manager.active_sessions
        
        # Close session
        manager.close_session(session_id)
        
        # Verify session is closed
        health = manager.get_session_health(session_id)
        assert health['status'] == 'not_found'
    
    def test_session_context_manager_integration(self, manager):
        """Test session context manager with real sessions."""
        with manager.session_context() as session_info:
            assert session_info.session_id is not None
            assert session_info.session_id in manager.active_sessions
        
        # Session should be released after context exit
        assert session_info.session_id not in manager.active_sessions
    
    def test_manager_context_manager_integration(self, manager):
        """Test manager context manager."""
        with manager:
            # Create a session within the context
            session_info = manager.create_session()
            assert session_info.session_id in manager.active_sessions
        
        # All sessions should be closed after context exit
        assert len(manager.active_sessions) == 0
    
    def test_session_pool_integration(self, manager):
        """Test session pooling with real sessions."""
        # Get multiple sessions
        session1 = manager.get_session()
        session2 = manager.get_session()
        
        assert session1.session_id != session2.session_id
        assert len(manager.active_sessions) == 2
        
        # Release sessions back to pool
        manager.release_session(session1)
        manager.release_session(session2)
        
        # Sessions should be in pool
        assert len(manager.session_pool.available_sessions) == 2
        
        # Get session from pool
        session3 = manager.get_session()
        assert session3.session_id in [session1.session_id, session2.session_id]
    
    def test_error_handling_integration(self, manager):
        """Test error handling with invalid configuration."""
        with pytest.raises(SessionCreationError):
            # Try to create session with invalid project ID
            with patch.object(manager, 'project_id', 'invalid-project-id'):
                manager.create_session()


class TestBrowserbaseIntegrationWithoutCredentials:
    """Tests that can run without Browserbase credentials."""
    
    def test_manager_initialization_missing_credentials(self):
        """Test that manager raises error when credentials are missing."""
        with pytest.raises(ConfigurationError):
            BrowserbaseManager(api_key=None, project_id=None)
    
    def test_manager_initialization_missing_api_key(self):
        """Test that manager raises error when API key is missing."""
        with pytest.raises(ConfigurationError):
            BrowserbaseManager(api_key=None, project_id="test-project")
    
    def test_manager_initialization_missing_project_id(self):
        """Test that manager raises error when project ID is missing."""
        with pytest.raises(ConfigurationError):
            BrowserbaseManager(api_key="test-key", project_id=None) 