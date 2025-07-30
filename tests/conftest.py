"""
Shared pytest configuration and fixtures for scraping_cli tests.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from typing import Dict, Any

# Configure test environment
os.environ.setdefault('TESTING', 'true')


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="scraping_cli_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def mock_browserbase_sdk():
    """Mock Browserbase SDK for unit tests."""
    with patch('scraping_cli.browserbase_manager.Browserbase') as mock_bb:
        # Mock session creation
        mock_session = Mock()
        mock_session.id = "test-session-123"
        mock_session.connect_url = "wss://test.browserbase.com/session/123"
        
        mock_bb.return_value.sessions.create.return_value = mock_session
        # Mock session deletion
        mock_bb.return_value.sessions.delete.return_value = None
        
        # Mock session listing
        mock_bb.return_value.sessions.list.return_value = []
        
        yield mock_bb


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Automatically cleanup sessions after each test."""
    yield
    # This fixture runs after each test to ensure cleanup
    # The actual cleanup is handled by the manager fixture in integration tests


@pytest.fixture(scope="session")
def mock_health_monitor():
    """Mock health monitor for unit tests."""
    with patch('scraping_cli.browserbase_manager.create_health_monitor') as mock_create:
        mock_monitor = Mock()
        mock_create.return_value = mock_monitor
        yield mock_monitor


@pytest.fixture
def sample_session_config():
    """Create a sample session configuration for testing."""
    from scraping_cli.browserbase_manager import SessionConfig
    
    return SessionConfig(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        stealth_mode=True,
        viewport_width=1366,
        viewport_height=768,
        timeout=30000,
        max_concurrent_sessions=5,
        session_ttl=1800,
        retry_attempts=3,
        retry_delay=1.0
    )


@pytest.fixture
def sample_session_info(sample_session_config):
    """Create a sample session info for testing."""
    from scraping_cli.browserbase_manager import SessionInfo, SessionStatus
    from datetime import datetime
    
    return SessionInfo(
        session_id="test-session-456",
        connect_url="wss://test.browserbase.com/session/456",
        status=SessionStatus.ACTIVE,
        created_at=datetime.now(),
        last_used=datetime.now(),
        config=sample_session_config
    )


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'BROWSERBASE_API_KEY': 'test-api-key',
        'BROWSERBASE_PROJECT_ID': 'test-project-id',
        'TESTING': 'true'
    }):
        yield


# Test markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (require external resources)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "browserbase: Tests requiring Browserbase API"
    )
    config.addinivalue_line(
        "markers", "async: Async tests"
    )


# Skip integration tests if no credentials
def pytest_collection_modifyitems(config, items):
    """Skip integration tests if no Browserbase credentials."""
    skip_integration = pytest.mark.skip(reason="No Browserbase credentials")
    skip_browserbase = pytest.mark.skip(reason="Browserbase API not available")
    
    for item in items:
        if "integration" in item.keywords and not (
            os.getenv('BROWSERBASE_API_KEY') and os.getenv('BROWSERBASE_PROJECT_ID')
        ):
            item.add_marker(skip_integration)
        
        if "browserbase" in item.keywords and not (
            os.getenv('BROWSERBASE_API_KEY') and os.getenv('BROWSERBASE_PROJECT_ID')
        ):
            item.add_marker(skip_browserbase) 