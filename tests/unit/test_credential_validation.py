"""
Unit tests for credential validation.

Tests that BrowserbaseManager properly validates credentials
without requiring real Browserbase API access.
"""

import pytest
import os
from unittest.mock import patch

from scraping_cli.browserbase_manager import BrowserbaseManager
from scraping_cli.exceptions import ConfigurationError


class TestCredentialValidation:
    """Test credential validation functionality."""
    
    def test_manager_initialization_missing_credentials(self):
        """Test that manager raises error when credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase API key is required"):
                BrowserbaseManager()
    
    def test_manager_initialization_missing_api_key(self):
        """Test that manager raises error when API key is missing."""
        with patch.dict(os.environ, {'BROWSERBASE_PROJECT_ID': 'test-project'}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase API key is required"):
                BrowserbaseManager()
    
    def test_manager_initialization_missing_project_id(self):
        """Test that manager raises error when project ID is missing."""
        with patch.dict(os.environ, {'BROWSERBASE_API_KEY': 'test-key'}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase project ID is required"):
                BrowserbaseManager()
    
    def test_manager_initialization_with_explicit_none(self):
        """Test that manager raises error when credentials are explicitly None."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase API key is required"):
                BrowserbaseManager(api_key=None, project_id=None)
    
    def test_manager_initialization_with_explicit_none_api_key(self):
        """Test that manager raises error when API key is explicitly None."""
        with patch.dict(os.environ, {'BROWSERBASE_PROJECT_ID': 'test-project'}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase API key is required"):
                BrowserbaseManager(api_key=None, project_id="test-project")
    
    def test_manager_initialization_with_explicit_none_project_id(self):
        """Test that manager raises error when project ID is explicitly None."""
        with patch.dict(os.environ, {'BROWSERBASE_API_KEY': 'test-key'}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase project ID is required"):
                BrowserbaseManager(api_key="test-key", project_id=None)
    
    def test_manager_initialization_with_empty_strings(self):
        """Test that manager raises error when credentials are empty strings."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase API key is required"):
                BrowserbaseManager(api_key="", project_id="")
    
    def test_manager_initialization_with_empty_api_key(self):
        """Test that manager raises error when API key is empty string."""
        with patch.dict(os.environ, {'BROWSERBASE_PROJECT_ID': 'test-project'}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase API key is required"):
                BrowserbaseManager(api_key="", project_id="test-project")
    
    def test_manager_initialization_with_empty_project_id(self):
        """Test that manager raises error when project ID is empty string."""
        with patch.dict(os.environ, {'BROWSERBASE_API_KEY': 'test-key'}, clear=True):
            with pytest.raises(ConfigurationError, match="Browserbase project ID is required"):
                BrowserbaseManager(api_key="test-key", project_id="") 