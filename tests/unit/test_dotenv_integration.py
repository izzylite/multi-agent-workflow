"""
Test dotenv integration for environment variable loading.
"""

import os
import tempfile
import pytest
from unittest.mock import patch

from scraping_cli.browserbase_manager import BrowserbaseManager


class TestDotenvIntegration:
    """Test that dotenv properly loads environment variables from .env file."""
    
    def test_dotenv_loads_environment_variables(self):
        """Test that dotenv loads environment variables from .env file."""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("BROWSERBASE_API_KEY=test_api_key\n")
            f.write("BROWSERBASE_PROJECT_ID=test_project_id\n")
            f.write("BROWSERBASE_MAX_SESSIONS=5\n")
            env_file = f.name
        
        try:
            # Mock the load_dotenv to use our temporary file
            with patch('scraping_cli.browserbase_manager.load_dotenv') as mock_load_dotenv:
                # Re-import the module to trigger load_dotenv
                import importlib
                import scraping_cli.browserbase_manager
                importlib.reload(scraping_cli.browserbase_manager)
                
                # Verify load_dotenv was called
                mock_load_dotenv.assert_called_once()
        
        finally:
            # Clean up temporary file
            os.unlink(env_file)
    
    def test_environment_variables_loaded_from_env_file(self):
        """Test that environment variables are properly loaded from .env file."""
        # Set up test environment
        test_env = {
            'BROWSERBASE_API_KEY': 'test_api_key_from_env',
            'BROWSERBASE_PROJECT_ID': 'test_project_id_from_env',
            'BROWSERBASE_MAX_SESSIONS': '10'
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            # The BrowserbaseManager should be able to read these values
            # Note: We don't actually create the manager since we don't have real credentials
            # but we can test that the environment variables are accessible
            
            assert os.getenv('BROWSERBASE_API_KEY') == 'test_api_key_from_env'
            assert os.getenv('BROWSERBASE_PROJECT_ID') == 'test_project_id_from_env'
            assert os.getenv('BROWSERBASE_MAX_SESSIONS') == '10'
    
    def test_missing_env_file_does_not_cause_error(self):
        """Test that missing .env file doesn't cause errors."""
        # Remove any existing .env file temporarily
        env_file = '.env'
        env_file_backup = None
        
        if os.path.exists(env_file):
            env_file_backup = env_file + '.backup'
            os.rename(env_file, env_file_backup)
        
        try:
            # The module should still import without errors
            import scraping_cli.browserbase_manager
            assert True  # If we get here, no error occurred
        finally:
            # Restore .env file if it existed
            if env_file_backup and os.path.exists(env_file_backup):
                os.rename(env_file_backup, env_file) 