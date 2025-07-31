"""
Unit tests for ProgressMonitor.

Tests the progress monitoring system with mock data and different display modes.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from scraping_cli.progress_monitor import (
    ProgressMonitor, ProgressStats, DisplayMode, StatusLevel,
    create_progress_monitor
)


class TestProgressStats:
    """Test ProgressStats dataclass."""
    
    def test_progress_stats_creation(self):
        """Test creating ProgressStats with default values."""
        stats = ProgressStats()
        assert stats.total_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0
        assert stats.current_task is None
        assert stats.agent_status == {}
        assert stats.browser_sessions == {}
        assert stats.errors == []
        assert stats.start_time is None
        assert stats.last_update is None
    
    def test_progress_stats_with_values(self):
        """Test creating ProgressStats with specific values."""
        start_time = datetime.now()
        stats = ProgressStats(
            total_tasks=10,
            completed_tasks=5,
            failed_tasks=1,
            current_task="test_task",
            start_time=start_time
        )
        assert stats.total_tasks == 10
        assert stats.completed_tasks == 5
        assert stats.failed_tasks == 1
        assert stats.current_task == "test_task"
        assert stats.start_time == start_time


class TestDisplayMode:
    """Test DisplayMode enum."""
    
    def test_display_mode_values(self):
        """Test DisplayMode enum values."""
        assert DisplayMode.SIMPLE.value == "simple"
        assert DisplayMode.DETAILED.value == "detailed"
        assert DisplayMode.DEBUG.value == "debug"
    
    def test_display_mode_from_string(self):
        """Test creating DisplayMode from string."""
        assert DisplayMode("simple") == DisplayMode.SIMPLE
        assert DisplayMode("detailed") == DisplayMode.DETAILED
        assert DisplayMode("debug") == DisplayMode.DEBUG


class TestStatusLevel:
    """Test StatusLevel enum."""
    
    def test_status_level_values(self):
        """Test StatusLevel enum values."""
        assert StatusLevel.INFO.value == "info"
        assert StatusLevel.WARNING.value == "warning"
        assert StatusLevel.ERROR.value == "error"
        assert StatusLevel.SUCCESS.value == "success"


class TestProgressMonitor:
    """Test ProgressMonitor functionality."""
    
    @pytest.fixture
    def mock_tqdm(self):
        """Mock tqdm for testing."""
        with patch('scraping_cli.progress_monitor.tqdm') as mock_tqdm:
            # Mock tqdm instance
            mock_bar = MagicMock()
            mock_tqdm.return_value = mock_bar
            yield mock_tqdm, mock_bar
    
    @pytest.fixture
    def mock_colorama(self):
        """Mock colorama for testing."""
        with patch('scraping_cli.progress_monitor.Fore') as mock_fore, \
             patch('scraping_cli.progress_monitor.Style') as mock_style, \
             patch('scraping_cli.progress_monitor.init') as mock_init:
            mock_fore.CYAN = '\033[36m'
            mock_fore.GREEN = '\033[32m'
            mock_fore.RED = '\033[31m'
            mock_fore.YELLOW = '\033[33m'
            mock_fore.BLUE = '\033[34m'
            mock_fore.WHITE = '\033[37m'
            mock_style.RESET_ALL = '\033[0m'
            yield mock_fore, mock_style, mock_init
    
    @pytest.fixture
    def mock_shutil(self):
        """Mock shutil for terminal size detection."""
        with patch('scraping_cli.progress_monitor.shutil') as mock_shutil:
            mock_shutil.get_terminal_size.return_value.columns = 80
            yield mock_shutil
    
    def test_progress_monitor_creation(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test creating ProgressMonitor with default values."""
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            assert monitor.mode == DisplayMode.SIMPLE
            assert monitor.stats.total_tasks == 0
            assert monitor.stats.start_time is not None
            assert monitor.term_width == 80
            assert monitor.disable is False
    
    def test_progress_monitor_with_custom_values(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test creating ProgressMonitor with custom values."""
        monitor = ProgressMonitor(
            total_tasks=10,
            mode=DisplayMode.DETAILED,
            disable=True
        )
        assert monitor.mode == DisplayMode.DETAILED
        assert monitor.stats.total_tasks == 10
        assert monitor.disable is True
    
    def test_progress_monitor_disable_auto_detection(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test auto-detection of disable flag."""
        with patch('sys.stdout.isatty', return_value=False):
            monitor = ProgressMonitor()
            assert monitor.disable is True
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            assert monitor.disable is False
    
    def test_terminal_width_detection_error(self, mock_tqdm, mock_colorama):
        """Test terminal width detection with error."""
        with patch('scraping_cli.progress_monitor.shutil') as mock_shutil:
            mock_shutil.get_terminal_size.side_effect = OSError("Terminal not found")
            monitor = ProgressMonitor()
            assert monitor.term_width == 80  # Default fallback
    
    def test_update_progress(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test updating progress."""
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor(total_tasks=5)
            
            # Update progress
            monitor.update_progress(
                n=2,
                task_name="test_task",
                agent_status={"agent1": "working"},
                browser_status={"session1": "active"},
                errors=["test error"]
            )
            
            assert monitor.stats.completed_tasks == 2
            assert monitor.stats.current_task == "test_task"
            assert monitor.stats.agent_status == {"agent1": "working"}
            assert monitor.stats.browser_sessions == {"session1": "active"}
            assert monitor.stats.errors == ["test error"]
            assert monitor.stats.failed_tasks == 1
            assert monitor.stats.last_update is not None
    
    def test_set_agent_status(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test setting agent status."""
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            
            monitor.set_agent_status("agent1", "working")
            assert monitor.stats.agent_status["agent1"] == "working"
            
            monitor.set_agent_status("agent2", "idle")
            assert monitor.stats.agent_status["agent2"] == "idle"
    
    def test_set_browser_session_status(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test setting browser session status."""
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            
            monitor.set_browser_session_status("session1", "active")
            assert monitor.stats.browser_sessions["session1"] == "active"
            
            monitor.set_browser_session_status("session2", "closed")
            assert monitor.stats.browser_sessions["session2"] == "closed"
    
    def test_add_error(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test adding errors."""
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            
            monitor.add_error("Test error 1")
            monitor.add_error("Test error 2")
            
            assert monitor.stats.errors == ["Test error 1", "Test error 2"]
            assert monitor.stats.failed_tasks == 2
    
    def test_get_summary(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test getting progress summary."""
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor(total_tasks=10)
            monitor.stats.start_time = datetime.now()
            monitor.stats.last_update = datetime.now()
            monitor.stats.completed_tasks = 8
            monitor.stats.failed_tasks = 1
            monitor.stats.errors = ["test error"]
            monitor.stats.agent_status = {"agent1": "working"}
            monitor.stats.browser_sessions = {"session1": "active"}
            
            summary = monitor.get_summary()
            
            assert summary['total_tasks'] == 10
            assert summary['completed_tasks'] == 8
            assert summary['failed_tasks'] == 1
            assert summary['success_rate'] == 80.0
            assert summary['errors'] == ["test error"]
            assert summary['agent_status'] == {"agent1": "working"}
            assert summary['browser_sessions'] == {"session1": "active"}
            assert summary['duration'] is not None
    
    def test_get_summary_zero_tasks(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test getting summary with zero tasks."""
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor(total_tasks=0)
            summary = monitor.get_summary()
            assert summary['success_rate'] == 0.0
    
    def test_context_manager(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test ProgressMonitor as context manager."""
        mock_bar = mock_tqdm[1]
        
        with patch('sys.stdout.isatty', return_value=True):
            with ProgressMonitor() as monitor:
                assert monitor is not None
            
            # Should call close on exit
            mock_bar.close.assert_called_once()
    
    def test_close_method(self, mock_tqdm, mock_colorama, mock_shutil):
        """Test close method."""
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            mock_bar = mock_tqdm[1]
            
            monitor.close()
            mock_bar.close.assert_called_once()


class TestProgressMonitorDisplayModes:
    """Test different display modes."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for display mode testing."""
        with patch('scraping_cli.progress_monitor.tqdm') as mock_tqdm, \
             patch('scraping_cli.progress_monitor.Fore') as mock_fore, \
             patch('scraping_cli.progress_monitor.Style') as mock_style, \
             patch('scraping_cli.progress_monitor.init') as mock_init, \
             patch('scraping_cli.progress_monitor.shutil') as mock_shutil:
            
            mock_shutil.get_terminal_size.return_value.columns = 80
            mock_fore.CYAN = '\033[36m'
            mock_fore.GREEN = '\033[32m'
            mock_fore.RED = '\033[31m'
            mock_fore.YELLOW = '\033[33m'
            mock_fore.BLUE = '\033[34m'
            mock_fore.WHITE = '\033[37m'
            mock_style.RESET_ALL = '\033[0m'
            
            yield mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil
    
    def test_simple_mode(self, mock_dependencies):
        """Test simple display mode."""
        mock_tqdm = mock_dependencies[0]
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor(mode=DisplayMode.SIMPLE)
            
            # Should create main progress bar
            mock_tqdm.assert_called_once()
            call_args = mock_tqdm.call_args
            assert call_args[1]['desc'] == '\033[36mOverall Progress\033[0m'
    
    def test_detailed_mode(self, mock_dependencies):
        """Test detailed display mode."""
        mock_tqdm = mock_dependencies[0]
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor(mode=DisplayMode.DETAILED)
            
            # Should create main progress bar
            mock_tqdm.assert_called_once()
            call_args = mock_tqdm.call_args
            assert call_args[1]['desc'] == '\033[36mOverall Progress\033[0m'
    
    def test_debug_mode(self, mock_dependencies):
        """Test debug display mode."""
        mock_tqdm = mock_dependencies[0]
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor(mode=DisplayMode.DEBUG)
            
            # Should create main and debug progress bars
            assert mock_tqdm.call_count == 2
            
            # Check debug bar
            debug_call = mock_tqdm.call_args_list[1]
            assert debug_call[1]['desc'] == '\033[36mDebug Info\033[0m'
            assert debug_call[1]['position'] == 1
    
    def test_disabled_mode(self, mock_dependencies):
        """Test disabled progress bars."""
        mock_tqdm = mock_dependencies[0]
        
        monitor = ProgressMonitor(disable=True)
        
        # Should not create any progress bars
        mock_tqdm.assert_not_called()


class TestProgressMonitorLogging:
    """Test logging functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for logging tests."""
        with patch('scraping_cli.progress_monitor.tqdm') as mock_tqdm, \
             patch('scraping_cli.progress_monitor.Fore') as mock_fore, \
             patch('scraping_cli.progress_monitor.Style') as mock_style, \
             patch('scraping_cli.progress_monitor.init') as mock_init, \
             patch('scraping_cli.progress_monitor.shutil') as mock_shutil, \
             patch('scraping_cli.progress_monitor.print') as mock_print:
            
            mock_shutil.get_terminal_size.return_value.columns = 80
            mock_fore.CYAN = '\033[36m'
            mock_fore.GREEN = '\033[32m'
            mock_fore.RED = '\033[31m'
            mock_fore.YELLOW = '\033[33m'
            mock_style.RESET_ALL = '\033[0m'
            
            yield mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print
    
    def test_log_info(self, mock_dependencies):
        """Test logging info messages."""
        mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print = mock_dependencies
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            monitor.log("Test info message", StatusLevel.INFO)
            
            # Should call tqdm.write with colored message
            mock_tqdm.write.assert_called_once_with('\033[36mTest info message\033[0m')
    
    def test_log_warning(self, mock_dependencies):
        """Test logging warning messages."""
        mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print = mock_dependencies
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            monitor.log("Test warning message", StatusLevel.WARNING)
            
            # Should call tqdm.write with colored message
            mock_tqdm.write.assert_called_once_with('\033[33mTest warning message\033[0m')
    
    def test_log_error(self, mock_dependencies):
        """Test logging error messages."""
        mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print = mock_dependencies
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            monitor.log("Test error message", StatusLevel.ERROR)
            
            # Should call tqdm.write with colored message
            mock_tqdm.write.assert_called_once_with('\033[31mTest error message\033[0m')
    
    def test_log_success(self, mock_dependencies):
        """Test logging success messages."""
        mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print = mock_dependencies
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor()
            monitor.log("Test success message", StatusLevel.SUCCESS)
            
            # Should call tqdm.write with colored message
            mock_tqdm.write.assert_called_once_with('\033[32mTest success message\033[0m')
    
    def test_log_disabled(self, mock_dependencies):
        """Test logging when progress bars are disabled."""
        mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print = mock_dependencies
        
        monitor = ProgressMonitor(disable=True)
        monitor.log("Test message")
        
        # Should call print instead of tqdm.write
        mock_print.assert_called_once_with("Test message")
    
    def test_log_debug_mode_timestamp(self, mock_dependencies):
        """Test logging in debug mode with timestamp."""
        mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print = mock_dependencies
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor(mode=DisplayMode.DEBUG)
            
            with patch('scraping_cli.progress_monitor.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "12:34:56"
                monitor.log("Test debug message")
                
                # Should include timestamp
                mock_tqdm.write.assert_called_once_with('\033[36m[12:34:56] Test debug message\033[0m')


class TestProgressMonitorSummary:
    """Test summary generation and display."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for summary tests."""
        with patch('scraping_cli.progress_monitor.tqdm') as mock_tqdm, \
             patch('scraping_cli.progress_monitor.Fore') as mock_fore, \
             patch('scraping_cli.progress_monitor.Style') as mock_style, \
             patch('scraping_cli.progress_monitor.init') as mock_init, \
             patch('scraping_cli.progress_monitor.shutil') as mock_shutil, \
             patch('scraping_cli.progress_monitor.print') as mock_print:
            
            mock_shutil.get_terminal_size.return_value.columns = 80
            mock_fore.CYAN = '\033[36m'
            mock_fore.GREEN = '\033[32m'
            mock_fore.RED = '\033[31m'
            mock_fore.YELLOW = '\033[33m'
            mock_fore.BLUE = '\033[34m'
            mock_fore.WHITE = '\033[37m'
            mock_style.RESET_ALL = '\033[0m'
            
            yield mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print
    
    def test_print_summary_colored(self, mock_dependencies):
        """Test printing colored summary."""
        mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print = mock_dependencies
        
        with patch('sys.stdout.isatty', return_value=True):
            monitor = ProgressMonitor(total_tasks=10)
            monitor.stats.completed_tasks = 8
            monitor.stats.failed_tasks = 1
            monitor.stats.errors = ["test error"]
            monitor.stats.agent_status = {"agent1": "working"}
            
            monitor.print_summary()
            
            # Should close bars and print summary
            mock_tqdm.return_value.close.assert_called()
            assert mock_print.call_count > 0
    
    def test_print_summary_plain(self, mock_dependencies):
        """Test printing plain summary when disabled."""
        mock_tqdm, mock_fore, mock_style, mock_init, mock_shutil, mock_print = mock_dependencies
        
        monitor = ProgressMonitor(total_tasks=10, disable=True)
        monitor.stats.completed_tasks = 8
        monitor.stats.failed_tasks = 1
        monitor.stats.errors = ["test error"]
        monitor.stats.agent_status = {"agent1": "working"}
        
        monitor.print_summary()
        
        # Should print plain summary without colors
        assert mock_print.call_count > 0
        # Check that no color codes are used
        for call in mock_print.call_args_list:
            args = call[0][0]
            assert '\033[' not in args


class TestCreateProgressMonitor:
    """Test factory function for creating progress monitors."""
    
    @pytest.fixture
    def mock_progress_monitor(self):
        """Mock ProgressMonitor class."""
        with patch('scraping_cli.progress_monitor.ProgressMonitor') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance
            yield mock_class, mock_instance
    
    def test_create_progress_monitor_defaults(self, mock_progress_monitor):
        """Test creating progress monitor with defaults."""
        mock_class, mock_instance = mock_progress_monitor
        
        result = create_progress_monitor()
        
        mock_class.assert_called_once_with(
            total_tasks=0,
            mode=DisplayMode.SIMPLE,
            disable=None
        )
        assert result == mock_instance
    
    def test_create_progress_monitor_with_values(self, mock_progress_monitor):
        """Test creating progress monitor with specific values."""
        mock_class, mock_instance = mock_progress_monitor
        
        result = create_progress_monitor(
            total_tasks=10,
            mode="detailed",
            disable=True
        )
        
        mock_class.assert_called_once_with(
            total_tasks=10,
            mode=DisplayMode.DETAILED,
            disable=True
        )
        assert result == mock_instance
    
    def test_create_progress_monitor_invalid_mode(self, mock_progress_monitor):
        """Test creating progress monitor with invalid mode."""
        mock_class, mock_instance = mock_progress_monitor
        
        result = create_progress_monitor(mode="invalid_mode")
        
        # Should fallback to SIMPLE mode
        mock_class.assert_called_once_with(
            total_tasks=0,
            mode=DisplayMode.SIMPLE,
            disable=None
        )
        assert result == mock_instance 