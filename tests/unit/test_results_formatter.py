"""
Unit tests for ResultsFormatter class.

Tests the enhanced results formatting capabilities including table formatting,
color-coded output, summary statistics, and customizable display templates.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from scraping_cli.results_formatter import (
    ResultsFormatter,
    DisplayConfig,
    DisplayTheme,
    TableFormat,
    SummaryStats,
    create_results_formatter
)


class TestResultsFormatter:
    """Test cases for ResultsFormatter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = [
            {
                "vendor": "Tesco",
                "category": "Groceries",
                "product_count": 150,
                "created_at": "2024-01-15T10:30:00",
                "file_size": 1024000,
                "compressed": True,
                "file_path": "/data/tesco_groceries_20240115.json"
            },
            {
                "vendor": "Asda",
                "category": "Electronics",
                "product_count": 75,
                "created_at": "2024-01-16T14:20:00",
                "file_size": 512000,
                "compressed": False,
                "file_path": "/data/asda_electronics_20240116.json"
            },
            {
                "vendor": "Tesco",
                "category": "Home",
                "product_count": 200,
                "created_at": "2024-01-17T09:15:00",
                "file_size": 2048000,
                "compressed": True,
                "file_path": "/data/tesco_home_20240117.json"
            }
        ]

    def test_init_default_config(self):
        """Test ResultsFormatter initialization with default config."""
        formatter = ResultsFormatter()
        assert formatter.config.theme == DisplayTheme.DEFAULT
        assert formatter.config.table_format == TableFormat.FANCY_GRID
        assert formatter.config.show_colors is True
        assert formatter.config.show_summary is True

    def test_init_custom_config(self):
        """Test ResultsFormatter initialization with custom config."""
        config = DisplayConfig(
            theme=DisplayTheme.DARK,
            table_format=TableFormat.SIMPLE,
            show_colors=False
        )
        formatter = ResultsFormatter(config)
        assert formatter.config.theme == DisplayTheme.DARK
        assert formatter.config.table_format == TableFormat.SIMPLE
        assert formatter.config.show_colors is False

    def test_format_results_table_basic(self):
        """Test basic table formatting."""
        formatter = ResultsFormatter()
        result = formatter.format_results_table(self.sample_data)
        
        assert "Tesco" in result
        assert "Asda" in result
        assert "Groceries" in result
        assert "Electronics" in result

    def test_format_results_table_empty_data(self):
        """Test table formatting with empty data."""
        formatter = ResultsFormatter()
        result = formatter.format_results_table([])
        
        assert "No data to display" in result

    def test_format_results_table_with_title(self):
        """Test table formatting with title."""
        formatter = ResultsFormatter()
        result = formatter.format_results_table(
            self.sample_data,
            title="Test Results"
        )
        
        assert "Test Results" in result

    def test_format_results_table_with_headers(self):
        """Test table formatting with custom headers."""
        formatter = ResultsFormatter()
        # Use headers that match the number of columns in sample_data
        headers = ["Store", "Type", "Count", "Date", "Size", "Compressed", "Path"]
        result = formatter.format_results_table(
            self.sample_data,
            headers=headers
        )
        
        assert "Store" in result
        assert "Type" in result
        assert "Count" in result

    def test_format_results_table_different_formats(self):
        """Test table formatting with different table formats."""
        formatter = ResultsFormatter()
        
        # Test simple format
        formatter.set_table_format(TableFormat.SIMPLE)
        simple_result = formatter.format_results_table(self.sample_data)
        assert "Tesco" in simple_result
        
        # Test grid format
        formatter.set_table_format(TableFormat.GRID)
        grid_result = formatter.format_results_table(self.sample_data)
        assert "Tesco" in grid_result

    def test_format_summary_stats_basic(self):
        """Test basic summary statistics formatting."""
        formatter = ResultsFormatter()
        result = formatter.format_summary_stats(self.sample_data)
        
        assert "Summary Statistics" in result
        # Check for colored text pattern instead of exact match
        assert "Total Items:" in result
        assert "3" in result
        assert "Unique Vendors:" in result
        assert "2" in result
        assert "Unique Categories:" in result
        assert "3" in result

    def test_format_summary_stats_empty_data(self):
        """Test summary statistics with empty data."""
        formatter = ResultsFormatter()
        result = formatter.format_summary_stats([])
        
        assert "No data for statistics" in result

    def test_format_summary_stats_with_title(self):
        """Test summary statistics with title."""
        formatter = ResultsFormatter()
        result = formatter.format_summary_stats(
            self.sample_data,
            title="Custom Title"
        )
        
        assert "Custom Title" in result

    def test_calculate_summary_stats(self):
        """Test summary statistics calculation."""
        formatter = ResultsFormatter()
        stats = formatter._calculate_summary_stats(self.sample_data)
        
        assert stats.total_items == 3
        assert stats.unique_vendors == 2
        assert stats.unique_categories == 3
        assert stats.total_size == 3584000  # 1024000 + 512000 + 2048000
        assert stats.average_items_per_vendor == 1.5
        assert len(stats.top_vendors) == 2
        assert len(stats.top_categories) == 3

    def test_calculate_summary_stats_empty_data(self):
        """Test summary statistics calculation with empty data."""
        formatter = ResultsFormatter()
        stats = formatter._calculate_summary_stats([])
        
        assert stats.total_items == 0
        assert stats.unique_vendors == 0
        assert stats.unique_categories == 0
        assert stats.total_size == 0
        assert stats.average_items_per_vendor == 0.0

    def test_format_custom_template_compact(self):
        """Test compact template formatting."""
        formatter = ResultsFormatter()
        result = formatter.format_custom_template("compact", self.sample_data)
        
        assert "Compact View" in result
        assert "Vendor" in result
        assert "Category" in result
        assert "Count" in result
        assert "Date" in result

    def test_format_custom_template_detailed(self):
        """Test detailed template formatting."""
        formatter = ResultsFormatter()
        result = formatter.format_custom_template("detailed", self.sample_data)
        
        assert "Detailed View" in result
        assert "Vendor" in result
        assert "Category" in result
        assert "Product Count" in result
        assert "Created At" in result
        assert "File Size" in result
        assert "Compressed" in result
        assert "File Path" in result

    def test_format_custom_template_summary(self):
        """Test summary template formatting."""
        formatter = ResultsFormatter()
        result = formatter.format_custom_template("summary", self.sample_data)
        
        assert "Data Summary" in result
        assert "Summary Statistics" in result

    def test_format_custom_template_invalid(self):
        """Test custom template with invalid template name."""
        formatter = ResultsFormatter()
        
        with pytest.raises(ValueError, match="Template 'invalid' not found"):
            formatter.format_custom_template("invalid", self.sample_data)

    def test_add_custom_template(self):
        """Test adding custom template."""
        formatter = ResultsFormatter()
        
        def custom_template(data, **kwargs):
            return f"Custom template with {len(data)} items"
        
        formatter.add_custom_template("custom", custom_template)
        result = formatter.format_custom_template("custom", self.sample_data)
        
        assert "Custom template with 3 items" in result

    def test_set_theme(self):
        """Test setting display theme."""
        formatter = ResultsFormatter()
        
        # Test dark theme
        formatter.set_theme(DisplayTheme.DARK)
        assert formatter.config.theme == DisplayTheme.DARK
        
        # Test light theme
        formatter.set_theme(DisplayTheme.LIGHT)
        assert formatter.config.theme == DisplayTheme.LIGHT

    def test_set_table_format(self):
        """Test setting table format."""
        formatter = ResultsFormatter()
        
        formatter.set_table_format(TableFormat.SIMPLE)
        assert formatter.config.table_format == TableFormat.SIMPLE

    def test_truncate_dataframe(self):
        """Test DataFrame truncation."""
        formatter = ResultsFormatter()
        formatter.config.max_truncate_length = 10
        
        data = [{"long_field": "This is a very long string that should be truncated"}]
        df = formatter._truncate_dataframe(pd.DataFrame(data))
        
        assert "This is a " in df.iloc[0]["long_field"]
        assert "..." in df.iloc[0]["long_field"]

    def test_format_file_size(self):
        """Test file size formatting."""
        formatter = ResultsFormatter()
        
        assert formatter._format_file_size(0) == "0 B"
        assert formatter._format_file_size(1024) == "1.0 KB"
        assert formatter._format_file_size(1048576) == "1.0 MB"
        assert formatter._format_file_size(1073741824) == "1.0 GB"

    def test_color_formatting(self):
        """Test color formatting."""
        formatter = ResultsFormatter()
        
        # Test with colors enabled
        colored_text = formatter._color("GREEN", "test")
        assert "test" in colored_text
        
        # Test with colors disabled
        formatter.config.show_colors = False
        formatter._setup_theme_colors()  # Re-setup colors after changing config
        plain_text = formatter._color("GREEN", "test")
        assert plain_text == "test"

    def test_theme_colors_setup(self):
        """Test theme color setup."""
        formatter = ResultsFormatter()
        
        # Test default theme
        formatter._setup_theme_colors()
        assert 'CYAN' in formatter._colors
        assert 'GREEN' in formatter._colors
        
        # Test minimal theme (no colors)
        formatter.config.theme = DisplayTheme.MINIMAL
        formatter._setup_theme_colors()
        assert formatter._colors['CYAN'] == ''

    def test_format_title(self):
        """Test title formatting."""
        formatter = ResultsFormatter()
        result = formatter._format_title("Test Title")
        
        assert "Test Title" in result
        assert "----------" in result

    def test_format_empty_message(self):
        """Test empty message formatting."""
        formatter = ResultsFormatter()
        result = formatter._format_empty_message("No data found")
        
        assert "No data found" in result

    @patch('colorama.init')
    def test_colorama_initialization(self, mock_init):
        """Test colorama initialization."""
        # Re-import to trigger colorama init
        import importlib
        import scraping_cli.results_formatter
        importlib.reload(scraping_cli.results_formatter)
        
        mock_init.assert_called_once()

    def test_create_results_formatter(self):
        """Test create_results_formatter factory function."""
        formatter = create_results_formatter()
        # Check if it has the expected attributes
        assert hasattr(formatter, 'config')
        assert hasattr(formatter, 'format_results_table')
        assert hasattr(formatter, 'format_summary_stats')
        
        config = DisplayConfig(theme=DisplayTheme.DARK)
        formatter_with_config = create_results_formatter(config)
        assert formatter_with_config.config.theme == DisplayTheme.DARK


class TestDisplayConfig:
    """Test cases for DisplayConfig dataclass."""

    def test_display_config_defaults(self):
        """Test DisplayConfig default values."""
        config = DisplayConfig()
        
        assert config.theme == DisplayTheme.DEFAULT
        assert config.table_format == TableFormat.FANCY_GRID
        assert config.show_colors is True
        assert config.show_summary is True
        assert config.max_width is None
        assert config.truncate_long_values is True
        assert config.max_truncate_length == 50

    def test_display_config_custom(self):
        """Test DisplayConfig with custom values."""
        config = DisplayConfig(
            theme=DisplayTheme.DARK,
            table_format=TableFormat.SIMPLE,
            show_colors=False,
            max_width=80
        )
        
        assert config.theme == DisplayTheme.DARK
        assert config.table_format == TableFormat.SIMPLE
        assert config.show_colors is False
        assert config.max_width == 80


class TestSummaryStats:
    """Test cases for SummaryStats dataclass."""

    def test_summary_stats_creation(self):
        """Test SummaryStats creation."""
        date_range = (datetime.now(), datetime.now())
        stats = SummaryStats(
            total_items=10,
            unique_vendors=3,
            unique_categories=5,
            date_range=date_range,
            total_size=1024,
            average_items_per_vendor=3.33,
            top_vendors=[("Vendor1", 5), ("Vendor2", 3)],
            top_categories=[("Category1", 4), ("Category2", 2)]
        )
        
        assert stats.total_items == 10
        assert stats.unique_vendors == 3
        assert stats.unique_categories == 5
        assert stats.total_size == 1024
        assert stats.average_items_per_vendor == 3.33
        assert len(stats.top_vendors) == 2
        assert len(stats.top_categories) == 2


class TestEnums:
    """Test cases for Enum classes."""

    def test_display_theme_enum(self):
        """Test DisplayTheme enum values."""
        assert DisplayTheme.DEFAULT.value == "default"
        assert DisplayTheme.DARK.value == "dark"
        assert DisplayTheme.LIGHT.value == "light"
        assert DisplayTheme.COLORFUL.value == "colorful"
        assert DisplayTheme.MINIMAL.value == "minimal"

    def test_table_format_enum(self):
        """Test TableFormat enum values."""
        assert TableFormat.FANCY_GRID.value == "fancy_grid"
        assert TableFormat.GRID.value == "grid"
        assert TableFormat.SIMPLE.value == "simple"
        assert TableFormat.PIPE.value == "pipe"
        assert TableFormat.HTML.value == "html" 