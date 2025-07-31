"""
Enhanced Results Formatter for CLI scraping results.

This module provides a comprehensive results formatting system with capabilities
for rich CLI output, table formatting, color-coded display, summary statistics,
and customizable display templates.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, Back, init

# Initialize colorama for cross-platform color support
init()


class DisplayTheme(Enum):
    """Available display themes."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    COLORFUL = "colorful"
    MINIMAL = "minimal"


class TableFormat(Enum):
    """Available table formats."""
    FANCY_GRID = "fancy_grid"
    GRID = "grid"
    SIMPLE = "simple"
    PIPE = "pipe"
    ORGTBL = "orgtbl"
    RST = "rst"
    MEDIAWIKI = "mediawiki"
    HTML = "html"
    LATEX = "latex"
    LATEX_RAW = "latex_raw"
    LATEX_BOOKTABS = "latex_booktabs"
    TEXTILE = "textile"
    JIRA = "jira"


@dataclass
class DisplayConfig:
    """Configuration for display formatting."""
    theme: DisplayTheme = DisplayTheme.DEFAULT
    table_format: TableFormat = TableFormat.FANCY_GRID
    show_colors: bool = True
    show_summary: bool = True
    max_width: Optional[int] = None
    truncate_long_values: bool = True
    max_truncate_length: int = 50


@dataclass
class SummaryStats:
    """Summary statistics for display."""
    total_items: int
    unique_vendors: int
    unique_categories: int
    date_range: Tuple[datetime, datetime]
    total_size: int
    average_items_per_vendor: float
    top_vendors: List[Tuple[str, int]]
    top_categories: List[Tuple[str, int]]


class ResultsFormatter:
    """
    Enhanced results formatter with rich CLI output capabilities.
    
    Provides table formatting, color-coded output, summary statistics,
    and customizable display templates.
    """

    def __init__(self, config: Optional[DisplayConfig] = None):
        """
        Initialize the ResultsFormatter.
        
        Args:
            config: Display configuration
        """
        self.config = config or DisplayConfig()
        self._setup_theme_colors()
        self._templates = self._load_default_templates()

    def format_results_table(
        self,
        data: List[Dict[str, Any]],
        headers: Optional[List[str]] = None,
        title: Optional[str] = None,
        show_index: bool = False
    ) -> str:
        """
        Format data as a table with rich formatting.
        
        Args:
            data: List of dictionaries to format
            headers: Optional custom headers
            title: Optional table title
            show_index: Whether to show row indices
            
        Returns:
            Formatted table string
        """
        if not data:
            return self._format_empty_message("No data to display")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Apply truncation if configured
        if self.config.truncate_long_values:
            df = self._truncate_dataframe(df)
        
        # Prepare headers
        if headers:
            df.columns = headers[:len(df.columns)]
        
        # Format the table
        table_str = tabulate(
            df,
            headers='keys',
            tablefmt=self.config.table_format.value,
            showindex=show_index,
            maxcolwidths=self.config.max_width
        )
        
        # Add title if provided
        if title:
            title_str = self._format_title(title)
            table_str = f"{title_str}\n{table_str}"
        
        return table_str

    def format_summary_stats(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None
    ) -> str:
        """
        Generate and format summary statistics.
        
        Args:
            data: Data to analyze
            title: Optional section title
            
        Returns:
            Formatted summary statistics
        """
        if not data:
            return self._format_empty_message("No data for statistics")
        
        # Calculate statistics
        stats = self._calculate_summary_stats(data)
        
        # Format the summary
        lines = []
        
        if title:
            lines.append(self._format_title(title))
        
        # Basic stats
        lines.append(f"{self._color('CYAN', 'Summary Statistics:')}")
        lines.append("=" * 50)
        lines.append(f"Total Items: {self._color('GREEN', str(stats.total_items))}")
        lines.append(f"Unique Vendors: {self._color('GREEN', str(stats.unique_vendors))}")
        lines.append(f"Unique Categories: {self._color('GREEN', str(stats.unique_categories))}")
        
        # Date range
        date_from, date_to = stats.date_range
        lines.append(f"Date Range: {self._color('YELLOW', f'{date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}')}")
        
        # File size
        size_str = self._format_file_size(stats.total_size)
        lines.append(f"Total Size: {self._color('GREEN', size_str)}")
        
        # Average items per vendor
        lines.append(f"Average Items/Vendor: {self._color('GREEN', f'{stats.average_items_per_vendor:.1f}')}")
        
        # Top vendors
        if stats.top_vendors:
            lines.append(f"\n{self._color('CYAN', 'Top Vendors:')}")
            vendor_data = [[vendor, count] for vendor, count in stats.top_vendors[:5]]
            vendor_table = tabulate(vendor_data, headers=["Vendor", "Count"], tablefmt="grid")
            lines.append(vendor_table)
        
        # Top categories
        if stats.top_categories:
            lines.append(f"\n{self._color('CYAN', 'Top Categories:')}")
            category_data = [[category, count] for category, count in stats.top_categories[:5]]
            category_table = tabulate(category_data, headers=["Category", "Count"], tablefmt="grid")
            lines.append(category_table)
        
        return "\n".join(lines)

    def format_custom_template(
        self,
        template_name: str,
        data: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        Format data using a custom template.
        
        Args:
            template_name: Name of the template to use
            data: Data to format
            **kwargs: Additional template parameters
            
        Returns:
            Formatted output string
        """
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self._templates[template_name]
        return template(data, **kwargs)

    def add_custom_template(
        self,
        name: str,
        template_func: Callable[[List[Dict[str, Any]], Dict[str, Any]], str]
    ) -> None:
        """
        Add a custom template function.
        
        Args:
            name: Template name
            template_func: Template function that takes data and kwargs, returns formatted string
        """
        self._templates[name] = template_func

    def set_theme(self, theme: DisplayTheme) -> None:
        """
        Set the display theme.
        
        Args:
            theme: Theme to apply
        """
        self.config.theme = theme
        self._setup_theme_colors()

    def set_table_format(self, table_format: TableFormat) -> None:
        """
        Set the table format.
        
        Args:
            table_format: Table format to use
        """
        self.config.table_format = table_format

    def _setup_theme_colors(self) -> None:
        """Setup color scheme based on current theme."""
        if not self.config.show_colors:
            self._colors = {
                'CYAN': '',
                'GREEN': '',
                'YELLOW': '',
                'RED': '',
                'BLUE': '',
                'MAGENTA': '',
                'RESET': ''
            }
            return
        
        if self.config.theme == DisplayTheme.DARK:
            self._colors = {
                'CYAN': Fore.CYAN,
                'GREEN': Fore.GREEN,
                'YELLOW': Fore.YELLOW,
                'RED': Fore.RED,
                'BLUE': Fore.BLUE,
                'MAGENTA': Fore.MAGENTA,
                'RESET': Style.RESET_ALL
            }
        elif self.config.theme == DisplayTheme.LIGHT:
            self._colors = {
                'CYAN': Fore.BLUE,
                'GREEN': Fore.GREEN,
                'YELLOW': Fore.YELLOW,
                'RED': Fore.RED,
                'BLUE': Fore.CYAN,
                'MAGENTA': Fore.MAGENTA,
                'RESET': Style.RESET_ALL
            }
        elif self.config.theme == DisplayTheme.COLORFUL:
            self._colors = {
                'CYAN': Fore.CYAN + Style.BRIGHT,
                'GREEN': Fore.GREEN + Style.BRIGHT,
                'YELLOW': Fore.YELLOW + Style.BRIGHT,
                'RED': Fore.RED + Style.BRIGHT,
                'BLUE': Fore.BLUE + Style.BRIGHT,
                'MAGENTA': Fore.MAGENTA + Style.BRIGHT,
                'RESET': Style.RESET_ALL
            }
        elif self.config.theme == DisplayTheme.MINIMAL:
            self._colors = {
                'CYAN': '',
                'GREEN': '',
                'YELLOW': '',
                'RED': '',
                'BLUE': '',
                'MAGENTA': '',
                'RESET': ''
            }
        else:  # DEFAULT
            self._colors = {
                'CYAN': Fore.CYAN,
                'GREEN': Fore.GREEN,
                'YELLOW': Fore.YELLOW,
                'RED': Fore.RED,
                'BLUE': Fore.BLUE,
                'MAGENTA': Fore.MAGENTA,
                'RESET': Style.RESET_ALL
            }

    def _load_default_templates(self) -> Dict[str, Callable]:
        """Load default template functions."""
        templates = {}
        
        # Compact template
        def compact_template(data: List[Dict[str, Any]], **kwargs) -> str:
            if not data:
                return self._format_empty_message("No data")
            
            # Show only key fields in a compact format
            compact_data = []
            for item in data:
                compact_item = {
                    'Vendor': item.get('vendor', 'N/A'),
                    'Category': item.get('category', 'N/A'),
                    'Count': item.get('product_count', 0),
                    'Date': item.get('created_at', 'N/A')
                }
                compact_data.append(compact_item)
            
            return self.format_results_table(compact_data, title="Compact View")
        
        # Detailed template
        def detailed_template(data: List[Dict[str, Any]], **kwargs) -> str:
            if not data:
                return self._format_empty_message("No data")
            
            # Show all fields in a detailed format
            detailed_data = []
            for item in data:
                detailed_item = {
                    'Vendor': item.get('vendor', 'N/A'),
                    'Category': item.get('category', 'N/A'),
                    'Product Count': item.get('product_count', 0),
                    'Created At': item.get('created_at', 'N/A'),
                    'File Size': self._format_file_size(item.get('file_size', 0)),
                    'Compressed': '✓' if item.get('compressed', False) else '✗',
                    'File Path': item.get('file_path', 'N/A')
                }
                detailed_data.append(detailed_item)
            
            return self.format_results_table(detailed_data, title="Detailed View")
        
        # Summary template
        def summary_template(data: List[Dict[str, Any]], **kwargs) -> str:
            return self.format_summary_stats(data, title="Data Summary")
        
        templates['compact'] = compact_template
        templates['detailed'] = detailed_template
        templates['summary'] = summary_template
        
        return templates

    def _calculate_summary_stats(self, data: List[Dict[str, Any]]) -> SummaryStats:
        """Calculate summary statistics from data."""
        if not data:
            return SummaryStats(
                total_items=0,
                unique_vendors=0,
                unique_categories=0,
                date_range=(datetime.now(), datetime.now()),
                total_size=0,
                average_items_per_vendor=0.0,
                top_vendors=[],
                top_categories=[]
            )
        
        # Basic counts
        total_items = len(data)
        vendors = [item.get('vendor', 'Unknown') for item in data]
        categories = [item.get('category', 'Unknown') for item in data]
        unique_vendors = len(set(vendors))
        unique_categories = len(set(categories))
        
        # Date range
        dates = []
        for item in data:
            created_at = item.get('created_at')
            if isinstance(created_at, str):
                try:
                    dates.append(datetime.fromisoformat(created_at.replace('Z', '+00:00')))
                except ValueError:
                    pass
            elif isinstance(created_at, datetime):
                dates.append(created_at)
        
        date_range = (min(dates), max(dates)) if dates else (datetime.now(), datetime.now())
        
        # File sizes
        total_size = sum(item.get('file_size', 0) for item in data)
        
        # Vendor statistics
        vendor_counts = {}
        for vendor in vendors:
            vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
        
        top_vendors = sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True)
        average_items_per_vendor = total_items / unique_vendors if unique_vendors > 0 else 0
        
        # Category statistics
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        return SummaryStats(
            total_items=total_items,
            unique_vendors=unique_vendors,
            unique_categories=unique_categories,
            date_range=date_range,
            total_size=total_size,
            average_items_per_vendor=average_items_per_vendor,
            top_vendors=top_vendors,
            top_categories=top_categories
        )

    def _truncate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Truncate long values in DataFrame."""
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).apply(
                    lambda x: x[:self.config.max_truncate_length] + '...' 
                    if len(str(x)) > self.config.max_truncate_length else x
                )
        return df

    def _format_title(self, title: str) -> str:
        """Format a section title."""
        return f"\n{self._color('CYAN', title)}\n{'-' * len(title)}"

    def _format_empty_message(self, message: str) -> str:
        """Format an empty state message."""
        return f"{self._color('YELLOW', message)}"

    def _color(self, color_name: str, text: str) -> str:
        """Apply color to text."""
        color_code = self._colors.get(color_name, '')
        return f"{color_code}{text}{self._colors['RESET']}"

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"


def create_results_formatter(config: Optional[DisplayConfig] = None) -> ResultsFormatter:
    """
    Create a ResultsFormatter instance.
    
    Args:
        config: Optional display configuration
        
    Returns:
        ResultsFormatter instance
    """
    return ResultsFormatter(config) 