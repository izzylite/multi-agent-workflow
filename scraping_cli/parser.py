"""
Command Parser Module

Handles all command-line argument parsing for the scraping CLI.
"""

import argparse
from typing import Optional, List


class CommandParser:
    """Handles command-line argument parsing for the scraping CLI."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Web scraping CLI using CrewAI and Browserbase",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python scraping_cli.py scrape --vendor tesco --urls "https://www.tesco.com/groceries"
  python scraping_cli.py list --format json
  python scraping_cli.py export --output results.csv
            """
        )
        
        # Global options
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        # Create subparsers for different commands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands'
        )
        
        # Add all command parsers
        self._add_scrape_parser(subparsers)
        self._add_list_parser(subparsers)
        self._add_export_parser(subparsers)
        self._add_view_parser(subparsers)
        self._add_delete_parser(subparsers)
        self._add_stats_parser(subparsers)
        
        return parser
    
    def _add_scrape_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add the scrape command parser."""
        scrape_parser = subparsers.add_parser(
            'scrape',
            help='Scrape product data from specified URLs'
        )
        scrape_parser.add_argument(
            '--vendor', '-V',
            required=True,
            choices=['tesco', 'asda', 'costco'],
            help='Target vendor platform'
        )
        # URL input group - only one method should be used
        url_group = scrape_parser.add_mutually_exclusive_group(required=True)
        url_group.add_argument(
            '--urls', '-u',
            nargs='+',
            help='URLs to scrape (space-separated)'
        )
        url_group.add_argument(
            '--url-file', '-F',
            help='File containing URLs (one per line)'
        )
        url_group.add_argument(
            '--urls-from-stdin',
            action='store_true',
            help='Read URLs from standard input (one per line)'
        )
        scrape_parser.add_argument(
            '--category', '-c',
            help='Product category for specialized scraping'
        )
        scrape_parser.add_argument(
            '--output', '-o',
            help='Output file path for results'
        )
        scrape_parser.add_argument(
            '--format', '-f',
            choices=['json', 'csv'],
            default='json',
            help='Output format (default: json)'
        )
    
    def _add_list_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add the list command parser."""
        list_parser = subparsers.add_parser(
            'list',
            help='List previous scraping results'
        )
        list_parser.add_argument(
            '--format', '-f',
            choices=['json', 'csv', 'table'],
            default='table',
            help='Output format (default: table)'
        )
        list_parser.add_argument(
            '--vendor', '-V',
            help='Filter by vendor'
        )
        list_parser.add_argument(
            '--category', '-c',
            help='Filter by category'
        )
        list_parser.add_argument(
            '--date-from',
            help='Filter by start date (YYYY-MM-DD)'
        )
        list_parser.add_argument(
            '--date-to',
            help='Filter by end date (YYYY-MM-DD)'
        )
        list_parser.add_argument(
            '--min-products',
            type=int,
            help='Minimum product count'
        )
        list_parser.add_argument(
            '--max-products',
            type=int,
            help='Maximum product count'
        )
        list_parser.add_argument(
            '--sort-by',
            choices=['vendor', 'category', 'product_count', 'created_at', 'file_size'],
            default='created_at',
            help='Sort by field (default: created_at)'
        )
        list_parser.add_argument(
            '--sort-order',
            choices=['asc', 'desc'],
            default='desc',
            help='Sort order (default: desc)'
        )
        list_parser.add_argument(
            '--page', '-p',
            type=int,
            default=1,
            help='Page number (default: 1)'
        )
        list_parser.add_argument(
            '--per-page',
            type=int,
            default=20,
            help='Results per page (default: 20)'
        )
    
    def _add_export_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add the export command parser."""
        export_parser = subparsers.add_parser(
            'export',
            help='Export results to file'
        )
        export_parser.add_argument(
            'file_path',
            help='Path to the result file to export'
        )
        export_parser.add_argument(
            '--output', '-o',
            required=True,
            help='Output file path'
        )
        export_parser.add_argument(
            '--format', '-f',
            choices=['json', 'csv', 'excel'],
            default='json',
            help='Output format (default: json)'
        )
        export_parser.add_argument(
            '--fields',
            nargs='+',
            help='Specific fields to export (default: all)'
        )
    
    def _add_view_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add the view command parser."""
        view_parser = subparsers.add_parser(
            'view',
            help='View detailed information about a specific result'
        )
        view_parser.add_argument(
            'file_path',
            help='Path to the result file to view'
        )
        view_parser.add_argument(
            '--fields', '-f',
            nargs='+',
            help='Specific fields to display (default: all)'
        )
    
    def _add_delete_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add the delete command parser."""
        delete_parser = subparsers.add_parser(
            'delete',
            help='Delete old or unwanted results'
        )
        delete_parser.add_argument(
            'file_path',
            help='Path to the result file to delete'
        )
        delete_parser.add_argument(
            '--confirm', '-y',
            action='store_true',
            help='Skip confirmation prompt'
        )
    
    def _add_stats_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add the stats command parser."""
        stats_parser = subparsers.add_parser(
            'stats',
            help='Show statistics about stored results'
        )
        stats_parser.add_argument(
            '--format', '-f',
            choices=['table', 'json'],
            default='table',
            help='Output format (default: table)'
        )
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        return self.parser.parse_args(args)
    
    def print_help(self) -> None:
        """Print the help message."""
        self.parser.print_help()
    
    def print_command_help(self, command: str) -> None:
        """Print help for a specific command."""
        if command in self.parser._subparsers._group_actions[0].choices:
            self.parser._subparsers._group_actions[0].choices[command].print_help()
        else:
            print(f"Unknown command: {command}")


def create_parser() -> CommandParser:
    """Create and return a new command parser instance."""
    return CommandParser() 