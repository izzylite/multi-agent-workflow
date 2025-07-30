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
    
    def _add_export_parser(self, subparsers: argparse._SubParsersAction) -> None:
        """Add the export command parser."""
        export_parser = subparsers.add_parser(
            'export',
            help='Export results to file'
        )
        export_parser.add_argument(
            '--output', '-o',
            required=True,
            help='Output file path'
        )
        export_parser.add_argument(
            '--format', '-f',
            choices=['json', 'csv'],
            default='json',
            help='Output format (default: json)'
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