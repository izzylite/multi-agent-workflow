#!/usr/bin/env python3
"""
Scraping CLI - Main Entry Point

A command-line interface for web scraping using CrewAI and Browserbase.
Supports scraping from Tesco, Asda, and Costco UK platforms.
"""

import sys
from typing import List, Optional

from scraping_cli.parser import create_parser
from scraping_cli.config import create_config_manager
from scraping_cli.logging_config import create_logging_manager


def handle_scrape_command(args, config_manager, logging_manager) -> None:
    """Handle the scrape command."""
    try:
        # Parse configuration
        scrape_config = config_manager.parse_scrape_config(args)
        
        # Log command start
        logging_manager.log_command_start(
            "scrape",
            vendor=scrape_config.vendor.value,
            urls=scrape_config.urls,
            category=scrape_config.category,
            output=scrape_config.output,
            format=scrape_config.format.value
        )
        
        # TODO: Implement actual scraping logic
        print(f"Scraping {scrape_config.vendor.value} from {len(scrape_config.urls)} URL(s)...")
        print("This is a placeholder - actual scraping logic will be implemented in later tasks")
        
        # Log command end
        logging_manager.log_command_end("scrape", success=True)
        
    except ValueError as e:
        logging_manager.log_error(e, "Configuration error")
        sys.exit(1)


def handle_list_command(args, config_manager, logging_manager) -> None:
    """Handle the list command."""
    try:
        # Parse configuration
        list_config = config_manager.parse_list_config(args)
        
        # Log command start
        logging_manager.log_command_start("list", format=list_config.format.value)
        
        # TODO: Implement actual listing logic
        print("Listing previous scraping results...")
        print("This is a placeholder - actual listing logic will be implemented in later tasks")
        
        # Log command end
        logging_manager.log_command_end("list", success=True)
        
    except ValueError as e:
        logging_manager.log_error(e, "Configuration error")
        sys.exit(1)


def handle_export_command(args, config_manager, logging_manager) -> None:
    """Handle the export command."""
    try:
        # Parse configuration
        export_config = config_manager.parse_export_config(args)
        
        # Log command start
        logging_manager.log_command_start(
            "export",
            output=export_config.output,
            format=export_config.format.value
        )
        
        # TODO: Implement actual export logic
        print(f"Exporting results to {export_config.output}...")
        print("This is a placeholder - actual export logic will be implemented in later tasks")
        
        # Log command end
        logging_manager.log_command_end("export", success=True)
        
    except ValueError as e:
        logging_manager.log_error(e, "Configuration error")
        sys.exit(1)


def main() -> int:
    """Main entry point for the CLI application."""
    parser = create_parser()
    config_manager = create_config_manager()
    logging_manager = create_logging_manager()
    
    try:
        args = parser.parse_args()
        
        # Parse global configuration
        global_config = config_manager.parse_global_config(args)
        
        # Setup logging
        logging_manager.setup_from_verbose_flag(global_config.verbose)
        
        # Check if a command was provided
        if not args.command:
            parser.print_help()
            return 1
        
        # Route to appropriate command handler
        if args.command == 'scrape':
            handle_scrape_command(args, config_manager, logging_manager)
        elif args.command == 'list':
            handle_list_command(args, config_manager, logging_manager)
        elif args.command == 'export':
            handle_export_command(args, config_manager, logging_manager)
        else:
            logging_manager.log_error(Exception(f"Unknown command: {args.command}"))
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        if logging_manager.logger:
            logging_manager.logger.info("Operation cancelled by user")
        else:
            print("Operation cancelled by user")
        return 130
    except Exception as e:
        if logging_manager.logger:
            logging_manager.log_error(e, "An error occurred")
            if hasattr(args, 'verbose') and args.verbose:
                logging_manager.logger.exception("Full traceback:")
        else:
            print(f"An error occurred: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main()) 