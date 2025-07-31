#!/usr/bin/env python3
"""
Scraping CLI - Main Entry Point

A command-line interface for web scraping using CrewAI and Browserbase.
Supports scraping from Tesco, Asda, and Costco UK platforms.
"""

import sys
import json
import csv
from datetime import datetime
from typing import List, Optional

from scraping_cli.parser import create_parser
from scraping_cli.config import create_config_manager
from scraping_cli.logging_config import create_logging_manager
from scraping_cli.url_input import create_url_input_handler
from scraping_cli.url_processor import create_url_processor
from scraping_cli.progress_monitor import create_progress_monitor, StatusLevel
from scraping_cli.results_manager import create_results_manager, ExportFormat, SortOrder


def handle_scrape_command(args, config_manager, logging_manager) -> None:
    """Handle the scrape command."""
    try:
        # Create URL input handler and processor
        url_input_handler = create_url_input_handler()
        url_processor = create_url_processor()
        
        # Get URLs from input method
        raw_urls = url_input_handler.validate_url_input(args)
        input_method = url_input_handler.get_input_method_description(args)
        
        # Process and validate URLs
        processed_urls = url_processor.process_urls(raw_urls, expected_vendor=args.vendor)
        
        # Remove duplicates
        unique_urls = url_processor.deduplicate_urls(processed_urls)
        
        # Parse configuration
        scrape_config = config_manager.parse_scrape_config(args)
        
        # Log command start with URL information
        logging_manager.log_command_start(
            "scrape",
            vendor=scrape_config.vendor.value,
            input_method=input_method,
            total_urls=len(raw_urls),
            unique_urls=len(unique_urls),
            category=scrape_config.category,
            output=scrape_config.output,
            format=scrape_config.format.value
        )
        
        # Log URL processing results
        logging_manager.log_debug(f"Input method: {input_method}")
        logging_manager.log_debug(f"Raw URLs: {len(raw_urls)}")
        logging_manager.log_debug(f"Unique URLs: {len(unique_urls)}")
        
        # Create progress monitor
        progress_mode = "debug" if args.verbose else "simple"
        progress_monitor = create_progress_monitor(
            total_tasks=len(unique_urls),
            mode=progress_mode,
            disable=not sys.stdout.isatty()
        )
        
        # Start scraping with progress monitoring
        with progress_monitor:
            progress_monitor.log(f"Starting scraping for {scrape_config.vendor.value}", StatusLevel.INFO)
            progress_monitor.log(f"Processing {len(unique_urls)} unique URLs", StatusLevel.INFO)
            progress_monitor.log(f"Input method: {input_method}", StatusLevel.INFO)
            
            # TODO: Implement actual scraping logic with progress updates
            # For now, simulate progress
            for i, url in enumerate(unique_urls):
                progress_monitor.update_progress(
                    n=1,
                    task_name=f"Scraping URL {i+1}/{len(unique_urls)}",
                    agent_status={"scraper": "working"},
                    browser_status={f"session_{i}": "active"}
                )
                
                # Simulate some processing time
                import time
                time.sleep(0.1)
                
                # Simulate occasional errors
                if i == 2:  # Simulate error on 3rd URL
                    progress_monitor.add_error(f"Failed to scrape URL {i+1}: Network timeout")
            
            progress_monitor.log("Scraping completed", StatusLevel.SUCCESS)
        
        # Log command end
        logging_manager.log_command_end("scrape", success=True)
        
    except ValueError as e:
        logging_manager.log_error(e, "Configuration error")
        sys.exit(1)
    except FileNotFoundError as e:
        logging_manager.log_error(e, "File not found")
        sys.exit(1)
    except Exception as e:
        logging_manager.log_error(e, "URL processing error")
        sys.exit(1)


def handle_list_command(args, config_manager, logging_manager) -> None:
    """Handle the list command."""
    try:
        # Parse configuration
        list_config = config_manager.parse_list_config(args)
        
        # Log command start
        logging_manager.log_command_start("list", format=list_config.format.value)
        
        # Create results manager
        results_manager = create_results_manager()
        
        # Parse date filters
        date_from = None
        date_to = None
        if args.date_from:
            date_from = datetime.strptime(args.date_from, "%Y-%m-%d")
        if args.date_to:
            date_to = datetime.strptime(args.date_to, "%Y-%m-%d")
        
        # Parse sort order
        sort_order = SortOrder.DESC if args.sort_order == "desc" else SortOrder.ASC
        
        # Create progress monitor for listing operation
        progress_monitor = create_progress_monitor(
            total_tasks=1,  # Single task for listing
            mode="simple",
            disable=not sys.stdout.isatty()
        )
        
        with progress_monitor:
            progress_monitor.log("Listing previous scraping results...", StatusLevel.INFO)
            
            # Get results with filtering and pagination
            results, total_count = results_manager.list_results(
                vendor=args.vendor,
                category=args.category,
                date_from=date_from,
                date_to=date_to,
                min_products=args.min_products,
                max_products=args.max_products,
                sort_by=args.sort_by,
                sort_order=sort_order,
                page=args.page,
                per_page=args.per_page
            )
            
            progress_monitor.update_progress(
                n=1,
                task_name="Loading results",
                agent_status={"lister": "working"}
            )
            
            # Display results
            if args.format == "table":
                results_manager.display_results_table(
                    results, total_count, args.page, args.per_page
                )
            else:
                # For JSON/CSV format, just print the data
                data = [
                    {
                        "vendor": r.vendor,
                        "category": r.category,
                        "product_count": r.product_count,
                        "created_at": r.created_at.isoformat(),
                        "file_size": r.file_size,
                        "compressed": r.compressed,
                        "file_path": r.file_path
                    }
                    for r in results
                ]
                if args.format == "json":
                    print(json.dumps(data, indent=2))
                else:  # CSV
                    writer = csv.DictWriter(sys.stdout, fieldnames=data[0].keys() if data else [])
                    writer.writeheader()
                    writer.writerows(data)
            
            progress_monitor.log("Listing completed", StatusLevel.SUCCESS)
        
        # Log command end
        logging_manager.log_command_end("list", success=True)
        
    except ValueError as e:
        logging_manager.log_error(e, "Configuration error")
        sys.exit(1)


def handle_export_command(args, config_manager, logging_manager) -> None:
    """Handle the export command."""
    try:
        # Log command start
        logging_manager.log_command_start(
            "export",
            output=args.output,
            format=args.format
        )
        
        # Create results manager
        results_manager = create_results_manager()
        
        # Parse export format
        export_format = ExportFormat.JSON
        if args.format == "csv":
            export_format = ExportFormat.CSV
        elif args.format == "excel":
            export_format = ExportFormat.EXCEL
        
        # Create progress monitor
        progress_monitor = create_progress_monitor(
            total_tasks=1,
            mode="simple",
            disable=not sys.stdout.isatty()
        )
        
        with progress_monitor:
            progress_monitor.log(f"Exporting results to {args.output}...", StatusLevel.INFO)
            
            # Export the result
            result_path = results_manager.export_result(
                args.file_path,
                args.output,
                export_format,
                fields=args.fields
            )
            
            progress_monitor.update_progress(
                n=1,
                task_name="Exporting",
                agent_status={"exporter": "working"}
            )
            
            progress_monitor.log(f"Export completed: {result_path}", StatusLevel.SUCCESS)
        
        # Log command end
        logging_manager.log_command_end("export", success=True)
        
    except FileNotFoundError as e:
        logging_manager.log_error(e, "File not found")
        sys.exit(1)
    except ValueError as e:
        logging_manager.log_error(e, "Configuration error")
        sys.exit(1)
    except Exception as e:
        logging_manager.log_error(e, "Export error")
        sys.exit(1)


def handle_view_command(args, config_manager, logging_manager) -> None:
    """Handle the view command."""
    try:
        # Log command start
        logging_manager.log_command_start("view", file_path=args.file_path)
        
        # Create results manager
        results_manager = create_results_manager()
        
        # Create progress monitor
        progress_monitor = create_progress_monitor(
            total_tasks=1,
            mode="simple",
            disable=not sys.stdout.isatty()
        )
        
        with progress_monitor:
            progress_monitor.log(f"Loading result file: {args.file_path}", StatusLevel.INFO)
            
            # View the result
            result = results_manager.view_result(args.file_path)
            
            progress_monitor.update_progress(
                n=1,
                task_name="Loading result",
                agent_status={"viewer": "working"}
            )
            
            # Display result details
            print(f"\nResult Details:")
            print(f"File Path: {result['file_path']}")
            print(f"Vendor: {result['vendor']}")
            print(f"Category: {result['category']}")
            print(f"Product Count: {result['product_count']}")
            print(f"Created At: {result['created_at']}")
            print(f"File Size: {results_manager._format_file_size(result['file_size'])}")
            print(f"Compressed: {'Yes' if result['compressed'] else 'No'}")
            
            # Display products if fields are specified
            if args.fields:
                print(f"\nProducts (showing fields: {', '.join(args.fields)}):")
                for i, product in enumerate(result['products'][:10]):  # Show first 10
                    filtered_product = {k: v for k, v in product.items() if k in args.fields}
                    print(f"  {i+1}. {filtered_product}")
                if len(result['products']) > 10:
                    print(f"  ... and {len(result['products']) - 10} more products")
            else:
                print(f"\nTotal Products: {result['product_count']}")
            
            progress_monitor.log("View completed", StatusLevel.SUCCESS)
        
        # Log command end
        logging_manager.log_command_end("view", success=True)
        
    except FileNotFoundError as e:
        logging_manager.log_error(e, "File not found")
        sys.exit(1)
    except Exception as e:
        logging_manager.log_error(e, "View error")
        sys.exit(1)


def handle_delete_command(args, config_manager, logging_manager) -> None:
    """Handle the delete command."""
    try:
        # Log command start
        logging_manager.log_command_start("delete", file_path=args.file_path)
        
        # Create results manager
        results_manager = create_results_manager()
        
        # Create progress monitor
        progress_monitor = create_progress_monitor(
            total_tasks=1,
            mode="simple",
            disable=not sys.stdout.isatty()
        )
        
        with progress_monitor:
            progress_monitor.log(f"Deleting result file: {args.file_path}", StatusLevel.INFO)
            
            # Delete the result
            deleted = results_manager.delete_result(args.file_path, confirm=args.confirm)
            
            progress_monitor.update_progress(
                n=1,
                task_name="Deleting",
                agent_status={"deleter": "working"}
            )
            
            if deleted:
                progress_monitor.log("File deleted successfully", StatusLevel.SUCCESS)
            else:
                progress_monitor.log("Deletion cancelled (use --confirm to skip prompt)", StatusLevel.WARNING)
            
            progress_monitor.log("Delete operation completed", StatusLevel.SUCCESS)
        
        # Log command end
        logging_manager.log_command_end("delete", success=True)
        
    except FileNotFoundError as e:
        logging_manager.log_error(e, "File not found")
        sys.exit(1)
    except Exception as e:
        logging_manager.log_error(e, "Delete error")
        sys.exit(1)


def handle_stats_command(args, config_manager, logging_manager) -> None:
    """Handle the stats command."""
    try:
        # Log command start
        logging_manager.log_command_start("stats", format=args.format)
        
        # Create results manager
        results_manager = create_results_manager()
        
        # Create progress monitor
        progress_monitor = create_progress_monitor(
            total_tasks=1,
            mode="simple",
            disable=not sys.stdout.isatty()
        )
        
        with progress_monitor:
            progress_monitor.log("Calculating statistics...", StatusLevel.INFO)
            
            # Get statistics
            stats = results_manager.get_stats()
            
            progress_monitor.update_progress(
                n=1,
                task_name="Calculating stats",
                agent_status={"stats": "working"}
            )
            
            # Display statistics
            if args.format == "table":
                results_manager.display_stats(stats)
            else:  # JSON format
                stats_data = {
                    "total_results": stats.total_results,
                    "total_products": stats.total_products,
                    "vendors": stats.vendors,
                    "categories": stats.categories,
                    "date_range": [
                        stats.date_range[0].isoformat(),
                        stats.date_range[1].isoformat()
                    ],
                    "total_size": stats.total_size,
                    "average_products_per_result": stats.average_products_per_result
                }
                print(json.dumps(stats_data, indent=2))
            
            progress_monitor.log("Statistics calculated", StatusLevel.SUCCESS)
        
        # Log command end
        logging_manager.log_command_end("stats", success=True)
        
    except Exception as e:
        logging_manager.log_error(e, "Stats error")
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
        elif args.command == 'view':
            handle_view_command(args, config_manager, logging_manager)
        elif args.command == 'delete':
            handle_delete_command(args, config_manager, logging_manager)
        elif args.command == 'stats':
            handle_stats_command(args, config_manager, logging_manager)
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