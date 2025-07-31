"""
Results Management System for CLI scraping results.

This module provides a comprehensive results management system with capabilities
for listing, viewing, exporting, and deleting scraping results with advanced
filtering, pagination, and sorting features.
"""

import os
import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama for cross-platform color support
init()


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"


class SortOrder(Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"


@dataclass
class ResultMetadata:
    """Metadata for a scraping result."""
    vendor: str
    category: str
    file_path: str
    product_count: int
    created_at: datetime
    file_size: int
    compressed: bool


@dataclass
class ResultStats:
    """Statistics about stored results."""
    total_results: int
    total_products: int
    vendors: Dict[str, int]
    categories: Dict[str, int]
    date_range: Tuple[datetime, datetime]
    total_size: int
    average_products_per_result: float


class ResultsManager:
    """
    Manages scraping results with operations for listing, viewing, exporting,
    and deleting results with advanced filtering and pagination.
    """

    def __init__(self, storage_dir: str = "data"):
        """
        Initialize the ResultsManager.
        
        Args:
            storage_dir: Directory containing stored results
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_results(
        self,
        vendor: Optional[str] = None,
        category: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        min_products: Optional[int] = None,
        max_products: Optional[int] = None,
        sort_by: str = "created_at",
        sort_order: SortOrder = SortOrder.DESC,
        page: int = 1,
        per_page: int = 20,
        table_format: str = "fancy_grid"
    ) -> Tuple[List[ResultMetadata], int]:
        """
        List available result sets with metadata and filtering.
        
        Args:
            vendor: Filter by vendor
            category: Filter by category
            date_from: Filter by start date
            date_to: Filter by end date
            min_products: Minimum product count
            max_products: Maximum product count
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            page: Page number (1-based)
            per_page: Results per page
            table_format: Tabulate format for display
            
        Returns:
            Tuple of (filtered results, total count)
        """
        all_results = self._scan_results()
        filtered_results = self._filter_results(
            all_results, vendor, category, date_from, date_to, 
            min_products, max_products
        )
        
        # Sort results
        reverse = sort_order == SortOrder.DESC
        filtered_results.sort(
            key=lambda x: getattr(x, sort_by, x.created_at),
            reverse=reverse
        )
        
        # Pagination
        total_count = len(filtered_results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = filtered_results[start_idx:end_idx]
        
        return paginated_results, total_count

    def view_result(self, file_path: str) -> Dict[str, Any]:
        """
        Display detailed information about a specific result.
        
        Args:
            file_path: Path to the result file
            
        Returns:
            Dictionary containing result details and products
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Result file not found: {file_path}")
        
        # Load the result data
        products = self._load_result_file(file_path)
        
        # Get file metadata
        stat = file_path.stat()
        created_at = datetime.fromtimestamp(stat.st_mtime)
        
        # Determine if file is compressed
        compressed = file_path.suffix == '.gz'
        
        # Extract vendor and category from path
        parts = file_path.parts
        vendor = "unknown"
        category = "unknown"
        
        # Try to extract vendor and category from path structure
        # Expected structure: .../vendor/category/filename.json
        if len(parts) >= 3:
            # Check if the path follows the expected structure
            potential_vendor = parts[-3]
            potential_category = parts[-2]
            
            # Simple heuristic: if these look like vendor/category names
            if str(potential_vendor).lower() in ['tesco', 'asda', 'costco']:
                vendor = potential_vendor
            if str(potential_category).lower() in ['groceries', 'electronics', 'clothing', 'home']:
                category = potential_category
        
        return {
            "file_path": str(file_path),
            "vendor": vendor,
            "category": category,
            "product_count": len(products),
            "created_at": created_at,
            "file_size": stat.st_size,
            "compressed": compressed,
            "products": products
        }

    def export_result(
        self,
        file_path: str,
        output_path: str,
        format: ExportFormat,
        fields: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        Export results to different formats.
        
        Args:
            file_path: Path to the result file
            output_path: Output file path
            format: Export format
            fields: Specific fields to export (None for all)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to the exported file
        """
        # Load the result data
        products = self._load_result_file(file_path)
        
        # Convert to DataFrame
        df = pd.DataFrame(products)
        
        # Select specific fields if requested
        if fields:
            available_fields = df.columns.tolist()
            valid_fields = [f for f in fields if f in available_fields]
            if valid_fields:
                df = df[valid_fields]
            else:
                raise ValueError(f"No valid fields found. Available: {available_fields}")
        
        # Export based on format
        if format == ExportFormat.JSON:
            df.to_json(output_path, orient='records', indent=2)
        elif format == ExportFormat.CSV:
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif format == ExportFormat.EXCEL:
            df.to_excel(output_path, index=False, sheet_name='Products')
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return output_path

    def delete_result(self, file_path: str, confirm: bool = False) -> bool:
        """
        Remove old or unwanted results.
        
        Args:
            file_path: Path to the result file
            confirm: Whether to skip confirmation
            
        Returns:
            True if deleted successfully
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Result file not found: {file_path}")
        
        if not confirm:
            # In a real CLI, this would prompt the user
            # For now, we'll just return False to indicate no deletion
            return False
        
        try:
            file_path.unlink()
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to delete file: {e}")

    def get_stats(self) -> ResultStats:
        """
        Show statistics about stored results.
        
        Returns:
            ResultStats object with comprehensive statistics
        """
        all_results = self._scan_results()
        
        if not all_results:
            return ResultStats(
                total_results=0,
                total_products=0,
                vendors={},
                categories={},
                date_range=(datetime.now(), datetime.now()),
                total_size=0,
                average_products_per_result=0.0
            )
        
        # Calculate statistics
        total_products = sum(r.product_count for r in all_results)
        total_size = sum(r.file_size for r in all_results)
        
        # Vendor and category counts
        vendors = {}
        categories = {}
        for result in all_results:
            vendors[result.vendor] = vendors.get(result.vendor, 0) + 1
            categories[result.category] = categories.get(result.category, 0) + 1
        
        # Date range
        dates = [r.created_at for r in all_results]
        date_range = (min(dates), max(dates))
        
        # Average products per result
        avg_products = total_products / len(all_results) if all_results else 0.0
        
        return ResultStats(
            total_results=len(all_results),
            total_products=total_products,
            vendors=vendors,
            categories=categories,
            date_range=date_range,
            total_size=total_size,
            average_products_per_result=avg_products
        )

    def display_results_table(
        self,
        results: List[ResultMetadata],
        total_count: int,
        page: int,
        per_page: int,
        table_format: str = "fancy_grid"
    ) -> None:
        """
        Display results in a formatted table.
        
        Args:
            results: List of result metadata
            total_count: Total number of results
            page: Current page number
            per_page: Results per page
            table_format: Tabulate format
        """
        if not results:
            print(f"{Fore.YELLOW}No results found matching the criteria.{Style.RESET_ALL}")
            return
        
        # Prepare table data
        table_data = []
        for result in results:
            # Format file size
            size_str = self._format_file_size(result.file_size)
            
            # Format date
            date_str = result.created_at.strftime("%Y-%m-%d %H:%M")
            
            # Format compression status
            compressed_str = "✓" if result.compressed else "✗"
            
            table_data.append([
                result.vendor,
                result.category,
                result.product_count,
                date_str,
                size_str,
                compressed_str,
                result.file_path
            ])
        
        # Create table
        headers = ["Vendor", "Category", "Products", "Created", "Size", "Compressed", "File Path"]
        table = tabulate(table_data, headers=headers, tablefmt=table_format)
        
        # Display pagination info
        start_idx = (page - 1) * per_page + 1
        end_idx = min(start_idx + per_page - 1, total_count)
        
        print(f"\n{Fore.CYAN}Results {start_idx}-{end_idx} of {total_count}:{Style.RESET_ALL}")
        print(table)
        
        # Show pagination controls if needed
        if total_count > per_page:
            total_pages = (total_count + per_page - 1) // per_page
            print(f"\n{Fore.BLUE}Page {page} of {total_pages}{Style.RESET_ALL}")

    def display_stats(self, stats: ResultStats) -> None:
        """
        Display statistics in a formatted way.
        
        Args:
            stats: ResultStats object
        """
        print(f"\n{Fore.CYAN}Results Statistics:{Style.RESET_ALL}")
        print("=" * 50)
        
        # Basic stats
        print(f"Total Results: {Fore.GREEN}{stats.total_results}{Style.RESET_ALL}")
        print(f"Total Products: {Fore.GREEN}{stats.total_products:,}{Style.RESET_ALL}")
        print(f"Average Products/Result: {Fore.GREEN}{stats.average_products_per_result:.1f}{Style.RESET_ALL}")
        print(f"Total Size: {Fore.GREEN}{self._format_file_size(stats.total_size)}{Style.RESET_ALL}")
        
        # Date range
        date_from, date_to = stats.date_range
        print(f"Date Range: {Fore.YELLOW}{date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}{Style.RESET_ALL}")
        
        # Vendor breakdown
        if stats.vendors:
            print(f"\n{Fore.CYAN}Vendor Breakdown:{Style.RESET_ALL}")
            vendor_data = [[vendor, count] for vendor, count in stats.vendors.items()]
            vendor_table = tabulate(vendor_data, headers=["Vendor", "Results"], tablefmt="grid")
            print(vendor_table)
        
        # Category breakdown
        if stats.categories:
            print(f"\n{Fore.CYAN}Category Breakdown:{Style.RESET_ALL}")
            category_data = [[category, count] for category, count in stats.categories.items()]
            category_table = tabulate(category_data, headers=["Category", "Results"], tablefmt="grid")
            print(category_table)

    def _scan_results(self) -> List[ResultMetadata]:
        """Scan the storage directory for result files."""
        results = []
        
        for file_path in self.storage_dir.rglob("*.json*"):
            if file_path.is_file():
                try:
                    # Get file metadata
                    stat = file_path.stat()
                    created_at = datetime.fromtimestamp(stat.st_mtime)
                    
                    # Determine if compressed
                    compressed = file_path.suffix == '.gz'
                    
                    # Extract vendor and category from path
                    parts = file_path.parts
                    vendor = "unknown"
                    category = "unknown"
                    
                    # Try to extract vendor and category from path structure
                    # Expected structure: .../vendor/category/filename.json
                    if len(parts) >= 3:
                        # Check if the path follows the expected structure
                        potential_vendor = parts[-3]
                        potential_category = parts[-2]
                        
                        # Simple heuristic: if these look like vendor/category names
                        if str(potential_vendor).lower() in ['tesco', 'asda', 'costco']:
                            vendor = potential_vendor
                        if str(potential_category).lower() in ['groceries', 'electronics', 'clothing', 'home']:
                            category = potential_category
                    
                    # Count products (load file to get count)
                    products = self._load_result_file(file_path)
                    product_count = len(products)
                    
                    results.append(ResultMetadata(
                        vendor=vendor,
                        category=category,
                        file_path=str(file_path),
                        product_count=product_count,
                        created_at=created_at,
                        file_size=stat.st_size,
                        compressed=compressed
                    ))
                except Exception as e:
                    # Skip files that can't be read
                    continue
        
        return results

    def _filter_results(
        self,
        results: List[ResultMetadata],
        vendor: Optional[str] = None,
        category: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        min_products: Optional[int] = None,
        max_products: Optional[int] = None
    ) -> List[ResultMetadata]:
        """Filter results based on criteria."""
        filtered = results
        
        if vendor:
            filtered = [r for r in filtered if str(r.vendor).lower() == str(vendor).lower()]
        
        if category:
            filtered = [r for r in filtered if str(r.category).lower() == str(category).lower()]
        
        if date_from:
            filtered = [r for r in filtered if r.created_at >= date_from]
        
        if date_to:
            filtered = [r for r in filtered if r.created_at <= date_to]
        
        if min_products is not None:
            filtered = [r for r in filtered if r.product_count >= min_products]
        
        if max_products is not None:
            filtered = [r for r in filtered if r.product_count <= max_products]
        
        return filtered

    def _load_result_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load a result file and return the products."""
        try:
            # Convert to Path if it's a string
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            if file_path.suffix == '.gz':
                # Compressed file
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # Regular JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, dict):
                return data.get('products', [])
            elif isinstance(data, list):
                return data
            else:
                return []
        except Exception as e:
            raise RuntimeError(f"Failed to load result file {file_path}: {e}")

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


def create_results_manager(storage_dir: str = "data") -> ResultsManager:
    """
    Factory function to create a ResultsManager instance.
    
    Args:
        storage_dir: Directory containing stored results
        
    Returns:
        ResultsManager instance
    """
    return ResultsManager(storage_dir) 