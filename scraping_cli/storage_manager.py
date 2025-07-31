"""
File-Based Data Storage System

Provides a comprehensive storage and retrieval system for scraped product data
with automatic organization, compression, and error handling.
"""

import os
import json
import gzip
import shutil
import logging
import threading
import tempfile
from typing import List, Optional, Dict, Any, Union, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager
import hashlib
import pickle

from pydantic import BaseModel, Field


@dataclass
class StorageConfig:
    """Configuration for the storage system."""
    base_directory: str = "data"
    compression_enabled: bool = True
    compression_threshold: int = 1024  # bytes
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    file_rotation_days: int = 30
    incremental_save_interval: int = 100  # save every N records
    backup_enabled: bool = True
    backup_directory: str = "data/backups"
    lock_timeout: int = 30  # seconds


@dataclass
class ProductData:
    """Structure for product data."""
    vendor: str
    category: str
    product_id: str
    title: str
    price: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    scraped_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.scraped_at is None:
            self.scraped_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class StorageQuery(BaseModel):
    """Query interface for retrieving stored data."""
    vendor: Optional[str] = None
    category: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    product_id: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None


class FileLock:
    """Simple file-based locking mechanism."""
    
    def __init__(self, lock_file: str, timeout: int = 30):
        self.lock_file = lock_file
        self.timeout = timeout
        self.lock_acquired = False
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def acquire(self):
        """Acquire the file lock with timeout."""
        start_time = datetime.now()
        
        while datetime.now() - start_time < timedelta(seconds=self.timeout):
            try:
                # Create lock file atomically
                with open(self.lock_file, 'x') as f:
                    f.write(str(os.getpid()))
                self.lock_acquired = True
                return
            except FileExistsError:
                # Check if lock is stale (process no longer running)
                try:
                    with open(self.lock_file, 'r') as f:
                        pid = int(f.read().strip())
                    # Check if process is still running
                    os.kill(pid, 0)
                    # Process is running, wait a bit
                    time.sleep(0.1)
                except (ValueError, OSError, ProcessLookupError):
                    # Lock is stale, remove it
                    try:
                        os.remove(self.lock_file)
                    except FileNotFoundError:
                        pass
                    continue
        
        raise TimeoutError(f"Could not acquire lock {self.lock_file} within {self.timeout} seconds")
    
    def release(self):
        """Release the file lock."""
        if self.lock_acquired:
            try:
                os.remove(self.lock_file)
            except FileNotFoundError:
                pass
            self.lock_acquired = False


class StorageManager:
    """
    File-based storage manager for scraped product data.
    
    Features:
    - Automatic directory organization by vendor/category/date
    - JSON serialization with custom encoders/decoders
    - Incremental saves to prevent data loss
    - Data compression for large datasets
    - File locking to prevent concurrent write issues
    - Query interface for retrieving stored results
    - Automatic file rotation and cleanup
    """
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self.logger = logging.getLogger(f"{__name__}.StorageManager")
        
        # Ensure base directory exists
        self.base_path = Path(self.config.base_directory)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize backup directory
        if self.config.backup_enabled:
            self.backup_path = Path(self.config.backup_directory)
            self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for incremental saves
        self._thread_local = threading.local()
        self._initialize_thread_local()
        
        # Statistics
        self.stats = {
            'total_saves': 0,
            'total_loads': 0,
            'compressed_files': 0,
            'backups_created': 0,
            'errors': 0
        }
    
    def _initialize_thread_local(self):
        """Initialize thread-local storage."""
        if not hasattr(self._thread_local, 'pending_data'):
            self._thread_local.pending_data = []
            self._thread_local.save_counter = 0
    
    def _get_directory_path(self, vendor: str, category: str, date: datetime) -> Path:
        """Get the directory path for organizing data."""
        date_str = date.strftime("%Y-%m-%d")
        return self.base_path / vendor / category / date_str
    
    def _get_file_path(self, vendor: str, category: str, date: datetime, 
                       filename: str = None) -> Path:
        """Get the file path for storing data."""
        directory = self._get_directory_path(vendor, category, date)
        directory.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = date.strftime("%H-%M-%S")
            filename = f"products_{timestamp}.json"
        
        return directory / filename
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if enabled and above threshold."""
        if not self.config.compression_enabled or len(data) < self.config.compression_threshold:
            return data
        
        compressed = gzip.compress(data)
        self.stats['compressed_files'] += 1
        self.logger.debug(f"Compressed data from {len(data)} to {len(compressed)} bytes")
        return compressed
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if it's compressed."""
        try:
            return gzip.decompress(data)
        except OSError:
            # Not compressed, return as-is
            return data
    
    def _serialize_data(self, data: List[ProductData]) -> bytes:
        """Serialize data to JSON with custom encoder."""
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        json_data = json.dumps([asdict(item) for item in data], cls=DateTimeEncoder, indent=2)
        return json_data.encode('utf-8')
    
    def _deserialize_data(self, data: bytes) -> List[ProductData]:
        """Deserialize data from JSON with custom decoder."""
        json_str = data.decode('utf-8')
        raw_data = json.loads(json_str)
        
        products = []
        for item in raw_data:
            # Convert ISO string back to datetime
            if 'scraped_at' in item and isinstance(item['scraped_at'], str):
                item['scraped_at'] = datetime.fromisoformat(item['scraped_at'])
            products.append(ProductData(**item))
        
        return products
    
    def _atomic_write(self, file_path: Path, data: bytes) -> None:
        """Write data atomically using temporary file."""
        temp_file = file_path.with_suffix('.tmp')
        
        try:
            # Write to temporary file
            with open(temp_file, 'wb') as f:
                f.write(data)
            
            # Atomic move
            temp_file.replace(file_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def _create_backup(self, file_path: Path) -> None:
        """Create a backup of the file."""
        if not self.config.backup_enabled:
            return
        
        try:
            backup_file = self.backup_path / f"{file_path.name}.backup"
            shutil.copy2(file_path, backup_file)
            self.stats['backups_created'] += 1
            self.logger.debug(f"Created backup: {backup_file}")
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
    
    @contextmanager
    def _file_lock(self, file_path: Path):
        """Context manager for file locking."""
        lock_file = file_path.with_suffix('.lock')
        with FileLock(str(lock_file), self.config.lock_timeout):
            yield
    
    def save_products(self, products: List[ProductData], 
                     vendor: str, category: str, 
                     date: datetime = None) -> str:
        """
        Save product data to file with automatic organization.
        
        Args:
            products: List of ProductData objects
            vendor: Vendor name (e.g., 'tesco', 'asda')
            category: Category name (e.g., 'grocery', 'household')
            date: Date for organization (defaults to current date)
        
        Returns:
            Path to the saved file
        """
        if date is None:
            date = datetime.now()
        
        file_path = self._get_file_path(vendor, category, date)
        
        try:
            with self._file_lock(file_path):
                # Create backup if file exists
                if file_path.exists():
                    self._create_backup(file_path)
                
                # Serialize and compress data
                serialized_data = self._serialize_data(products)
                compressed_data = self._compress_data(serialized_data)
                
                # Write atomically
                self._atomic_write(file_path, compressed_data)
                
                self.stats['total_saves'] += 1
                self.logger.info(f"Saved {len(products)} products to {file_path}")
                
                return str(file_path)
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Failed to save products: {e}")
            raise
    
    def save_incremental(self, product: ProductData) -> None:
        """
        Add a product to the incremental save buffer.
        Automatically saves when buffer reaches threshold.
        """
        self._initialize_thread_local()
        
        self._thread_local.pending_data.append(product)
        self._thread_local.save_counter += 1
        
        # Save if threshold reached
        if self._thread_local.save_counter >= self.config.incremental_save_interval:
            self._flush_incremental_data()
    
    def _flush_incremental_data(self) -> None:
        """Flush the incremental data buffer."""
        if not hasattr(self._thread_local, 'pending_data') or not self._thread_local.pending_data:
            return
        
        # Group by vendor/category/date
        grouped_data = {}
        for product in self._thread_local.pending_data:
            key = (product.vendor, product.category, product.scraped_at.date())
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(product)
        
        # Save each group
        for (vendor, category, date), products in grouped_data.items():
            date_obj = datetime.combine(date, datetime.min.time())
            self.save_products(products, vendor, category, date_obj)
        
        # Clear buffer
        self._thread_local.pending_data = []
        self._thread_local.save_counter = 0
    
    def load_products(self, file_path: Union[str, Path]) -> List[ProductData]:
        """
        Load product data from file.
        
        Args:
            file_path: Path to the file to load
        
        Returns:
            List of ProductData objects
        """
        file_path = Path(file_path)
        
        try:
            with self._file_lock(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Decompress if needed
                decompressed_data = self._decompress_data(data)
                
                # Deserialize
                products = self._deserialize_data(decompressed_data)
                
                self.stats['total_loads'] += 1
                self.logger.debug(f"Loaded {len(products)} products from {file_path}")
                
                return products
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Failed to load products from {file_path}: {e}")
            raise
    
    def query_products(self, query: StorageQuery) -> List[ProductData]:
        """
        Query stored products based on criteria.
        
        Args:
            query: StorageQuery object with search criteria
        
        Returns:
            List of ProductData objects matching the query
        """
        results = []
        
        # Build search path - files are organized as base_path/vendor/category/date/products_timestamp.json
        search_path = self.base_path
        if query.vendor:
            search_path = search_path / query.vendor
            if query.category:
                search_path = search_path / query.category
        elif query.category:
            # If only category is specified, we need to search all vendors
            search_path = search_path
        
        if not search_path.exists():
            return results
        
        # Find all JSON files recursively
        json_files = list(search_path.rglob("*.json"))
        
        for file_path in json_files:
            try:
                products = self.load_products(file_path)
                
                # Apply filters
                filtered_products = self._apply_query_filters(products, query)
                results.extend(filtered_products)
                
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        # Apply limit and offset
        if query.offset:
            results = results[query.offset:]
        if query.limit:
            results = results[:query.limit]
        
        return results
    
    def _apply_query_filters(self, products: List[ProductData], 
                           query: StorageQuery) -> List[ProductData]:
        """Apply query filters to products."""
        filtered = []
        
        for product in products:
            # Vendor filter
            if query.vendor and product.vendor != query.vendor:
                continue
            
            # Category filter
            if query.category and product.category != query.category:
                continue
            
            # Product ID filter
            if query.product_id and product.product_id != query.product_id:
                continue
            
            # Date range filter
            if query.date_from and product.scraped_at < query.date_from:
                continue
            if query.date_to and product.scraped_at > query.date_to:
                continue
            
            filtered.append(product)
        
        return filtered
    
    def cleanup_old_files(self, days: int = None) -> int:
        """
        Clean up old files based on rotation policy.
        
        Args:
            days: Number of days to keep (defaults to config setting)
        
        Returns:
            Number of files removed
        """
        if days is None:
            days = self.config.file_rotation_days
        
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0
        
        # Look for JSON files in the organized directory structure
        for file_path in self.base_path.rglob("*.json"):
            try:
                # Check file modification time
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if mtime < cutoff_date:
                    file_path.unlink()
                    removed_count += 1
                    self.logger.debug(f"Removed old file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {file_path}: {e}")
        
        self.logger.info(f"Cleanup removed {removed_count} old files")
        return removed_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = self.stats.copy()
        
        # Add file system stats
        total_files = 0
        total_size = 0
        
        for file_path in self.base_path.rglob("*.json"):
            if file_path.exists():
                total_files += 1
                total_size += file_path.stat().st_size
        
        stats.update({
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024)
        })
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush any pending data."""
        self._flush_incremental_data()


def create_storage_manager(config: StorageConfig = None) -> StorageManager:
    """Factory function to create a StorageManager instance."""
    return StorageManager(config)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create storage manager
    config = StorageConfig(
        base_directory="test_data",
        compression_enabled=True,
        incremental_save_interval=5
    )
    
    with StorageManager(config) as storage:
        # Create sample product data
        products = [
            ProductData(
                vendor="tesco",
                category="grocery",
                product_id="12345",
                title="Organic Bananas",
                price="£1.50",
                image_url="https://example.com/banana.jpg",
                description="Fresh organic bananas",
                url="https://tesco.com/bananas"
            ),
            ProductData(
                vendor="asda",
                category="household",
                product_id="67890",
                title="Dish Soap",
                price="£2.00",
                image_url="https://example.com/soap.jpg",
                description="Eco-friendly dish soap",
                url="https://asda.com/soap"
            )
        ]
        
        # Save products
        file_path = storage.save_products(products, "tesco", "grocery")
        print(f"Saved products to: {file_path}")
        
        # Load products
        loaded_products = storage.load_products(file_path)
        print(f"Loaded {len(loaded_products)} products")
        
        # Query products
        query = StorageQuery(vendor="tesco", category="grocery")
        results = storage.query_products(query)
        print(f"Query returned {len(results)} products")
        
        # Print stats
        stats = storage.get_storage_stats()
        print(f"Storage stats: {stats}") 