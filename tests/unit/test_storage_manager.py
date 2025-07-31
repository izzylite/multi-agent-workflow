"""
Unit tests for StorageManager.

Tests file operations, JSON serialization/deserialization, compression,
file locking, and query interface with mock filesystem.
"""

import pytest
import tempfile
import shutil
import os
import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from scraping_cli.storage_manager import (
    StorageManager, StorageConfig, ProductData, StorageQuery, FileLock, create_storage_manager
)


class TestStorageConfig:
    """Test StorageConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StorageConfig()
        assert config.base_directory == "data"
        assert config.compression_enabled is True
        assert config.compression_threshold == 1024
        assert config.max_file_size == 50 * 1024 * 1024
        assert config.file_rotation_days == 30
        assert config.incremental_save_interval == 100
        assert config.backup_enabled is True
        assert config.backup_directory == "data/backups"
        assert config.lock_timeout == 30
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = StorageConfig(
            base_directory="custom_data",
            compression_enabled=False,
            compression_threshold=2048,
            max_file_size=100 * 1024 * 1024,
            file_rotation_days=60,
            incremental_save_interval=50,
            backup_enabled=False,
            backup_directory="custom_backups",
            lock_timeout=60
        )
        
        assert config.base_directory == "custom_data"
        assert config.compression_enabled is False
        assert config.compression_threshold == 2048
        assert config.max_file_size == 100 * 1024 * 1024
        assert config.file_rotation_days == 60
        assert config.incremental_save_interval == 50
        assert config.backup_enabled is False
        assert config.backup_directory == "custom_backups"
        assert config.lock_timeout == 60


class TestProductData:
    """Test ProductData dataclass."""
    
    def test_product_data_creation(self):
        """Test creating ProductData with all fields."""
        product = ProductData(
            vendor="tesco",
            category="grocery",
            product_id="12345",
            title="Organic Bananas",
            price="£1.50",
            image_url="https://example.com/banana.jpg",
            description="Fresh organic bananas",
            url="https://tesco.com/bananas",
            metadata={"organic": True, "weight": "1kg"}
        )
        
        assert product.vendor == "tesco"
        assert product.category == "grocery"
        assert product.product_id == "12345"
        assert product.title == "Organic Bananas"
        assert product.price == "£1.50"
        assert product.image_url == "https://example.com/banana.jpg"
        assert product.description == "Fresh organic bananas"
        assert product.url == "https://tesco.com/bananas"
        assert product.metadata == {"organic": True, "weight": "1kg"}
        assert isinstance(product.scraped_at, datetime)
    
    def test_product_data_defaults(self):
        """Test ProductData with default values."""
        product = ProductData(
            vendor="asda",
            category="household",
            product_id="67890",
            title="Dish Soap"
        )
        
        assert product.vendor == "asda"
        assert product.category == "household"
        assert product.product_id == "67890"
        assert product.title == "Dish Soap"
        assert product.price is None
        assert product.image_url is None
        assert product.description is None
        assert product.url is None
        assert isinstance(product.scraped_at, datetime)
        assert product.metadata == {}


class TestFileLock:
    """Test FileLock functionality."""
    
    def test_file_lock_creation(self):
        """Test FileLock creation."""
        lock = FileLock("test.lock", timeout=10)
        assert lock.lock_file == "test.lock"
        assert lock.timeout == 10
        assert lock.lock_acquired is False
    
    def test_file_lock_context_manager(self, tmp_path):
        """Test FileLock as context manager."""
        lock_file = tmp_path / "test.lock"
        
        with FileLock(str(lock_file), timeout=5):
            assert lock_file.exists()
            with open(lock_file, 'r') as f:
                pid = int(f.read().strip())
                assert pid == os.getpid()
        
        # Lock should be released
        assert not lock_file.exists()
    
    def test_file_lock_timeout(self, tmp_path):
        """Test FileLock timeout behavior."""
        lock_file = tmp_path / "test.lock"
        
        # Create a lock file manually
        with open(lock_file, 'w') as f:
            f.write("99999")  # Non-existent PID
        
        # Should acquire lock immediately since PID doesn't exist
        with FileLock(str(lock_file), timeout=1):
            assert lock_file.exists()
    
    def test_file_lock_stale_lock_cleanup(self, tmp_path):
        """Test cleanup of stale lock files."""
        lock_file = tmp_path / "test.lock"
        
        # Create a lock file with non-existent PID
        with open(lock_file, 'w') as f:
            f.write("99999")
        
        # Should clean up stale lock and acquire
        with FileLock(str(lock_file), timeout=1):
            assert lock_file.exists()
            with open(lock_file, 'r') as f:
                pid = int(f.read().strip())
                assert pid == os.getpid()


class TestStorageManager:
    """Test StorageManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage_manager(self, temp_dir):
        """Create StorageManager instance for testing."""
        config = StorageConfig(
            base_directory=temp_dir,
            compression_enabled=True,
            backup_enabled=True,
            backup_directory=os.path.join(temp_dir, "backups"),
            incremental_save_interval=5
        )
        return StorageManager(config)
    
    @pytest.fixture
    def sample_products(self):
        """Create sample product data for testing."""
        return [
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
    
    def test_storage_manager_initialization(self, temp_dir):
        """Test StorageManager initialization."""
        config = StorageConfig(base_directory=temp_dir)
        storage = StorageManager(config)
        
        assert storage.config == config
        assert storage.base_path == Path(temp_dir)
        assert storage.base_path.exists()
        assert storage.stats['total_saves'] == 0
        assert storage.stats['total_loads'] == 0
    
    def test_directory_creation(self, storage_manager):
        """Test automatic directory creation."""
        vendor = "tesco"
        category = "grocery"
        date = datetime.now()
        
        directory_path = storage_manager._get_directory_path(vendor, category, date)
        assert not directory_path.exists()
        
        file_path = storage_manager._get_file_path(vendor, category, date)
        assert directory_path.exists()
        assert file_path.parent == directory_path
    
    def test_data_serialization_deserialization(self, storage_manager, sample_products):
        """Test JSON serialization and deserialization."""
        # Serialize
        serialized = storage_manager._serialize_data(sample_products)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = storage_manager._deserialize_data(serialized)
        assert len(deserialized) == len(sample_products)
        
        # Check data integrity
        for original, deserialized_product in zip(sample_products, deserialized):
            assert original.vendor == deserialized_product.vendor
            assert original.category == deserialized_product.category
            assert original.product_id == deserialized_product.product_id
            assert original.title == deserialized_product.title
            assert original.price == deserialized_product.price
    
    def test_compression_decompression(self, storage_manager):
        """Test data compression and decompression."""
        test_data = b"x" * 2000  # Above compression threshold
        
        # Compress
        compressed = storage_manager._compress_data(test_data)
        assert len(compressed) < len(test_data)
        assert compressed != test_data
        
        # Decompress
        decompressed = storage_manager._decompress_data(compressed)
        assert decompressed == test_data
    
    def test_no_compression_below_threshold(self, storage_manager):
        """Test that data below threshold is not compressed."""
        test_data = b"x" * 500  # Below compression threshold
        
        compressed = storage_manager._compress_data(test_data)
        assert compressed == test_data
    
    def test_atomic_write(self, storage_manager, temp_dir):
        """Test atomic write functionality."""
        file_path = Path(temp_dir) / "test.json"
        test_data = b"test data"
        
        storage_manager._atomic_write(file_path, test_data)
        
        assert file_path.exists()
        with open(file_path, 'rb') as f:
            assert f.read() == test_data
    
    def test_atomic_write_error_cleanup(self, storage_manager, temp_dir):
        """Test atomic write cleanup on error."""
        file_path = Path(temp_dir) / "test.json"
        test_data = b"test data"
        
        # Create a directory with the same name to cause write error
        file_path.mkdir(parents=True)
        
        with pytest.raises(Exception):
            storage_manager._atomic_write(file_path, test_data)
        
        # Temp file should be cleaned up
        temp_file = file_path.with_suffix('.tmp')
        assert not temp_file.exists()
    
    def test_save_and_load_products(self, storage_manager, sample_products):
        """Test saving and loading products."""
        vendor = "tesco"
        category = "grocery"
        date = datetime.now()
        
        # Save products
        file_path = storage_manager.save_products(sample_products, vendor, category, date)
        assert Path(file_path).exists()
        assert storage_manager.stats['total_saves'] == 1
        
        # Load products
        loaded_products = storage_manager.load_products(file_path)
        assert len(loaded_products) == len(sample_products)
        assert storage_manager.stats['total_loads'] == 1
        
        # Check data integrity
        for original, loaded in zip(sample_products, loaded_products):
            assert original.vendor == loaded.vendor
            assert original.category == loaded.category
            assert original.product_id == loaded.product_id
            assert original.title == loaded.title
    
    def test_incremental_save(self, storage_manager):
        """Test incremental save functionality."""
        # Add products to incremental buffer
        for i in range(3):
            product = ProductData(
                vendor="tesco",
                category="grocery",
                product_id=f"product_{i}",
                title=f"Product {i}"
            )
            storage_manager.save_incremental(product)
        
        # Buffer should not be flushed yet (threshold is 5)
        assert len(storage_manager._thread_local.pending_data) == 3
        
                # Add more products to trigger flush
        for i in range(3, 6):
            product = ProductData(
                vendor="tesco",
                category="grocery",
                product_id=f"product_{i}",
                title=f"Product {i}"
            )
            storage_manager.save_incremental(product)

        # Buffer should be flushed, but the last product that triggered the flush remains
        assert len(storage_manager._thread_local.pending_data) == 1
    
    def test_context_manager_flush(self, storage_manager):
        """Test that context manager flushes pending data."""
        # Add product to buffer
        product = ProductData(
            vendor="tesco",
            category="grocery",
            product_id="test_product",
            title="Test Product"
        )
        storage_manager.save_incremental(product)
        
        # Exit context manager
        storage_manager.__exit__(None, None, None)
        
        # Buffer should be flushed
        assert len(storage_manager._thread_local.pending_data) == 0
    
    def test_query_products(self, storage_manager, sample_products):
        """Test query interface."""
        # Create separate product lists for each vendor
        tesco_products = [
            ProductData(
                vendor="tesco",
                category="grocery",
                product_id="12345",
                title="Organic Bananas",
                price="£1.50",
                image_url="https://example.com/banana.jpg",
                description="Fresh organic bananas",
                url="https://tesco.com/bananas"
            )
        ]
        
        asda_products = [
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
        storage_manager.save_products(tesco_products, "tesco", "grocery")
        storage_manager.save_products(asda_products, "asda", "household")

        # Query by vendor
        query = StorageQuery(vendor="tesco")
        results = storage_manager.query_products(query)
        assert len(results) == 1
        assert results[0].vendor == "tesco"

        # Query by category
        query = StorageQuery(category="grocery")
        results = storage_manager.query_products(query)
        assert len(results) == 1
        assert results[0].category == "grocery"

        # Query by vendor and category
        query = StorageQuery(vendor="tesco", category="grocery")
        results = storage_manager.query_products(query)
        assert len(results) == 1
        assert results[0].vendor == "tesco"
        assert results[0].category == "grocery"

        # Query with limit
        query = StorageQuery(limit=1)
        results = storage_manager.query_products(query)
        assert len(results) == 1
    
    def test_query_filters(self, storage_manager):
        """Test query filtering functionality."""
        # Create products with different dates
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)
        
        products = [
            ProductData(
                vendor="tesco",
                category="grocery",
                product_id="1",
                title="Product 1",
                scraped_at=yesterday
            ),
            ProductData(
                vendor="tesco",
                category="grocery",
                product_id="2",
                title="Product 2",
                scraped_at=now
            ),
            ProductData(
                vendor="tesco",
                category="grocery",
                product_id="3",
                title="Product 3",
                scraped_at=tomorrow
            )
        ]
        
        storage_manager.save_products(products, "tesco", "grocery")
        
        # Test date range filter
        query = StorageQuery(
            vendor="tesco",
            date_from=now,
            date_to=tomorrow
        )
        results = storage_manager.query_products(query)
        assert len(results) == 2  # Only products from now and tomorrow
        
        # Test product ID filter
        query = StorageQuery(product_id="2")
        results = storage_manager.query_products(query)
        assert len(results) == 1
        assert results[0].product_id == "2"
    
    def test_cleanup_old_files(self, storage_manager, temp_dir):
        """Test cleanup of old files."""
        # Create old files
        old_date = datetime.now() - timedelta(days=35)
        old_file = storage_manager._get_file_path("tesco", "grocery", old_date)
        old_file.parent.mkdir(parents=True, exist_ok=True)
        old_file.write_text("old data")
        
        # Set the file modification time to the old date
        import os
        os.utime(old_file, (old_date.timestamp(), old_date.timestamp()))
        
        # Create recent files
        recent_date = datetime.now()
        recent_file = storage_manager._get_file_path("tesco", "grocery", recent_date)
        recent_file.parent.mkdir(parents=True, exist_ok=True)
        recent_file.write_text("recent data")
        
        # Cleanup old files (default 30 days)
        removed_count = storage_manager.cleanup_old_files()
        
        assert removed_count == 1
        assert not old_file.exists()
        assert recent_file.exists()
    
    def test_storage_stats(self, storage_manager, sample_products):
        """Test storage statistics."""
        # Save some products
        storage_manager.save_products(sample_products, "tesco", "grocery")
        
        stats = storage_manager.get_storage_stats()
        
        assert stats['total_saves'] == 1
        assert stats['total_loads'] == 0
        assert stats['total_files'] == 1
        assert stats['total_size_bytes'] > 0
        assert stats['total_size_mb'] > 0
    
    def test_backup_creation(self, storage_manager, sample_products):
        """Test backup file creation."""
        # Save products twice to trigger backup
        storage_manager.save_products(sample_products, "tesco", "grocery")
        storage_manager.save_products(sample_products, "tesco", "grocery")
        
        # Check backup was created
        backup_files = list(storage_manager.backup_path.glob("*.backup"))
        assert len(backup_files) == 1
        assert storage_manager.stats['backups_created'] == 1
    
    def test_error_handling(self, storage_manager):
        """Test error handling in storage operations."""
        # Test save with invalid data
        with pytest.raises(Exception):
            storage_manager.save_products("invalid_data", "tesco", "grocery")
        
        assert storage_manager.stats['errors'] == 1
        
        # Test load from non-existent file
        with pytest.raises(Exception):
            storage_manager.load_products("non_existent_file.json")
        
        assert storage_manager.stats['errors'] == 2


class TestStorageQuery:
    """Test StorageQuery functionality."""
    
    def test_storage_query_creation(self):
        """Test StorageQuery creation."""
        query = StorageQuery(
            vendor="tesco",
            category="grocery",
            date_from=datetime.now(),
            date_to=datetime.now() + timedelta(days=1),
            product_id="12345",
            limit=10,
            offset=5
        )
        
        assert query.vendor == "tesco"
        assert query.category == "grocery"
        assert query.date_from is not None
        assert query.date_to is not None
        assert query.product_id == "12345"
        assert query.limit == 10
        assert query.offset == 5
    
    def test_storage_query_defaults(self):
        """Test StorageQuery with default values."""
        query = StorageQuery()
        
        assert query.vendor is None
        assert query.category is None
        assert query.date_from is None
        assert query.date_to is None
        assert query.product_id is None
        assert query.limit is None
        assert query.offset is None


def test_create_storage_manager():
    """Test factory function for creating StorageManager."""
    config = StorageConfig(base_directory="test_data")
    storage = create_storage_manager(config)
    
    assert isinstance(storage, StorageManager)
    assert storage.config == config


if __name__ == "__main__":
    pytest.main([__file__]) 