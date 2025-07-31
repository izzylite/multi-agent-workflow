"""
Unit tests for the Results Management System.
"""

import json
import gzip
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from scraping_cli.results_manager import (
    ResultsManager,
    ResultMetadata,
    ResultStats,
    ExportFormat,
    SortOrder,
    create_results_manager
)


class TestExportFormat:
    """Test ExportFormat enum."""
    
    def test_export_format_values(self):
        """Test that ExportFormat has the expected values."""
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.EXCEL.value == "excel"


class TestSortOrder:
    """Test SortOrder enum."""
    
    def test_sort_order_values(self):
        """Test that SortOrder has the expected values."""
        assert SortOrder.ASC.value == "asc"
        assert SortOrder.DESC.value == "desc"


class TestResultMetadata:
    """Test ResultMetadata dataclass."""
    
    def test_result_metadata_creation(self):
        """Test creating a ResultMetadata instance."""
        now = datetime.now()
        metadata = ResultMetadata(
            vendor="tesco",
            category="groceries",
            file_path="/path/to/file.json",
            product_count=100,
            created_at=now,
            file_size=1024,
            compressed=False
        )
        
        assert metadata.vendor == "tesco"
        assert metadata.category == "groceries"
        assert metadata.file_path == "/path/to/file.json"
        assert metadata.product_count == 100
        assert metadata.created_at == now
        assert metadata.file_size == 1024
        assert metadata.compressed is False


class TestResultStats:
    """Test ResultStats dataclass."""
    
    def test_result_stats_creation(self):
        """Test creating a ResultStats instance."""
        now = datetime.now()
        stats = ResultStats(
            total_results=10,
            total_products=1000,
            vendors={"tesco": 5, "asda": 5},
            categories={"groceries": 8, "electronics": 2},
            date_range=(now, now + timedelta(days=1)),
            total_size=1024000,
            average_products_per_result=100.0
        )
        
        assert stats.total_results == 10
        assert stats.total_products == 1000
        assert stats.vendors == {"tesco": 5, "asda": 5}
        assert stats.categories == {"groceries": 8, "electronics": 2}
        assert stats.date_range == (now, now + timedelta(days=1))
        assert stats.total_size == 1024000
        assert stats.average_products_per_result == 100.0


class TestResultsManager:
    """Test ResultsManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def results_manager(self, temp_dir):
        """Create a ResultsManager instance with temporary directory."""
        return ResultsManager(temp_dir)
    
    @pytest.fixture
    def sample_products(self):
        """Create sample product data."""
        return [
            {
                "id": "1",
                "name": "Product 1",
                "price": 10.99,
                "vendor": "tesco",
                "category": "groceries"
            },
            {
                "id": "2", 
                "name": "Product 2",
                "price": 15.50,
                "vendor": "tesco",
                "category": "groceries"
            }
        ]
    
    def test_results_manager_initialization(self, temp_dir):
        """Test ResultsManager initialization."""
        manager = ResultsManager(temp_dir)
        assert manager.storage_dir == Path(temp_dir)
        assert manager.storage_dir.exists()
    
    def test_results_manager_creates_directory(self, temp_dir):
        """Test that ResultsManager creates the storage directory."""
        new_dir = Path(temp_dir) / "new_storage"
        manager = ResultsManager(str(new_dir))
        assert new_dir.exists()
    
    def test_scan_results_empty_directory(self, results_manager):
        """Test scanning results in an empty directory."""
        results = results_manager._scan_results()
        assert results == []
    
    def test_scan_results_with_files(self, results_manager, sample_products):
        """Test scanning results with actual files."""
        # Create test directory structure
        test_dir = results_manager.storage_dir / "tesco" / "groceries"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a test file
        test_file = test_dir / "products.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        # Set file modification time
        timestamp = datetime.now().timestamp()
        os.utime(test_file, (timestamp, timestamp))
        
        results = results_manager._scan_results()
        assert len(results) == 1
        
        result = results[0]
        assert result.vendor == "tesco"
        assert result.category == "groceries"
        assert result.product_count == 2
        assert result.compressed is False
    
    def test_scan_results_with_compressed_files(self, results_manager, sample_products):
        """Test scanning results with compressed files."""
        # Create test directory structure
        test_dir = results_manager.storage_dir / "asda" / "electronics"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a compressed test file
        test_file = test_dir / "products.json.gz"
        with gzip.open(test_file, 'wt', encoding='utf-8') as f:
            json.dump(sample_products, f)
        
        # Set file modification time
        timestamp = datetime.now().timestamp()
        os.utime(test_file, (timestamp, timestamp))
        
        results = results_manager._scan_results()
        assert len(results) == 1
        
        result = results[0]
        assert result.vendor == "asda"
        assert result.category == "electronics"
        assert result.product_count == 2
        assert result.compressed is True
    
    def test_filter_results_by_vendor(self, results_manager):
        """Test filtering results by vendor."""
        # Create test results
        results = [
            ResultMetadata("tesco", "groceries", "/path1", 10, datetime.now(), 1024, False),
            ResultMetadata("asda", "groceries", "/path2", 15, datetime.now(), 2048, False),
            ResultMetadata("tesco", "electronics", "/path3", 5, datetime.now(), 512, False)
        ]
        
        filtered = results_manager._filter_results(results, vendor="tesco")
        assert len(filtered) == 2
        assert all(r.vendor == "tesco" for r in filtered)
    
    def test_filter_results_by_category(self, results_manager):
        """Test filtering results by category."""
        # Create test results
        results = [
            ResultMetadata("tesco", "groceries", "/path1", 10, datetime.now(), 1024, False),
            ResultMetadata("asda", "groceries", "/path2", 15, datetime.now(), 2048, False),
            ResultMetadata("tesco", "electronics", "/path3", 5, datetime.now(), 512, False)
        ]
        
        filtered = results_manager._filter_results(results, category="groceries")
        assert len(filtered) == 2
        assert all(r.category == "groceries" for r in filtered)
    
    def test_filter_results_by_date_range(self, results_manager):
        """Test filtering results by date range."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)
        
        # Create test results
        results = [
            ResultMetadata("tesco", "groceries", "/path1", 10, yesterday, 1024, False),
            ResultMetadata("asda", "groceries", "/path2", 15, now, 2048, False),
            ResultMetadata("tesco", "electronics", "/path3", 5, tomorrow, 512, False)
        ]
        
        filtered = results_manager._filter_results(results, date_from=now)
        assert len(filtered) == 2
        assert all(r.created_at >= now for r in filtered)
    
    def test_filter_results_by_product_count(self, results_manager):
        """Test filtering results by product count."""
        # Create test results
        results = [
            ResultMetadata("tesco", "groceries", "/path1", 5, datetime.now(), 1024, False),
            ResultMetadata("asda", "groceries", "/path2", 15, datetime.now(), 2048, False),
            ResultMetadata("tesco", "electronics", "/path3", 25, datetime.now(), 512, False)
        ]
        
        filtered = results_manager._filter_results(results, min_products=10, max_products=20)
        assert len(filtered) == 1
        assert filtered[0].product_count == 15
    
    def test_load_result_file_json(self, results_manager, sample_products):
        """Test loading a regular JSON file."""
        # Create test file
        test_file = results_manager.storage_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        products = results_manager._load_result_file(test_file)
        assert len(products) == 2
        assert products[0]["name"] == "Product 1"
        assert products[1]["name"] == "Product 2"
    
    def test_load_result_file_compressed(self, results_manager, sample_products):
        """Test loading a compressed JSON file."""
        # Create compressed test file
        test_file = results_manager.storage_dir / "test.json.gz"
        with gzip.open(test_file, 'wt', encoding='utf-8') as f:
            json.dump(sample_products, f)
        
        products = results_manager._load_result_file(test_file)
        assert len(products) == 2
        assert products[0]["name"] == "Product 1"
        assert products[1]["name"] == "Product 2"
    
    def test_load_result_file_dict_format(self, results_manager, sample_products):
        """Test loading a file with dict format."""
        # Create test file with dict format
        test_file = results_manager.storage_dir / "test.json"
        data = {"products": sample_products, "metadata": {"count": 2}}
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        products = results_manager._load_result_file(test_file)
        assert len(products) == 2
        assert products[0]["name"] == "Product 1"
    
    def test_load_result_file_not_found(self, results_manager):
        """Test loading a non-existent file."""
        test_file = results_manager.storage_dir / "nonexistent.json"
        
        with pytest.raises(RuntimeError, match="Failed to load result file"):
            results_manager._load_result_file(test_file)
    
    def test_format_file_size(self, results_manager):
        """Test file size formatting."""
        assert results_manager._format_file_size(512) == "512.0 B"
        assert results_manager._format_file_size(1024) == "1.0 KB"
        assert results_manager._format_file_size(1024 * 1024) == "1.0 MB"
        assert results_manager._format_file_size(1024 * 1024 * 1024) == "1.0 GB"
    
    def test_list_results_basic(self, results_manager):
        """Test basic list_results functionality."""
        # Mock _scan_results to return test data
        test_results = [
            ResultMetadata("tesco", "groceries", "/path1", 10, datetime.now(), 1024, False)
        ]
        
        with patch.object(results_manager, '_scan_results', return_value=test_results):
            results, total = results_manager.list_results()
            assert len(results) == 1
            assert total == 1
    
    def test_list_results_with_filtering(self, results_manager):
        """Test list_results with filtering."""
        # Create test results
        now = datetime.now()
        test_results = [
            ResultMetadata("tesco", "groceries", "/path1", 10, now, 1024, False),
            ResultMetadata("asda", "groceries", "/path2", 15, now, 2048, False),
            ResultMetadata("tesco", "electronics", "/path3", 5, now, 512, False)
        ]
        
        with patch.object(results_manager, '_scan_results', return_value=test_results):
            results, total = results_manager.list_results(vendor="tesco")
            assert len(results) == 2
            assert total == 2
            assert all(r.vendor == "tesco" for r in results)
    
    def test_list_results_with_pagination(self, results_manager):
        """Test list_results with pagination."""
        # Create test results
        test_results = [
            ResultMetadata("tesco", "groceries", f"/path{i}", 10, datetime.now(), 1024, False)
            for i in range(25)
        ]
        
        with patch.object(results_manager, '_scan_results', return_value=test_results):
            results, total = results_manager.list_results(page=2, per_page=10)
            assert len(results) == 10
            assert total == 25
    
    def test_view_result(self, results_manager, sample_products):
        """Test viewing a specific result."""
        # Create test file
        test_file = results_manager.storage_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        result = results_manager.view_result(str(test_file))
        assert result["product_count"] == 2
        assert result["vendor"] == "unknown"  # No vendor in path
        assert result["category"] == "unknown"  # No category in path
        assert len(result["products"]) == 2
    
    def test_view_result_not_found(self, results_manager):
        """Test viewing a non-existent result."""
        with pytest.raises(FileNotFoundError):
            results_manager.view_result("/nonexistent/path.json")
    
    def test_export_result_json(self, results_manager, sample_products):
        """Test exporting to JSON format."""
        # Create test file
        test_file = results_manager.storage_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        output_file = results_manager.storage_dir / "export.json"
        result_path = results_manager.export_result(
            str(test_file), str(output_file), ExportFormat.JSON
        )
        
        assert Path(result_path).exists()
        
        # Verify exported data
        with open(result_path, 'r') as f:
            exported_data = json.load(f)
        assert len(exported_data) == 2
        assert exported_data[0]["name"] == "Product 1"
    
    def test_export_result_csv(self, results_manager, sample_products):
        """Test exporting to CSV format."""
        # Create test file
        test_file = results_manager.storage_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        output_file = results_manager.storage_dir / "export.csv"
        result_path = results_manager.export_result(
            str(test_file), str(output_file), ExportFormat.CSV
        )
        
        assert Path(result_path).exists()
        
        # Verify exported data
        df = pd.read_csv(result_path)
        assert len(df) == 2
        assert df.iloc[0]["name"] == "Product 1"
    
    def test_export_result_excel(self, results_manager, sample_products):
        """Test exporting to Excel format."""
        # Create test file
        test_file = results_manager.storage_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        output_file = results_manager.storage_dir / "export.xlsx"
        result_path = results_manager.export_result(
            str(test_file), str(output_file), ExportFormat.EXCEL
        )
        
        assert Path(result_path).exists()
        
        # Verify exported data
        df = pd.read_excel(result_path)
        assert len(df) == 2
        assert df.iloc[0]["name"] == "Product 1"
    
    def test_export_result_with_fields(self, results_manager, sample_products):
        """Test exporting with specific fields."""
        # Create test file
        test_file = results_manager.storage_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        output_file = results_manager.storage_dir / "export.json"
        result_path = results_manager.export_result(
            str(test_file), str(output_file), ExportFormat.JSON, fields=["name", "price"]
        )
        
        # Verify exported data has only specified fields
        with open(result_path, 'r') as f:
            exported_data = json.load(f)
        assert len(exported_data) == 2
        assert set(exported_data[0].keys()) == {"name", "price"}
    
    def test_export_result_invalid_format(self, results_manager, sample_products):
        """Test exporting with invalid format."""
        # Create test file
        test_file = results_manager.storage_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        output_file = results_manager.storage_dir / "export.txt"
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            results_manager.export_result(
                str(test_file), str(output_file), "invalid_format"
            )
    
    def test_delete_result(self, results_manager, sample_products):
        """Test deleting a result."""
        # Create test file
        test_file = results_manager.storage_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(sample_products, f)
        
        # Test without confirmation (should not delete)
        result = results_manager.delete_result(str(test_file), confirm=False)
        assert result is False
        assert test_file.exists()
        
        # Test with confirmation (should delete)
        result = results_manager.delete_result(str(test_file), confirm=True)
        assert result is True
        assert not test_file.exists()
    
    def test_delete_result_not_found(self, results_manager):
        """Test deleting a non-existent result."""
        with pytest.raises(FileNotFoundError):
            results_manager.delete_result("/nonexistent/path.json")
    
    def test_get_stats_empty(self, results_manager):
        """Test getting stats with no results."""
        stats = results_manager.get_stats()
        assert stats.total_results == 0
        assert stats.total_products == 0
        assert stats.vendors == {}
        assert stats.categories == {}
        assert stats.total_size == 0
        assert stats.average_products_per_result == 0.0
    
    def test_get_stats_with_data(self, results_manager):
        """Test getting stats with actual data."""
        # Mock _scan_results to return test data
        now = datetime.now()
        test_results = [
            ResultMetadata("tesco", "groceries", "/path1", 10, now, 1024, False),
            ResultMetadata("asda", "groceries", "/path2", 15, now, 2048, False),
            ResultMetadata("tesco", "electronics", "/path3", 5, now, 512, False)
        ]
        
        with patch.object(results_manager, '_scan_results', return_value=test_results):
            stats = results_manager.get_stats()
            assert stats.total_results == 3
            assert stats.total_products == 30
            assert stats.vendors == {"tesco": 2, "asda": 1}
            assert stats.categories == {"groceries": 2, "electronics": 1}
            assert stats.total_size == 3584
            assert stats.average_products_per_result == 10.0


class TestResultsManagerDisplay:
    """Test ResultsManager display methods."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def results_manager(self, temp_dir):
        """Create a ResultsManager instance with temporary directory."""
        return ResultsManager(temp_dir)
    
    def test_display_results_table_empty(self, results_manager):
        """Test displaying empty results table."""
        with patch('builtins.print') as mock_print:
            results_manager.display_results_table([], 0, 1, 20)
            mock_print.assert_called_with(
                '\x1b[33mNo results found matching the criteria.\x1b[0m'
            )
    
    def test_display_results_table_with_data(self, results_manager):
        """Test displaying results table with data."""
        now = datetime.now()
        results = [
            ResultMetadata("tesco", "groceries", "/path1", 10, now, 1024, False),
            ResultMetadata("asda", "electronics", "/path2", 15, now, 2048, True)
        ]
        
        with patch('builtins.print') as mock_print:
            results_manager.display_results_table(results, 2, 1, 20)
            
            # Verify that print was called (we don't check exact output due to colors)
            assert mock_print.call_count > 0
    
    def test_display_stats_empty(self, results_manager):
        """Test displaying stats with no data."""
        stats = ResultStats(
            total_results=0,
            total_products=0,
            vendors={},
            categories={},
            date_range=(datetime.now(), datetime.now()),
            total_size=0,
            average_products_per_result=0.0
        )
        
        with patch('builtins.print') as mock_print:
            results_manager.display_stats(stats)
            
            # Verify that print was called
            assert mock_print.call_count > 0
    
    def test_display_stats_with_data(self, results_manager):
        """Test displaying stats with data."""
        now = datetime.now()
        stats = ResultStats(
            total_results=5,
            total_products=100,
            vendors={"tesco": 3, "asda": 2},
            categories={"groceries": 4, "electronics": 1},
            date_range=(now, now + timedelta(days=1)),
            total_size=1024000,
            average_products_per_result=20.0
        )
        
        with patch('builtins.print') as mock_print:
            results_manager.display_stats(stats)
            
            # Verify that print was called
            assert mock_print.call_count > 0


def test_create_results_manager():
    """Test the create_results_manager factory function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = create_results_manager(temp_dir)
        assert isinstance(manager, ResultsManager)
        assert manager.storage_dir == Path(temp_dir) 