"""
Unit tests for ExportManager class.

Tests the enhanced export management capabilities including multi-format support,
configuration options, filtering, sorting, and validation functionality.
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from scraping_cli.export_manager import (
    ExportManager,
    ExportConfig,
    CSVConfig,
    ExcelConfig,
    MarkdownConfig,
    ExportFormat,
    CompressionFormat,
    ExportTemplate,
    create_export_manager
)


class TestExportManager:
    """Test cases for ExportManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_manager = ExportManager(self.temp_dir)
        
        self.sample_data = [
            {
                "id": 1,
                "name": "Product A",
                "price": 10.99,
                "category": "Electronics",
                "vendor": "Tesco",
                "created_at": "2024-01-15T10:30:00"
            },
            {
                "id": 2,
                "name": "Product B",
                "price": 25.50,
                "category": "Home",
                "vendor": "Asda",
                "created_at": "2024-01-16T14:20:00"
            },
            {
                "id": 3,
                "name": "Product C",
                "price": 5.99,
                "category": "Electronics",
                "vendor": "Tesco",
                "created_at": "2024-01-17T09:15:00"
            }
        ]

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test ExportManager initialization."""
        assert self.export_manager.output_dir.exists()
        assert self.export_manager.output_dir.is_dir()

    def test_export_data_json(self):
        """Test JSON export."""
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "test.json"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        assert Path(result).exists()
        assert Path(result).suffix == '.json'
        
        # Verify JSON content
        with open(result, 'r') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "data" in data
        assert len(data["data"]) == 3

    def test_export_data_csv(self):
        """Test CSV export."""
        config = ExportConfig(format=ExportFormat.CSV)
        output_path = Path(self.temp_dir) / "test.csv"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        assert Path(result).exists()
        assert Path(result).suffix == '.csv'
        
        # Verify CSV content
        df = pd.read_csv(result)
        assert len(df) == 3
        assert "name" in df.columns

    def test_export_data_excel(self):
        """Test Excel export."""
        config = ExportConfig(format=ExportFormat.EXCEL)
        output_path = Path(self.temp_dir) / "test.xlsx"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        assert Path(result).exists()
        assert Path(result).suffix == '.xlsx'
        
        # Verify Excel content
        df = pd.read_excel(result)
        assert len(df) == 3
        assert "name" in df.columns

    def test_export_data_markdown(self):
        """Test Markdown export."""
        config = ExportConfig(format=ExportFormat.MARKDOWN)
        output_path = Path(self.temp_dir) / "test.md"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        assert Path(result).exists()
        assert Path(result).suffix == '.md'
        
        # Verify Markdown content
        with open(result, 'r') as f:
            content = f.read()
        
        assert "# Data Export" in content
        assert "Total Records:** 3" in content

    def test_export_data_with_fields(self):
        """Test export with field selection."""
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "test_fields.json"
        fields = ["name", "price"]
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config, fields=fields
        )
        
        with open(result, 'r') as f:
            data = json.load(f)
        
        exported_data = data["data"]
        assert len(exported_data) == 3
        assert "name" in exported_data[0]
        assert "price" in exported_data[0]
        assert "id" not in exported_data[0]

    def test_export_data_with_filters(self):
        """Test export with filters."""
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "test_filtered.json"
        filters = {"vendor": "Tesco"}
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config, filters=filters
        )
        
        with open(result, 'r') as f:
            data = json.load(f)
        
        exported_data = data["data"]
        assert len(exported_data) == 2  # Only Tesco products
        assert all(item["vendor"] == "Tesco" for item in exported_data)

    def test_export_data_with_sorting(self):
        """Test export with sorting."""
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "test_sorted.json"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config, 
            sort_by="price", sort_ascending=False
        )
        
        with open(result, 'r') as f:
            data = json.load(f)
        
        exported_data = data["data"]
        prices = [item["price"] for item in exported_data]
        assert prices == sorted(prices, reverse=True)

    def test_export_data_with_csv_config(self):
        """Test CSV export with custom configuration."""
        config = ExportConfig(format=ExportFormat.CSV)
        csv_config = CSVConfig(delimiter=";", encoding="utf-8")
        output_path = Path(self.temp_dir) / "test_custom.csv"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config, csv_config=csv_config
        )
        
        assert Path(result).exists()
        
        # Verify custom delimiter
        with open(result, 'r') as f:
            first_line = f.readline()
            assert ';' in first_line

    def test_export_data_with_excel_config(self):
        """Test Excel export with custom configuration."""
        config = ExportConfig(format=ExportFormat.EXCEL)
        excel_config = ExcelConfig(sheet_name="Custom", auto_filter=False)
        output_path = Path(self.temp_dir) / "test_custom.xlsx"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config, excel_config=excel_config
        )
        
        assert Path(result).exists()

    def test_export_data_with_markdown_config(self):
        """Test Markdown export with custom configuration."""
        config = ExportConfig(format=ExportFormat.MARKDOWN)
        markdown_config = MarkdownConfig(max_rows=2, include_summary=False)
        output_path = Path(self.temp_dir) / "test_custom.md"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config, markdown_config=markdown_config
        )
        
        assert Path(result).exists()
        
        with open(result, 'r') as f:
            content = f.read()
        
        # Should show limited rows message
        assert "Showing first 2 records" in content

    def test_export_data_compression(self):
        """Test export with compression."""
        config = ExportConfig(
            format=ExportFormat.JSON,
            compression=CompressionFormat.GZIP
        )
        output_path = Path(self.temp_dir) / "test_compressed.json"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        assert Path(result).exists()
        assert result.endswith('.gz')

    def test_export_data_zip_compression(self):
        """Test export with ZIP compression."""
        config = ExportConfig(
            format=ExportFormat.JSON,
            compression=CompressionFormat.ZIP
        )
        output_path = Path(self.temp_dir) / "test_compressed.json"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        assert Path(result).exists()
        assert result.endswith('.zip')

    def test_progress_tracking_with_status(self):
        """Test progress tracking with status messages."""
        progress_calls = []
        
        def progress_callback(current, total, status):
            progress_calls.append((current, total, status))
        
        config = ExportConfig(
            format=ExportFormat.JSON,
            progress_callback=progress_callback
        )
        output_path = Path(self.temp_dir) / "test_progress.json"
        
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        assert Path(result).exists()
        assert len(progress_calls) > 0
        # Check that status messages are included
        assert any(len(call) == 3 for call in progress_calls)

    def test_export_data_empty_data(self):
        """Test export with empty data."""
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "test_empty.json"
        
        with pytest.raises(ValueError, match="No data to export"):
            self.export_manager.export_data([], str(output_path), config)

    def test_export_data_invalid_format(self):
        """Test export with invalid format."""
        config = ExportConfig(format="invalid")
        output_path = Path(self.temp_dir) / "test_invalid.json"
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.export_manager.export_data(
                self.sample_data, str(output_path), config
            )

    def test_export_result_file(self):
        """Test exporting a result file."""
        # Create a temporary result file
        result_file = Path(self.temp_dir) / "test_result.json"
        with open(result_file, 'w') as f:
            json.dump(self.sample_data, f)
        
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "exported.json"
        
        result = self.export_manager.export_result_file(
            str(result_file), str(output_path), config
        )
        
        assert Path(result).exists()

    def test_batch_export(self):
        """Test batch export functionality."""
        # Create temporary result files
        result_files = []
        for i in range(3):
            result_file = Path(self.temp_dir) / f"test_result_{i}.json"
            with open(result_file, 'w') as f:
                json.dump(self.sample_data, f)
            result_files.append(str(result_file))
        
        config = ExportConfig(format=ExportFormat.JSON)
        output_dir = Path(self.temp_dir) / "batch_export"
        
        results = self.export_manager.batch_export(
            result_files, str(output_dir), config
        )
        
        assert len(results) == 3
        for result in results:
            assert Path(result).exists()

    def test_incremental_export_json(self):
        """Test incremental JSON export."""
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "test_incremental.json"
        
        result = self.export_manager.incremental_export(
            self.sample_data, str(output_path), config, chunk_size=2
        )
        
        assert Path(result).exists()
        
        # Verify JSON content
        with open(result, 'r') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert "data" in data
        assert len(data["data"]) == 3
        assert data["metadata"]["incremental"] is True
        assert data["metadata"]["chunk_size"] == 2

    def test_incremental_export_csv(self):
        """Test incremental CSV export."""
        config = ExportConfig(format=ExportFormat.CSV)
        output_path = Path(self.temp_dir) / "test_incremental.csv"
        
        result = self.export_manager.incremental_export(
            self.sample_data, str(output_path), config, chunk_size=2
        )
        
        assert Path(result).exists()
        
        # Verify CSV content
        df = pd.read_csv(result)
        assert len(df) == 3
        assert "name" in df.columns

    def test_incremental_export_with_resume(self):
        """Test incremental export with resume functionality."""
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "test_incremental_resume.json"
        
        # Export with resume from index 1
        result = self.export_manager.incremental_export(
            self.sample_data, str(output_path), config, 
            chunk_size=1, resume_from=1
        )
        
        assert Path(result).exists()
        
        # Verify only partial data was exported
        with open(result, 'r') as f:
            data = json.load(f)
        
        # Should have 2 records (starting from index 1)
        assert len(data["data"]) == 2

    def test_validate_export_json(self):
        """Test export validation for JSON."""
        config = ExportConfig(format=ExportFormat.JSON)
        output_path = Path(self.temp_dir) / "test_validation.json"
        
        # Export data
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        # Validate export
        validation = self.export_manager.validate_export(
            result, self.sample_data
        )
        
        assert validation["file_exists"]
        assert validation["row_count"] == 3
        assert validation["data_integrity"]

    def test_validate_export_csv(self):
        """Test export validation for CSV."""
        config = ExportConfig(format=ExportFormat.CSV)
        output_path = Path(self.temp_dir) / "test_validation.csv"
        
        # Export data
        result = self.export_manager.export_data(
            self.sample_data, str(output_path), config
        )
        
        # Validate export
        validation = self.export_manager.validate_export(
            result, self.sample_data
        )
        
        assert validation["file_exists"]
        assert validation["row_count"] == 3
        assert validation["data_integrity"]

    def test_validate_export_nonexistent_file(self):
        """Test validation of non-existent file."""
        validation = self.export_manager.validate_export(
            "nonexistent.json", self.sample_data
        )
        
        assert not validation["file_exists"]
        assert "does not exist" in validation["errors"][0]

    def test_apply_filters_simple(self):
        """Test simple filter application."""
        df = pd.DataFrame(self.sample_data)
        filters = {"vendor": "Tesco"}
        
        filtered_df = self.export_manager._apply_filters(df, filters)
        
        assert len(filtered_df) == 2
        assert all(vendor == "Tesco" for vendor in filtered_df["vendor"])

    def test_apply_filters_complex(self):
        """Test complex filter application."""
        df = pd.DataFrame(self.sample_data)
        filters = {
            "vendor": {"operator": "contains", "value": "Tesco"},
            "price": {"operator": "between", "min": 5, "max": 15}
        }
        
        filtered_df = self.export_manager._apply_filters(df, filters)
        
        assert len(filtered_df) == 2  # Two products match both filters (Tesco products with prices 5.99 and 10.99)

    def test_select_fields(self):
        """Test field selection."""
        df = pd.DataFrame(self.sample_data)
        fields = ["name", "price"]
        
        selected_df = self.export_manager._select_fields(df, fields)
        
        assert list(selected_df.columns) == fields
        assert len(selected_df) == 3

    def test_select_fields_invalid(self):
        """Test field selection with invalid fields."""
        df = pd.DataFrame(self.sample_data)
        fields = ["invalid_field"]
        
        with pytest.raises(ValueError, match="No valid fields found"):
            self.export_manager._select_fields(df, fields)

    def test_compress_file(self):
        """Test file compression."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        with open(test_file, 'w') as f:
            f.write("test content")
        
        compressed_path = self.export_manager._compress_file(test_file, 'gzip')
        
        assert Path(compressed_path).exists()
        assert compressed_path.endswith('.gz')
        assert not test_file.exists()  # Original file should be removed

    def test_load_result_file(self):
        """Test loading result file."""
        # Create a test result file
        result_file = Path(self.temp_dir) / "test_result.json"
        with open(result_file, 'w') as f:
            json.dump(self.sample_data, f)
        
        loaded_data = self.export_manager._load_result_file(result_file)
        
        assert len(loaded_data) == 3
        assert loaded_data[0]["name"] == "Product A"

    def test_load_result_file_compressed(self):
        """Test loading compressed result file."""
        # Create a compressed test file
        import gzip
        result_file = Path(self.temp_dir) / "test_result.json.gz"
        with gzip.open(result_file, 'wt') as f:
            json.dump(self.sample_data, f)
        
        loaded_data = self.export_manager._load_result_file(result_file)
        
        assert len(loaded_data) == 3
        assert loaded_data[0]["name"] == "Product A"

    def test_load_result_file_not_found(self):
        """Test loading non-existent result file."""
        with pytest.raises(FileNotFoundError):
            self.export_manager._load_result_file("nonexistent.json")


class TestExportConfig:
    """Test cases for ExportConfig dataclass."""

    def test_export_config_defaults(self):
        """Test ExportConfig default values."""
        config = ExportConfig(format=ExportFormat.JSON)
        
        assert config.format == ExportFormat.JSON
        assert config.compression == CompressionFormat.NONE
        assert config.pretty_print is True
        assert config.include_metadata is True
        assert config.max_file_size is None
        assert config.chunk_size == 1000
        assert config.progress_callback is None

    def test_export_config_custom(self):
        """Test ExportConfig with custom values."""
        config = ExportConfig(
            format=ExportFormat.CSV,
            compression=CompressionFormat.GZIP,
            pretty_print=False,
            chunk_size=500
        )
        
        assert config.format == ExportFormat.CSV
        assert config.compression == CompressionFormat.GZIP
        assert config.pretty_print is False
        assert config.chunk_size == 500


class TestCSVConfig:
    """Test cases for CSVConfig dataclass."""

    def test_csv_config_defaults(self):
        """Test CSVConfig default values."""
        config = CSVConfig()
        
        assert config.delimiter == ","
        assert config.quotechar == '"'
        assert config.quoting == 0  # csv.QUOTE_MINIMAL
        assert config.encoding == "utf-8"
        assert config.include_header is True

    def test_csv_config_custom(self):
        """Test CSVConfig with custom values."""
        config = CSVConfig(
            delimiter=";",
            encoding="latin-1",
            include_header=False
        )
        
        assert config.delimiter == ";"
        assert config.encoding == "latin-1"
        assert config.include_header is False


class TestExcelConfig:
    """Test cases for ExcelConfig dataclass."""

    def test_excel_config_defaults(self):
        """Test ExcelConfig default values."""
        config = ExcelConfig()
        
        assert config.sheet_name == "Products"
        assert config.auto_filter is True
        assert config.freeze_panes is True
        assert config.header_style is True
        assert config.column_widths is None
        assert config.number_format == "#,##0"

    def test_excel_config_custom(self):
        """Test ExcelConfig with custom values."""
        config = ExcelConfig(
            sheet_name="Custom",
            auto_filter=False,
            column_widths={"name": 20, "price": 15}
        )
        
        assert config.sheet_name == "Custom"
        assert config.auto_filter is False
        assert config.column_widths == {"name": 20, "price": 15}


class TestMarkdownConfig:
    """Test cases for MarkdownConfig dataclass."""

    def test_markdown_config_defaults(self):
        """Test MarkdownConfig default values."""
        config = MarkdownConfig()
        
        assert config.include_table is True
        assert config.include_summary is True
        assert config.max_rows is None
        assert config.sort_by is None
        assert config.sort_ascending is True

    def test_markdown_config_custom(self):
        """Test MarkdownConfig with custom values."""
        config = MarkdownConfig(
            include_table=False,
            max_rows=10,
            sort_by="name"
        )
        
        assert config.include_table is False
        assert config.max_rows == 10
        assert config.sort_by == "name"


class TestEnums:
    """Test cases for Enum classes."""

    def test_export_format_enum(self):
        """Test ExportFormat enum values."""
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.EXCEL.value == "excel"
        assert ExportFormat.MARKDOWN.value == "markdown"

    def test_compression_format_enum(self):
        """Test CompressionFormat enum values."""
        assert CompressionFormat.NONE.value == "none"
        assert CompressionFormat.GZIP.value == "gzip"
        assert CompressionFormat.ZIP.value == "zip"


def test_create_export_manager():
    """Test create_export_manager factory function."""
    export_manager = create_export_manager("test_exports")
    
    assert isinstance(export_manager, ExportManager)
    assert export_manager.output_dir.name == "test_exports"


class TestExportTemplates:
    """Test cases for export template functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.export_manager = ExportManager(self.temp_dir)
        self.sample_data = [
            {
                "vendor": "Tesco",
                "category": "Groceries",
                "created_at": "2024-01-01",
                "price": 5.99
            },
            {
                "vendor": "Asda",
                "category": "Groceries", 
                "created_at": "2024-01-02",
                "price": 4.99
            }
        ]

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_template_management(self):
        """Test template management functionality."""
        # Test listing templates
        templates = self.export_manager.list_templates()
        assert len(templates) >= 4  # Should have default templates
        
        # Test getting a template
        summary_template = self.export_manager.get_template("summary")
        assert summary_template is not None
        assert summary_template.name == "summary"
        
        # Test getting non-existent template
        assert self.export_manager.get_template("nonexistent") is None
        
        # Test adding a custom template
        custom_template = ExportTemplate(
            name="custom",
            description="Custom template",
            config=ExportConfig(format=ExportFormat.JSON)
        )
        self.export_manager.add_template(custom_template)
        
        # Verify template was added
        assert self.export_manager.get_template("custom") is not None
        
        # Test removing template
        assert self.export_manager.remove_template("custom") is True
        assert self.export_manager.get_template("custom") is None
        
        # Test removing non-existent template
        assert self.export_manager.remove_template("nonexistent") is False

    def test_export_with_template(self):
        """Test exporting with a template."""
        output_path = Path(self.temp_dir) / "test_template_export.json"
        
        result = self.export_manager.export_with_template(
            self.sample_data, str(output_path), "summary"
        )
        
        assert Path(result).exists()
        # Verify it's JSON format (summary template uses JSON)
        assert result.endswith('.json')

    def test_export_with_template_overrides(self):
        """Test exporting with template and parameter overrides."""
        output_path = Path(self.temp_dir) / "test_template_override.json"
        
        result = self.export_manager.export_with_template(
            self.sample_data, 
            str(output_path), 
            "summary",
            fields=["vendor", "category"]  # Override template fields
        )
        
        assert Path(result).exists()

    def test_export_with_invalid_template(self):
        """Test exporting with invalid template name."""
        output_path = Path(self.temp_dir) / "test_invalid_template.json"
        
        with pytest.raises(ValueError, match="Template 'invalid' not found"):
            self.export_manager.export_with_template(
                self.sample_data, str(output_path), "invalid"
            )

    def test_export_result_file_with_template(self):
        """Test exporting result file with template."""
        # Create a test result file
        result_file = Path(self.temp_dir) / "test_result.json"
        with open(result_file, 'w') as f:
            json.dump(self.sample_data, f)
        
        output_path = Path(self.temp_dir) / "test_result_template_export.json"
        
        result = self.export_manager.export_result_file_with_template(
            str(result_file), str(output_path), "summary"
        )
        
        assert Path(result).exists()

    def test_default_templates(self):
        """Test that default templates are loaded correctly."""
        templates = self.export_manager.list_templates()
        template_names = [t.name for t in templates]
        
        # Check that all default templates are present
        assert "summary" in template_names
        assert "detailed" in template_names
        assert "csv_standard" in template_names
        assert "compressed" in template_names
        
        # Check template descriptions
        summary_template = self.export_manager.get_template("summary")
        assert "summary statistics" in summary_template.description.lower()
        
        detailed_template = self.export_manager.get_template("detailed")
        assert "detailed formatting" in detailed_template.description.lower() 