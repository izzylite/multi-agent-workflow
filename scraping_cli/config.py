"""
Configuration Module

Manages CLI flags, options, and default values for the scraping CLI.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Vendor(Enum):
    """Supported vendor platforms."""
    TESCO = "tesco"
    ASDA = "asda"
    COSTCO = "costco"


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    CSV = "csv"
    TABLE = "table"


@dataclass
class ScrapeConfig:
    """Configuration for scrape command."""
    vendor: Vendor
    urls: Optional[List[str]] = None
    url_file: Optional[str] = None
    urls_from_stdin: bool = False
    category: Optional[str] = None
    output: Optional[str] = None
    format: OutputFormat = OutputFormat.JSON
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # At least one URL input method must be provided
        input_methods = sum([
            1 if self.urls else 0,
            1 if self.url_file else 0,
            1 if self.urls_from_stdin else 0
        ])
        
        if input_methods == 0:
            raise ValueError("No URL input method specified. Use --urls, --url-file, or --urls-from-stdin")
        
        if input_methods > 1:
            raise ValueError("Multiple URL input methods specified. Use only one of --urls, --url-file, or --urls-from-stdin")


@dataclass
class ListConfig:
    """Configuration for list command."""
    format: OutputFormat = OutputFormat.TABLE


@dataclass
class ExportConfig:
    """Configuration for export command."""
    output: str
    format: OutputFormat = OutputFormat.JSON
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.output:
            raise ValueError("Output file path must be provided")


@dataclass
class ViewConfig:
    """Configuration for view command."""
    file_path: str
    fields: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.file_path:
            raise ValueError("File path must be provided")


@dataclass
class DeleteConfig:
    """Configuration for delete command."""
    file_path: str
    confirm: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.file_path:
            raise ValueError("File path must be provided")


@dataclass
class StatsConfig:
    """Configuration for stats command."""
    format: OutputFormat = OutputFormat.TABLE


@dataclass
class GlobalConfig:
    """Global configuration for the CLI application."""
    verbose: bool = False
    log_level: str = field(init=False)
    
    def __post_init__(self):
        """Set log level based on verbose flag."""
        self.log_level = "DEBUG" if self.verbose else "INFO"


class ConfigurationManager:
    """Manages configuration for the scraping CLI."""
    
    def __init__(self):
        self.global_config: Optional[GlobalConfig] = None
        self.scrape_config: Optional[ScrapeConfig] = None
        self.list_config: Optional[ListConfig] = None
        self.export_config: Optional[ExportConfig] = None
        self.view_config: Optional[ViewConfig] = None
        self.delete_config: Optional[DeleteConfig] = None
        self.stats_config: Optional[StatsConfig] = None
    
    def parse_scrape_config(self, args) -> ScrapeConfig:
        """Parse scrape command configuration from arguments."""
        try:
            vendor = Vendor(args.vendor)
            format_enum = OutputFormat(args.format)
            
            config = ScrapeConfig(
                vendor=vendor,
                urls=getattr(args, 'urls', None),
                url_file=getattr(args, 'url_file', None),
                urls_from_stdin=getattr(args, 'urls_from_stdin', False),
                category=args.category,
                output=args.output,
                format=format_enum
            )
            
            self.scrape_config = config
            return config
            
        except ValueError as e:
            raise ValueError(f"Invalid scrape configuration: {e}")
    
    def parse_list_config(self, args) -> ListConfig:
        """Parse list command configuration from arguments."""
        try:
            format_enum = OutputFormat(args.format)
            
            config = ListConfig(format=format_enum)
            self.list_config = config
            return config
            
        except ValueError as e:
            raise ValueError(f"Invalid list configuration: {e}")
    
    def parse_export_config(self, args) -> ExportConfig:
        """Parse export command configuration from arguments."""
        try:
            format_enum = OutputFormat(args.format)
            
            config = ExportConfig(
                output=args.output,
                format=format_enum
            )
            
            self.export_config = config
            return config
            
        except ValueError as e:
            raise ValueError(f"Invalid export configuration: {e}")
    
    def parse_view_config(self, args) -> ViewConfig:
        """Parse view command configuration from arguments."""
        try:
            config = ViewConfig(
                file_path=args.file_path,
                fields=getattr(args, 'fields', None)
            )
            
            self.view_config = config
            return config
            
        except ValueError as e:
            raise ValueError(f"Invalid view configuration: {e}")
    
    def parse_delete_config(self, args) -> DeleteConfig:
        """Parse delete command configuration from arguments."""
        try:
            config = DeleteConfig(
                file_path=args.file_path,
                confirm=getattr(args, 'confirm', False)
            )
            
            self.delete_config = config
            return config
            
        except ValueError as e:
            raise ValueError(f"Invalid delete configuration: {e}")
    
    def parse_stats_config(self, args) -> StatsConfig:
        """Parse stats command configuration from arguments."""
        try:
            format_enum = OutputFormat(args.format)
            
            config = StatsConfig(
                format=format_enum
            )
            
            self.stats_config = config
            return config
            
        except ValueError as e:
            raise ValueError(f"Invalid stats configuration: {e}")
    
    def parse_global_config(self, args) -> GlobalConfig:
        """Parse global configuration from arguments."""
        config = GlobalConfig(verbose=args.verbose)
        self.global_config = config
        return config
    
    def get_default_output_path(self, vendor: Vendor, format_enum: OutputFormat) -> str:
        """Generate a default output path based on vendor and format."""
        timestamp = self._get_timestamp()
        filename = f"{vendor.value}_{timestamp}.{format_enum.value}"
        return os.path.join("results", filename)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for file naming."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def validate_vendor(self, vendor_str: str) -> Vendor:
        """Validate and convert vendor string to Vendor enum."""
        try:
            return Vendor(vendor_str)
        except ValueError:
            raise ValueError(f"Unsupported vendor: {vendor_str}. Supported vendors: {[v.value for v in Vendor]}")
    
    def validate_format(self, format_str: str) -> OutputFormat:
        """Validate and convert format string to OutputFormat enum."""
        try:
            return OutputFormat(format_str)
        except ValueError:
            raise ValueError(f"Unsupported format: {format_str}. Supported formats: {[f.value for f in OutputFormat]}")
    
    def get_supported_vendors(self) -> List[str]:
        """Get list of supported vendor names."""
        return [vendor.value for vendor in Vendor]
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported format names."""
        return [format.value for format in OutputFormat]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert current configuration to dictionary."""
        config_dict = {
            "global": self.global_config.__dict__ if self.global_config else None,
            "scrape": self.scrape_config.__dict__ if self.scrape_config else None,
            "list": self.list_config.__dict__ if self.list_config else None,
            "export": self.export_config.__dict__ if self.export_config else None
        }
        return config_dict


def create_config_manager() -> ConfigurationManager:
    """Create and return a new configuration manager instance."""
    return ConfigurationManager() 