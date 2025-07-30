"""
Logging Configuration Module

Provides configurable logging setup for the scraping CLI.
"""

import logging
import sys
from typing import Optional
from enum import Enum


class LogLevel(Enum):
    """Available log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormatter:
    """Custom log formatter with consistent formatting."""
    
    def __init__(self, include_timestamp: bool = True, include_module: bool = True):
        self.include_timestamp = include_timestamp
        self.include_module = include_module
    
    def get_format_string(self) -> str:
        """Get the format string for logging."""
        parts = []
        
        if self.include_timestamp:
            parts.append("%(asctime)s")
        
        if self.include_module:
            parts.append("%(name)s")
        
        parts.extend([
            "%(levelname)s",
            "%(message)s"
        ])
        
        return " - ".join(parts)


class LoggingManager:
    """Manages logging configuration for the scraping CLI."""
    
    def __init__(self):
        self.logger = None
        self.formatter = LogFormatter()
        self._configured = False
    
    def setup_logging(self, 
                     level: LogLevel = LogLevel.INFO,
                     include_timestamp: bool = True,
                     include_module: bool = True,
                     output_stream: Optional[str] = None) -> logging.Logger:
        """Setup logging with the specified configuration."""
        
        # Create logger
        logger = logging.getLogger("scraping_cli")
        logger.setLevel(level.value)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = LogFormatter(include_timestamp, include_module)
        log_format = formatter.get_format_string()
        formatter_obj = logging.Formatter(log_format)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter_obj)
        logger.addHandler(console_handler)
        
        # Create file handler if output_stream is specified
        if output_stream:
            file_handler = logging.FileHandler(output_stream)
            file_handler.setFormatter(formatter_obj)
            logger.addHandler(file_handler)
        
        self.logger = logger
        self._configured = True
        
        return logger
    
    def setup_from_verbose_flag(self, verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
        """Setup logging based on verbose flag."""
        level = LogLevel.DEBUG if verbose else LogLevel.INFO
        return self.setup_logging(level=level, output_stream=log_file)
    
    def get_logger(self) -> Optional[logging.Logger]:
        """Get the configured logger."""
        if not self._configured:
            raise RuntimeError("Logging not configured. Call setup_logging() first.")
        return self.logger
    
    def set_level(self, level: LogLevel) -> None:
        """Set the logging level."""
        if self.logger:
            self.logger.setLevel(level.value)
    
    def add_file_handler(self, file_path: str) -> None:
        """Add a file handler to the logger."""
        if not self.logger:
            raise RuntimeError("Logger not configured. Call setup_logging() first.")
        
        formatter = LogFormatter()
        log_format = formatter.get_format_string()
        formatter_obj = logging.Formatter(log_format)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter_obj)
        self.logger.addHandler(file_handler)
    
    def log_command_start(self, command: str, **kwargs) -> None:
        """Log the start of a command with parameters."""
        if not self.logger:
            return
        
        self.logger.info(f"Starting {command} command")
        for key, value in kwargs.items():
            if value is not None:
                self.logger.info(f"{key}: {value}")
    
    def log_command_end(self, command: str, success: bool = True) -> None:
        """Log the end of a command."""
        if not self.logger:
            return
        
        status = "completed successfully" if success else "failed"
        self.logger.info(f"{command} command {status}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with context."""
        if not self.logger:
            return
        
        if context:
            self.logger.error(f"{context}: {error}")
        else:
            self.logger.error(f"Error: {error}")
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        if not self.logger:
            return
        
        self.logger.warning(message)
    
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        if not self.logger:
            return
        
        self.logger.debug(message)


def create_logging_manager() -> LoggingManager:
    """Create and return a new logging manager instance."""
    return LoggingManager()


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Convenience function to setup logging quickly."""
    manager = create_logging_manager()
    return manager.setup_from_verbose_flag(verbose, log_file) 