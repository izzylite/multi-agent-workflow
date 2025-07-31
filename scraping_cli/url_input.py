"""
URL Input Handler Module

Handles multiple URL input methods for the scraping CLI.
"""

import sys
import os
from typing import List, Optional
from pathlib import Path


class URLInputHandler:
    """Handles different URL input methods."""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.urls', '.list'}
    
    def get_urls_from_args(self, urls: List[str]) -> List[str]:
        """Extract URLs from command-line arguments."""
        if not urls:
            return []
        
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            url = url.strip()
            if url:  # Skip empty strings
                cleaned_urls.append(url)
        
        return cleaned_urls
    
    def get_urls_from_file(self, file_path: str) -> List[str]:
        """Read URLs from a file (one per line)."""
        if not file_path:
            raise ValueError("File path cannot be empty")
        
        # Validate file path
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"URL file not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check file extension
        if str(path.suffix).lower() not in self.supported_extensions:
            print(f"Warning: File extension '{path.suffix}' is not in supported extensions: {self.supported_extensions}")
        
        # Read URLs from file
        urls = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        urls.append(line)
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(path, 'r', encoding='latin-1') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            urls.append(line)
            except Exception as e:
                raise ValueError(f"Error reading file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
        
        if not urls:
            raise ValueError(f"No valid URLs found in file: {file_path}")
        
        return urls
    
    def get_urls_from_stdin(self) -> List[str]:
        """Read URLs from standard input (one per line)."""
        urls = []
        
        # Check if stdin is a terminal (interactive mode)
        if sys.stdin.isatty():
            print("Enter URLs (one per line, press Ctrl+D or Ctrl+Z when done):")
        
        try:
            for line in sys.stdin:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    urls.append(line)
        except KeyboardInterrupt:
            print("\nInput interrupted by user")
            return []
        except EOFError:
            pass  # Normal end of input
        
        if not urls:
            raise ValueError("No URLs provided via stdin")
        
        return urls
    
    def validate_url_input(self, args) -> List[str]:
        """Validate and extract URLs based on the provided arguments."""
        urls = []
        
        # Check which input method was used
        if hasattr(args, 'urls') and args.urls:
            urls = self.get_urls_from_args(args.urls)
        elif hasattr(args, 'url_file') and args.url_file:
            urls = self.get_urls_from_file(args.url_file)
        elif hasattr(args, 'urls_from_stdin') and args.urls_from_stdin:
            urls = self.get_urls_from_stdin()
        else:
            raise ValueError("No URL input method specified")
        
        # Validate that we have URLs
        if not urls:
            raise ValueError("No valid URLs provided")
        
        return urls
    
    def get_input_method_description(self, args) -> str:
        """Get a description of the input method used."""
        if hasattr(args, 'urls') and args.urls:
            return f"command-line arguments ({len(args.urls)} URLs)"
        elif hasattr(args, 'url_file') and args.url_file:
            return f"file: {args.url_file}"
        elif hasattr(args, 'urls_from_stdin') and args.urls_from_stdin:
            return "standard input"
        else:
            return "unknown"
    
    def get_supported_file_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.supported_extensions)


def create_url_input_handler() -> URLInputHandler:
    """Create and return a new URL input handler instance."""
    return URLInputHandler() 