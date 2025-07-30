"""
URL Processing Module

Handles URL validation, parsing, and normalization for the scraping CLI.
"""

import re
import sys
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, urljoin, urlunparse, parse_qs, urlencode
from dataclasses import dataclass
from enum import Enum

import validators


class URLType(Enum):
    """Types of URLs that can be processed."""
    PRODUCT = "product"
    CATEGORY = "category"
    SEARCH = "search"
    UNKNOWN = "unknown"


@dataclass
class ParsedURL:
    """Represents a parsed and validated URL."""
    original: str
    normalized: str
    scheme: str
    netloc: str
    path: str
    query: str
    fragment: str
    url_type: URLType
    vendor: Optional[str] = None
    
    def __post_init__(self):
        """Validate the parsed URL."""
        if not self.normalized:
            raise ValueError("Normalized URL cannot be empty")
        
        if not self.scheme or not self.netloc:
            raise ValueError("URL must have both scheme and netloc")


class URLValidator:
    """Handles URL validation and parsing."""
    
    def __init__(self):
        self.tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'ref', 'source', 'campaign'
        }
    
    def validate_url(self, url: str) -> bool:
        """Validate if a URL is properly formatted."""
        if not url or not isinstance(url, str):
            return False
        
        # Use validators library for basic URL validation
        return validators.url(url)
    
    def parse_url(self, url: str) -> urlparse:
        """Parse a URL using urllib.parse."""
        return urlparse(url)
    
    def normalize_url(self, url: str) -> str:
        """Normalize a URL by adding protocol and cleaning parameters."""
        if not url:
            raise ValueError("URL cannot be empty")
        
        # Add https:// if no scheme is provided
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Parse the URL
        parsed = urlparse(url)
        
        # Clean query parameters (remove tracking params)
        if parsed.query:
            query_params = parse_qs(parsed.query)
            cleaned_params = {
                k: v for k, v in query_params.items() 
                if k.lower() not in self.tracking_params
            }
            
            # Rebuild query string
            if cleaned_params:
                query = urlencode(cleaned_params, doseq=True)
            else:
                query = ''
        else:
            query = ''
        
        # Rebuild URL without tracking parameters
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            query,
            parsed.fragment
        ))
        
        return normalized
    
    def extract_domain(self, url: str) -> str:
        """Extract the domain from a URL."""
        parsed = urlparse(url)
        return parsed.netloc.lower()
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """Check if two URLs belong to the same domain."""
        domain1 = self.extract_domain(url1)
        domain2 = self.extract_domain(url2)
        return domain1 == domain2


class VendorURLValidator:
    """Validates URLs for specific vendors."""
    
    def __init__(self):
        self.vendor_patterns = {
            'tesco': {
                'domain': r'tesco\.com',
                'patterns': [
                    r'/groceries/',
                    r'/food/',
                    r'/product/',
                    r'/search\?'
                ]
            },
            'asda': {
                'domain': r'asda\.com',
                'patterns': [
                    r'/groceries/',
                    r'/food/',
                    r'/product/',
                    r'/search\?'
                ]
            },
            'costco': {
                'domain': r'costco\.co\.uk',
                'patterns': [
                    r'/groceries/',
                    r'/food/',
                    r'/product/',
                    r'/search\?'
                ]
            }
        }
    
    def validate_vendor_url(self, url: str, vendor: str) -> bool:
        """Validate if a URL matches the expected pattern for a vendor."""
        if vendor not in self.vendor_patterns:
            return False
        
        vendor_config = self.vendor_patterns[vendor]
        
        # Check domain
        domain_pattern = vendor_config['domain']
        if not re.search(domain_pattern, url, re.IGNORECASE):
            return False
        
        # Check if URL matches any of the vendor patterns
        for pattern in vendor_config['patterns']:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        return False
    
    def get_vendor_from_url(self, url: str) -> Optional[str]:
        """Determine the vendor from a URL."""
        for vendor, config in self.vendor_patterns.items():
            if re.search(config['domain'], url, re.IGNORECASE):
                return vendor
        return None


class URLCategorizer:
    """Categorizes URLs based on their structure and content."""
    
    def __init__(self):
        self.product_patterns = [
            r'/product/',
            r'/item/',
            r'/p/',
            r'product_id=',
            r'item_id='
        ]
        
        self.category_patterns = [
            r'/category/',
            r'/department/',
            r'/aisle/',
            r'/section/'
        ]
        
        self.search_patterns = [
            r'/search',
            r'search=',
            r'q=',
            r'query='
        ]
    
    def categorize_url(self, url: str) -> URLType:
        """Categorize a URL based on its structure."""
        url_lower = url.lower()
        
        # Check for product patterns
        for pattern in self.product_patterns:
            if re.search(pattern, url_lower):
                return URLType.PRODUCT
        
        # Check for category patterns
        for pattern in self.category_patterns:
            if re.search(pattern, url_lower):
                return URLType.CATEGORY
        
        # Check for search patterns
        for pattern in self.search_patterns:
            if re.search(pattern, url_lower):
                return URLType.SEARCH
        
        return URLType.UNKNOWN


class URLProcessor:
    """Main URL processing class that orchestrates validation, parsing, and categorization."""
    
    def __init__(self):
        self.validator = URLValidator()
        self.vendor_validator = VendorURLValidator()
        self.categorizer = URLCategorizer()
    
    def process_url(self, url: str, expected_vendor: Optional[str] = None) -> ParsedURL:
        """Process a single URL through the complete pipeline."""
        # Validate URL format
        if not self.validator.validate_url(url):
            raise ValueError(f"Invalid URL format: {url}")
        
        # Normalize URL
        normalized_url = self.validator.normalize_url(url)
        
        # Parse URL components
        parsed = self.validator.parse_url(normalized_url)
        
        # Determine vendor
        vendor = self.vendor_validator.get_vendor_from_url(normalized_url)
        
        # Validate vendor if expected
        if expected_vendor and vendor != expected_vendor:
            raise ValueError(f"URL does not match expected vendor '{expected_vendor}'. Found: {vendor}")
        
        # Categorize URL
        url_type = self.categorizer.categorize_url(normalized_url)
        
        return ParsedURL(
            original=url,
            normalized=normalized_url,
            scheme=parsed.scheme,
            netloc=parsed.netloc,
            path=parsed.path,
            query=parsed.query,
            fragment=parsed.fragment,
            url_type=url_type,
            vendor=vendor
        )
    
    def process_urls(self, urls: List[str], expected_vendor: Optional[str] = None) -> List[ParsedURL]:
        """Process multiple URLs."""
        processed_urls = []
        errors = []
        
        for i, url in enumerate(urls):
            try:
                processed_url = self.process_url(url, expected_vendor)
                processed_urls.append(processed_url)
            except ValueError as e:
                errors.append(f"URL {i+1}: {e}")
        
        if errors:
            raise ValueError(f"URL processing errors:\n" + "\n".join(errors))
        
        return processed_urls
    
    def deduplicate_urls(self, urls: List[ParsedURL]) -> List[ParsedURL]:
        """Remove duplicate URLs from the list."""
        seen = set()
        unique_urls = []
        
        for url in urls:
            if url.normalized not in seen:
                seen.add(url.normalized)
                unique_urls.append(url)
        
        return unique_urls


def create_url_processor() -> URLProcessor:
    """Create and return a new URL processor instance."""
    return URLProcessor() 