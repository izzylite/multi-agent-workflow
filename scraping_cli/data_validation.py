"""
Data Validation and Cleaning System

Provides comprehensive data validation and cleaning capabilities for scraped product data
using Pydantic for schema validation and custom cleaning logic.
"""

import re
import json
import logging
from typing import List, Optional, Dict, Any, Union, Type, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from urllib.parse import urlparse

from pydantic import BaseModel, Field, validator, ValidationError
from pydantic.types import HttpUrl


class ValidationStrictness(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"


class ValidationErrorType(Enum):
    """Types of validation errors."""
    SCHEMA_ERROR = "schema_error"
    TYPE_ERROR = "type_error"
    REQUIRED_FIELD_ERROR = "required_field_error"
    FORMAT_ERROR = "format_error"
    CROSS_FIELD_ERROR = "cross_field_error"
    VENDOR_SPECIFIC_ERROR = "vendor_specific_error"


@dataclass
class ValidationError:
    """Structured validation error information."""
    error_type: ValidationErrorType
    field: str
    message: str
    value: Any = None
    expected_type: Optional[str] = None
    vendor: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    cleaned_data: Optional[Dict[str, Any]] = None
    vendor: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


# Base Pydantic models for validation
class BaseProductSchema(BaseModel):
    """Base schema for product data validation."""
    vendor: str = Field(..., description="Product vendor name")
    category: str = Field(..., description="Product category")
    product_id: str = Field(..., description="Unique product identifier")
    title: str = Field(..., description="Product title")
    price: Optional[str] = Field(None, description="Product price")
    image_url: Optional[HttpUrl] = Field(None, description="Product image URL")
    description: Optional[str] = Field(None, description="Product description")
    url: Optional[HttpUrl] = Field(None, description="Product URL")
    scraped_at: datetime = Field(default_factory=datetime.now, description="Scraping timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('vendor')
    def validate_vendor(cls, v):
        """Validate vendor name."""
        if not v or not v.strip():
            raise ValueError("Vendor name cannot be empty")
        return v.strip()

    @validator('category')
    def validate_category(cls, v):
        """Validate category name."""
        if not v or not v.strip():
            raise ValueError("Category cannot be empty")
        return v.strip()

    @validator('product_id')
    def validate_product_id(cls, v):
        """Validate product ID."""
        if not v or not v.strip():
            raise ValueError("Product ID cannot be empty")
        return v.strip()

    @validator('title')
    def validate_title(cls, v):
        """Validate product title."""
        if not v or not v.strip():
            raise ValueError("Product title cannot be empty")
        return v.strip()

    @validator('price')
    def validate_price(cls, v):
        """Validate price format."""
        if v is not None:
            # Remove currency symbols and whitespace
            cleaned_price = re.sub(r'[£$€¥\s]', '', str(v))
            # Check if it's a valid number
            if not re.match(r'^\d+(\.\d{1,2})?$', cleaned_price):
                raise ValueError("Invalid price format")
        return v

    @validator('description')
    def validate_description(cls, v):
        """Validate and clean description."""
        if v is not None:
            # Remove excessive whitespace
            v = re.sub(r'\s+', ' ', v.strip())
            if len(v) > 10000:  # Limit description length
                v = v[:10000] + "..."
        return v

    class Config:
        extra = "allow"  # Allow additional fields


class TescoProductSchema(BaseProductSchema):
    """Tesco-specific product schema."""
    specifications: Optional[Dict[str, Any]] = Field(default_factory=dict)
    availability: Optional[str] = None
    stock_status: Optional[str] = None
    unit_price: Optional[str] = None
    offer_type: Optional[str] = None
    clubcard_price: Optional[str] = None
    nutrition_info: Optional[Dict[str, Any]] = Field(default_factory=dict)
    allergens: Optional[List[str]] = Field(default_factory=list)
    dietary_info: Optional[List[str]] = Field(default_factory=list)

    @validator('clubcard_price')
    def validate_clubcard_price(cls, v):
        """Validate Tesco Clubcard price."""
        if v is not None:
            cleaned_price = re.sub(r'[£$€¥\s]', '', str(v))
            if not re.match(r'^\d+(\.\d{1,2})?$', cleaned_price):
                raise ValueError("Invalid Clubcard price format")
        return v

    @validator('allergens', 'dietary_info')
    def validate_lists(cls, v):
        """Validate list fields."""
        if v is None:
            return []
        return [item.strip() for item in v if item and item.strip()]

    @validator('stock_status')
    def validate_stock_status(cls, v):
        """Validate Tesco stock status."""
        if v is not None:
            valid_statuses = ['in_stock', 'out_of_stock', 'limited_stock', 'available_for_delivery']
            if str(v).lower() not in valid_statuses:
                raise ValueError(f"Invalid stock status. Must be one of: {valid_statuses}")
        return v

    @validator('offer_type')
    def validate_offer_type(cls, v):
        """Validate Tesco offer types."""
        if v is not None:
            valid_offers = ['clubcard_price', 'buy_one_get_one', 'multibuy', 'clearance', 'reduced']
            if str(v).lower() not in valid_offers:
                raise ValueError(f"Invalid offer type. Must be one of: {valid_offers}")
        return v

    @validator('nutrition_info')
    def validate_nutrition_info(cls, v):
        """Validate Tesco nutrition information."""
        if v is not None:
            valid_nutrition_fields = ['energy', 'fat', 'saturates', 'carbohydrates', 'sugars', 'protein', 'salt']
            for key in v.keys():
                if str(key).lower() not in valid_nutrition_fields:
                    raise ValueError(f"Invalid nutrition field: {key}")
        return v


class AsdaProductSchema(BaseProductSchema):
    """Asda-specific product schema."""
    specifications: Optional[Dict[str, Any]] = Field(default_factory=dict)
    availability: Optional[str] = None
    stock_status: Optional[str] = None
    unit_price: Optional[str] = None
    offer_type: Optional[str] = None
    asda_price: Optional[str] = None
    nutrition_info: Optional[Dict[str, Any]] = Field(default_factory=dict)
    allergens: Optional[List[str]] = Field(default_factory=list)
    dietary_info: Optional[List[str]] = Field(default_factory=list)
    product_code: Optional[str] = None

    @validator('asda_price')
    def validate_asda_price(cls, v):
        """Validate Asda price."""
        if v is not None:
            cleaned_price = re.sub(r'[£$€¥\s]', '', str(v))
            if not re.match(r'^\d+(\.\d{1,2})?$', cleaned_price):
                raise ValueError("Invalid Asda price format")
        return v

    @validator('product_code')
    def validate_product_code(cls, v):
        """Validate Asda product code."""
        if v is not None and not re.match(r'^\d+$', str(v)):
            raise ValueError("Invalid product code format")
        return v

    @validator('stock_status')
    def validate_stock_status(cls, v):
        """Validate Asda stock status."""
        if v is not None:
            valid_statuses = ['in_stock', 'out_of_stock', 'limited_stock', 'available_for_delivery', 'available_for_click_collect']
            if str(v).lower() not in valid_statuses:
                raise ValueError(f"Invalid stock status. Must be one of: {valid_statuses}")
        return v

    @validator('offer_type')
    def validate_offer_type(cls, v):
        """Validate Asda offer types."""
        if v is not None:
            valid_offers = ['asda_price', 'rollback', 'buy_one_get_one', 'multibuy', 'clearance', 'reduced']
            if str(v).lower() not in valid_offers:
                raise ValueError(f"Invalid offer type. Must be one of: {valid_offers}")
        return v

    @validator('availability')
    def validate_availability(cls, v):
        """Validate Asda availability."""
        if v is not None:
            valid_availability = ['available', 'unavailable', 'limited', 'coming_soon']
            if str(v).lower() not in valid_availability:
                raise ValueError(f"Invalid availability. Must be one of: {valid_availability}")
        return v


class CostcoProductSchema(BaseProductSchema):
    """Costco-specific product schema."""
    specifications: Optional[Dict[str, Any]] = Field(default_factory=dict)
    availability: Optional[str] = None
    stock_status: Optional[str] = None
    unit_price: Optional[str] = None
    offer_type: Optional[str] = None
    costco_price: Optional[str] = None
    nutrition_info: Optional[Dict[str, Any]] = Field(default_factory=dict)
    allergens: Optional[List[str]] = Field(default_factory=list)
    dietary_info: Optional[List[str]] = Field(default_factory=list)
    product_code: Optional[str] = None
    membership_required: Optional[bool] = None
    bulk_quantity: Optional[str] = None
    warehouse_location: Optional[str] = None
    online_only: Optional[bool] = None
    in_warehouse_only: Optional[bool] = None

    @validator('costco_price')
    def validate_costco_price(cls, v):
        """Validate Costco price."""
        if v is not None:
            cleaned_price = re.sub(r'[£$€¥\s]', '', str(v))
            if not re.match(r'^\d+(\.\d{1,2})?$', cleaned_price):
                raise ValueError("Invalid Costco price format")
        return v

    @validator('bulk_quantity')
    def validate_bulk_quantity(cls, v):
        """Validate bulk quantity format."""
        if v is not None:
            # Common bulk quantity patterns: "2x500ml", "Pack of 6", "500g x 2"
            if not re.match(r'^(\d+x?\d*[a-zA-Z]*|\d+\s*[a-zA-Z]+\s*x\s*\d+|[a-zA-Z\s]+\d+)$', str(v), re.IGNORECASE):
                raise ValueError("Invalid bulk quantity format")
        return v

    @validator('stock_status')
    def validate_stock_status(cls, v):
        """Validate Costco stock status."""
        if v is not None:
            valid_statuses = ['in_stock', 'out_of_stock', 'limited_stock', 'available_for_delivery', 'warehouse_only']
            if str(v).lower() not in valid_statuses:
                raise ValueError(f"Invalid stock status. Must be one of: {valid_statuses}")
        return v

    @validator('offer_type')
    def validate_offer_type(cls, v):
        """Validate Costco offer types."""
        if v is not None:
            valid_offers = ['costco_price', 'member_price', 'bulk_discount', 'clearance', 'reduced']
            if str(v).lower() not in valid_offers:
                raise ValueError(f"Invalid offer type. Must be one of: {valid_offers}")
        return v

    @validator('warehouse_location')
    def validate_warehouse_location(cls, v):
        """Validate Costco warehouse location."""
        if v is not None:
            # UK Costco warehouse locations
            valid_locations = [
                'birmingham', 'bristol', 'cardiff', 'chester', 'crawley', 'croydon', 'derby',
                'edinburgh', 'glasgow', 'haydock', 'leeds', 'liverpool', 'manchester',
                'newcastle', 'nottingham', 'reading', 'sheffield', 'southampton', 'watford'
            ]
            if str(v).lower() not in valid_locations:
                raise ValueError(f"Invalid warehouse location. Must be one of: {valid_locations}")
        return v

    @validator('membership_required')
    def validate_membership_required(cls, v):
        """Validate Costco membership requirement."""
        if v is not None and not isinstance(v, bool):
            raise ValueError("Membership required must be a boolean value")
        return v


class DataValidator:
    """
    Core data validator using Pydantic for schema validation.
    
    Provides comprehensive validation for product data with configurable strictness
    levels and vendor-specific validation rules.
    """
    
    def __init__(self, strictness: ValidationStrictness = ValidationStrictness.NORMAL):
        """
        Initialize DataValidator.
        
        Args:
            strictness: Validation strictness level
        """
        self.strictness = strictness
        self.logger = logging.getLogger(f"{__name__}.DataValidator")
        
        # Schema mapping for different vendors
        self.schemas = {
            "Tesco": TescoProductSchema,
            "Asda": AsdaProductSchema,
            "Costco": CostcoProductSchema,
            "default": BaseProductSchema
        }
        
        # Validation patterns
        self.patterns = {
            "price": re.compile(r'^[£$€¥]?\s*\d+(\.\d{1,2})?$'),
            "url": re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            "product_id": re.compile(r'^[a-zA-Z0-9_-]+$'),
            "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        }
    
    def validate_product(self, data: Dict[str, Any], vendor: Optional[str] = None) -> ValidationResult:
        """
        Validate product data against appropriate schema.
        
        Args:
            data: Product data dictionary
            vendor: Vendor name for schema selection
            
        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(
            is_valid=True,
            vendor=vendor or data.get('vendor', 'unknown'),
            timestamp=datetime.now()
        )
        
        try:
            # Determine schema based on vendor
            schema_class = self._get_schema_class(vendor or data.get('vendor'))
            
            # Validate against schema
            validated_data = schema_class(**data)
            result.cleaned_data = validated_data.dict()
            
            # Additional cross-field validations
            cross_field_errors = self._validate_cross_fields(data)
            result.errors.extend(cross_field_errors)
            
            # Vendor-specific validations
            vendor_specific_errors = self.validate_vendor_specific_rules(data, vendor or data.get('vendor', 'unknown'))
            result.errors.extend(vendor_specific_errors)
            
            # Update validity based on errors
            result.is_valid = len(result.errors) == 0
            
        except ValidationError as e:
            # Convert Pydantic validation errors to our format
            for error in e.errors():
                result.errors.append(ValidationError(
                    error_type=ValidationErrorType.SCHEMA_ERROR,
                    field=error['loc'][0] if error['loc'] else 'unknown',
                    message=error['msg'],
                    value=error.get('input'),
                    expected_type=str(error.get('type')),
                    vendor=vendor
                ))
            result.is_valid = False
            
        except Exception as e:
            result.errors.append(ValidationError(
                error_type=ValidationErrorType.SCHEMA_ERROR,
                field='unknown',
                message=f"Unexpected validation error: {str(e)}",
                vendor=vendor
            ))
            result.is_valid = False
        
        return result
    
    def _get_schema_class(self, vendor: str) -> Type[BaseModel]:
        """Get appropriate schema class for vendor."""
        return self.schemas.get(vendor, self.schemas["default"])
    
    def _validate_cross_fields(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Perform cross-field validations."""
        errors = []
        
        # Price consistency checks
        price_fields = ['price', 'unit_price', 'clubcard_price', 'asda_price', 'costco_price']
        prices = [data.get(field) for field in price_fields if data.get(field)]
        
        if len(prices) > 1:
            # Check if prices are consistent (within reasonable range)
            try:
                numeric_prices = []
                for price in prices:
                    cleaned = re.sub(r'[£$€¥\s]', '', str(price))
                    numeric_prices.append(float(cleaned))
                
                if max(numeric_prices) - min(numeric_prices) > 100:  # £100 difference threshold
                    errors.append(ValidationError(
                        error_type=ValidationErrorType.CROSS_FIELD_ERROR,
                        field='price',
                        message="Price fields show significant inconsistency",
                        value=prices
                    ))
            except (ValueError, TypeError):
                pass
        
        # URL consistency checks
        if data.get('url') and data.get('image_url'):
            url_domain = urlparse(data['url']).netloc
            image_domain = urlparse(data['image_url']).netloc
            
            if url_domain != image_domain:
                errors.append(ValidationError(
                    error_type=ValidationErrorType.CROSS_FIELD_ERROR,
                    field='image_url',
                    message="Image URL domain doesn't match product URL domain",
                    value=data['image_url']
                ))
        
        return errors
    
    def validate_batch(self, products: List[Dict[str, Any]], vendor: Optional[str] = None) -> List[ValidationResult]:
        """
        Validate a batch of products.
        
        Args:
            products: List of product data dictionaries
            vendor: Vendor name for schema selection
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        for product in products:
            result = self.validate_product(product, vendor)
            results.append(result)
        return results
    
        def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate validation summary from batch results.
        
        Args:
            results: List of ValidationResult objects
            
        Returns:
            Summary statistics
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        invalid = total - valid
        
        error_types = {}
        for result in results:
            for error in result.errors:
                error_types[error.error_type.value] = error_types.get(error.error_type.value, 0) + 1
        
        return {
            "total_products": total,
            "valid_products": valid,
            "invalid_products": invalid,
            "success_rate": valid / total if total > 0 else 0,
            "error_types": error_types,
            "vendors": list(set(r.vendor for r in results if r.vendor))
        }

    def validate_vendor_specific_rules(self, data: Dict[str, Any], vendor: str) -> List[ValidationError]:
        """
        Apply vendor-specific validation rules beyond schema validation.
        
        Args:
            data: Product data dictionary
            vendor: Vendor name
            
        Returns:
            List of vendor-specific validation errors
        """
        errors = []
        
        if vendor == "Tesco":
            errors.extend(self._validate_tesco_specific(data))
        elif vendor == "Asda":
            errors.extend(self._validate_asda_specific(data))
        elif vendor == "Costco":
            errors.extend(self._validate_costco_specific(data))
        
        return errors

    def _validate_tesco_specific(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate Tesco-specific business rules."""
        errors = []
        
        # Check if Clubcard price is lower than regular price
        if data.get('clubcard_price') and data.get('price'):
            try:
                clubcard = float(re.sub(r'[£$€¥\s]', '', str(data['clubcard_price'])))
                regular = float(re.sub(r'[£$€¥\s]', '', str(data['price'])))
                if clubcard >= regular:
                    errors.append(ValidationError(
                        error_type=ValidationErrorType.VENDOR_SPECIFIC_ERROR,
                        field='clubcard_price',
                        message="Clubcard price should be lower than regular price",
                        value=data['clubcard_price'],
                        vendor="Tesco"
                    ))
            except (ValueError, TypeError):
                pass
        
        # Check if product has valid Tesco URL
        if data.get('url'):
            if 'tesco.com' not in str(data['url']).lower():
                errors.append(ValidationError(
                    error_type=ValidationErrorType.VENDOR_SPECIFIC_ERROR,
                    field='url',
                    message="URL must be from tesco.com domain",
                    value=data['url'],
                    vendor="Tesco"
                ))
        
        return errors

    def _validate_asda_specific(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate Asda-specific business rules."""
        errors = []
        
        # Check if Asda price is lower than regular price
        if data.get('asda_price') and data.get('price'):
            try:
                asda_price = float(re.sub(r'[£$€¥\s]', '', str(data['asda_price'])))
                regular = float(re.sub(r'[£$€¥\s]', '', str(data['price'])))
                if asda_price >= regular:
                    errors.append(ValidationError(
                        error_type=ValidationErrorType.VENDOR_SPECIFIC_ERROR,
                        field='asda_price',
                        message="Asda price should be lower than regular price",
                        value=data['asda_price'],
                        vendor="Asda"
                    ))
            except (ValueError, TypeError):
                pass
        
        # Check if product has valid Asda URL
        if data.get('url'):
            if 'asda.com' not in str(data['url']).lower():
                errors.append(ValidationError(
                    error_type=ValidationErrorType.VENDOR_SPECIFIC_ERROR,
                    field='url',
                    message="URL must be from asda.com domain",
                    value=data['url'],
                    vendor="Asda"
                ))
        
        return errors

    def _validate_costco_specific(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate Costco-specific business rules."""
        errors = []
        
        # Check if Costco price is lower than regular price
        if data.get('costco_price') and data.get('price'):
            try:
                costco_price = float(re.sub(r'[£$€¥\s]', '', str(data['costco_price'])))
                regular = float(re.sub(r'[£$€¥\s]', '', str(data['price'])))
                if costco_price >= regular:
                    errors.append(ValidationError(
                        error_type=ValidationErrorType.VENDOR_SPECIFIC_ERROR,
                        field='costco_price',
                        message="Costco price should be lower than regular price",
                        value=data['costco_price'],
                        vendor="Costco"
                    ))
            except (ValueError, TypeError):
                pass
        
        # Check if product has valid Costco URL
        if data.get('url'):
            if 'costco.co.uk' not in str(data['url']).lower():
                errors.append(ValidationError(
                    error_type=ValidationErrorType.VENDOR_SPECIFIC_ERROR,
                    field='url',
                    message="URL must be from costco.co.uk domain",
                    value=data['url'],
                    vendor="Costco"
                ))
        
        # Check if membership products have required flag
        if data.get('membership_required') is None and data.get('costco_price'):
            errors.append(ValidationError(
                error_type=ValidationErrorType.VENDOR_SPECIFIC_ERROR,
                field='membership_required',
                message="Costco products with special pricing should specify membership requirement",
                vendor="Costco"
            ))
        
        return errors


class DataCleaner:
    """
    Data cleaning and normalization system for product data.
    
    Provides comprehensive cleaning capabilities including text normalization,
    price normalization, unit conversion, HTML tag removal, and duplicate detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaner.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.DataCleaner")
        
        # Currency symbols and their codes
        self.currency_symbols = {
            '£': 'GBP',
            '$': 'USD',
            '€': 'EUR',
            '¥': 'JPY'
        }
        
        # Common unit conversions
        self.unit_conversions = {
            'weight': {
                'g': 1,
                'kg': 1000,
                'oz': 28.35,
                'lb': 453.59
            },
            'volume': {
                'ml': 1,
                'l': 1000,
                'fl oz': 29.57,
                'pt': 473.18
            }
        }
        
        # HTML tag patterns
        self.html_patterns = [
            re.compile(r'<[^>]+>'),  # HTML tags
            re.compile(r'&[a-zA-Z]+;'),  # HTML entities
            re.compile(r'&#[0-9]+;'),  # Numeric HTML entities
        ]
        
        # Text normalization patterns
        self.text_patterns = {
            'whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s\-.,!?()]'),
            'multiple_spaces': re.compile(r' +'),
            'newlines': re.compile(r'\n+'),
        }
    
    def clean_product(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a single product data dictionary.
        
        Args:
            data: Product data dictionary
            
        Returns:
            Cleaned product data dictionary
        """
        cleaned_data = data.copy()
        
        # Clean text fields
        text_fields = ['title', 'description', 'category', 'vendor']
        for field in text_fields:
            if field in cleaned_data and cleaned_data[field]:
                cleaned_data[field] = self._normalize_text(cleaned_data[field])
        
        # Clean price fields
        price_fields = ['price', 'unit_price', 'clubcard_price', 'asda_price', 'costco_price']
        for field in price_fields:
            if field in cleaned_data and cleaned_data[field]:
                cleaned_data[field] = self._normalize_price(cleaned_data[field])
        
        # Clean specifications
        if 'specifications' in cleaned_data and cleaned_data['specifications']:
            cleaned_data['specifications'] = self._clean_specifications(cleaned_data['specifications'])
        
        # Clean nutrition info
        if 'nutrition_info' in cleaned_data and cleaned_data['nutrition_info']:
            cleaned_data['nutrition_info'] = self._clean_nutrition_info(cleaned_data['nutrition_info'])
        
        # Clean allergens and dietary info
        for field in ['allergens', 'dietary_info']:
            if field in cleaned_data and cleaned_data[field]:
                cleaned_data[field] = self._clean_list_field(cleaned_data[field])
        
        # Clean URLs
        url_fields = ['url', 'image_url']
        for field in url_fields:
            if field in cleaned_data and cleaned_data[field]:
                cleaned_data[field] = self._clean_url(cleaned_data[field])
        
        return cleaned_data
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing HTML tags and normalizing whitespace."""
        if not text:
            return text
        
        # Remove HTML tags
        for pattern in self.html_patterns:
            text = pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.text_patterns['whitespace'].sub(' ', text)
        text = self.text_patterns['multiple_spaces'].sub(' ', text)
        text = self.text_patterns['newlines'].sub(' ', text)
        
        # Remove excessive special characters
        text = self.text_patterns['special_chars'].sub('', text)
        
        return text.strip()
    
    def _normalize_price(self, price: str) -> str:
        """Normalize price format and currency."""
        if not price:
            return price
        
        # Remove currency symbols and normalize
        price_str = str(price).strip()
        
        # Extract currency symbol
        currency = 'GBP'  # Default to GBP
        for symbol, code in self.currency_symbols.items():
            if symbol in price_str:
                currency = code
                break
        
        # Remove currency symbols and whitespace
        cleaned_price = re.sub(r'[£$€¥\s]', '', price_str)
        
        # Ensure it's a valid number
        try:
            float(cleaned_price)
            return f"{currency} {cleaned_price}"
        except ValueError:
            return price  # Return original if not a valid number
    
    def _clean_specifications(self, specs: Dict[str, Any]) -> Dict[str, Any]:
        """Clean product specifications."""
        cleaned_specs = {}
        
        for key, value in specs.items():
            if isinstance(value, str):
                cleaned_key = self._normalize_text(key)
                cleaned_value = self._normalize_text(value)
                cleaned_specs[cleaned_key] = cleaned_value
            elif isinstance(value, dict):
                cleaned_specs[key] = self._clean_specifications(value)
            else:
                cleaned_specs[key] = value
        
        return cleaned_specs
    
    def _clean_nutrition_info(self, nutrition: Dict[str, Any]) -> Dict[str, Any]:
        """Clean nutrition information."""
        cleaned_nutrition = {}
        
        for key, value in nutrition.items():
            if isinstance(value, str):
                cleaned_key = self._normalize_text(key)
                cleaned_value = self._normalize_text(value)
                cleaned_nutrition[cleaned_key] = cleaned_value
            elif isinstance(value, (int, float)):
                cleaned_nutrition[key] = value
            else:
                cleaned_nutrition[key] = value
        
        return cleaned_nutrition
    
    def _clean_list_field(self, items: List[str]) -> List[str]:
        """Clean list fields like allergens and dietary info."""
        if not items:
            return []
        
        cleaned_items = []
        for item in items:
            if item and isinstance(item, str):
                cleaned_item = self._normalize_text(item)
                if cleaned_item and cleaned_item not in cleaned_items:
                    cleaned_items.append(cleaned_item)
        
        return cleaned_items
    
    def _clean_url(self, url: str) -> str:
        """Clean and validate URL."""
        if not url:
            return url
        
        url_str = str(url).strip()
        
        # Ensure URL has protocol
        if not url_str.startswith(('http://', 'https://')):
            url_str = 'https://' + url_str
        
        # Basic URL validation
        try:
            parsed = urlparse(url_str)
            if parsed.netloc:
                return url_str
        except Exception:
            pass
        
        return url
    
    def detect_duplicates(self, products: List[Dict[str, Any]], 
                         key_fields: Optional[List[str]] = None) -> List[List[int]]:
        """
        Detect duplicate products based on key fields.
        
        Args:
            products: List of product dictionaries
            key_fields: Fields to use for duplicate detection (default: ['title', 'vendor'])
            
        Returns:
            List of duplicate groups (each group contains indices of duplicate products)
        """
        if key_fields is None:
            key_fields = ['title', 'vendor']
        
        # Create signature for each product
        signatures = []
        for i, product in enumerate(products):
            signature_parts = []
            for field in key_fields:
                value = product.get(field, '')
                if isinstance(value, str):
                    # Normalize for comparison
                    normalized = self._normalize_text(value).lower()
                    signature_parts.append(normalized)
                else:
                    signature_parts.append(str(value).lower())
            
            signature = '|'.join(signature_parts)
            signatures.append((signature, i))
        
        # Group by signature
        groups = {}
        for signature, index in signatures:
            if signature in groups:
                groups[signature].append(index)
            else:
                groups[signature] = [index]
        
        # Return only groups with duplicates
        return [indices for indices in groups.values() if len(indices) > 1]
    
    def merge_duplicates(self, products: List[Dict[str, Any]], 
                        duplicate_groups: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Merge duplicate products using intelligent merging strategy.
        
        Args:
            products: List of product dictionaries
            duplicate_groups: Groups of duplicate indices from detect_duplicates
            
        Returns:
            List of merged products
        """
        # Create a set of indices to remove
        indices_to_remove = set()
        merged_products = products.copy()
        
        for group in duplicate_groups:
            if len(group) < 2:
                continue
            
            # Use the first product as the base
            base_index = group[0]
            base_product = merged_products[base_index]
            
            # Merge information from other duplicates
            for duplicate_index in group[1:]:
                duplicate_product = merged_products[duplicate_index]
                base_product = self._merge_product_data(base_product, duplicate_product)
                indices_to_remove.add(duplicate_index)
            
            # Update the base product
            merged_products[base_index] = base_product
        
        # Remove duplicates
        return [product for i, product in enumerate(merged_products) if i not in indices_to_remove]
    
    def _merge_product_data(self, base: Dict[str, Any], duplicate: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two product dictionaries intelligently."""
        merged = base.copy()
        
        # Merge fields with different strategies
        for field, value in duplicate.items():
            if field not in merged or not merged[field]:
                # If base doesn't have the field or it's empty, use duplicate
                merged[field] = value
            elif isinstance(value, list) and isinstance(merged[field], list):
                # Merge lists
                merged[field] = list(set(merged[field] + value))
            elif isinstance(value, dict) and isinstance(merged[field], dict):
                # Recursively merge dictionaries
                merged[field] = self._merge_product_data(merged[field], value)
            elif isinstance(value, str) and isinstance(merged[field], str):
                # For strings, prefer the longer one (more information)
                if len(value) > len(merged[field]):
                    merged[field] = value
        
        return merged
    
    def clean_batch(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean a batch of products.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of cleaned product dictionaries
        """
        cleaned_products = []
        
        for product in products:
            try:
                cleaned_product = self.clean_product(product)
                cleaned_products.append(cleaned_product)
            except Exception as e:
                self.logger.warning(f"Error cleaning product {product.get('product_id', 'unknown')}: {e}")
                # Add original product if cleaning fails
                cleaned_products.append(product)
        
        return cleaned_products
    
    def get_cleaning_summary(self, original_products: List[Dict[str, Any]], 
                           cleaned_products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate cleaning summary statistics.
        
        Args:
            original_products: Original product data
            cleaned_products: Cleaned product data
            
        Returns:
            Summary statistics
        """
        total_products = len(original_products)
        
        # Count changes
        changes = {
            'text_normalized': 0,
            'prices_normalized': 0,
            'urls_cleaned': 0,
            'specifications_cleaned': 0,
            'duplicates_found': 0
        }
        
        # Detect duplicates
        duplicate_groups = self.detect_duplicates(original_products)
        changes['duplicates_found'] = sum(len(group) - 1 for group in duplicate_groups)
        
        # Count other changes (simplified)
        for i, (original, cleaned) in enumerate(zip(original_products, cleaned_products)):
            if original != cleaned:
                changes['text_normalized'] += 1  # Simplified counting
        
                 return {
             "total_products": total_products,
             "cleaned_products": len(cleaned_products),
             "changes_made": changes,
             "duplicate_groups": len(duplicate_groups),
             "total_duplicates": changes['duplicates_found']
         }


class DataEnricher:
    """
    Data enrichment system for product data.
    
    Provides capabilities for category standardization, brand extraction,
    size/weight parsing, and nutritional information extraction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataEnricher.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.DataEnricher")
        
        # Category standardization mappings
        self.category_mappings = {
            'fresh_food': ['fresh', 'produce', 'vegetables', 'fruit', 'meat', 'fish', 'dairy'],
            'pantry': ['pantry', 'dry goods', 'canned', 'baking', 'condiments', 'sauces'],
            'dairy': ['dairy', 'milk', 'cheese', 'yogurt', 'butter', 'eggs'],
            'frozen': ['frozen', 'ice cream', 'frozen meals', 'frozen vegetables'],
            'household': ['household', 'cleaning', 'laundry', 'paper', 'kitchen'],
            'drinks': ['drinks', 'beverages', 'soft drinks', 'juice', 'water', 'alcohol'],
            'health': ['health', 'pharmacy', 'vitamins', 'supplements', 'personal care'],
            'baby': ['baby', 'infant', 'nappies', 'baby food', 'baby care'],
            'bakery': ['bakery', 'bread', 'pastries', 'cakes', 'baked goods'],
            'meat_fish': ['meat', 'fish', 'poultry', 'seafood', 'protein'],
            'fruit_vegetables': ['fruit', 'vegetables', 'produce', 'fresh produce']
        }
        
        # Brand extraction patterns
        self.brand_patterns = [
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:brand|product|food|drink)\b', re.IGNORECASE),
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:organic|natural|premium)\b', re.IGNORECASE),
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:co\.|company|ltd|limited)\b', re.IGNORECASE),
        ]
        
        # Size/weight parsing patterns
        self.size_patterns = {
            'weight': [
                re.compile(r'(\d+(?:\.\d+)?)\s*(g|kg|oz|lb)', re.IGNORECASE),
                re.compile(r'(\d+(?:\.\d+)?)\s*(gram|kilo|ounce|pound)', re.IGNORECASE),
            ],
            'volume': [
                re.compile(r'(\d+(?:\.\d+)?)\s*(ml|l|fl\s*oz|pt)', re.IGNORECASE),
                re.compile(r'(\d+(?:\.\d+)?)\s*(milliliter|liter|fluid\s*ounce|pint)', re.IGNORECASE),
            ],
            'count': [
                re.compile(r'(\d+)\s*(pack|piece|item|unit)', re.IGNORECASE),
                re.compile(r'pack\s*of\s*(\d+)', re.IGNORECASE),
            ]
        }
        
        # Nutritional information patterns
        self.nutrition_patterns = {
            'energy': [
                re.compile(r'(\d+(?:\.\d+)?)\s*(kcal|cal|kj)', re.IGNORECASE),
                re.compile(r'energy[:\s]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            ],
            'fat': [
                re.compile(r'(\d+(?:\.\d+)?)\s*g\s*fat', re.IGNORECASE),
                re.compile(r'fat[:\s]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            ],
            'protein': [
                re.compile(r'(\d+(?:\.\d+)?)\s*g\s*protein', re.IGNORECASE),
                re.compile(r'protein[:\s]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            ],
            'carbohydrates': [
                re.compile(r'(\d+(?:\.\d+)?)\s*g\s*carb', re.IGNORECASE),
                re.compile(r'carbohydrates[:\s]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            ],
            'sugar': [
                re.compile(r'(\d+(?:\.\d+)?)\s*g\s*sugar', re.IGNORECASE),
                re.compile(r'sugar[:\s]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            ],
            'salt': [
                re.compile(r'(\d+(?:\.\d+)?)\s*g\s*salt', re.IGNORECASE),
                re.compile(r'salt[:\s]*(\d+(?:\.\d+)?)', re.IGNORECASE),
            ]
        }
    
    def enrich_product(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single product with additional extracted information.
        
        Args:
            data: Product data dictionary
            
        Returns:
            Enriched product data dictionary
        """
        enriched_data = data.copy()
        
        # Extract and standardize category
        if 'category' in enriched_data:
            enriched_data['standardized_category'] = self._standardize_category(enriched_data['category'])
        
        # Extract brand information
        if 'title' in enriched_data:
            brand = self._extract_brand(enriched_data['title'])
            if brand:
                enriched_data['extracted_brand'] = brand
        
        # Extract size/weight information
        if 'title' in enriched_data or 'description' in enriched_data:
            size_info = self._extract_size_weight(enriched_data.get('title', '') + ' ' + enriched_data.get('description', ''))
            if size_info:
                enriched_data['extracted_size'] = size_info
        
        # Extract nutritional information
        if 'description' in enriched_data or 'specifications' in enriched_data:
            nutrition_info = self._extract_nutrition_info(
                enriched_data.get('description', '') + ' ' + str(enriched_data.get('specifications', ''))
            )
            if nutrition_info:
                enriched_data['extracted_nutrition'] = nutrition_info
        
        return enriched_data
    
    def _standardize_category(self, category: str) -> str:
        """Standardize product category using predefined mappings."""
        if not category:
            return category
        
        category_lower = str(category).lower()
        
        # Find matching standardized category
        for standard_category, keywords in self.category_mappings.items():
            for keyword in keywords:
                if keyword in category_lower:
                    return standard_category
        
        # If no match found, return original category
        return category
    
    def _extract_brand(self, text: str) -> Optional[str]:
        """Extract brand information from product text."""
        if not text:
            return None
        
        # Try different brand extraction patterns
        for pattern in self.brand_patterns:
            match = pattern.search(text)
            if match:
                brand = match.group(1).strip()
                # Clean up the brand name
                brand = re.sub(r'\s+', ' ', brand)
                return brand
        
        # Look for common brand patterns at the beginning of the title
        words = text.split()
        if len(words) >= 2:
            # Check if first word looks like a brand (capitalized, not too long)
            potential_brand = words[0]
            if (potential_brand[0].isupper() and 
                len(potential_brand) > 2 and 
                len(potential_brand) < 20):
                return potential_brand
        
        return None
    
    def _extract_size_weight(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract size and weight information from product text."""
        if not text:
            return None
        
        size_info = {}
        
        # Extract weight information
        for pattern in self.size_patterns['weight']:
            matches = pattern.findall(text)
            for match in matches:
                value, unit = match
                size_info['weight'] = {
                    'value': float(value),
                    'unit': str(unit).lower(),
                    'original_text': f"{value}{unit}"
                }
                break
        
        # Extract volume information
        for pattern in self.size_patterns['volume']:
            matches = pattern.findall(text)
            for match in matches:
                value, unit = match
                size_info['volume'] = {
                    'value': float(value),
                    'unit': str(unit).lower(),
                    'original_text': f"{value}{unit}"
                }
                break
        
        # Extract count information
        for pattern in self.size_patterns['count']:
            matches = pattern.findall(text)
            for match in matches:
                value = match
                size_info['count'] = {
                    'value': int(value),
                    'unit': 'pieces',
                    'original_text': f"{value} pieces"
                }
                break
        
        return size_info if size_info else None
    
    def _extract_nutrition_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract nutritional information from product text."""
        if not text:
            return None
        
        nutrition_info = {}
        
        # Extract each type of nutritional information
        for nutrient, patterns in self.nutrition_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    value = float(match.group(1))
                    nutrition_info[nutrient] = {
                        'value': value,
                        'unit': self._get_nutrition_unit(nutrient, match.group(0)),
                        'original_text': match.group(0)
                    }
                    break
        
        return nutrition_info if nutrition_info else None
    
    def _get_nutrition_unit(self, nutrient: str, original_text: str) -> str:
        """Get the appropriate unit for nutritional information."""
        unit_mappings = {
            'energy': 'kcal',
            'fat': 'g',
            'protein': 'g',
            'carbohydrates': 'g',
            'sugar': 'g',
            'salt': 'g'
        }
        
        # Check if unit is specified in the original text
        if 'kcal' in str(original_text).lower():
            return 'kcal'
        elif 'kj' in str(original_text).lower():
            return 'kj'
        elif 'cal' in str(original_text).lower():
            return 'cal'
        
        return unit_mappings.get(nutrient, 'g')
    
    def enrich_batch(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich a batch of products.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of enriched product dictionaries
        """
        enriched_products = []
        
        for product in products:
            try:
                enriched_product = self.enrich_product(product)
                enriched_products.append(enriched_product)
            except Exception as e:
                self.logger.warning(f"Error enriching product {product.get('product_id', 'unknown')}: {e}")
                # Add original product if enrichment fails
                enriched_products.append(product)
        
        return enriched_products
    
    def get_enrichment_summary(self, original_products: List[Dict[str, Any]], 
                             enriched_products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate enrichment summary statistics.
        
        Args:
            original_products: Original product data
            enriched_products: Enriched product data
            
        Returns:
            Summary statistics
        """
        total_products = len(original_products)
        
        # Count enrichments
        enrichments = {
            'categories_standardized': 0,
            'brands_extracted': 0,
            'sizes_extracted': 0,
            'nutrition_extracted': 0
        }
        
        for enriched in enriched_products:
            if 'standardized_category' in enriched:
                enrichments['categories_standardized'] += 1
            if 'extracted_brand' in enriched:
                enrichments['brands_extracted'] += 1
            if 'extracted_size' in enriched:
                enrichments['sizes_extracted'] += 1
            if 'extracted_nutrition' in enriched:
                enrichments['nutrition_extracted'] += 1
        
                 return {
             "total_products": total_products,
             "enriched_products": len(enriched_products),
             "enrichments_made": enrichments,
             "enrichment_rate": sum(enrichments.values()) / (total_products * 4) if total_products > 0 else 0
         }


class ValidationReporter:
    """
    Comprehensive validation reporting and error logging system.
    
    Provides structured reporting, error logging, and monitoring capabilities
    for the data validation and cleaning pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ValidationReporter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ValidationReporter")
        
        # Reporting configuration
        self.report_levels = {
            'summary': 'basic',
            'detailed': 'comprehensive',
            'debug': 'verbose'
        }
        
        # Error severity levels
        self.severity_levels = {
            'critical': 1,
            'error': 2,
            'warning': 3,
            'info': 4
        }
        
        # Initialize error tracking
        self.error_history = []
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'error_counts': {},
            'vendor_stats': {},
            'field_error_stats': {}
        }
    
    def log_validation_result(self, result: ValidationResult) -> None:
        """
        Log a validation result for tracking and reporting.
        
        Args:
            result: ValidationResult object
        """
        # Update statistics
        self.validation_stats['total_validations'] += 1
        
        if result.is_valid:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        # Track vendor statistics
        vendor = result.vendor or 'unknown'
        if vendor not in self.validation_stats['vendor_stats']:
            self.validation_stats['vendor_stats'][vendor] = {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'error_types': {}
            }
        
        self.validation_stats['vendor_stats'][vendor]['total'] += 1
        if result.is_valid:
            self.validation_stats['vendor_stats'][vendor]['successful'] += 1
        else:
            self.validation_stats['vendor_stats'][vendor]['failed'] += 1
        
        # Track errors
        for error in result.errors:
            # Error type tracking
            error_type = error.error_type.value
            if error_type not in self.validation_stats['error_counts']:
                self.validation_stats['error_counts'][error_type] = 0
            self.validation_stats['error_counts'][error_type] += 1
            
            # Field error tracking
            field = error.field
            if field not in self.validation_stats['field_error_stats']:
                self.validation_stats['field_error_stats'][field] = 0
            self.validation_stats['field_error_stats'][field] += 1
            
            # Vendor error tracking
            if error.vendor:
                if error_type not in self.validation_stats['vendor_stats'][error.vendor]['error_types']:
                    self.validation_stats['vendor_stats'][error.vendor]['error_types'][error_type] = 0
                self.validation_stats['vendor_stats'][error.vendor]['error_types'][error_type] += 1
        
        # Add to error history
        self.error_history.append({
            'timestamp': result.timestamp,
            'vendor': result.vendor,
            'is_valid': result.is_valid,
            'error_count': len(result.errors),
            'errors': [self._serialize_error(error) for error in result.errors]
        })
    
    def _serialize_error(self, error: ValidationError) -> Dict[str, Any]:
        """Serialize validation error for logging."""
        return {
            'error_type': error.error_type.value,
            'field': error.field,
            'message': error.message,
            'value': str(error.value) if error.value is not None else None,
            'expected_type': error.expected_type,
            'vendor': error.vendor,
            'timestamp': error.timestamp.isoformat()
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Summary report dictionary
        """
        total = self.validation_stats['total_validations']
        successful = self.validation_stats['successful_validations']
        failed = self.validation_stats['failed_validations']
        
        report = {
            'summary': {
                'total_validations': total,
                'successful_validations': successful,
                'failed_validations': failed,
                'success_rate': successful / total if total > 0 else 0,
                'failure_rate': failed / total if total > 0 else 0
            },
            'error_analysis': {
                'error_types': self.validation_stats['error_counts'],
                'field_errors': self.validation_stats['field_error_stats'],
                'top_error_fields': self._get_top_error_fields(),
                'top_error_types': self._get_top_error_types()
            },
            'vendor_analysis': {
                'vendor_stats': self.validation_stats['vendor_stats'],
                'vendor_success_rates': self._calculate_vendor_success_rates(),
                'vendor_error_patterns': self._analyze_vendor_error_patterns()
            },
            'trends': {
                'recent_errors': self._get_recent_errors(10),
                'error_trends': self._analyze_error_trends()
            }
        }
        
        return report
    
    def _get_top_error_fields(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top error-prone fields."""
        sorted_fields = sorted(
            self.validation_stats['field_error_stats'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [{'field': field, 'error_count': count} for field, count in sorted_fields[:limit]]
    
    def _get_top_error_types(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top error types."""
        sorted_types = sorted(
            self.validation_stats['error_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [{'error_type': error_type, 'count': count} for error_type, count in sorted_types[:limit]]
    
    def _calculate_vendor_success_rates(self) -> Dict[str, float]:
        """Calculate success rates for each vendor."""
        rates = {}
        for vendor, stats in self.validation_stats['vendor_stats'].items():
            total = stats['total']
            successful = stats['successful']
            rates[vendor] = successful / total if total > 0 else 0
        return rates
    
    def _analyze_vendor_error_patterns(self) -> Dict[str, Dict[str, int]]:
        """Analyze error patterns by vendor."""
        patterns = {}
        for vendor, stats in self.validation_stats['vendor_stats'].items():
            patterns[vendor] = stats['error_types']
        return patterns
    
    def _get_recent_errors(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent validation errors."""
        return self.error_history[-limit:] if self.error_history else []
    
    def _analyze_error_trends(self) -> Dict[str, Any]:
        """Analyze error trends over time."""
        if len(self.error_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple trend analysis
        recent_errors = sum(1 for entry in self.error_history[-10:] if not entry['is_valid'])
        earlier_errors = sum(1 for entry in self.error_history[-20:-10] if not entry['is_valid'])
        
        if recent_errors > earlier_errors:
            trend = 'increasing'
        elif recent_errors < earlier_errors:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'recent_error_rate': recent_errors / 10,
            'earlier_error_rate': earlier_errors / 10
        }
    
    def generate_detailed_report(self, vendor: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a detailed report with specific vendor focus.
        
        Args:
            vendor: Optional vendor filter
            
        Returns:
            Detailed report dictionary
        """
        base_report = self.generate_summary_report()
        
        if vendor:
            # Filter for specific vendor
            vendor_errors = [
                entry for entry in self.error_history
                if entry['vendor'] == vendor
            ]
            
            detailed_report = {
                'vendor': vendor,
                'vendor_specific_stats': self.validation_stats['vendor_stats'].get(vendor, {}),
                'vendor_errors': vendor_errors,
                'vendor_error_analysis': self._analyze_vendor_errors(vendor)
            }
            
            base_report['detailed_analysis'] = detailed_report
        
        return base_report
    
    def _analyze_vendor_errors(self, vendor: str) -> Dict[str, Any]:
        """Analyze errors for a specific vendor."""
        vendor_errors = [
            entry for entry in self.error_history
            if entry['vendor'] == vendor
        ]
        
        if not vendor_errors:
            return {'error_count': 0, 'common_errors': []}
        
        # Analyze common error patterns
        all_errors = []
        for entry in vendor_errors:
            all_errors.extend(entry['errors'])
        
        error_counts = {}
        for error in all_errors:
            error_type = error['error_type']
            if error_type not in error_counts:
                error_counts[error_type] = 0
            error_counts[error_type] += 1
        
        return {
            'error_count': len(all_errors),
            'common_errors': sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def export_report(self, report_type: str = 'summary', 
                     format: str = 'json', 
                     file_path: Optional[str] = None) -> str:
        """
        Export report in specified format.
        
        Args:
            report_type: Type of report ('summary', 'detailed')
            format: Export format ('json', 'csv', 'html')
            file_path: Optional file path for export
            
        Returns:
            Exported report content or file path
        """
        if report_type == 'summary':
            report_data = self.generate_summary_report()
        else:
            report_data = self.generate_detailed_report()
        
        if format == 'json':
            content = json.dumps(report_data, indent=2, default=str)
        elif format == 'csv':
            content = self._export_to_csv(report_data)
        elif format == 'html':
            content = self._export_to_html(report_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return file_path
        else:
            return content
    
    def _export_to_csv(self, report_data: Dict[str, Any]) -> str:
        """Export report to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write summary
        writer.writerow(['Metric', 'Value'])
        summary = report_data.get('summary', {})
        for key, value in summary.items():
            writer.writerow([key, value])
        
        # Write error analysis
        writer.writerow([])
        writer.writerow(['Error Type', 'Count'])
        error_types = report_data.get('error_analysis', {}).get('error_types', {})
        for error_type, count in error_types.items():
            writer.writerow([error_type, count])
        
        return output.getvalue()
    
    def _export_to_html(self, report_data: Dict[str, Any]) -> str:
        """Export report to HTML format."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; padding: 10px; border: 1px solid #ccc; }
                .metric { margin: 10px 0; }
                .error { color: red; }
                .success { color: green; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Data Validation Report</h1>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">Total Validations: {total_validations}</div>
                <div class="metric">Successful: <span class="success">{successful}</span></div>
                <div class="metric">Failed: <span class="error">{failed}</span></div>
                <div class="metric">Success Rate: {success_rate:.2%}</div>
            </div>
            
            <div class="section">
                <h2>Error Analysis</h2>
                <table>
                    <tr><th>Error Type</th><th>Count</th></tr>
                    {error_rows}
                </table>
            </div>
            
            <div class="section">
                <h2>Vendor Analysis</h2>
                <table>
                    <tr><th>Vendor</th><th>Success Rate</th></tr>
                    {vendor_rows}
                </table>
            </div>
        </body>
        </html>
        """
        
        # Format the template
        summary = report_data.get('summary', {})
        error_types = report_data.get('error_analysis', {}).get('error_types', {})
        vendor_rates = report_data.get('vendor_analysis', {}).get('vendor_success_rates', {})
        
        error_rows = ''.join([
            f'<tr><td>{error_type}</td><td>{count}</td></tr>'
            for error_type, count in error_types.items()
        ])
        
        vendor_rows = ''.join([
            f'<tr><td>{vendor}</td><td>{rate:.2%}</td></tr>'
            for vendor, rate in vendor_rates.items()
        ])
        
        return html_template.format(
            total_validations=summary.get('total_validations', 0),
            successful=summary.get('successful_validations', 0),
            failed=summary.get('failed_validations', 0),
            success_rate=summary.get('success_rate', 0),
            error_rows=error_rows,
            vendor_rows=vendor_rows
        )
    
    def clear_history(self) -> None:
        """Clear error history and reset statistics."""
        self.error_history = []
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'error_counts': {},
            'vendor_stats': {},
            'field_error_stats': {}
        } 