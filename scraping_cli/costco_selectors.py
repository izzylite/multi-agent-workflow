"""
Costco Selector Mapping Module

Provides comprehensive CSS selector and XPath mapping for Costco's UK wholesale platform.
This module centralizes all Costco-specific selectors with fallback strategies for robust
element selection and resilience against site structure changes.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class CostcoElementType(Enum):
    """Types of elements that can be selected on Costco's website."""
    PRODUCT_CARD = "product_card"
    PRODUCT_TITLE = "product_title"
    PRODUCT_PRICE = "product_price"
    PRODUCT_IMAGE = "product_image"
    PRODUCT_LINK = "product_link"
    PRODUCT_DESCRIPTION = "product_description"
    PRODUCT_SPECIFICATIONS = "product_specifications"
    PRODUCT_NUTRITION = "product_nutrition"
    PRODUCT_ALLERGENS = "product_allergens"
    PRODUCT_DIETARY = "product_dietary"
    PRODUCT_CODE = "product_code"
    PRODUCT_AVAILABILITY = "product_availability"
    PRODUCT_STOCK_STATUS = "product_stock_status"
    PRODUCT_UNIT_PRICE = "product_unit_price"
    PRODUCT_OFFER_TYPE = "product_offer_type"
    PRODUCT_COSTCO_PRICE = "product_costco_price"
    PRODUCT_MEMBERSHIP_REQUIRED = "product_membership_required"
    PRODUCT_BULK_QUANTITY = "product_bulk_quantity"
    PRODUCT_WAREHOUSE_LOCATION = "product_warehouse_location"
    PRODUCT_ONLINE_ONLY = "product_online_only"
    PRODUCT_IN_WAREHOUSE_ONLY = "product_in_warehouse_only"
    SEARCH_INPUT = "search_input"
    SEARCH_BUTTON = "search_button"
    CATEGORY_MENU = "category_menu"
    CATEGORY_LINK = "category_link"
    PAGINATION_NEXT = "pagination_next"
    PAGINATION_INFO = "pagination_info"
    LOAD_MORE = "load_more"
    INFINITE_SCROLL = "infinite_scroll"
    COOKIE_CONSENT = "cookie_consent"
    LOGIN_PROMPT = "login_prompt"
    MEMBERSHIP_VERIFICATION = "membership_verification"
    LOCATION_SELECTOR = "location_selector"
    CAPTCHA = "captcha"
    LOADING_INDICATOR = "loading_indicator"


@dataclass
class CostcoSelector:
    """Represents a selector with priority and fallback options."""
    primary: str
    fallbacks: List[str]
    element_type: CostcoElementType
    description: str
    priority: int = 1


class CostcoSelectorMapping:
    """
    Comprehensive selector mapping for Costco's website elements.
    """
    
    def __init__(self):
        self.selectors = self._initialize_selectors()
    
    def _initialize_selectors(self) -> Dict[CostcoElementType, CostcoSelector]:
        """Initialize all Costco selectors with primary and fallback options."""
        return {
            CostcoElementType.PRODUCT_CARD: CostcoSelector(
                primary=".product-item",
                fallbacks=[
                    ".product-card",
                    "[data-testid='product-card']",
                    ".product-tile",
                    ".item-card",
                    "[data-testid='product-item']",
                    ".product-grid .item",
                    ".product-list .product"
                ],
                element_type=CostcoElementType.PRODUCT_CARD,
                description="Product card container elements"
            ),
            
            CostcoElementType.PRODUCT_TITLE: CostcoSelector(
                primary=".product-title",
                fallbacks=[
                    "h1",
                    "[data-testid='product-title']",
                    ".item-title",
                    "h2",
                    ".product-name",
                    ".product-heading",
                    ".item-name"
                ],
                element_type=CostcoElementType.PRODUCT_TITLE,
                description="Product title elements"
            ),
            
            CostcoElementType.PRODUCT_PRICE: CostcoSelector(
                primary=".price",
                fallbacks=[
                    "[data-testid='product-price']",
                    "span.price",
                    ".costco-price",
                    ".member-price",
                    ".price-value",
                    ".product-price",
                    ".item-price"
                ],
                element_type=CostcoElementType.PRODUCT_PRICE,
                description="Product price elements"
            ),
            
            CostcoElementType.PRODUCT_IMAGE: CostcoSelector(
                primary=".product-image img",
                fallbacks=[
                    "img",
                    "[data-testid='product-image']",
                    ".item-image img",
                    ".product-photo img",
                    ".product-thumbnail img",
                    ".item-photo img"
                ],
                element_type=CostcoElementType.PRODUCT_IMAGE,
                description="Product image elements"
            ),
            
            CostcoElementType.PRODUCT_LINK: CostcoSelector(
                primary=".product-item a",
                fallbacks=[
                    "a",
                    ".product-card a",
                    ".item-link",
                    ".product-link",
                    "[data-testid='product-link']"
                ],
                element_type=CostcoElementType.PRODUCT_LINK,
                description="Product link elements"
            ),
            
            CostcoElementType.PRODUCT_DESCRIPTION: CostcoSelector(
                primary=".product-description",
                fallbacks=[
                    "[data-testid='product-description']",
                    ".item-description",
                    ".product-summary",
                    ".description",
                    ".product-details .summary"
                ],
                element_type=CostcoElementType.PRODUCT_DESCRIPTION,
                description="Product description elements"
            ),
            
            CostcoElementType.PRODUCT_SPECIFICATIONS: CostcoSelector(
                primary=".product-specifications",
                fallbacks=[
                    "[data-testid='product-specifications']",
                    ".product-details",
                    ".specifications",
                    ".product-info",
                    ".item-specs"
                ],
                element_type=CostcoElementType.PRODUCT_SPECIFICATIONS,
                description="Product specifications elements"
            ),
            
            CostcoElementType.PRODUCT_NUTRITION: CostcoSelector(
                primary=".nutrition-table",
                fallbacks=[
                    "[data-testid='nutrition-info']",
                    ".nutrition-information",
                    ".nutrition-details",
                    ".nutrition-facts",
                    ".nutrition-label"
                ],
                element_type=CostcoElementType.PRODUCT_NUTRITION,
                description="Product nutrition information elements"
            ),
            
            CostcoElementType.PRODUCT_ALLERGENS: CostcoSelector(
                primary=".allergen-information",
                fallbacks=[
                    "[data-testid='allergens']",
                    ".allergens",
                    ".allergen-list",
                    ".allergen-details",
                    ".allergen-info"
                ],
                element_type=CostcoElementType.PRODUCT_ALLERGENS,
                description="Product allergen information elements"
            ),
            
            CostcoElementType.PRODUCT_DIETARY: CostcoSelector(
                primary=".dietary-information",
                fallbacks=[
                    "[data-testid='dietary-info']",
                    ".dietary-labels",
                    ".dietary-tags",
                    ".dietary-badges",
                    ".dietary-info"
                ],
                element_type=CostcoElementType.PRODUCT_DIETARY,
                description="Product dietary information elements"
            ),
            
            CostcoElementType.PRODUCT_CODE: CostcoSelector(
                primary=".product-code",
                fallbacks=[
                    "[data-testid='product-code']",
                    ".sku",
                    ".item-code",
                    ".product-sku",
                    ".product-id"
                ],
                element_type=CostcoElementType.PRODUCT_CODE,
                description="Product code/SKU elements"
            ),
            
            CostcoElementType.PRODUCT_AVAILABILITY: CostcoSelector(
                primary=".availability",
                fallbacks=[
                    "[data-testid='availability']",
                    ".stock-status",
                    ".availability-status",
                    ".in-stock",
                    ".stock-info"
                ],
                element_type=CostcoElementType.PRODUCT_AVAILABILITY,
                description="Product availability elements"
            ),
            
            CostcoElementType.PRODUCT_STOCK_STATUS: CostcoSelector(
                primary=".stock-status",
                fallbacks=[
                    "[data-testid='stock-status']",
                    ".availability",
                    ".stock-indicator",
                    ".inventory-status",
                    ".stock-level"
                ],
                element_type=CostcoElementType.PRODUCT_STOCK_STATUS,
                description="Product stock status elements"
            ),
            
            CostcoElementType.PRODUCT_UNIT_PRICE: CostcoSelector(
                primary=".unit-price",
                fallbacks=[
                    "[data-testid='unit-price']",
                    ".price-per-unit",
                    ".unit-cost",
                    ".price-per-item",
                    ".unit-pricing"
                ],
                element_type=CostcoElementType.PRODUCT_UNIT_PRICE,
                description="Product unit price elements"
            ),
            
            CostcoElementType.PRODUCT_OFFER_TYPE: CostcoSelector(
                primary=".offer-type",
                fallbacks=[
                    "[data-testid='offer-type']",
                    ".deal-type",
                    ".promotion-type",
                    ".offer-badge",
                    ".deal-badge"
                ],
                element_type=CostcoElementType.PRODUCT_OFFER_TYPE,
                description="Product offer/deal type elements"
            ),
            
            CostcoElementType.PRODUCT_COSTCO_PRICE: CostcoSelector(
                primary=".costco-price",
                fallbacks=[
                    "[data-testid='costco-price']",
                    ".member-price",
                    ".costco-member-price",
                    ".wholesale-price",
                    ".bulk-price"
                ],
                element_type=CostcoElementType.PRODUCT_COSTCO_PRICE,
                description="Costco-specific price elements"
            ),
            
            CostcoElementType.PRODUCT_MEMBERSHIP_REQUIRED: CostcoSelector(
                primary=".membership-required",
                fallbacks=[
                    "[data-testid='membership-info']",
                    ".membership-notice",
                    ".member-only",
                    ".membership-badge",
                    ".member-required"
                ],
                element_type=CostcoElementType.PRODUCT_MEMBERSHIP_REQUIRED,
                description="Membership requirement elements"
            ),
            
            CostcoElementType.PRODUCT_BULK_QUANTITY: CostcoSelector(
                primary=".bulk-quantity",
                fallbacks=[
                    "[data-testid='quantity-info']",
                    ".quantity-notice",
                    ".bulk-size",
                    ".quantity-info",
                    ".bulk-info"
                ],
                element_type=CostcoElementType.PRODUCT_BULK_QUANTITY,
                description="Bulk quantity information elements"
            ),
            
            CostcoElementType.PRODUCT_WAREHOUSE_LOCATION: CostcoSelector(
                primary=".warehouse-location",
                fallbacks=[
                    "[data-testid='location-info']",
                    ".location-notice",
                    ".store-location",
                    ".warehouse-info",
                    ".location-details"
                ],
                element_type=CostcoElementType.PRODUCT_WAREHOUSE_LOCATION,
                description="Warehouse location elements"
            ),
            
            CostcoElementType.PRODUCT_ONLINE_ONLY: CostcoSelector(
                primary=".online-only",
                fallbacks=[
                    "[data-testid='online-only']",
                    ".web-only",
                    ".online-exclusive",
                    ".online-badge",
                    ".web-exclusive"
                ],
                element_type=CostcoElementType.PRODUCT_ONLINE_ONLY,
                description="Online-only product indicators"
            ),
            
            CostcoElementType.PRODUCT_IN_WAREHOUSE_ONLY: CostcoSelector(
                primary=".in-warehouse-only",
                fallbacks=[
                    "[data-testid='warehouse-only']",
                    ".store-only",
                    ".warehouse-exclusive",
                    ".in-store-only",
                    ".warehouse-badge"
                ],
                element_type=CostcoElementType.PRODUCT_IN_WAREHOUSE_ONLY,
                description="Warehouse-only product indicators"
            ),
            
            CostcoElementType.SEARCH_INPUT: CostcoSelector(
                primary="input[type='search']",
                fallbacks=[
                    "input[name='search']",
                    "#search-input",
                    ".search-input",
                    "[data-testid='search-input']",
                    ".search-box input",
                    "input[placeholder*='search']"
                ],
                element_type=CostcoElementType.SEARCH_INPUT,
                description="Search input elements"
            ),
            
            CostcoElementType.SEARCH_BUTTON: CostcoSelector(
                primary="button[type='submit']",
                fallbacks=[
                    ".search-button",
                    "[data-testid='search-button']",
                    ".search-submit",
                    "button.search-btn",
                    ".search-icon"
                ],
                element_type=CostcoElementType.SEARCH_BUTTON,
                description="Search button elements"
            ),
            
            CostcoElementType.CATEGORY_MENU: CostcoSelector(
                primary=".category-menu",
                fallbacks=[
                    ".main-navigation",
                    "nav",
                    ".department-menu",
                    ".category-nav",
                    "[data-testid='category-menu']",
                    ".main-menu"
                ],
                element_type=CostcoElementType.CATEGORY_MENU,
                description="Category navigation menu elements"
            ),
            
            CostcoElementType.CATEGORY_LINK: CostcoSelector(
                primary=".category-menu a",
                fallbacks=[
                    ".main-navigation a",
                    "nav a",
                    ".department-menu a",
                    ".category-nav a",
                    "[data-testid='category-link']"
                ],
                element_type=CostcoElementType.CATEGORY_LINK,
                description="Category link elements"
            ),
            
            CostcoElementType.PAGINATION_NEXT: CostcoSelector(
                primary=".pagination-next",
                fallbacks=[
                    ".next-page",
                    "[data-testid='pagination-next']",
                    ".pagination .next",
                    "a[rel='next']",
                    ".load-more",
                    ".next-button"
                ],
                element_type=CostcoElementType.PAGINATION_NEXT,
                description="Next page pagination elements"
            ),
            
            CostcoElementType.PAGINATION_INFO: CostcoSelector(
                primary=".pagination-info",
                fallbacks=[
                    ".page-info",
                    "[data-testid='pagination-info']",
                    ".results-info",
                    ".page-numbers",
                    ".pagination-details"
                ],
                element_type=CostcoElementType.PAGINATION_INFO,
                description="Pagination information elements"
            ),
            
            CostcoElementType.LOAD_MORE: CostcoSelector(
                primary=".load-more",
                fallbacks=[
                    ".infinite-scroll",
                    "[data-testid='load-more']",
                    ".show-more",
                    ".load-more-button",
                    ".infinite-load"
                ],
                element_type=CostcoElementType.LOAD_MORE,
                description="Load more content elements"
            ),
            
            CostcoElementType.INFINITE_SCROLL: CostcoSelector(
                primary=".infinite-scroll",
                fallbacks=[
                    ".load-more",
                    "[data-testid='infinite-scroll']",
                    ".scroll-load",
                    ".auto-load",
                    ".continuous-load"
                ],
                element_type=CostcoElementType.INFINITE_SCROLL,
                description="Infinite scroll elements"
            ),
            
            CostcoElementType.COOKIE_CONSENT: CostcoSelector(
                primary=".cookie-banner",
                fallbacks=[
                    ".cookie-consent",
                    "[data-testid='cookie-banner']",
                    ".gdpr-banner",
                    ".cookie-notice",
                    ".cookie-popup"
                ],
                element_type=CostcoElementType.COOKIE_CONSENT,
                description="Cookie consent banner elements"
            ),
            
            CostcoElementType.LOGIN_PROMPT: CostcoSelector(
                primary=".login-modal",
                fallbacks=[
                    ".signin-modal",
                    "[data-testid='login-modal']",
                    ".auth-modal",
                    ".login-overlay",
                    ".signin-overlay"
                ],
                element_type=CostcoElementType.LOGIN_PROMPT,
                description="Login prompt modal elements"
            ),
            
            CostcoElementType.MEMBERSHIP_VERIFICATION: CostcoSelector(
                primary=".membership-required",
                fallbacks=[
                    ".member-verification",
                    "[data-testid='membership-prompt']",
                    ".login-required",
                    ".member-only",
                    ".membership-check"
                ],
                element_type=CostcoElementType.MEMBERSHIP_VERIFICATION,
                description="Membership verification elements"
            ),
            
            CostcoElementType.LOCATION_SELECTOR: CostcoSelector(
                primary=".location-selector",
                fallbacks=[
                    ".postcode-input",
                    "[data-testid='location-input']",
                    ".store-locator",
                    ".location-prompt",
                    ".location-input"
                ],
                element_type=CostcoElementType.LOCATION_SELECTOR,
                description="Location selector elements"
            ),
            
            CostcoElementType.CAPTCHA: CostcoSelector(
                primary=".captcha",
                fallbacks=[
                    "[data-testid='captcha']",
                    ".recaptcha",
                    ".g-recaptcha",
                    ".captcha-challenge",
                    ".captcha-form"
                ],
                element_type=CostcoElementType.CAPTCHA,
                description="CAPTCHA challenge elements"
            ),
            
            CostcoElementType.LOADING_INDICATOR: CostcoSelector(
                primary=".loading",
                fallbacks=[
                    ".spinner",
                    "[data-testid='loading']",
                    ".loading-indicator",
                    ".loading-spinner",
                    ".loading-animation"
                ],
                element_type=CostcoElementType.LOADING_INDICATOR,
                description="Loading indicator elements"
            )
        }
    
    def get_selectors(self, element_type: CostcoElementType) -> List[str]:
        """
        Get all selectors for a specific element type.
        
        Args:
            element_type: The type of element to get selectors for
            
        Returns:
            List of selectors (primary + fallbacks)
        """
        if element_type not in self.selectors:
            return []
        
        selector_obj = self.selectors[element_type]
        return [selector_obj.primary] + selector_obj.fallbacks
    
    def get_primary_selector(self, element_type: CostcoElementType) -> Optional[str]:
        """
        Get the primary selector for a specific element type.
        
        Args:
            element_type: The type of element to get the primary selector for
            
        Returns:
            Primary selector string or None if not found
        """
        if element_type not in self.selectors:
            return None
        
        return self.selectors[element_type].primary
    
    def get_fallback_selectors(self, element_type: CostcoElementType) -> List[str]:
        """
        Get fallback selectors for a specific element type.
        
        Args:
            element_type: The type of element to get fallback selectors for
            
        Returns:
            List of fallback selectors
        """
        if element_type not in self.selectors:
            return []
        
        return self.selectors[element_type].fallbacks
    
    def add_selector(self, element_type: CostcoElementType, primary: str,
                    fallbacks: List[str], description: str, priority: int = 1):
        """
        Add a new selector mapping.
        
        Args:
            element_type: The type of element
            primary: Primary selector
            fallbacks: List of fallback selectors
            description: Description of the selector
            priority: Priority level (lower is higher priority)
        """
        self.selectors[element_type] = CostcoSelector(
            primary=primary,
            fallbacks=fallbacks,
            element_type=element_type,
            description=description,
            priority=priority
        )
    
    def update_selector(self, element_type: CostcoElementType, primary: str = None,
                       fallbacks: List[str] = None, description: str = None):
        """
        Update an existing selector mapping.
        
        Args:
            element_type: The type of element to update
            primary: New primary selector (optional)
            fallbacks: New fallback selectors (optional)
            description: New description (optional)
        """
        if element_type not in self.selectors:
            return
        
        current = self.selectors[element_type]
        
        if primary is not None:
            current.primary = primary
        if fallbacks is not None:
            current.fallbacks = fallbacks
        if description is not None:
            current.description = description
    
    def get_all_element_types(self) -> List[CostcoElementType]:
        """
        Get all available element types.
        
        Returns:
            List of all element types
        """
        return list(self.selectors.keys())
    
    def get_selector_info(self, element_type: CostcoElementType) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a selector.
        
        Args:
            element_type: The type of element to get info for
            
        Returns:
            Dictionary with selector information or None if not found
        """
        if element_type not in self.selectors:
            return None
        
        selector = self.selectors[element_type]
        return {
            'primary': selector.primary,
            'fallbacks': selector.fallbacks,
            'description': selector.description,
            'priority': selector.priority,
            'element_type': selector.element_type.value
        }


# Global instance
costco_selectors = CostcoSelectorMapping()


def get_costco_selectors() -> CostcoSelectorMapping:
    """
    Get the global Costco selector mapping instance.
    
    Returns:
        CostcoSelectorMapping instance
    """
    return costco_selectors 