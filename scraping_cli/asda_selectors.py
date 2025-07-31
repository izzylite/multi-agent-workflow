"""
Asda Selector Mapping Module

Provides comprehensive CSS selectors and XPath expressions for Asda's website elements,
including product cards, prices, details, images, and specifications. Includes fallback
strategies for selector changes and maintains a robust selector mapping system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class AsdaElementType(Enum):
    """Types of elements that can be selected on Asda's website."""
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
    PRODUCT_ASDA_PRICE = "product_asda_price"
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
    AGE_VERIFICATION = "age_verification"
    LOCATION_SELECTOR = "location_selector"


@dataclass
class AsdaSelector:
    """Represents a selector with priority and fallback options."""
    primary: str
    fallbacks: List[str]
    element_type: AsdaElementType
    description: str
    priority: int = 1  # Lower number = higher priority


class AsdaSelectorMapping:
    """
    Comprehensive selector mapping for Asda's website elements.
    
    Provides robust selectors with fallback strategies for all major elements
    on Asda's grocery platform, ensuring reliable data extraction even when
    the site structure changes.
    """
    
    def __init__(self):
        self.selectors = self._initialize_selectors()
    
    def _initialize_selectors(self) -> Dict[AsdaElementType, AsdaSelector]:
        """Initialize all selectors for Asda elements."""
        return {
            # Product Card Selectors
            AsdaElementType.PRODUCT_CARD: AsdaSelector(
                primary=".product-list .product-item",
                fallbacks=[
                    "[data-testid='product-card']",
                    ".product-card",
                    ".product-item",
                    "[data-testid='product-item']",
                    ".product-tile",
                    "[data-testid='product-tile']"
                ],
                element_type=AsdaElementType.PRODUCT_CARD,
                description="Product card container elements"
            ),
            
            # Product Title Selectors
            AsdaElementType.PRODUCT_TITLE: AsdaSelector(
                primary="h1[data-testid='product-title']",
                fallbacks=[
                    ".product-title",
                    "h1",
                    "[data-testid='product-name']",
                    ".product-name",
                    ".product-title h1",
                    "[data-testid='title']"
                ],
                element_type=AsdaElementType.PRODUCT_TITLE,
                description="Product title/name elements"
            ),
            
            # Product Price Selectors
            AsdaElementType.PRODUCT_PRICE: AsdaSelector(
                primary="span.price",
                fallbacks=[
                    "[data-testid='product-price']",
                    ".price",
                    ".product-price",
                    "[data-testid='price']",
                    ".current-price",
                    "[data-testid='current-price']"
                ],
                element_type=AsdaElementType.PRODUCT_PRICE,
                description="Product price elements"
            ),
            
            # Product Image Selectors
            AsdaElementType.PRODUCT_IMAGE: AsdaSelector(
                primary="img[data-testid='product-image']",
                fallbacks=[
                    ".product-image img",
                    "img",
                    "[data-testid='product-image'] img",
                    ".product-img img",
                    "img[alt*='product']"
                ],
                element_type=AsdaElementType.PRODUCT_IMAGE,
                description="Product image elements"
            ),
            
            # Product Link Selectors
            AsdaElementType.PRODUCT_LINK: AsdaSelector(
                primary="a[href*='/product/']",
                fallbacks=[
                    ".product-card a",
                    ".product-item a",
                    "a[data-testid='product-link']",
                    ".product-link",
                    "a[href*='groceries.asda.com']"
                ],
                element_type=AsdaElementType.PRODUCT_LINK,
                description="Product detail page links"
            ),
            
            # Product Description Selectors
            AsdaElementType.PRODUCT_DESCRIPTION: AsdaSelector(
                primary=".product-description",
                fallbacks=[
                    "[data-testid='product-description']",
                    ".description",
                    ".product-summary",
                    "[data-testid='description']",
                    ".product-info .description"
                ],
                element_type=AsdaElementType.PRODUCT_DESCRIPTION,
                description="Product description elements"
            ),
            
            # Product Specifications Selectors
            AsdaElementType.PRODUCT_SPECIFICATIONS: AsdaSelector(
                primary=".product-specifications",
                fallbacks=[
                    "[data-testid='product-specifications']",
                    ".product-details",
                    ".specifications",
                    "[data-testid='specifications']",
                    ".product-info .specs"
                ],
                element_type=AsdaElementType.PRODUCT_SPECIFICATIONS,
                description="Product specifications elements"
            ),
            
            # Product Nutrition Selectors
            AsdaElementType.PRODUCT_NUTRITION: AsdaSelector(
                primary=".nutrition-table",
                fallbacks=[
                    "[data-testid='nutrition-info']",
                    ".nutrition-information",
                    ".nutrition-details",
                    "[data-testid='nutrition']",
                    ".product-nutrition"
                ],
                element_type=AsdaElementType.PRODUCT_NUTRITION,
                description="Product nutrition information elements"
            ),
            
            # Product Allergens Selectors
            AsdaElementType.PRODUCT_ALLERGENS: AsdaSelector(
                primary=".allergen-information",
                fallbacks=[
                    "[data-testid='allergens']",
                    ".allergens",
                    ".allergen-details",
                    "[data-testid='allergen-info']",
                    ".product-allergens"
                ],
                element_type=AsdaElementType.PRODUCT_ALLERGENS,
                description="Product allergen information elements"
            ),
            
            # Product Dietary Selectors
            AsdaElementType.PRODUCT_DIETARY: AsdaSelector(
                primary=".dietary-information",
                fallbacks=[
                    "[data-testid='dietary-info']",
                    ".dietary-labels",
                    ".dietary-details",
                    "[data-testid='dietary']",
                    ".product-dietary"
                ],
                element_type=AsdaElementType.PRODUCT_DIETARY,
                description="Product dietary information elements"
            ),
            
            # Product Code Selectors
            AsdaElementType.PRODUCT_CODE: AsdaSelector(
                primary=".product-code",
                fallbacks=[
                    "[data-testid='product-code']",
                    ".sku",
                    ".product-sku",
                    "[data-testid='sku']",
                    ".item-code"
                ],
                element_type=AsdaElementType.PRODUCT_CODE,
                description="Product code/SKU elements"
            ),
            
            # Product Availability Selectors
            AsdaElementType.PRODUCT_AVAILABILITY: AsdaSelector(
                primary=".product-availability",
                fallbacks=[
                    "[data-testid='availability']",
                    ".availability",
                    ".stock-status",
                    "[data-testid='stock-status']",
                    ".product-stock"
                ],
                element_type=AsdaElementType.PRODUCT_AVAILABILITY,
                description="Product availability elements"
            ),
            
            # Product Stock Status Selectors
            AsdaElementType.PRODUCT_STOCK_STATUS: AsdaSelector(
                primary=".stock-status",
                fallbacks=[
                    "[data-testid='stock-status']",
                    ".stock",
                    ".availability-status",
                    "[data-testid='availability']",
                    ".product-availability"
                ],
                element_type=AsdaElementType.PRODUCT_STOCK_STATUS,
                description="Product stock status elements"
            ),
            
            # Product Unit Price Selectors
            AsdaElementType.PRODUCT_UNIT_PRICE: AsdaSelector(
                primary=".unit-price",
                fallbacks=[
                    "[data-testid='unit-price']",
                    ".price-per-unit",
                    ".unit-cost",
                    "[data-testid='unit-cost']",
                    ".product-unit-price"
                ],
                element_type=AsdaElementType.PRODUCT_UNIT_PRICE,
                description="Product unit price elements"
            ),
            
            # Product Offer Type Selectors
            AsdaElementType.PRODUCT_OFFER_TYPE: AsdaSelector(
                primary=".offer-type",
                fallbacks=[
                    "[data-testid='offer-type']",
                    ".promotion",
                    ".deal-type",
                    "[data-testid='promotion']",
                    ".product-offer"
                ],
                element_type=AsdaElementType.PRODUCT_OFFER_TYPE,
                description="Product offer/promotion type elements"
            ),
            
            # Product Asda Price Selectors
            AsdaElementType.PRODUCT_ASDA_PRICE: AsdaSelector(
                primary=".asda-price",
                fallbacks=[
                    "[data-testid='asda-price']",
                    ".asda-deal-price",
                    ".asda-offer",
                    "[data-testid='asda-deal']",
                    ".product-asda-price"
                ],
                element_type=AsdaElementType.PRODUCT_ASDA_PRICE,
                description="Asda-specific pricing elements"
            ),
            
            # Search Input Selectors
            AsdaElementType.SEARCH_INPUT: AsdaSelector(
                primary="input[data-testid='search-input']",
                fallbacks=[
                    "input[name='search']",
                    "#search-input",
                    ".search-input",
                    "input[type='search']",
                    "[data-testid='search']"
                ],
                element_type=AsdaElementType.SEARCH_INPUT,
                description="Search input field elements"
            ),
            
            # Search Button Selectors
            AsdaElementType.SEARCH_BUTTON: AsdaSelector(
                primary="button[type='submit']",
                fallbacks=[
                    ".search-button",
                    "[data-testid='search-button']",
                    "button.search",
                    ".search-submit",
                    "[data-testid='search-submit']"
                ],
                element_type=AsdaElementType.SEARCH_BUTTON,
                description="Search button elements"
            ),
            
            # Category Menu Selectors
            AsdaElementType.CATEGORY_MENU: AsdaSelector(
                primary=".category-menu",
                fallbacks=[
                    ".main-navigation",
                    "nav",
                    "[data-testid='category-menu']",
                    ".breadcrumb",
                    ".category-nav"
                ],
                element_type=AsdaElementType.CATEGORY_MENU,
                description="Category navigation menu elements"
            ),
            
            # Category Link Selectors
            AsdaElementType.CATEGORY_LINK: AsdaSelector(
                primary="a[href*='/category/']",
                fallbacks=[
                    ".category-link",
                    "[data-testid='category-link']",
                    "a.category",
                    ".menu-item a",
                    "a[href*='groceries.asda.com']"
                ],
                element_type=AsdaElementType.CATEGORY_LINK,
                description="Category link elements"
            ),
            
            # Pagination Next Selectors
            AsdaElementType.PAGINATION_NEXT: AsdaSelector(
                primary=".pagination-next",
                fallbacks=[
                    "[data-testid='pagination-next']",
                    ".next-page",
                    "a[aria-label='Next page']",
                    ".pagination .next",
                    "[data-testid='next-page']"
                ],
                element_type=AsdaElementType.PAGINATION_NEXT,
                description="Next page pagination elements"
            ),
            
            # Pagination Info Selectors
            AsdaElementType.PAGINATION_INFO: AsdaSelector(
                primary=".pagination-info",
                fallbacks=[
                    "[data-testid='pagination-info']",
                    ".page-info",
                    ".pagination-details",
                    "[data-testid='page-info']",
                    ".pagination-summary"
                ],
                element_type=AsdaElementType.PAGINATION_INFO,
                description="Pagination information elements"
            ),
            
            # Load More Selectors
            AsdaElementType.LOAD_MORE: AsdaSelector(
                primary=".load-more",
                fallbacks=[
                    "[data-testid='load-more']",
                    ".load-more-button",
                    ".infinite-scroll-trigger",
                    "[data-testid='load-more-button']",
                    ".load-more-trigger"
                ],
                element_type=AsdaElementType.LOAD_MORE,
                description="Load more button elements"
            ),
            
            # Infinite Scroll Selectors
            AsdaElementType.INFINITE_SCROLL: AsdaSelector(
                primary=".infinite-scroll",
                fallbacks=[
                    "[data-testid='infinite-scroll']",
                    ".scroll-container",
                    ".infinite-scroll-container",
                    "[data-testid='scroll-container']",
                    ".scroll-trigger"
                ],
                element_type=AsdaElementType.INFINITE_SCROLL,
                description="Infinite scroll container elements"
            ),
            
            # Cookie Consent Selectors
            AsdaElementType.COOKIE_CONSENT: AsdaSelector(
                primary=".cookie-consent",
                fallbacks=[
                    "[data-testid='cookie-consent']",
                    ".cookie-banner",
                    ".cookie-notice",
                    "[data-testid='cookie-banner']",
                    ".cookie-popup"
                ],
                element_type=AsdaElementType.COOKIE_CONSENT,
                description="Cookie consent dialog elements"
            ),
            
            # Login Prompt Selectors
            AsdaElementType.LOGIN_PROMPT: AsdaSelector(
                primary=".login-prompt",
                fallbacks=[
                    "[data-testid='login-prompt']",
                    ".login-modal",
                    ".login-overlay",
                    "[data-testid='login-modal']",
                    ".login-dialog"
                ],
                element_type=AsdaElementType.LOGIN_PROMPT,
                description="Login prompt/modal elements"
            ),
            
            # Age Verification Selectors
            AsdaElementType.AGE_VERIFICATION: AsdaSelector(
                primary=".age-verification",
                fallbacks=[
                    "[data-testid='age-verification']",
                    ".age-check",
                    ".age-modal",
                    "[data-testid='age-check']",
                    ".age-overlay"
                ],
                element_type=AsdaElementType.AGE_VERIFICATION,
                description="Age verification dialog elements"
            ),
            
            # Location Selector Elements
            AsdaElementType.LOCATION_SELECTOR: AsdaSelector(
                primary=".location-selector",
                fallbacks=[
                    "[data-testid='location-selector']",
                    ".postcode-selector",
                    ".location-picker",
                    "[data-testid='postcode-selector']",
                    ".location-modal"
                ],
                element_type=AsdaElementType.LOCATION_SELECTOR,
                description="Location/postcode selector elements"
            )
        }
    
    def get_selectors(self, element_type: AsdaElementType) -> List[str]:
        """
        Get all selectors for a specific element type, ordered by priority.
        
        Args:
            element_type: The type of element to get selectors for
            
        Returns:
            List of selectors ordered by priority (primary first, then fallbacks)
        """
        if element_type not in self.selectors:
            return []
        
        selector_obj = self.selectors[element_type]
        return [selector_obj.primary] + selector_obj.fallbacks
    
    def get_primary_selector(self, element_type: AsdaElementType) -> Optional[str]:
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
    
    def get_fallback_selectors(self, element_type: AsdaElementType) -> List[str]:
        """
        Get the fallback selectors for a specific element type.
        
        Args:
            element_type: The type of element to get fallback selectors for
            
        Returns:
            List of fallback selectors
        """
        if element_type not in self.selectors:
            return []
        
        return self.selectors[element_type].fallbacks
    
    def add_selector(self, element_type: AsdaElementType, primary: str, 
                    fallbacks: List[str], description: str, priority: int = 1):
        """
        Add a new selector mapping.
        
        Args:
            element_type: The type of element
            primary: Primary selector
            fallbacks: List of fallback selectors
            description: Description of the element
            priority: Priority level (lower = higher priority)
        """
        self.selectors[element_type] = AsdaSelector(
            primary=primary,
            fallbacks=fallbacks,
            element_type=element_type,
            description=description,
            priority=priority
        )
    
    def update_selector(self, element_type: AsdaElementType, primary: str = None,
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
    
    def get_all_element_types(self) -> List[AsdaElementType]:
        """Get all available element types."""
        return list(self.selectors.keys())
    
    def get_selector_info(self, element_type: AsdaElementType) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a selector mapping.
        
        Args:
            element_type: The type of element
            
        Returns:
            Dictionary with selector information or None if not found
        """
        if element_type not in self.selectors:
            return None
        
        selector = self.selectors[element_type]
        return {
            'element_type': element_type.value,
            'primary': selector.primary,
            'fallbacks': selector.fallbacks,
            'description': selector.description,
            'priority': selector.priority
        }


# Global instance for easy access
asda_selectors = AsdaSelectorMapping()


def get_asda_selectors() -> AsdaSelectorMapping:
    """
    Get the global Asda selector mapping instance.
    
    Returns:
        AsdaSelectorMapping instance
    """
    return asda_selectors 