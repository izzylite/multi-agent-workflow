"""
Vendor-Specific Tool Extensions

Provides specialized tool extensions for Tesco, Asda, and Costco that adapt
core browser tools for each vendor's unique site structure, navigation patterns,
and anti-bot requirements.
"""

import asyncio
import logging
import time
import random
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from .browser_tools import (
    BrowserbaseTool, NavigationTool, InteractionTool, ExtractionTool,
    ScreenshotTool, WaitingTool, AntiBotConfig, BrowserbaseToolInput
)
from .browserbase_manager import BrowserbaseManager
from .storage_manager import StorageManager, StorageConfig, ProductData
from .asda_selectors import AsdaElementType, get_asda_selectors
from .costco_selectors import CostcoElementType, get_costco_selectors


@dataclass
class VendorConfig:
    """Configuration for vendor-specific behavior."""
    name: str
    base_url: str
    search_selectors: List[str]
    product_card_selectors: List[str]
    product_title_selectors: List[str]
    product_price_selectors: List[str]
    product_image_selectors: List[str]
    pagination_selectors: List[str]
    category_selectors: List[str]
    anti_bot_delays: Dict[str, tuple]  # (min_delay, max_delay) for different operations
    session_timeout: int = 30000
    max_retries: int = 3
    enable_stealth: bool = True


class VendorTool(BrowserbaseTool):
    """
    Base class for vendor-specific tools with common vendor behavior.
    """
    
    def __init__(self, 
                 browser_manager: BrowserbaseManager,
                 vendor_config: VendorConfig,
                 anti_bot_config: Optional[AntiBotConfig] = None,
                 storage_manager: Optional[StorageManager] = None,
                 **kwargs):
        super().__init__(browser_manager, anti_bot_config, **kwargs)
        self.vendor_config = vendor_config
        self.storage_manager = storage_manager
        self.logger = logging.getLogger(f"{__name__}.{vendor_config.name}")
    
    async def _apply_vendor_specific_delays(self, operation: str) -> None:
        """Apply vendor-specific delays for different operations."""
        if operation in self.vendor_config.anti_bot_delays:
            min_delay, max_delay = self.vendor_config.anti_bot_delays[operation]
            delay = random.uniform(min_delay, max_delay)
            self.logger.debug(f"Applying {self.vendor_config.name} delay for {operation}: {delay:.2f}s")
            await asyncio.sleep(delay)
    
    async def _retry_operation(self, operation_func, *args, max_retries: int = None, **kwargs):
        """Retry an operation with vendor-specific error handling."""
        max_retries = max_retries or self.vendor_config.max_retries
        
        for attempt in range(max_retries):
            try:
                return await operation_func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"{self.vendor_config.name} operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(1, 3))
                else:
                    raise e
    
    def _convert_to_product_data(self, product_dict: Dict[str, Any], category: str) -> ProductData:
        """Convert scraped product dictionary to ProductData object."""
        return ProductData(
            vendor=str(self.vendor_config.name).lower(),
            category=category,
            product_id=product_dict.get('product_id', ''),
            title=product_dict.get('title', ''),
            price=product_dict.get('price'),
            image_url=product_dict.get('image_url'),
            description=product_dict.get('description'),
            url=product_dict.get('url'),
            metadata=product_dict.get('metadata', {})
        )
    
    def save_products(self, products: List[Dict[str, Any]], category: str) -> Optional[str]:
        """Save scraped products to storage if storage manager is available."""
        if not self.storage_manager:
            self.logger.warning("No storage manager available, skipping product save")
            return None
        
        try:
            # Convert to ProductData objects
            product_data_list = [
                self._convert_to_product_data(product, category)
                for product in products
            ]
            
            # Save to storage
            file_path = self.storage_manager.save_products(
                product_data_list,
                str(self.vendor_config.name).lower(),
                category
            )
            
            self.logger.info(f"Saved {len(products)} products to storage: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save products: {e}")
            return None
    
    def save_product_incremental(self, product: Dict[str, Any], category: str) -> None:
        """Save a single product incrementally."""
        if not self.storage_manager:
            return
        
        try:
            product_data = self._convert_to_product_data(product, category)
            self.storage_manager.save_incremental(product_data)
        except Exception as e:
            self.logger.error(f"Failed to save product incrementally: {e}")


class TescoTool(VendorTool):
    """
    Tesco-specific tool extensions with Tesco's navigation patterns and anti-bot measures.
    """
    
    def __init__(self, browser_manager: BrowserbaseManager, **kwargs):
        vendor_config = VendorConfig(
            name="Tesco",
            base_url="https://www.tesco.com",
            search_selectors=[
                "input[data-testid='search-input']",
                "input[name='search']",
                "#search-input"
            ],
            product_card_selectors=[
                ".product-list-grid .product-card",
                "[data-testid='product-card']",
                ".product-item"
            ],
            product_title_selectors=[
                "h1[data-testid='product-title']",
                ".product-title",
                "h1"
            ],
            product_price_selectors=[
                "span.price",
                "[data-testid='product-price']",
                ".price"
            ],
            product_image_selectors=[
                "img[data-testid='product-image']",
                ".product-image img",
                "img"
            ],
            pagination_selectors=[
                ".pagination-next",
                "[data-testid='pagination-next']",
                ".next-page"
            ],
            category_selectors=[
                "nav[aria-label='Main menu']",
                ".category-menu",
                ".main-navigation"
            ],
            anti_bot_delays={
                'navigation': (2, 5),
                'interaction': (1, 3),
                'extraction': (0.5, 2),
                'search': (1, 4)
            }
        )
        super().__init__(browser_manager, vendor_config, **kwargs)
    
    async def navigate_to_category(self, category_name: str) -> Dict[str, Any]:
        """Navigate to a Tesco category using mega menu navigation."""
        try:
            # Apply Tesco-specific delays
            await self._apply_vendor_specific_delays('navigation')
            
            # Navigate to homepage first
            await self._retry_operation(
                self.browser_operations.navigate,
                url=self.vendor_config.base_url,
                wait_until="networkidle"
            )
            
            # Wait for main navigation to load
            await self.browser_operations.wait_for_element(
                selector=self.vendor_config.category_selectors[0],
                timeout=10000
            )
            
            # Click on category menu
            await self._retry_operation(
                self.browser_operations.click,
                selector="nav[aria-label='Main menu']",
                timeout=5000
            )
            
            # Find and click the specific category
            category_selector = f"a[href*='{str(category_name).lower()}'], a:contains('{category_name}')"
            await self._retry_operation(
                self.browser_operations.click,
                selector=category_selector,
                timeout=5000
            )
            
            # Wait for product grid to load
            await self.browser_operations.wait_for_element(
                selector=self.vendor_config.product_card_selectors[0],
                timeout=15000
            )
            
            return {
                'status': 'success',
                'category': category_name,
                'url': self.browser_operations.get_current_url()
            }
            
        except Exception as e:
            return await self._handle_error(e, f"tesco_category_navigation_{category_name}")
    
    async def search_products(self, query: str) -> Dict[str, Any]:
        """Search for products on Tesco with proper delays and error handling."""
        try:
            await self._apply_vendor_specific_delays('search')
            
            # Navigate to homepage
            await self._retry_operation(
                self.browser_operations.navigate,
                url=self.vendor_config.base_url,
                wait_until="networkidle"
            )
            
            # Find and fill search input
            search_input = None
            for selector in self.vendor_config.search_selectors:
                try:
                    await self.browser_operations.wait_for_element(selector, timeout=5000)
                    search_input = selector
                    break
                except:
                    continue
            
            if not search_input:
                raise ValueError("Could not find search input on Tesco")
            
            # Type search query
            await self._retry_operation(
                self.browser_operations.type_text,
                selector=search_input,
                text=query,
                timeout=5000
            )
            
            # Submit search
            await self._retry_operation(
                self.browser_operations.click,
                selector="button[type='submit'], .search-button",
                timeout=5000
            )
            
            # Wait for results
            await self.browser_operations.wait_for_element(
                selector=self.vendor_config.product_card_selectors[0],
                timeout=15000
            )
            
            return {
                'status': 'success',
                'query': query,
                'url': self.browser_operations.get_current_url()
            }
            
        except Exception as e:
            return await self._handle_error(e, f"tesco_search_{query}")
    
    async def extract_product_cards(self) -> List[Dict[str, Any]]:
        """Extract product cards from Tesco product listing pages."""
        try:
            await self._apply_vendor_specific_delays('extraction')
            
            products = []
            
            # Find all product cards
            for card_selector in self.vendor_config.product_card_selectors:
                try:
                    cards = await self.browser_operations.find_elements(card_selector)
                    if cards:
                        break
                except:
                    continue
            
            if not cards:
                return []
            
            # Extract data from each card
            for card in cards:
                try:
                    # Extract title
                    title = None
                    for title_selector in self.vendor_config.product_title_selectors:
                        try:
                            title_elem = await card.find_element(title_selector)
                            title = await title_elem.text()
                            break
                        except:
                            continue
                    
                    # Extract price
                    price = None
                    for price_selector in self.vendor_config.product_price_selectors:
                        try:
                            price_elem = await card.find_element(price_selector)
                            price = await price_elem.text()
                            break
                        except:
                            continue
                    
                    # Extract image
                    image = None
                    for image_selector in self.vendor_config.product_image_selectors:
                        try:
                            img_elem = await card.find_element(image_selector)
                            image = await img_elem.get_attribute('src')
                            break
                        except:
                            continue
                    
                    # Extract link
                    link = None
                    try:
                        link_elem = await card.find_element('a')
                        link = await link_elem.get_attribute('href')
                    except:
                        pass
                    
                    if title or price:  # Only include if we have at least title or price
                        products.append({
                            'title': title,
                            'price': price,
                            'image': image,
                            'link': link,
                            'vendor': 'Tesco'
                        })
                
                except Exception as e:
                    self.logger.warning(f"Failed to extract product card: {e}")
                    continue
            
            return products
            
        except Exception as e:
            self.logger.error(f"Failed to extract Tesco product cards: {e}")
            return []


class AsdaTool(VendorTool):
    """
    Asda-specific tool extensions with Asda's navigation patterns and anti-bot measures.
    """
    
    def __init__(self, browser_manager: BrowserbaseManager, **kwargs):
        vendor_config = VendorConfig(
            name="Asda",
            base_url="https://groceries.asda.com",
            search_selectors=[
                "input[data-testid='search-input']",
                "input[name='search']",
                "#search-input",
                ".search-input",
                "input[type='search']"
            ],
            product_card_selectors=[
                ".product-list .product-item",
                "[data-testid='product-card']",
                ".product-card",
                ".product-item",
                "[data-testid='product-item']"
            ],
            product_title_selectors=[
                "h1[data-testid='product-title']",
                ".product-title",
                "h1",
                "[data-testid='product-name']",
                ".product-name"
            ],
            product_price_selectors=[
                "span.price",
                "[data-testid='product-price']",
                ".price",
                ".product-price",
                "[data-testid='price']"
            ],
            product_image_selectors=[
                "img[data-testid='product-image']",
                ".product-image img",
                "img",
                "[data-testid='product-image'] img"
            ],
            pagination_selectors=[
                ".pagination-next",
                "[data-testid='pagination-next']",
                ".next-page",
                "a[aria-label='Next page']",
                ".pagination .next"
            ],
            category_selectors=[
                ".category-menu",
                ".main-navigation",
                "nav",
                ".breadcrumb",
                "[data-testid='category-menu']"
            ],
            anti_bot_delays={
                'navigation': (1.5, 4),
                'interaction': (0.8, 2.5),
                'extraction': (0.3, 1.5),
                'search': (0.8, 3),
                'pagination': (2, 5),
                'infinite_scroll': (1, 3)
            }
        )
        super().__init__(browser_manager, vendor_config, **kwargs)
        
        # Asda-specific navigation state
        self.current_page = 1
        self.total_pages = None
        self.has_infinite_scroll = False
        self.last_scroll_position = 0
        
        # Initialize selector mapping
        self.selector_mapping = get_asda_selectors()
    
    async def navigate_to_category(self, category_name: str) -> Dict[str, Any]:
        """Navigate to an Asda category using multiple navigation strategies."""
        try:
            await self._apply_vendor_specific_delays('navigation')
            
            # Navigate to homepage
            await self._retry_operation(
                self.browser_operations.navigate,
                url=self.vendor_config.base_url,
                wait_until="networkidle"
            )
            
            # Wait for navigation to load
            await self.browser_operations.wait_for_element(
                selector=self.vendor_config.category_selectors[0],
                timeout=10000
            )
            
            # Try multiple navigation strategies
            navigation_successful = False
            
            # Strategy 1: Direct URL navigation
            try:
                category_url = f"{self.vendor_config.base_url}/category/{str(category_name).lower().replace(' ', '-')}"
                await self._retry_operation(
                    self.browser_operations.navigate,
                    url=category_url,
                    wait_until="networkidle"
                )
                
                # Check if we're on a product page
                await self.browser_operations.wait_for_element(
                    selector=self.vendor_config.product_card_selectors[0],
                    timeout=10000
                )
                navigation_successful = True
                
            except Exception as e:
                self.logger.debug(f"Direct URL navigation failed: {e}")
            
            # Strategy 2: Breadcrumb navigation
            if not navigation_successful:
                try:
                    category_selector = f"a[href*='{str(category_name).lower()}'], a:contains('{category_name}')"
                    await self._retry_operation(
                        self.browser_operations.click,
                        selector=category_selector,
                        timeout=5000
                    )
                    
                    # Wait for product grid
                    await self.browser_operations.wait_for_element(
                        selector=self.vendor_config.product_card_selectors[0],
                        timeout=15000
                    )
                    navigation_successful = True
                    
                except Exception as e:
                    self.logger.debug(f"Breadcrumb navigation failed: {e}")
            
            # Strategy 3: Menu navigation
            if not navigation_successful:
                try:
                    # Look for category in main menu
                    menu_selectors = [
                        f"a[href*='{str(category_name).lower()}']",
                        f".menu-item a:contains('{category_name}')",
                        f"[data-testid='menu-item'] a:contains('{category_name}')"
                    ]
                    
                    for selector in menu_selectors:
                        try:
                            await self._retry_operation(
                                self.browser_operations.click,
                                selector=selector,
                                timeout=3000
                            )
                            
                            # Wait for product grid
                            await self.browser_operations.wait_for_element(
                                selector=self.vendor_config.product_card_selectors[0],
                                timeout=10000
                            )
                            navigation_successful = True
                            break
                            
                        except:
                            continue
                            
                except Exception as e:
                    self.logger.debug(f"Menu navigation failed: {e}")
            
            if not navigation_successful:
                raise NavigationError(f"Failed to navigate to category {category_name} using all strategies")
            
            # Reset navigation state
            self.current_page = 1
            self.total_pages = None
            self.has_infinite_scroll = False
            
            return {
                'status': 'success',
                'category': category_name,
                'url': self.browser_operations.get_current_url()
            }
            
        except Exception as e:
            return await self._handle_error(e, f"asda_category_navigation_{category_name}")
    
    async def search_products(self, query: str) -> Dict[str, Any]:
        """Search for products on Asda with proper delays."""
        try:
            await self._apply_vendor_specific_delays('search')
            
            # Navigate to homepage
            await self._retry_operation(
                self.browser_operations.navigate,
                url=self.vendor_config.base_url,
                wait_until="networkidle"
            )
            
            # Find search input
            search_input = None
            for selector in self.vendor_config.search_selectors:
                try:
                    await self.browser_operations.wait_for_element(selector, timeout=5000)
                    search_input = selector
                    break
                except:
                    continue
            
            if not search_input:
                raise ValueError("Could not find search input on Asda")
            
            # Type search query
            await self._retry_operation(
                self.browser_operations.type_text,
                selector=search_input,
                text=query,
                timeout=5000
            )
            
            # Submit search
            await self._retry_operation(
                self.browser_operations.click,
                selector="button[type='submit'], .search-button",
                timeout=5000
            )
            
            # Wait for results
            await self.browser_operations.wait_for_element(
                selector=self.vendor_config.product_card_selectors[0],
                timeout=15000
            )
            
            return {
                'status': 'success',
                'query': query,
                'url': self.browser_operations.get_current_url()
            }
            
        except Exception as e:
            return await self._handle_error(e, f"asda_search_{query}")
    
    async def extract_product_cards(self) -> List[Dict[str, Any]]:
        """Extract product cards from Asda product listing pages using enhanced selector mapping."""
        try:
            await self._apply_vendor_specific_delays('extraction')
            
            products = []
            
            # Find product cards using selector mapping
            cards = await self._find_elements_with_selectors(AsdaElementType.PRODUCT_CARD)
            
            if not cards:
                self.logger.warning("No product cards found on current page")
                return []
            
            self.logger.info(f"Found {len(cards)} product cards to extract")
            
            # Extract data from each card
            for i, card in enumerate(cards):
                try:
                    product_data = await self._extract_single_product_card(card, i)
                    if product_data:
                        products.append(product_data)
                
                except Exception as e:
                    self.logger.warning(f"Failed to extract product card {i}: {e}")
                    continue
            
            self.logger.info(f"Successfully extracted {len(products)} products")
            return products
            
        except Exception as e:
            self.logger.error(f"Failed to extract Asda product cards: {e}")
            return []
    
    async def _extract_single_product_card(self, card: Any, index: int) -> Optional[Dict[str, Any]]:
        """
        Extract data from a single product card.
        
        Args:
            card: Product card element
            index: Index of the card for logging
            
        Returns:
            Dictionary with product data or None if extraction fails
        """
        try:
            product_data = {
                'vendor': 'Asda',
                'extracted_at': datetime.now().isoformat(),
                'card_index': index
            }
            
            # Extract basic product information using selector mapping
            title = await self._extract_text_from_card(card, AsdaElementType.PRODUCT_TITLE)
            price = await self._extract_text_from_card(card, AsdaElementType.PRODUCT_PRICE)
            image_url = await self._extract_attribute_from_card(card, AsdaElementType.PRODUCT_IMAGE, 'src')
            product_url = await self._extract_attribute_from_card(card, AsdaElementType.PRODUCT_LINK, 'href')
            description = await self._extract_text_from_card(card, AsdaElementType.PRODUCT_DESCRIPTION)
            
            # Extract additional pricing information
            unit_price = await self._extract_text_from_card(card, AsdaElementType.PRODUCT_UNIT_PRICE)
            offer_type = await self._extract_text_from_card(card, AsdaElementType.PRODUCT_OFFER_TYPE)
            asda_price = await self._extract_text_from_card(card, AsdaElementType.PRODUCT_ASDA_PRICE)
            
            # Extract availability information
            availability = await self._extract_text_from_card(card, AsdaElementType.PRODUCT_AVAILABILITY)
            stock_status = await self._extract_text_from_card(card, AsdaElementType.PRODUCT_STOCK_STATUS)
            
            # Only include product if we have at least title or price
            if not title and not price:
                return None
            
            # Add extracted data to product dictionary
            product_data.update({
                'title': title,
                'price': price,
                'image_url': image_url,
                'product_url': product_url,
                'description': description,
                'unit_price': unit_price,
                'offer_type': offer_type,
                'asda_price': asda_price,
                'availability': availability,
                'stock_status': stock_status
            })
            
            # Generate product ID if not available
            if title:
                import hashlib
                product_data['product_id'] = hashlib.md5(title.encode()).hexdigest()[:12]
            
            return product_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract product card {index}: {e}")
            return None
    
    async def _extract_text_from_card(self, card: Any, element_type: AsdaElementType) -> Optional[str]:
        """
        Extract text from a specific element type within a product card.
        
        Args:
            card: Product card element
            element_type: Type of element to extract
            
        Returns:
            Extracted text or None if not found
        """
        try:
            selectors = self.selector_mapping.get_selectors(element_type)
            
            for selector in selectors:
                try:
                    # Try to find element within the card
                    element = await card.find_element(selector)
                    if element:
                        text = await self.browser_operations.get_text(element)
                        if text and text.strip():
                            return text.strip()
                except:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to extract text for {element_type.value}: {e}")
            return None
    
    async def _extract_attribute_from_card(self, card: Any, element_type: AsdaElementType, 
                                         attribute: str) -> Optional[str]:
        """
        Extract attribute from a specific element type within a product card.
        
        Args:
            card: Product card element
            element_type: Type of element to extract
            attribute: Attribute name to extract
            
        Returns:
            Extracted attribute value or None if not found
        """
        try:
            selectors = self.selector_mapping.get_selectors(element_type)
            
            for selector in selectors:
                try:
                    # Try to find element within the card
                    element = await card.find_element(selector)
                    if element:
                        attr_value = await self.browser_operations.get_attribute(element, attribute)
                        if attr_value and attr_value.strip():
                            return attr_value.strip()
                except:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Failed to extract attribute {attribute} for {element_type.value}: {e}")
            return None
    
    async def extract_product_details(self, product_url: str) -> Optional[Dict[str, Any]]:
        """
        Extract detailed product information from a product detail page.
        
        Args:
            product_url: URL of the product detail page
            
        Returns:
            Dictionary with detailed product information or None if extraction fails
        """
        try:
            await self._apply_vendor_specific_delays('extraction')
            
            # Navigate to product detail page
            nav_result = await self.navigate_to_product_detail(product_url)
            if nav_result.get('status') != 'success':
                self.logger.error(f"Failed to navigate to product detail page: {product_url}")
                return None
            
            # Extract detailed product information
            product_details = {
                'vendor': 'Asda',
                'product_url': product_url,
                'extracted_at': datetime.now().isoformat()
            }
            
            # Extract basic product information
            title = await self._get_text_with_selectors(AsdaElementType.PRODUCT_TITLE)
            price = await self._get_text_with_selectors(AsdaElementType.PRODUCT_PRICE)
            description = await self._get_text_with_selectors(AsdaElementType.PRODUCT_DESCRIPTION)
            image_url = await self._get_attribute_with_selectors(AsdaElementType.PRODUCT_IMAGE, 'src')
            
            # Extract detailed specifications
            specifications = await self._get_text_with_selectors(AsdaElementType.PRODUCT_SPECIFICATIONS)
            nutrition_info = await self._get_text_with_selectors(AsdaElementType.PRODUCT_NUTRITION)
            allergens = await self._get_text_with_selectors(AsdaElementType.PRODUCT_ALLERGENS)
            dietary_info = await self._get_text_with_selectors(AsdaElementType.PRODUCT_DIETARY)
            product_code = await self._get_text_with_selectors(AsdaElementType.PRODUCT_CODE)
            
            # Extract pricing information
            unit_price = await self._get_text_with_selectors(AsdaElementType.PRODUCT_UNIT_PRICE)
            offer_type = await self._get_text_with_selectors(AsdaElementType.PRODUCT_OFFER_TYPE)
            asda_price = await self._get_text_with_selectors(AsdaElementType.PRODUCT_ASDA_PRICE)
            
            # Extract availability information
            availability = await self._get_text_with_selectors(AsdaElementType.PRODUCT_AVAILABILITY)
            stock_status = await self._get_text_with_selectors(AsdaElementType.PRODUCT_STOCK_STATUS)
            
            # Add extracted data to product details
            product_details.update({
                'title': title,
                'price': price,
                'description': description,
                'image_url': image_url,
                'specifications': specifications,
                'nutrition_info': nutrition_info,
                'allergens': allergens,
                'dietary_info': dietary_info,
                'product_code': product_code,
                'unit_price': unit_price,
                'offer_type': offer_type,
                'asda_price': asda_price,
                'availability': availability,
                'stock_status': stock_status
            })
            
            # Generate product ID if not available
            if title:
                import hashlib
                product_details['product_id'] = hashlib.md5(title.encode()).hexdigest()[:12]
            
            return product_details
            
        except Exception as e:
            self.logger.error(f"Failed to extract product details from {product_url}: {e}")
            return None
    
    async def extract_multiple_product_details(self, product_urls: List[str]) -> List[Dict[str, Any]]:
        """
        Extract detailed information from multiple product detail pages.
        
        Args:
            product_urls: List of product detail page URLs
            
        Returns:
            List of dictionaries with detailed product information
        """
        try:
            self.logger.info(f"Extracting details from {len(product_urls)} product pages")
            
            detailed_products = []
            
            for i, url in enumerate(product_urls):
                try:
                    self.logger.info(f"Extracting product {i+1}/{len(product_urls)}: {url}")
                    
                    product_details = await self.extract_product_details(url)
                    if product_details:
                        detailed_products.append(product_details)
                    
                    # Apply anti-bot delays between extractions
                    await self._apply_vendor_specific_delays('extraction')
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract product details from {url}: {e}")
                    continue
            
            self.logger.info(f"Successfully extracted details from {len(detailed_products)} products")
            return detailed_products
            
        except Exception as e:
            self.logger.error(f"Failed to extract multiple product details: {e}")
            return []
    
    async def handle_asda_challenges(self) -> Dict[str, Any]:
        """
        Handle Asda-specific challenges including cookie consent, login prompts, 
        age verification, and location selection.
        
        Returns:
            Dictionary with challenge handling results
        """
        try:
            self.logger.info("Handling Asda-specific challenges")
            
            challenges_handled = {
                'cookie_consent': False,
                'login_prompt': False,
                'age_verification': False,
                'location_selector': False,
                'dynamic_content': False
            }
            
            # Handle cookie consent
            cookie_result = await self._handle_cookie_consent()
            challenges_handled['cookie_consent'] = cookie_result
            
            # Handle login prompts
            login_result = await self._handle_login_prompt()
            challenges_handled['login_prompt'] = login_result
            
            # Handle age verification
            age_result = await self._handle_age_verification()
            challenges_handled['age_verification'] = age_result
            
            # Handle location selection
            location_result = await self._handle_location_selector()
            challenges_handled['location_selector'] = location_result
            
            # Handle dynamic content loading
            dynamic_result = await self._handle_dynamic_content()
            challenges_handled['dynamic_content'] = dynamic_result
            
            self.logger.info(f"Challenge handling completed: {challenges_handled}")
            return {
                'status': 'success',
                'challenges_handled': challenges_handled
            }
            
        except Exception as e:
            self.logger.error(f"Failed to handle Asda challenges: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _handle_cookie_consent(self) -> bool:
        """Handle cookie consent dialogs."""
        try:
            # Check for cookie consent dialogs
            cookie_selectors = self.selector_mapping.get_selectors(AsdaElementType.COOKIE_CONSENT)
            
            for selector in cookie_selectors:
                try:
                    cookie_dialog = await self.browser_operations.find_element(selector, timeout=3000)
                    if cookie_dialog:
                        self.logger.info("Found cookie consent dialog, attempting to accept")
                        
                        # Try to find and click accept button
                        accept_selectors = [
                            "button[data-testid='accept-cookies']",
                            ".accept-cookies",
                            "button:contains('Accept')",
                            "button:contains('Accept All')",
                            ".cookie-accept"
                        ]
                        
                        for accept_selector in accept_selectors:
                            try:
                                accept_button = await self.browser_operations.find_element(accept_selector, timeout=2000)
                                if accept_button:
                                    await self.browser_operations.click(accept_button)
                                    await asyncio.sleep(1)
                                    self.logger.info("Successfully accepted cookies")
                                    return True
                            except:
                                continue
                        
                        # If no accept button found, try to close the dialog
                        close_selectors = [
                            ".close-cookie",
                            "button[data-testid='close-cookie']",
                            ".cookie-close",
                            "button:contains('Close')"
                        ]
                        
                        for close_selector in close_selectors:
                            try:
                                close_button = await self.browser_operations.find_element(close_selector, timeout=2000)
                                if close_button:
                                    await self.browser_operations.click(close_button)
                                    await asyncio.sleep(1)
                                    self.logger.info("Successfully closed cookie dialog")
                                    return True
                            except:
                                continue
                        
                        break
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to handle cookie consent: {e}")
            return False
    
    async def _handle_login_prompt(self) -> bool:
        """Handle login prompts and modals."""
        try:
            # Check for login prompts
            login_selectors = self.selector_mapping.get_selectors(AsdaElementType.LOGIN_PROMPT)
            
            for selector in login_selectors:
                try:
                    login_dialog = await self.browser_operations.find_element(selector, timeout=3000)
                    if login_dialog:
                        self.logger.info("Found login prompt, attempting to close")
                        
                        # Try to find and click close button
                        close_selectors = [
                            ".close-login",
                            "button[data-testid='close-login']",
                            ".login-close",
                            "button:contains('Close')",
                            ".modal-close"
                        ]
                        
                        for close_selector in close_selectors:
                            try:
                                close_button = await self.browser_operations.find_element(close_selector, timeout=2000)
                                if close_button:
                                    await self.browser_operations.click(close_button)
                                    await asyncio.sleep(1)
                                    self.logger.info("Successfully closed login prompt")
                                    return True
                            except:
                                continue
                        
                        # Try to click outside the modal
                        try:
                            await self.browser_operations.execute_script(
                                "document.querySelector('.modal-overlay').click();"
                            )
                            await asyncio.sleep(1)
                            self.logger.info("Successfully closed login prompt by clicking outside")
                            return True
                        except:
                            pass
                        
                        break
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to handle login prompt: {e}")
            return False
    
    async def _handle_age_verification(self) -> bool:
        """Handle age verification dialogs."""
        try:
            # Check for age verification dialogs
            age_selectors = self.selector_mapping.get_selectors(AsdaElementType.AGE_VERIFICATION)
            
            for selector in age_selectors:
                try:
                    age_dialog = await self.browser_operations.find_element(selector, timeout=3000)
                    if age_dialog:
                        self.logger.info("Found age verification dialog, attempting to verify")
                        
                        # Try to find and click verify button
                        verify_selectors = [
                            "button[data-testid='verify-age']",
                            ".verify-age",
                            "button:contains('Yes')",
                            "button:contains('I am over 18')",
                            ".age-yes"
                        ]
                        
                        for verify_selector in verify_selectors:
                            try:
                                verify_button = await self.browser_operations.find_element(verify_selector, timeout=2000)
                                if verify_button:
                                    await self.browser_operations.click(verify_button)
                                    await asyncio.sleep(1)
                                    self.logger.info("Successfully verified age")
                                    return True
                            except:
                                continue
                        
                        break
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to handle age verification: {e}")
            return False
    
    async def _handle_location_selector(self) -> bool:
        """Handle location/postcode selector dialogs."""
        try:
            # Check for location selector dialogs
            location_selectors = self.selector_mapping.get_selectors(AsdaElementType.LOCATION_SELECTOR)
            
            for selector in location_selectors:
                try:
                    location_dialog = await self.browser_operations.find_element(selector, timeout=3000)
                    if location_dialog:
                        self.logger.info("Found location selector, attempting to handle")
                        
                        # Try to find and fill postcode input
                        postcode_selectors = [
                            "input[data-testid='postcode-input']",
                            ".postcode-input",
                            "input[name='postcode']",
                            "input[placeholder*='postcode']"
                        ]
                        
                        for postcode_selector in postcode_selectors:
                            try:
                                postcode_input = await self.browser_operations.find_element(postcode_selector, timeout=2000)
                                if postcode_input:
                                    # Use a default postcode (London)
                                    await self.browser_operations.type_text(postcode_input, "SW1A 1AA")
                                    await asyncio.sleep(1)
                                    
                                    # Try to find and click submit button
                                    submit_selectors = [
                                        "button[data-testid='submit-postcode']",
                                        ".submit-postcode",
                                        "button:contains('Submit')",
                                        "button:contains('Find')"
                                    ]
                                    
                                    for submit_selector in submit_selectors:
                                        try:
                                            submit_button = await self.browser_operations.find_element(submit_selector, timeout=2000)
                                            if submit_button:
                                                await self.browser_operations.click(submit_button)
                                                await asyncio.sleep(2)
                                                self.logger.info("Successfully set location")
                                                return True
                                        except:
                                            continue
                                    
                                    break
                            except:
                                continue
                        
                        break
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to handle location selector: {e}")
            return False
    
    async def _handle_dynamic_content(self) -> bool:
        """Handle dynamic content loading and JavaScript-heavy pages."""
        try:
            self.logger.info("Handling dynamic content loading")
            
            # Wait for page to be fully loaded
            await self.browser_operations.wait_for_element(
                selector="body",
                timeout=10000
            )
            
            # Wait for any loading indicators to disappear
            loading_selectors = [
                ".loading",
                "[data-testid='loading']",
                ".spinner",
                ".loader"
            ]
            
            for selector in loading_selectors:
                try:
                    await self.browser_operations.wait_for_element_not_present(selector, timeout=5000)
                except:
                    pass  # Loading indicator might not be present
            
            # Scroll to trigger lazy loading
            await self.browser_operations.execute_script(
                "window.scrollTo(0, document.body.scrollHeight / 2);"
            )
            await asyncio.sleep(2)
            
            # Scroll back to top
            await self.browser_operations.execute_script(
                "window.scrollTo(0, 0);"
            )
            await asyncio.sleep(1)
            
            self.logger.info("Successfully handled dynamic content loading")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to handle dynamic content: {e}")
            return False
    
    async def _find_element_with_selectors(self, element_type: CostcoElementType, 
                                         timeout: int = 5000) -> Optional[Any]:
        """
        Find a single element using fallback selectors.
        
        Args:
            element_type: The type of element to find
            timeout: Timeout in milliseconds
            
        Returns:
            Element if found, None otherwise
        """
        selectors = self.selector_mapping.get_selectors(element_type)
        
        for selector in selectors:
            try:
                element = await self.browser_operations.find_element(selector)
                if element:
                    return element
            except:
                continue
        
        return None
    
    async def _find_elements_with_selectors(self, element_type: CostcoElementType, 
                                          timeout: int = 5000) -> List[Any]:
        """
        Find multiple elements using fallback selectors.
        
        Args:
            element_type: The type of element to find
            timeout: Timeout in milliseconds
            
        Returns:
            List of elements found
        """
        selectors = self.selector_mapping.get_selectors(element_type)
        
        for selector in selectors:
            try:
                elements = await self.browser_operations.find_elements(selector)
                if elements:
                    return elements
            except:
                continue
        
        return []
    
    async def _get_text_with_selectors(self, element_type: CostcoElementType, 
                                     timeout: int = 5000) -> Optional[str]:
        """
        Get text from an element using fallback selectors.
        
        Args:
            element_type: The type of element to get text from
            timeout: Timeout in milliseconds
            
        Returns:
            Text content if found, None otherwise
        """
        selectors = self.selector_mapping.get_selectors(element_type)
        
        for selector in selectors:
            try:
                text = await self.browser_operations.get_text(selector)
                if text and text.strip():
                    return text.strip()
            except:
                continue
        
        return None
    
    async def _get_attribute_with_selectors(self, element_type: CostcoElementType, 
                                          attribute: str, timeout: int = 5000) -> Optional[str]:
        """
        Get attribute from an element using fallback selectors.
        
        Args:
            element_type: The type of element to get attribute from
            attribute: Attribute name to get
            timeout: Timeout in milliseconds
            
        Returns:
            Attribute value if found, None otherwise
        """
        selectors = self.selector_mapping.get_selectors(element_type)
        
        for selector in selectors:
            try:
                value = await self.browser_operations.get_attribute(selector, attribute)
                if value:
                    return value
            except:
                continue
        
        return None
    
    async def apply_enhanced_anti_bot_measures(self) -> None:
        """Apply enhanced anti-bot measures for Asda."""
        try:
            self.logger.info("Applying enhanced anti-bot measures")
            
            # Random mouse movements
            await self._simulate_human_mouse_movements()
            
            # Random scrolling
            await self._simulate_human_scrolling()
            
            # Random delays
            await self._apply_random_delays()
            
            self.logger.info("Enhanced anti-bot measures applied")
            
        except Exception as e:
            self.logger.warning(f"Failed to apply enhanced anti-bot measures: {e}")
    
    async def _simulate_human_mouse_movements(self) -> None:
        """Simulate human-like mouse movements."""
        try:
            # Get viewport dimensions
            viewport_width = await self.browser_operations.execute_script(
                "return window.innerWidth;"
            )
            viewport_height = await self.browser_operations.execute_script(
                "return window.innerHeight;"
            )
            
            # Generate random mouse movements
            for _ in range(random.randint(2, 5)):
                x = random.randint(100, viewport_width - 100)
                y = random.randint(100, viewport_height - 100)
                
                await self.browser_operations.execute_script(
                    f"document.elementFromPoint({x}, {y}).dispatchEvent(new MouseEvent('mouseover'));"
                )
                await asyncio.sleep(random.uniform(0.1, 0.3))
            
        except Exception as e:
            self.logger.debug(f"Failed to simulate mouse movements: {e}")
    
    async def _simulate_human_scrolling(self) -> None:
        """Simulate human-like scrolling behavior."""
        try:
            # Get current scroll position
            current_scroll = await self.browser_operations.execute_script(
                "return window.pageYOffset;"
            )
            
            # Simulate natural scrolling
            scroll_distance = random.randint(100, 500)
            scroll_direction = random.choice([-1, 1])
            
            await self.browser_operations.execute_script(
                f"window.scrollBy(0, {scroll_distance * scroll_direction});"
            )
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Scroll back to original position
            await self.browser_operations.execute_script(
                f"window.scrollTo(0, {current_scroll});"
            )
            await asyncio.sleep(random.uniform(0.3, 0.8))
            
        except Exception as e:
            self.logger.debug(f"Failed to simulate scrolling: {e}")
    
    async def _apply_random_delays(self) -> None:
        """Apply random delays to simulate human behavior."""
        try:
            # Random delay between 0.5 and 2 seconds
            delay = random.uniform(0.5, 2.0)
            await asyncio.sleep(delay)
            
        except Exception as e:
            self.logger.debug(f"Failed to apply random delay: {e}")
    
    async def detect_and_handle_captcha(self) -> bool:
        """Detect and handle CAPTCHA challenges."""
        try:
            # Check for CAPTCHA elements
            captcha_selectors = [
                ".captcha",
                "[data-testid='captcha']",
                ".recaptcha",
                "iframe[src*='recaptcha']"
            ]
            
            for selector in captcha_selectors:
                try:
                    captcha_element = await self.browser_operations.find_element(selector, timeout=3000)
                    if captcha_element:
                        self.logger.warning("CAPTCHA detected - manual intervention may be required")
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Failed to detect CAPTCHA: {e}")
            return False
    
    async def _has_next_page(self) -> bool:
        """Check if there's a next page available."""
        try:
            # Check for pagination buttons
            for selector in self.vendor_config.pagination_selectors:
                try:
                    next_button = await self.browser_operations.find_element(selector)
                    if next_button:
                        # Check if button is enabled and not disabled
                        is_disabled = await next_button.get_attribute('disabled')
                        if not is_disabled:
                            return True
                except:
                    continue
            
            # Check for infinite scroll indicators
            scroll_indicators = [
                ".load-more",
                "[data-testid='load-more']",
                ".infinite-scroll-trigger"
            ]
            
            for selector in scroll_indicators:
                try:
                    indicator = await self.browser_operations.find_element(selector)
                    if indicator:
                        self.has_infinite_scroll = True
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check for next page: {e}")
            return False
    
    async def _navigate_to_next_page(self) -> bool:
        """Navigate to the next page using pagination or infinite scroll."""
        try:
            await self._apply_vendor_specific_delays('pagination')
            
            # Try pagination first
            for selector in self.vendor_config.pagination_selectors:
                try:
                    await self._retry_operation(
                        self.browser_operations.click,
                        selector=selector,
                        timeout=5000
                    )
                    
                    # Wait for new content to load
                    await self.browser_operations.wait_for_element(
                        selector=self.vendor_config.product_card_selectors[0],
                        timeout=15000
                    )
                    
                    self.current_page += 1
                    return True
                    
                except:
                    continue
            
            # Try infinite scroll if pagination failed
            if self.has_infinite_scroll:
                return await self._handle_infinite_scroll()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to navigate to next page: {e}")
            return False
    
    async def _handle_infinite_scroll(self) -> bool:
        """Handle infinite scroll by scrolling down and waiting for new content."""
        try:
            await self._apply_vendor_specific_delays('infinite_scroll')
            
            # Get current scroll position
            current_position = await self.browser_operations.execute_script(
                "return window.pageYOffset;"
            )
            
            # Scroll down
            await self.browser_operations.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            
            # Wait for new content to load
            await asyncio.sleep(2)
            
            # Check if new content was loaded
            new_position = await self.browser_operations.execute_script(
                "return window.pageYOffset;"
            )
            
            if new_position > current_position:
                self.current_page += 1
                self.last_scroll_position = new_position
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to handle infinite scroll: {e}")
            return False
    
    async def navigate_to_product_detail(self, product_url: str) -> Dict[str, Any]:
        """Navigate to a product detail page."""
        try:
            await self._apply_vendor_specific_delays('navigation')
            
            await self._retry_operation(
                self.browser_operations.navigate,
                url=product_url,
                wait_until="networkidle"
            )
            
            # Wait for product details to load
            detail_selectors = [
                ".product-details",
                "[data-testid='product-details']",
                ".product-info"
            ]
            
            for selector in detail_selectors:
                try:
                    await self.browser_operations.wait_for_element(selector, timeout=10000)
                    break
                except:
                    continue
            
            return {
                'status': 'success',
                'url': product_url,
                'current_url': self.browser_operations.get_current_url()
            }
            
        except Exception as e:
            return await self._handle_error(e, f"asda_product_detail_navigation_{product_url}")
    
    async def get_total_pages(self) -> Optional[int]:
        """Get the total number of pages if pagination is available."""
        try:
            # Look for pagination information
            pagination_info_selectors = [
                ".pagination-info",
                "[data-testid='pagination-info']",
                ".page-info"
            ]
            
            for selector in pagination_info_selectors:
                try:
                    info_element = await self.browser_operations.find_element(selector)
                    if info_element:
                        info_text = await self.browser_operations.get_text(selector)
                        # Extract page numbers from text like "Page 1 of 5"
                        import re
                        match = re.search(r'of\s+(\d+)', info_text)
                        if match:
                            return int(match.group(1))
                except:
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get total pages: {e}")
            return None
    
    async def _find_element_with_selectors(self, element_type: AsdaElementType, 
                                         timeout: int = 5000) -> Optional[Any]:
        """
        Find an element using the selector mapping with fallback strategies.
        
        Args:
            element_type: The type of element to find
            timeout: Timeout for element waiting
            
        Returns:
            Element if found, None otherwise
        """
        selectors = self.selector_mapping.get_selectors(element_type)
        
        for selector in selectors:
            try:
                element = await self.browser_operations.find_element(selector, timeout=timeout)
                if element:
                    return element
            except:
                continue
        
        return None
    
    async def _find_elements_with_selectors(self, element_type: AsdaElementType, 
                                          timeout: int = 5000) -> List[Any]:
        """
        Find elements using the selector mapping with fallback strategies.
        
        Args:
            element_type: The type of element to find
            timeout: Timeout for element waiting
            
        Returns:
            List of elements found
        """
        selectors = self.selector_mapping.get_selectors(element_type)
        
        for selector in selectors:
            try:
                elements = await self.browser_operations.find_elements(selector, timeout=timeout)
                if elements:
                    return elements
            except:
                continue
        
        return []
    
    async def _get_text_with_selectors(self, element_type: AsdaElementType, 
                                     timeout: int = 5000) -> Optional[str]:
        """
        Get text from an element using the selector mapping with fallback strategies.
        
        Args:
            element_type: The type of element to get text from
            timeout: Timeout for element waiting
            
        Returns:
            Text content if found, None otherwise
        """
        element = await self._find_element_with_selectors(element_type, timeout)
        if element:
            try:
                return await self.browser_operations.get_text(element)
            except:
                pass
        
        return None
    
    async def _get_attribute_with_selectors(self, element_type: AsdaElementType, 
                                          attribute: str, timeout: int = 5000) -> Optional[str]:
        """
        Get attribute from an element using the selector mapping with fallback strategies.
        
        Args:
            element_type: The type of element to get attribute from
            attribute: Attribute name to get
            timeout: Timeout for element waiting
            
        Returns:
            Attribute value if found, None otherwise
        """
        element = await self._find_element_with_selectors(element_type, timeout)
        if element:
            try:
                return await self.browser_operations.get_attribute(element, attribute)
            except:
                pass
        
        return None


class CostcoTool(VendorTool):
    """
    Costco-specific tool extensions with robust error handling for Costco's unreliable site.
    """
    
    def __init__(self, browser_manager: BrowserbaseManager, **kwargs):
        vendor_config = VendorConfig(
            name="Costco",
            base_url="https://www.costco.co.uk",
            search_selectors=[
                "input[type='search']",
                "input[name='search']",
                "#search-input",
                ".search-input",
                "[data-testid='search-input']"
            ],
            product_card_selectors=[
                ".product-item",
                ".product-card",
                "[data-testid='product-card']",
                ".product-tile",
                ".item-card",
                "[data-testid='product-item']"
            ],
            product_title_selectors=[
                "h1",
                ".product-title",
                "[data-testid='product-title']",
                ".item-title",
                "h2",
                ".product-name"
            ],
            product_price_selectors=[
                ".price",
                "[data-testid='product-price']",
                "span.price",
                ".costco-price",
                ".member-price",
                ".price-value"
            ],
            product_image_selectors=[
                "img",
                ".product-image img",
                "[data-testid='product-image']",
                ".item-image img",
                ".product-photo img"
            ],
            pagination_selectors=[
                ".pagination-next",
                ".next-page",
                "[data-testid='pagination-next']",
                ".pagination .next",
                "a[rel='next']",
                ".load-more"
            ],
            category_selectors=[
                ".category-menu",
                ".main-navigation",
                "nav",
                ".department-menu",
                ".category-nav",
                "[data-testid='category-menu']"
            ],
            anti_bot_delays={
                'navigation': (3, 8),  # Longer delays for Costco's unreliable site
                'interaction': (2, 5),
                'extraction': (1, 3),
                'search': (2, 6),
                'pagination': (2, 4),
                'infinite_scroll': (1, 3)
            },
            max_retries=5  # More retries for Costco
        )
        super().__init__(browser_manager, vendor_config, **kwargs)
        
        # Costco-specific navigation state
        self.current_page = 1
        self.total_pages = None
        self.has_infinite_scroll = False
        self.last_scroll_position = 0
        
        # Initialize Costco selector mapping
        self.selector_mapping = get_costco_selectors()
    
    async def navigate_to_category(self, category_name: str) -> Dict[str, Any]:
        """Navigate to a Costco category with robust error handling and multiple strategies."""
        try:
            await self._apply_vendor_specific_delays('navigation')
            
            # Handle Costco-specific challenges first
            await self.handle_costco_challenges()
            
            # Strategy 1: Direct URL navigation
            try:
                category_url = f"{self.vendor_config.base_url}/category/{str(category_name).lower().replace(' ', '-')}"
                await self._retry_operation(
                    self.browser_operations.navigate,
                    url=category_url,
                    wait_until="networkidle"
                )
                
                # Check if we're on a valid category page
                if await self._is_valid_category_page():
                    self.current_page = 1
                    return {
                        'status': 'success',
                        'category': category_name,
                        'url': self.browser_operations.get_current_url(),
                        'strategy': 'direct_url'
                    }
            except Exception as e:
                self.logger.warning(f"Direct URL navigation failed for {category_name}: {e}")
            
            # Strategy 2: Homepage navigation with menu
            try:
                await self._retry_operation(
                    self.browser_operations.navigate,
                    url=self.vendor_config.base_url,
                    wait_until="networkidle"
                )
                
                # Wait for navigation menu
                await self.browser_operations.wait_for_element(
                    selector=self.vendor_config.category_selectors[0],
                    timeout=20000
                )
                
                # Try multiple category selectors
                category_found = False
                for category_selector in [
                    f"a[href*='{str(category_name).lower()}']",
                    f"a:contains('{category_name}')",
                    f"a[href*='{str(category_name).lower().replace(' ', '-')}']",
                    f".category-menu a:contains('{category_name}')",
                    f".department-menu a:contains('{category_name}')"
                ]:
                    try:
                        await self._retry_operation(
                            self.browser_operations.click,
                            selector=category_selector,
                            timeout=10000
                        )
                        category_found = True
                        break
                    except:
                        continue
                
                if category_found:
                    # Wait for product grid
                    await self.browser_operations.wait_for_element(
                        selector=self.vendor_config.product_card_selectors[0],
                        timeout=25000
                    )
                    
                    self.current_page = 1
                    return {
                        'status': 'success',
                        'category': category_name,
                        'url': self.browser_operations.get_current_url(),
                        'strategy': 'menu_navigation'
                    }
                    
            except Exception as e:
                self.logger.warning(f"Menu navigation failed for {category_name}: {e}")
            
            # Strategy 3: Search-based navigation
            try:
                search_result = await self.search_products(category_name)
                if search_result.get('status') == 'success':
                    self.current_page = 1
                    return {
                        'status': 'success',
                        'category': category_name,
                        'url': self.browser_operations.get_current_url(),
                        'strategy': 'search_navigation'
                    }
            except Exception as e:
                self.logger.warning(f"Search navigation failed for {category_name}: {e}")
            
            # If all strategies fail
            raise NavigationError(f"All navigation strategies failed for category {category_name}")
            
        except Exception as e:
            return await self._handle_error(e, f"costco_category_navigation_{category_name}")
    
    async def _is_valid_category_page(self) -> bool:
        """Check if current page is a valid category page."""
        try:
            # Check for product cards
            for selector in self.vendor_config.product_card_selectors:
                try:
                    elements = await self.browser_operations.find_elements(selector)
                    if elements:
                        return True
                except:
                    continue
            
            # Check for category indicators
            category_indicators = [
                ".category-title",
                ".department-title",
                "[data-testid='category-header']",
                "h1"
            ]
            
            for selector in category_indicators:
                try:
                    element = await self.browser_operations.find_element(selector)
                    if element:
                        return True
                except:
                    continue
            
            return False
        except:
            return False
    
    async def search_products(self, query: str) -> Dict[str, Any]:
        """Search for products on Costco with robust error handling."""
        try:
            await self._apply_vendor_specific_delays('search')
            
            # Navigate to homepage
            await self._retry_operation(
                self.browser_operations.navigate,
                url=self.vendor_config.base_url,
                wait_until="networkidle"
            )
            
            # Find search input
            search_input = None
            for selector in self.vendor_config.search_selectors:
                try:
                    await self.browser_operations.wait_for_element(selector, timeout=10000)
                    search_input = selector
                    break
                except:
                    continue
            
            if not search_input:
                raise ValueError("Could not find search input on Costco")
            
            # Type search query
            await self._retry_operation(
                self.browser_operations.type_text,
                selector=search_input,
                text=query,
                timeout=10000
            )
            
            # Submit search
            await self._retry_operation(
                self.browser_operations.click,
                selector="button[type='submit'], .search-button",
                timeout=10000
            )
            
            # Wait for results with longer timeout
            await self.browser_operations.wait_for_element(
                selector=self.vendor_config.product_card_selectors[0],
                timeout=25000
            )
            
            return {
                'status': 'success',
                'query': query,
                'url': self.browser_operations.get_current_url()
            }
            
        except Exception as e:
            return await self._handle_error(e, f"costco_search_{query}")
    
    async def _has_next_page(self) -> bool:
        """Check if there's a next page available."""
        try:
            # Check for pagination buttons
            for selector in self.vendor_config.pagination_selectors:
                try:
                    next_button = await self.browser_operations.find_element(selector)
                    if next_button:
                        # Check if button is enabled and visible
                        is_enabled = await next_button.is_enabled()
                        is_displayed = await next_button.is_displayed()
                        if is_enabled and is_displayed:
                            return True
                except:
                    continue
            
            # Check for infinite scroll indicators
            scroll_indicators = [
                ".load-more",
                ".infinite-scroll",
                "[data-testid='load-more']",
                ".show-more"
            ]
            
            for selector in scroll_indicators:
                try:
                    indicator = await self.browser_operations.find_element(selector)
                    if indicator:
                        return True
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _navigate_to_next_page(self) -> bool:
        """Navigate to the next page or load more content."""
        try:
            await self._apply_vendor_specific_delays('pagination')
            
            # Try pagination first
            for selector in self.vendor_config.pagination_selectors:
                try:
                    next_button = await self.browser_operations.find_element(selector)
                    if next_button and await next_button.is_enabled():
                        await self.browser_operations.click(selector)
                        await self.browser_operations.wait_for_element(
                            selector=self.vendor_config.product_card_selectors[0],
                            timeout=15000
                        )
                        self.current_page += 1
                        return True
                except:
                    continue
            
            # Try infinite scroll
            if await self._handle_infinite_scroll():
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to navigate to next page: {e}")
            return False
    
    async def _handle_infinite_scroll(self) -> bool:
        """Handle infinite scroll loading."""
        try:
            await self._apply_vendor_specific_delays('infinite_scroll')
            
            # Get current scroll position
            current_position = await self.browser_operations.execute_script("return window.pageYOffset;")
            
            # Scroll down
            await self.browser_operations.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # Wait for new content to load
            await asyncio.sleep(2)
            
            # Check if new content was loaded
            new_position = await self.browser_operations.execute_script("return window.pageYOffset;")
            
            if new_position > current_position:
                self.last_scroll_position = new_position
                self.has_infinite_scroll = True
                return True
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to handle infinite scroll: {e}")
            return False
    
    async def navigate_to_product_detail(self, product_url: str) -> Dict[str, Any]:
        """Navigate to a specific product detail page."""
        try:
            await self._apply_vendor_specific_delays('navigation')
            
            await self._retry_operation(
                self.browser_operations.navigate,
                url=product_url,
                wait_until="networkidle"
            )
            
            # Wait for product details to load
            await self.browser_operations.wait_for_element(
                selector=self.vendor_config.product_title_selectors[0],
                timeout=15000
            )
            
            return {
                'status': 'success',
                'url': product_url
            }
        except Exception as e:
            return await self._handle_error(e, f"costco_product_detail_navigation_{product_url}")
    
    async def get_total_pages(self) -> Optional[int]:
        """Get total number of pages if available."""
        try:
            # Look for pagination info
            pagination_info_selectors = [
                ".pagination-info",
                ".page-info",
                "[data-testid='pagination-info']",
                ".results-info"
            ]
            
            for selector in pagination_info_selectors:
                try:
                    info_element = await self.browser_operations.find_element(selector)
                    if info_element:
                        info_text = await self.browser_operations.get_text(selector)
                        # Extract page numbers from text like "Page 1 of 5"
                        import re
                        match = re.search(r'(\d+)\s+of\s+(\d+)', info_text)
                        if match:
                            return int(match.group(2))
                except:
                    continue
            
            return None
        except:
            return None
    
    async def extract_product_cards(self) -> List[Dict[str, Any]]:
        """Extract product cards from Costco with robust error handling using selector mapping."""
        try:
            await self._apply_vendor_specific_delays('extraction')
            
            # Find product cards using selector mapping
            cards = await self._find_elements_with_selectors(CostcoElementType.PRODUCT_CARD)
            
            if not cards:
                self.logger.warning("No product cards found on Costco page")
                return []
            
            products = []
            
            # Extract data from each card
            for index, card in enumerate(cards):
                try:
                    product_data = await self._extract_single_product_card(card, index)
                    if product_data:
                        products.append(product_data)
                
                except Exception as e:
                    self.logger.warning(f"Failed to extract Costco product card {index}: {e}")
                    continue
            
            return products
            
        except Exception as e:
            self.logger.error(f"Failed to extract Costco product cards: {e}")
            return []
    
    async def _extract_single_product_card(self, card: Any, index: int) -> Optional[Dict[str, Any]]:
        """
        Extract detailed data from a single product card.
        
        Args:
            card: Product card element
            index: Index of the card
            
        Returns:
            Dictionary with product data or None if extraction fails
        """
        try:
            # Extract title
            title = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_TITLE)
            
            # Extract price
            price = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_PRICE)
            
            # Extract Costco-specific price
            costco_price = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_COSTCO_PRICE)
            
            # Extract image
            image_url = await self._extract_attribute_from_card(card, CostcoElementType.PRODUCT_IMAGE, 'src')
            
            # Extract link
            product_url = await self._extract_attribute_from_card(card, CostcoElementType.PRODUCT_LINK, 'href')
            
            # Extract description
            description = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_DESCRIPTION)
            
            # Extract unit price
            unit_price = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_UNIT_PRICE)
            
            # Extract offer type
            offer_type = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_OFFER_TYPE)
            
            # Extract availability
            availability = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_AVAILABILITY)
            
            # Extract stock status
            stock_status = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_STOCK_STATUS)
            
            # Extract membership requirement
            membership_required = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_MEMBERSHIP_REQUIRED)
            
            # Extract bulk quantity
            bulk_quantity = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_BULK_QUANTITY)
            
            # Extract warehouse location
            warehouse_location = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_WAREHOUSE_LOCATION)
            
            # Extract online-only indicator
            online_only = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_ONLINE_ONLY)
            
            # Extract warehouse-only indicator
            in_warehouse_only = await self._extract_text_from_card(card, CostcoElementType.PRODUCT_IN_WAREHOUSE_ONLY)
            
            # Only return if we have at least title or price
            if title or price:
                return {
                    'title': title,
                    'price': price,
                    'costco_price': costco_price,
                    'image': image_url,
                    'link': product_url,
                    'description': description,
                    'unit_price': unit_price,
                    'offer_type': offer_type,
                    'availability': availability,
                    'stock_status': stock_status,
                    'membership_required': bool(membership_required),
                    'bulk_quantity': bulk_quantity,
                    'warehouse_location': warehouse_location,
                    'online_only': bool(online_only),
                    'in_warehouse_only': bool(in_warehouse_only),
                    'vendor': 'Costco'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to extract single product card: {e}")
            return None
    
    async def _extract_text_from_card(self, card: Any, element_type: CostcoElementType) -> Optional[str]:
        """
        Extract text from an element within a product card.
        
        Args:
            card: Product card element
            element_type: Type of element to extract
            
        Returns:
            Extracted text or None
        """
        try:
            selectors = self.selector_mapping.get_selectors(element_type)
            
            for selector in selectors:
                try:
                    # Try to find element within the card
                    element = await card.find_element(selector)
                    if element:
                        text = await element.text()
                        if text and text.strip():
                            return text.strip()
                except:
                    continue
            
            return None
        except Exception as e:
            self.logger.debug(f"Failed to extract text for {element_type.value}: {e}")
            return None
    
    async def _extract_attribute_from_card(self, card: Any, element_type: CostcoElementType, 
                                         attribute: str) -> Optional[str]:
        """
        Extract attribute from an element within a product card.
        
        Args:
            card: Product card element
            element_type: Type of element to extract
            attribute: Attribute name to extract
            
        Returns:
            Extracted attribute value or None
        """
        try:
            selectors = self.selector_mapping.get_selectors(element_type)
            
            for selector in selectors:
                try:
                    # Try to find element within the card
                    element = await card.find_element(selector)
                    if element:
                        value = await element.get_attribute(attribute)
                        if value:
                            return value
                except:
                    continue
            
            return None
        except Exception as e:
            self.logger.debug(f"Failed to extract attribute {attribute} for {element_type.value}: {e}")
            return None
    
    async def extract_product_details(self, product_url: str) -> Optional[Dict[str, Any]]:
        """
        Extract detailed product information from a product detail page.
        
        Args:
            product_url: URL of the product detail page
            
        Returns:
            Dictionary with detailed product information or None if extraction fails
        """
        try:
            # Navigate to product detail page
            nav_result = await self.navigate_to_product_detail(product_url)
            if nav_result.get('status') != 'success':
                return None
            
            # Extract detailed information using selector mapping
            details = {}
            
            # Extract specifications
            specifications = await self._extract_specifications()
            details['specifications'] = specifications
            
            # Extract nutrition information
            nutrition_info = await self._extract_nutrition_info()
            details['nutrition_info'] = nutrition_info
            
            # Extract allergens
            allergens = await self._extract_allergens()
            details['allergens'] = allergens
            
            # Extract dietary information
            dietary_info = await self._extract_dietary_info()
            details['dietary_info'] = dietary_info
            
            # Extract product code
            product_code = await self._get_text_with_selectors(CostcoElementType.PRODUCT_CODE)
            details['product_code'] = product_code
            
            # Extract additional Costco-specific fields
            membership_required = await self._get_text_with_selectors(CostcoElementType.PRODUCT_MEMBERSHIP_REQUIRED)
            details['membership_required'] = bool(membership_required)
            
            bulk_quantity = await self._get_text_with_selectors(CostcoElementType.PRODUCT_BULK_QUANTITY)
            details['bulk_quantity'] = bulk_quantity
            
            warehouse_location = await self._get_text_with_selectors(CostcoElementType.PRODUCT_WAREHOUSE_LOCATION)
            details['warehouse_location'] = warehouse_location
            
            online_only = await self._get_text_with_selectors(CostcoElementType.PRODUCT_ONLINE_ONLY)
            details['online_only'] = bool(online_only)
            
            in_warehouse_only = await self._get_text_with_selectors(CostcoElementType.PRODUCT_IN_WAREHOUSE_ONLY)
            details['in_warehouse_only'] = bool(in_warehouse_only)
            
            return details
            
        except Exception as e:
            self.logger.error(f"Failed to extract product details from {product_url}: {e}")
            return None
    
    async def extract_multiple_product_details(self, product_urls: List[str]) -> List[Dict[str, Any]]:
        """
        Extract detailed information from multiple product URLs.
        
        Args:
            product_urls: List of product URLs to extract details from
            
        Returns:
            List of dictionaries with product details
        """
        details_list = []
        
        for i, url in enumerate(product_urls):
            try:
                self.logger.info(f"Extracting details from product {i+1}/{len(product_urls)}: {url}")
                
                details = await self.extract_product_details(url)
                if details:
                    details['url'] = url
                    details_list.append(details)
                
                # Apply delays between extractions
                await self._apply_vendor_specific_delays('extraction')
                
            except Exception as e:
                self.logger.warning(f"Failed to extract details from {url}: {e}")
                continue
        
        return details_list
    
    async def _extract_specifications(self) -> Dict[str, Any]:
        """Extract product specifications."""
        try:
            specifications = {}
            
            # Find specifications container
            spec_element = await self._find_element_with_selectors(CostcoElementType.PRODUCT_SPECIFICATIONS)
            if not spec_element:
                return specifications
            
            # Extract specification key-value pairs
            # This is a simplified approach - in practice, you'd need to parse the specific structure
            spec_text = await spec_element.text()
            if spec_text:
                # Parse specifications based on common patterns
                lines = spec_text.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        specifications[key.strip()] = value.strip()
            
            return specifications
            
        except Exception as e:
            self.logger.error(f"Failed to extract specifications: {e}")
            return {}
    
    async def _extract_nutrition_info(self) -> Dict[str, Any]:
        """Extract nutrition information."""
        try:
            nutrition_info = {}
            
            # Find nutrition table
            nutrition_element = await self._find_element_with_selectors(CostcoElementType.PRODUCT_NUTRITION)
            if not nutrition_element:
                return nutrition_info
            
            # Extract nutrition data
            nutrition_text = await nutrition_element.text()
            if nutrition_text:
                # Parse nutrition information based on common patterns
                lines = nutrition_text.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        nutrition_info[key.strip()] = value.strip()
            
            return nutrition_info
            
        except Exception as e:
            self.logger.error(f"Failed to extract nutrition info: {e}")
            return {}
    
    async def _extract_allergens(self) -> List[str]:
        """Extract allergen information."""
        try:
            allergens = []
            
            # Find allergen information
            allergen_element = await self._find_element_with_selectors(CostcoElementType.PRODUCT_ALLERGENS)
            if not allergen_element:
                return allergens
            
            # Extract allergen text
            allergen_text = await allergen_element.text()
            if allergen_text:
                # Parse allergens (comma-separated or listed)
                allergen_list = allergen_text.split(',')
                for allergen in allergen_list:
                    allergen = allergen.strip()
                    if allergen:
                        allergens.append(allergen)
            
            return allergens
            
        except Exception as e:
            self.logger.error(f"Failed to extract allergens: {e}")
            return []
    
    async def _extract_dietary_info(self) -> List[str]:
        """Extract dietary information."""
        try:
            dietary_info = []
            
            # Find dietary information
            dietary_element = await self._find_element_with_selectors(CostcoElementType.PRODUCT_DIETARY)
            if not dietary_element:
                return dietary_info
            
            # Extract dietary text
            dietary_text = await dietary_element.text()
            if dietary_text:
                # Parse dietary information (comma-separated or listed)
                dietary_list = dietary_text.split(',')
                for dietary in dietary_list:
                    dietary = dietary.strip()
                    if dietary:
                        dietary_info.append(dietary)
            
            return dietary_info
            
        except Exception as e:
            self.logger.error(f"Failed to extract dietary info: {e}")
            return []
    
    async def handle_costco_challenges(self) -> Dict[str, Any]:
        """Handle Costco-specific challenges like cookie consent, login prompts, etc."""
        try:
            challenges_handled = {
                'cookie_consent': False,
                'login_prompt': False,
                'membership_verification': False,
                'location_selector': False,
                'dynamic_content': False,
                'age_verification': False,
                'postal_code_prompt': False,
                'warehouse_selection': False,
                'rate_limiting': False
            }
            
            # Handle each challenge type with enhanced logging
            self.logger.info("Starting Costco challenge handling...")
            
            challenges_handled['cookie_consent'] = await self._handle_cookie_consent()
            if challenges_handled['cookie_consent']:
                self.logger.info("Successfully handled cookie consent")
            
            challenges_handled['login_prompt'] = await self._handle_login_prompt()
            if challenges_handled['login_prompt']:
                self.logger.info("Successfully handled login prompt")
            
            challenges_handled['membership_verification'] = await self._handle_membership_verification()
            if challenges_handled['membership_verification']:
                self.logger.info("Successfully handled membership verification")
            
            challenges_handled['location_selector'] = await self._handle_location_selector()
            if challenges_handled['location_selector']:
                self.logger.info("Successfully handled location selector")
            
            challenges_handled['dynamic_content'] = await self._handle_dynamic_content()
            if challenges_handled['dynamic_content']:
                self.logger.info("Successfully handled dynamic content")
            
            # Additional Costco-specific challenges
            challenges_handled['age_verification'] = await self._handle_age_verification()
            if challenges_handled['age_verification']:
                self.logger.info("Successfully handled age verification")
            
            challenges_handled['postal_code_prompt'] = await self._handle_postal_code_prompt()
            if challenges_handled['postal_code_prompt']:
                self.logger.info("Successfully handled postal code prompt")
            
            challenges_handled['warehouse_selection'] = await self._handle_warehouse_selection()
            if challenges_handled['warehouse_selection']:
                self.logger.info("Successfully handled warehouse selection")
            
            challenges_handled['rate_limiting'] = await self._handle_rate_limiting()
            if challenges_handled['rate_limiting']:
                self.logger.info("Successfully handled rate limiting")
            
            self.logger.info(f"Costco challenge handling completed: {challenges_handled}")
            
            return {
                'status': 'success',
                'challenges_handled': challenges_handled
            }
            
        except Exception as e:
            return await self._handle_error(e, "costco_challenge_handling")
    
    async def _handle_cookie_consent(self) -> bool:
        """Handle cookie consent dialogs."""
        try:
            cookie_selectors = [
                ".cookie-banner",
                ".cookie-consent",
                "[data-testid='cookie-banner']",
                ".gdpr-banner",
                ".cookie-notice"
            ]
            
            for selector in cookie_selectors:
                try:
                    cookie_banner = await self.browser_operations.find_element(selector)
                    if cookie_banner:
                        # Try to find accept button
                        accept_buttons = [
                            "button:contains('Accept')",
                            "button:contains('Accept All')",
                            "button:contains('Allow')",
                            ".accept-cookies",
                            "[data-testid='accept-cookies']"
                        ]
                        
                        for button_selector in accept_buttons:
                            try:
                                accept_button = await self.browser_operations.find_element(button_selector)
                                if accept_button:
                                    await self.browser_operations.click(button_selector)
                                    await asyncio.sleep(1)
                                    return True
                            except:
                                continue
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _handle_login_prompt(self) -> bool:
        """Handle login prompts and modals."""
        try:
            login_selectors = [
                ".login-modal",
                ".signin-modal",
                "[data-testid='login-modal']",
                ".auth-modal",
                ".login-overlay"
            ]
            
            for selector in login_selectors:
                try:
                    login_modal = await self.browser_operations.find_element(selector)
                    if login_modal:
                        # Try to find close button
                        close_buttons = [
                            ".close",
                            ".modal-close",
                            "[data-testid='close-modal']",
                            ".cancel",
                            "button:contains('Close')"
                        ]
                        
                        for button_selector in close_buttons:
                            try:
                                close_button = await self.browser_operations.find_element(button_selector)
                                if close_button:
                                    await self.browser_operations.click(button_selector)
                                    await asyncio.sleep(1)
                                    return True
                            except:
                                continue
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _handle_membership_verification(self) -> bool:
        """Handle membership verification prompts."""
        try:
            membership_selectors = [
                ".membership-required",
                ".member-verification",
                "[data-testid='membership-prompt']",
                ".login-required",
                ".member-only"
            ]
            
            for selector in membership_selectors:
                try:
                    membership_prompt = await self.browser_operations.find_element(selector)
                    if membership_prompt:
                        # Try to find continue as guest or skip button
                        skip_buttons = [
                            "button:contains('Continue as Guest')",
                            "button:contains('Skip')",
                            "button:contains('Browse')",
                            ".guest-continue",
                            "[data-testid='skip-login']"
                        ]
                        
                        for button_selector in skip_buttons:
                            try:
                                skip_button = await self.browser_operations.find_element(button_selector)
                                if skip_button:
                                    await self.browser_operations.click(button_selector)
                                    await asyncio.sleep(1)
                                    return True
                            except:
                                continue
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _handle_location_selector(self) -> bool:
        """Handle location/postcode selectors."""
        try:
            location_selectors = [
                ".location-selector",
                ".postcode-input",
                "[data-testid='location-input']",
                ".store-locator",
                ".location-prompt"
            ]
            
            for selector in location_selectors:
                try:
                    location_input = await self.browser_operations.find_element(selector)
                    if location_input:
                        # Try to find skip or continue button
                        skip_buttons = [
                            "button:contains('Skip')",
                            "button:contains('Continue')",
                            "button:contains('Browse')",
                            ".skip-location",
                            "[data-testid='skip-location']"
                        ]
                        
                        for button_selector in skip_buttons:
                            try:
                                skip_button = await self.browser_operations.find_element(button_selector)
                                if skip_button:
                                    await self.browser_operations.click(button_selector)
                                    await asyncio.sleep(1)
                                    return True
                            except:
                                continue
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _handle_dynamic_content(self) -> bool:
        """Handle dynamic content loading."""
        try:
            # Wait for page to be fully loaded
            await asyncio.sleep(2)
            
            # Check for loading indicators
            loading_selectors = [
                ".loading",
                ".spinner",
                "[data-testid='loading']",
                ".loading-indicator"
            ]
            
            for selector in loading_selectors:
                try:
                    loading_element = await self.browser_operations.find_element(selector)
                    if loading_element:
                        # Wait for loading to complete
                        await self.browser_operations.wait_for_element_not_present(selector, timeout=10000)
                except:
                    continue
            
            # Perform some scrolling to trigger lazy loading
            await self.browser_operations.execute_script("window.scrollTo(0, 100);")
            await asyncio.sleep(1)
            await self.browser_operations.execute_script("window.scrollTo(0, 0);")
            
            return True
        except:
            return False
    
    async def _handle_age_verification(self) -> bool:
        """Handle age verification prompts."""
        try:
            age_selectors = [
                ".age-verification",
                ".age-check",
                "[data-testid='age-verification']",
                ".age-gate",
                ".age-prompt"
            ]
            
            for selector in age_selectors:
                try:
                    age_prompt = await self.browser_operations.find_element(selector)
                    if age_prompt:
                        # Try to find "Yes" or "I am over 18" button
                        yes_buttons = [
                            "button:contains('Yes')",
                            "button:contains('I am over 18')",
                            "button:contains('Continue')",
                            ".age-yes",
                            "[data-testid='age-yes']"
                        ]
                        
                        for button_selector in yes_buttons:
                            try:
                                yes_button = await self.browser_operations.find_element(button_selector)
                                if yes_button:
                                    await self.browser_operations.click(button_selector)
                                    await asyncio.sleep(1)
                                    return True
                            except:
                                continue
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _handle_postal_code_prompt(self) -> bool:
        """Handle postal code input prompts."""
        try:
            postal_selectors = [
                ".postal-code-input",
                ".postcode-input",
                "[data-testid='postal-input']",
                ".zip-input",
                ".location-input"
            ]
            
            for selector in postal_selectors:
                try:
                    postal_input = await self.browser_operations.find_element(selector)
                    if postal_input:
                        # Try to find skip or continue button
                        skip_buttons = [
                            "button:contains('Skip')",
                            "button:contains('Continue')",
                            "button:contains('Browse')",
                            ".skip-postal",
                            "[data-testid='skip-postal']"
                        ]
                        
                        for button_selector in skip_buttons:
                            try:
                                skip_button = await self.browser_operations.find_element(button_selector)
                                if skip_button:
                                    await self.browser_operations.click(button_selector)
                                    await asyncio.sleep(1)
                                    return True
                            except:
                                continue
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _handle_warehouse_selection(self) -> bool:
        """Handle warehouse selection prompts."""
        try:
            warehouse_selectors = [
                ".warehouse-selector",
                ".store-selector",
                "[data-testid='warehouse-selector']",
                ".location-selector",
                ".store-locator"
            ]
            
            for selector in warehouse_selectors:
                try:
                    warehouse_prompt = await self.browser_operations.find_element(selector)
                    if warehouse_prompt:
                        # Try to find skip or continue button
                        skip_buttons = [
                            "button:contains('Skip')",
                            "button:contains('Continue')",
                            "button:contains('Browse')",
                            ".skip-warehouse",
                            "[data-testid='skip-warehouse']"
                        ]
                        
                        for button_selector in skip_buttons:
                            try:
                                skip_button = await self.browser_operations.find_element(button_selector)
                                if skip_button:
                                    await self.browser_operations.click(button_selector)
                                    await asyncio.sleep(1)
                                    return True
                            except:
                                continue
                except:
                    continue
            
            return False
        except:
            return False
    
    async def _handle_rate_limiting(self) -> bool:
        """Handle rate limiting and temporary blocks."""
        try:
            # Check for rate limiting indicators
            rate_limit_selectors = [
                ".rate-limit",
                ".too-many-requests",
                "[data-testid='rate-limit']",
                ".temporary-block",
                ".please-wait"
            ]
            
            for selector in rate_limit_selectors:
                try:
                    rate_limit_element = await self.browser_operations.find_element(selector)
                    if rate_limit_element:
                        self.logger.warning("Rate limiting detected on Costco page")
                        # Wait for rate limiting to pass
                        await asyncio.sleep(30)  # Wait 30 seconds
                        return True
                except:
                    continue
            
            return False
        except:
            return False
    
    async def apply_enhanced_anti_bot_measures(self) -> None:
        """Apply enhanced anti-bot measures for Costco."""
        try:
            self.logger.info("Applying enhanced anti-bot measures for Costco...")
            
            # Simulate human mouse movements
            await self._simulate_human_mouse_movements()
            
            # Simulate human scrolling
            await self._simulate_human_scrolling()
            
            # Apply random delays
            await self._apply_random_delays()
            
            # Detect and handle CAPTCHA
            captcha_detected = await self.detect_and_handle_captcha()
            if captcha_detected:
                self.logger.warning("CAPTCHA detected and handled")
            
            # Apply Costco-specific measures
            await self._apply_costco_specific_measures()
            
            self.logger.info("Enhanced anti-bot measures applied successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to apply anti-bot measures: {e}")
    
    async def _apply_costco_specific_measures(self) -> None:
        """Apply Costco-specific anti-bot measures."""
        try:
            # Simulate page interaction patterns common to Costco users
            await self._simulate_costco_user_behavior()
            
            # Apply longer delays for Costco (more conservative)
            await asyncio.sleep(2)  # Additional delay for Costco
            
            # Simulate form interactions (if any forms are present)
            await self._simulate_form_interactions()
            
        except Exception as e:
            self.logger.error(f"Failed to apply Costco-specific measures: {e}")
    
    async def _simulate_costco_user_behavior(self) -> None:
        """Simulate behavior patterns common to Costco users."""
        try:
            # Simulate reading product descriptions
            await self.browser_operations.execute_script("window.scrollTo(0, 200);")
            await asyncio.sleep(1)
            
            # Simulate checking prices
            await self.browser_operations.execute_script("window.scrollTo(0, 400);")
            await asyncio.sleep(1)
            
            # Simulate looking at product images
            await self.browser_operations.execute_script("window.scrollTo(0, 100);")
            await asyncio.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Failed to simulate Costco user behavior: {e}")
    
    async def _simulate_form_interactions(self) -> None:
        """Simulate form interactions if forms are present."""
        try:
            # Check for any form elements
            form_selectors = [
                "form",
                ".search-form",
                ".filter-form",
                ".sort-form"
            ]
            
            for selector in form_selectors:
                try:
                    form = await self.browser_operations.find_element(selector)
                    if form:
                        # Simulate form focus/blur
                        await self.browser_operations.execute_script(f"""
                            var form = document.querySelector('{selector}');
                            if (form) {{
                                form.dispatchEvent(new Event('focus'));
                                setTimeout(() => {{
                                    form.dispatchEvent(new Event('blur'));
                                }}, 500);
                            }}
                        """)
                        await asyncio.sleep(0.5)
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Failed to simulate form interactions: {e}")
    
    async def _simulate_human_mouse_movements(self) -> None:
        """Simulate human-like mouse movements."""
        try:
            # Get viewport dimensions
            viewport_width = await self.browser_operations.execute_script("return window.innerWidth;")
            viewport_height = await self.browser_operations.execute_script("return window.innerHeight;")
            
            # Generate random mouse movements
            import random
            for _ in range(3):
                x = random.randint(100, viewport_width - 100)
                y = random.randint(100, viewport_height - 100)
                
                await self.browser_operations.execute_script(f"""
                    var event = new MouseEvent('mousemove', {{
                        'view': window,
                        'bubbles': true,
                        'cancelable': true,
                        'clientX': {x},
                        'clientY': {y}
                    }});
                    document.dispatchEvent(event);
                """)
                
                await asyncio.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            self.logger.error(f"Failed to simulate mouse movements: {e}")
    
    async def _simulate_human_scrolling(self) -> None:
        """Simulate human-like scrolling behavior."""
        try:
            import random
            
            # Get page height
            page_height = await self.browser_operations.execute_script("return document.body.scrollHeight;")
            
            # Perform random scrolling
            for _ in range(2):
                scroll_position = random.randint(100, min(page_height - 100, 1000))
                await self.browser_operations.execute_script(f"window.scrollTo(0, {scroll_position});")
                await asyncio.sleep(random.uniform(1, 3))
            
            # Scroll back to top
            await self.browser_operations.execute_script("window.scrollTo(0, 0);")
            await asyncio.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Failed to simulate scrolling: {e}")
    
    async def _apply_random_delays(self) -> None:
        """Apply random delays between actions."""
        try:
            import random
            delay = random.uniform(1, 3)
            await asyncio.sleep(delay)
        except Exception as e:
            self.logger.error(f"Failed to apply random delays: {e}")
    
    async def detect_and_handle_captcha(self) -> bool:
        """Detect and handle CAPTCHA challenges."""
        try:
            captcha_selectors = [
                ".captcha",
                "[data-testid='captcha']",
                ".recaptcha",
                ".g-recaptcha",
                ".captcha-challenge"
            ]
            
            for selector in captcha_selectors:
                try:
                    captcha_element = await self.browser_operations.find_element(selector)
                    if captcha_element:
                        self.logger.warning("CAPTCHA detected on Costco page")
                        return True
                except:
                    continue
            
            return False
        except:
            return False


# Factory functions for creating vendor-specific tools
def create_tesco_tool(browser_manager: BrowserbaseManager, 
                     anti_bot_config: Optional[AntiBotConfig] = None,
                     storage_manager: Optional[StorageManager] = None) -> TescoTool:
    """Create a Tesco-specific tool instance."""
    return TescoTool(browser_manager, anti_bot_config=anti_bot_config, storage_manager=storage_manager)


def create_asda_tool(browser_manager: BrowserbaseManager,
                    anti_bot_config: Optional[AntiBotConfig] = None,
                    storage_manager: Optional[StorageManager] = None) -> AsdaTool:
    """Create an Asda-specific tool instance."""
    return AsdaTool(browser_manager, anti_bot_config=anti_bot_config, storage_manager=storage_manager)


def create_costco_tool(browser_manager: BrowserbaseManager,
                      anti_bot_config: Optional[AntiBotConfig] = None,
                      storage_manager: Optional[StorageManager] = None) -> CostcoTool:
    """Create a Costco-specific tool instance."""
    return CostcoTool(browser_manager, anti_bot_config=anti_bot_config, storage_manager=storage_manager)


def create_all_vendor_tools(browser_manager: BrowserbaseManager,
                           anti_bot_config: Optional[AntiBotConfig] = None,
                           storage_manager: Optional[StorageManager] = None) -> Dict[str, VendorTool]:
    """Create all vendor-specific tools with the same configuration."""
    return {
        'tesco': create_tesco_tool(browser_manager, anti_bot_config, storage_manager),
        'asda': create_asda_tool(browser_manager, anti_bot_config, storage_manager),
        'costco': create_costco_tool(browser_manager, anti_bot_config, storage_manager)
    } 