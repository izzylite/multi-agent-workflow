"""
Costco Integration Module

Provides Costco-specific scraping agent and product data structures for
Browserbase-based scraping of Costco's UK wholesale platform.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from crewai import Agent, Task
from crewai.tools import BaseTool

from .crewai_integration import ScrapingAgent, AgentConfig, AgentRole, TaskConfig, TaskType
from .vendor_tools import CostcoTool, create_costco_tool
from .browserbase_manager import BrowserbaseManager
from .storage_manager import ProductData, StorageManager
from .exceptions import SessionCreationError, NavigationError, ElementNotFoundError


class CostcoCategory(Enum):
    """Costco product categories."""
    FRESH_FOOD = "fresh_food"
    PANTRY = "pantry"
    DAIRY = "dairy"
    FROZEN = "frozen"
    HOUSEHOLD = "household"
    DRINKS = "drinks"
    HEALTH = "health"
    BABY = "baby"
    BAKERY = "bakery"
    MEAT_FISH = "meat_fish"
    FRUIT_VEG = "fruit_vegetables"
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    AUTOMOTIVE = "automotive"
    GARDEN = "garden"
    OFFICE = "office"
    SPORTS = "sports"
    TOYS = "toys"
    BOOKS = "books"
    PHARMACY = "pharmacy"


@dataclass
class CostcoProduct(ProductData):
    """Costco-specific product data structure."""
    specifications: Optional[Dict[str, Any]] = None
    availability: Optional[str] = None
    stock_status: Optional[str] = None
    unit_price: Optional[str] = None
    offer_type: Optional[str] = None
    costco_price: Optional[str] = None
    nutrition_info: Optional[Dict[str, Any]] = None
    allergens: Optional[List[str]] = None
    dietary_info: Optional[List[str]] = None
    product_code: Optional[str] = None
    membership_required: Optional[bool] = None
    bulk_quantity: Optional[str] = None
    warehouse_location: Optional[str] = None
    online_only: Optional[bool] = None
    in_warehouse_only: Optional[bool] = None

    def __post_init__(self):
        super().__post_init__()
        if self.vendor is None:
            self.vendor = "Costco"
        if self.specifications is None:
            self.specifications = {}
        if self.nutrition_info is None:
            self.nutrition_info = {}
        if self.allergens is None:
            self.allergens = []
        if self.dietary_info is None:
            self.dietary_info = []


class CostcoAgent(ScrapingAgent):
    """
    Costco-specific scraping agent extending ScrapingAgent with CostcoTool integration.
    
    This agent provides specialized functionality for scraping Costco's UK wholesale platform,
    including category navigation, product extraction, and data processing.
    """
    
    def __init__(self, 
                 browser_manager: BrowserbaseManager,
                 storage_manager: Optional[StorageManager] = None,
                 anti_bot_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize CostcoAgent.
        
        Args:
            browser_manager: BrowserbaseManager for session management
            storage_manager: Optional StorageManager for data persistence
            anti_bot_config: Optional anti-bot configuration
            **kwargs: Additional arguments for ScrapingAgent
        """
        # Create Costco-specific agent configuration
        config = AgentConfig(
            role=AgentRole.SCRAPER,
            name="CostcoScraper",
            goal="Scrape comprehensive product data from Costco's UK wholesale platform with high accuracy and efficiency",
            backstory="""You are an expert Costco web scraper with deep knowledge of Costco's website structure, 
            navigation patterns, and product data extraction. You excel at handling Costco's membership requirements, 
            bulk product layouts, and anti-bot measures. You ensure data quality and completeness while 
            maintaining respectful scraping practices.""",
            verbose=True,
            allow_delegation=True
        )
        
        super().__init__(config)
        
        self.browser_manager = browser_manager
        self.storage_manager = storage_manager
        self.anti_bot_config = anti_bot_config or {}
        self.costco_tool = create_costco_tool(
            browser_manager=browser_manager,
            anti_bot_config=anti_bot_config,
            storage_manager=storage_manager
        )
        
        # Add CostcoTool to agent's tools
        if self.config.tools is None:
            self.config.tools = []
        self.config.tools.append(self.costco_tool)
        
        # Recreate agent with updated tools
        self.agent = self._create_agent()
        
        self.logger = logging.getLogger(f"{__name__}.CostcoAgent")
        
        # Costco-specific category mapping
        self.category_mapping = {
            CostcoCategory.FRESH_FOOD: ["fresh-food", "fruit-vegetables", "meat-fish"],
            CostcoCategory.PANTRY: ["pantry", "food-cupboard", "baking"],
            CostcoCategory.DAIRY: ["dairy", "milk", "cheese", "yogurt"],
            CostcoCategory.FROZEN: ["frozen", "frozen-food"],
            CostcoCategory.HOUSEHOLD: ["household", "cleaning", "toiletries"],
            CostcoCategory.DRINKS: ["drinks", "soft-drinks", "alcohol"],
            CostcoCategory.HEALTH: ["health", "pharmacy", "beauty"],
            CostcoCategory.BABY: ["baby", "baby-food", "nappies"],
            CostcoCategory.BAKERY: ["bakery", "bread", "cakes"],
            CostcoCategory.MEAT_FISH: ["meat-fish", "meat", "fish"],
            CostcoCategory.FRUIT_VEG: ["fruit-vegetables", "fruit", "vegetables"],
            CostcoCategory.ELECTRONICS: ["electronics", "computers", "phones"],
            CostcoCategory.CLOTHING: ["clothing", "apparel", "fashion"],
            CostcoCategory.AUTOMOTIVE: ["automotive", "car", "vehicle"],
            CostcoCategory.GARDEN: ["garden", "outdoor", "lawn"],
            CostcoCategory.OFFICE: ["office", "business", "supplies"],
            CostcoCategory.SPORTS: ["sports", "fitness", "exercise"],
            CostcoCategory.TOYS: ["toys", "games", "entertainment"],
            CostcoCategory.BOOKS: ["books", "media", "entertainment"],
            CostcoCategory.PHARMACY: ["pharmacy", "health", "medicine"]
        }
    
    async def navigate_to_category(self, category: Union[str, CostcoCategory]) -> Dict[str, Any]:
        """
        Navigate to a specific Costco category.
        
        Args:
            category: Category name or CostcoCategory enum
            
        Returns:
            Navigation result with status and metadata
        """
        try:
            if isinstance(category, CostcoCategory):
                category_name = category.value
            else:
                category_name = category
            
            self.logger.info(f"Navigating to Costco category: {category_name}")
            
            result = await self.costco_tool.navigate_to_category(category_name)
            
            # Log navigation event
            self._log_memory_event("costco_category_navigation", details={
                "category": category_name,
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to navigate to category {category}: {e}")
            return {
                'status': 'error',
                'category': category,
                'error': str(e)
            }
    
    async def search_products(self, query: str) -> Dict[str, Any]:
        """
        Search for products on Costco.
        
        Args:
            query: Search query string
            
        Returns:
            Search result with status and metadata
        """
        try:
            self.logger.info(f"Searching Costco for: {query}")
            
            result = await self.costco_tool.search_products(query)
            
            # Log search event
            self._log_memory_event("costco_product_search", details={
                "query": query,
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to search for {query}: {e}")
            return {
                'status': 'error',
                'query': query,
                'error': str(e)
            }
    
    async def extract_product_cards(self) -> List[CostcoProduct]:
        """
        Extract product cards from current Costco page.
        
        Returns:
            List of CostcoProduct objects
        """
        try:
            self.logger.info("Extracting product cards from Costco page")
            
            raw_products = await self.costco_tool.extract_product_cards()
            
            # Convert to CostcoProduct objects
            costco_products = []
            for product_dict in raw_products:
                costco_product = self._convert_to_costco_product(product_dict)
                if costco_product:
                    costco_products.append(costco_product)
            
            # Log extraction event
            self._log_memory_event("costco_product_extraction", details={
                "products_count": len(costco_products),
                "raw_products_count": len(raw_products)
            })
            
            return costco_products
            
        except Exception as e:
            self.logger.error(f"Failed to extract product cards: {e}")
            return []
    
    async def scrape_category(self, category: Union[str, CostcoCategory], 
                            max_pages: int = 5) -> List[CostcoProduct]:
        """
        Scrape all products from a Costco category.
        
        Args:
            category: Category name or CostcoCategory enum
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of CostcoProduct objects
        """
        try:
            self.logger.info(f"Starting category scrape for: {category}")
            
            # Navigate to category
            nav_result = await self.navigate_to_category(category)
            if nav_result.get('status') != 'success':
                raise NavigationError(f"Failed to navigate to category {category}")
            
            all_products = []
            page_count = 0
            
            while page_count < max_pages:
                # Extract products from current page
                products = await self.extract_product_cards()
                all_products.extend(products)
                
                # Check if there's a next page
                has_next = await self.costco_tool._has_next_page()
                if not has_next:
                    break
                
                # Navigate to next page
                await self.costco_tool._navigate_to_next_page()
                page_count += 1
            
            # Log scraping event
            self._log_memory_event("costco_category_scrape", details={
                "category": category,
                "products_count": len(all_products),
                "pages_scraped": page_count + 1
            })
            
            return all_products
            
        except Exception as e:
            self.logger.error(f"Failed to scrape category {category}: {e}")
            return []
    
    async def scrape_search_results(self, query: str, max_pages: int = 3) -> List[CostcoProduct]:
        """
        Scrape products from search results.
        
        Args:
            query: Search query
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of CostcoProduct objects
        """
        try:
            self.logger.info(f"Starting search scrape for: {query}")
            
            # Perform search
            search_result = await self.search_products(query)
            if search_result.get('status') != 'success':
                raise NavigationError(f"Failed to search for {query}")
            
            all_products = []
            page_count = 0
            
            while page_count < max_pages:
                # Extract products from current page
                products = await self.extract_product_cards()
                all_products.extend(products)
                
                # Check if there's a next page
                has_next = await self.costco_tool._has_next_page()
                if not has_next:
                    break
                
                # Navigate to next page
                await self.costco_tool._navigate_to_next_page()
                page_count += 1
            
            # Log scraping event
            self._log_memory_event("costco_search_scrape", details={
                "query": query,
                "products_count": len(all_products),
                "pages_scraped": page_count + 1
            })
            
            return all_products
            
        except Exception as e:
            self.logger.error(f"Failed to scrape search results for {query}: {e}")
            return []
    
    async def get_detailed_products(self, products: List[CostcoProduct]) -> List[CostcoProduct]:
        """
        Get detailed information for a list of products by visiting each product page.
        
        Args:
            products: List of CostcoProduct objects with basic information
            
        Returns:
            List of CostcoProduct objects with detailed information
        """
        try:
            self.logger.info(f"Getting detailed information for {len(products)} products")
            
            detailed_products = []
            
            for product in products:
                if product.url:
                    try:
                        # Navigate to product page
                        await self.costco_tool.browser_operations.navigate(
                            url=product.url,
                            wait_until="networkidle"
                        )
                        
                        # Extract detailed information
                        detailed_product = await self._extract_product_details(product)
                        if detailed_product:
                            detailed_products.append(detailed_product)
                        
                        # Apply anti-bot delays
                        await self.costco_tool._apply_vendor_specific_delays('extraction')
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to get details for product {product.product_id}: {e}")
                        detailed_products.append(product)  # Keep original product
            
            # Log detailed extraction event
            self._log_memory_event("costco_detailed_extraction", details={
                "total_products": len(products),
                "successful_extractions": len(detailed_products)
            })
            
            return detailed_products
            
        except Exception as e:
            self.logger.error(f"Failed to get detailed products: {e}")
            return products
    
    def _convert_to_costco_product(self, product_dict: Dict[str, Any]) -> Optional[CostcoProduct]:
        """
        Convert raw product dictionary to CostcoProduct object.
        
        Args:
            product_dict: Raw product data dictionary
            
        Returns:
            CostcoProduct object or None if conversion fails
        """
        try:
            # Extract basic product information
            title = product_dict.get('title', '')
            price = product_dict.get('price', '')
            image_url = product_dict.get('image', '')
            product_url = product_dict.get('link', '')
            description = product_dict.get('description', '')
            
            # Generate product ID if not present
            product_id = product_dict.get('product_id')
            if not product_id and title:
                import hashlib
                product_id = hashlib.md5(title.encode()).hexdigest()[:12]
            
            # Extract Costco-specific fields
            specifications = product_dict.get('specifications', {})
            availability = product_dict.get('availability', '')
            stock_status = product_dict.get('stock_status', '')
            unit_price = product_dict.get('unit_price', '')
            offer_type = product_dict.get('offer_type', '')
            costco_price = product_dict.get('costco_price', '')
            nutrition_info = product_dict.get('nutrition_info', {})
            allergens = product_dict.get('allergens', [])
            dietary_info = product_dict.get('dietary_info', [])
            product_code = product_dict.get('product_code', '')
            membership_required = product_dict.get('membership_required', True)  # Default to True for Costco
            bulk_quantity = product_dict.get('bulk_quantity', '')
            warehouse_location = product_dict.get('warehouse_location', '')
            online_only = product_dict.get('online_only', False)
            in_warehouse_only = product_dict.get('in_warehouse_only', False)
            
            return CostcoProduct(
                vendor="Costco",
                category=product_dict.get('category', 'unknown'),
                product_id=product_id,
                title=title,
                price=price,
                image_url=image_url,
                url=product_url,
                description=description,
                specifications=specifications,
                availability=availability,
                stock_status=stock_status,
                unit_price=unit_price,
                offer_type=offer_type,
                costco_price=costco_price,
                nutrition_info=nutrition_info,
                allergens=allergens,
                dietary_info=dietary_info,
                product_code=product_code,
                membership_required=membership_required,
                bulk_quantity=bulk_quantity,
                warehouse_location=warehouse_location,
                online_only=online_only,
                in_warehouse_only=in_warehouse_only,
                scraped_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to convert product dict to CostcoProduct: {e}")
            return None
    
    async def _extract_product_details(self, product: CostcoProduct) -> Optional[CostcoProduct]:
        """
        Extract detailed information from a product page.
        
        Args:
            product: CostcoProduct object with basic information
            
        Returns:
            Updated CostcoProduct object with detailed information
        """
        try:
            # Extract detailed specifications
            specs_selectors = [
                ".product-specifications",
                "[data-testid='product-specifications']",
                ".product-details"
            ]
            
            for selector in specs_selectors:
                try:
                    specs_element = await self.costco_tool.browser_operations.find_element(selector)
                    if specs_element:
                        specs_text = await self.costco_tool.browser_operations.get_text(selector)
                        product.specifications['full_specifications'] = specs_text
                        break
                except:
                    continue
            
            # Extract nutrition information
            nutrition_selectors = [
                ".nutrition-table",
                "[data-testid='nutrition-info']",
                ".nutrition-information"
            ]
            
            for selector in nutrition_selectors:
                try:
                    nutrition_element = await self.costco_tool.browser_operations.find_element(selector)
                    if nutrition_element:
                        nutrition_text = await self.costco_tool.browser_operations.get_text(selector)
                        product.nutrition_info['nutrition_table'] = nutrition_text
                        break
                except:
                    continue
            
            # Extract allergens
            allergen_selectors = [
                ".allergen-information",
                "[data-testid='allergens']",
                ".allergens"
            ]
            
            for selector in allergen_selectors:
                try:
                    allergen_element = await self.costco_tool.browser_operations.find_element(selector)
                    if allergen_element:
                        allergen_text = await self.costco_tool.browser_operations.get_text(selector)
                        product.allergens = [allergen.strip() for allergen in allergen_text.split(',')]
                        break
                except:
                    continue
            
            # Extract dietary information
            dietary_selectors = [
                ".dietary-information",
                "[data-testid='dietary-info']",
                ".dietary-labels"
            ]
            
            for selector in dietary_selectors:
                try:
                    dietary_element = await self.costco_tool.browser_operations.find_element(selector)
                    if dietary_element:
                        dietary_text = await self.costco_tool.browser_operations.get_text(selector)
                        product.dietary_info = [diet.strip() for diet in dietary_text.split(',')]
                        break
                except:
                    continue
            
            # Extract product code
            product_code_selectors = [
                ".product-code",
                "[data-testid='product-code']",
                ".sku"
            ]
            
            for selector in product_code_selectors:
                try:
                    code_element = await self.costco_tool.browser_operations.find_element(selector)
                    if code_element:
                        product.product_code = await self.costco_tool.browser_operations.get_text(selector)
                        break
                except:
                    continue
            
            # Extract membership requirements
            membership_selectors = [
                ".membership-required",
                "[data-testid='membership-info']",
                ".membership-notice"
            ]
            
            for selector in membership_selectors:
                try:
                    membership_element = await self.costco_tool.browser_operations.find_element(selector)
                    if membership_element:
                        membership_text = await self.costco_tool.browser_operations.get_text(selector)
                        product.membership_required = "membership" in membership_text.lower()
                        break
                except:
                    continue
            
            # Extract bulk quantity information
            bulk_selectors = [
                ".bulk-quantity",
                "[data-testid='quantity-info']",
                ".quantity-notice"
            ]
            
            for selector in bulk_selectors:
                try:
                    bulk_element = await self.costco_tool.browser_operations.find_element(selector)
                    if bulk_element:
                        product.bulk_quantity = await self.costco_tool.browser_operations.get_text(selector)
                        break
                except:
                    continue
            
            # Extract warehouse location
            location_selectors = [
                ".warehouse-location",
                "[data-testid='location-info']",
                ".location-notice"
            ]
            
            for selector in location_selectors:
                try:
                    location_element = await self.costco_tool.browser_operations.find_element(selector)
                    if location_element:
                        product.warehouse_location = await self.costco_tool.browser_operations.get_text(selector)
                        break
                except:
                    continue
            
            return product
            
        except Exception as e:
            self.logger.error(f"Failed to extract product details: {e}")
            return product
    
    def create_crewai_agent(self) -> Agent:
        """
        Create a CrewAI agent specifically for Costco scraping.
        
        Returns:
            CrewAI Agent configured for Costco scraping
        """
        return Agent(
            role="Costco Product Scraper",
            goal="Extract comprehensive product data from Costco's UK wholesale platform with high accuracy",
            backstory="""You are a specialized Costco web scraper with deep expertise in Costco's website structure, 
            navigation patterns, and product data extraction. You excel at handling Costco's membership requirements, 
            bulk product layouts, and anti-bot measures. You ensure data quality and completeness while 
            maintaining respectful scraping practices. You have extensive experience with Costco's product categories, 
            pricing structures, and wholesale specifications.""",
            verbose=True,
            allow_delegation=True,
            tools=[self.costco_tool]
        )


def create_costco_agent(browser_manager: BrowserbaseManager,
                       storage_manager: Optional[StorageManager] = None,
                       anti_bot_config: Optional[Dict[str, Any]] = None) -> CostcoAgent:
    """
    Factory function to create a CostcoAgent instance.
    
    Args:
        browser_manager: BrowserbaseManager for session management
        storage_manager: Optional StorageManager for data persistence
        anti_bot_config: Optional anti-bot configuration
        
    Returns:
        Configured CostcoAgent instance
    """
    return CostcoAgent(
        browser_manager=browser_manager,
        storage_manager=storage_manager,
        anti_bot_config=anti_bot_config
    ) 