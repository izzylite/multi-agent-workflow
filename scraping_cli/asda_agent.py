"""
Asda Integration Module

Provides Asda-specific scraping agent and product data structures for
Browserbase-based scraping of Asda's grocery platform.
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
from .vendor_tools import AsdaTool, create_asda_tool
from .browserbase_manager import BrowserbaseManager
from .storage_manager import ProductData, StorageManager
from .exceptions import SessionCreationError, NavigationError, ElementNotFoundError


class AsdaCategory(Enum):
    """Asda product categories."""
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


@dataclass
class AsdaProduct(ProductData):
    """Asda-specific product data structure."""
    # Inherit from ProductData and add Asda-specific fields
    specifications: Optional[Dict[str, Any]] = None
    availability: Optional[str] = None
    stock_status: Optional[str] = None
    unit_price: Optional[str] = None
    offer_type: Optional[str] = None
    asda_price: Optional[str] = None
    nutrition_info: Optional[Dict[str, Any]] = None
    allergens: Optional[List[str]] = None
    dietary_info: Optional[List[str]] = None
    product_code: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.vendor is None:
            self.vendor = "Asda"
        if self.specifications is None:
            self.specifications = {}
        if self.nutrition_info is None:
            self.nutrition_info = {}
        if self.allergens is None:
            self.allergens = []
        if self.dietary_info is None:
            self.dietary_info = []


class AsdaAgent(ScrapingAgent):
    """
    Asda-specific scraping agent extending ScrapingAgent with AsdaTool integration.
    
    This agent provides specialized functionality for scraping Asda's grocery platform,
    including category navigation, product extraction, and data processing.
    """
    
    def __init__(self, 
                 browser_manager: BrowserbaseManager,
                 storage_manager: Optional[StorageManager] = None,
                 anti_bot_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize AsdaAgent.
        
        Args:
            browser_manager: BrowserbaseManager for session management
            storage_manager: Optional StorageManager for data persistence
            anti_bot_config: Optional anti-bot configuration
            **kwargs: Additional arguments for ScrapingAgent
        """
        # Create Asda-specific agent configuration
        config = AgentConfig(
            role=AgentRole.SCRAPER,
            name="AsdaScraper",
            goal="Scrape comprehensive product data from Asda's grocery platform with high accuracy and efficiency",
            backstory="""You are an expert Asda web scraper with deep knowledge of Asda's website structure, 
            navigation patterns, and product data extraction. You excel at handling Asda's breadcrumb navigation, 
            faceted filtering, and anti-bot measures. You ensure data quality and completeness while 
            maintaining respectful scraping practices.""",
            verbose=True,
            allow_delegation=True
        )
        
        super().__init__(config)
        
        self.browser_manager = browser_manager
        self.storage_manager = storage_manager
        self.anti_bot_config = anti_bot_config or {}
        self.asda_tool = create_asda_tool(
            browser_manager=browser_manager,
            anti_bot_config=anti_bot_config,
            storage_manager=storage_manager
        )
        
        # Add AsdaTool to agent's tools
        if self.config.tools is None:
            self.config.tools = []
        self.config.tools.append(self.asda_tool)
        
        # Recreate agent with updated tools
        self.agent = self._create_agent()
        
        self.logger = logging.getLogger(f"{__name__}.AsdaAgent")
        
        # Asda-specific category mapping
        self.category_mapping = {
            AsdaCategory.FRESH_FOOD: ["fresh-food", "fruit-vegetables", "meat-fish"],
            AsdaCategory.PANTRY: ["pantry", "food-cupboard", "baking"],
            AsdaCategory.DAIRY: ["dairy", "milk", "cheese", "yogurt"],
            AsdaCategory.FROZEN: ["frozen", "frozen-food"],
            AsdaCategory.HOUSEHOLD: ["household", "cleaning", "toiletries"],
            AsdaCategory.DRINKS: ["drinks", "soft-drinks", "alcohol"],
            AsdaCategory.HEALTH: ["health", "pharmacy", "beauty"],
            AsdaCategory.BABY: ["baby", "baby-food", "nappies"],
            AsdaCategory.BAKERY: ["bakery", "bread", "cakes"],
            AsdaCategory.MEAT_FISH: ["meat-fish", "meat", "fish"],
            AsdaCategory.FRUIT_VEG: ["fruit-vegetables", "fruit", "vegetables"]
        }
    
    async def navigate_to_category(self, category: Union[str, AsdaCategory]) -> Dict[str, Any]:
        """
        Navigate to a specific Asda category.
        
        Args:
            category: Category name or AsdaCategory enum
            
        Returns:
            Navigation result with status and metadata
        """
        try:
            if isinstance(category, AsdaCategory):
                category_name = category.value
            else:
                category_name = category
            
            self.logger.info(f"Navigating to Asda category: {category_name}")
            
            result = await self.asda_tool.navigate_to_category(category_name)
            
            # Log navigation event
            self._log_memory_event("asda_category_navigation", details={
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
        Search for products on Asda.
        
        Args:
            query: Search query string
            
        Returns:
            Search result with status and metadata
        """
        try:
            self.logger.info(f"Searching Asda for: {query}")
            
            result = await self.asda_tool.search_products(query)
            
            # Log search event
            self._log_memory_event("asda_product_search", details={
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
    
    async def extract_product_cards(self) -> List[AsdaProduct]:
        """
        Extract product cards from current Asda page.
        
        Returns:
            List of AsdaProduct objects
        """
        try:
            self.logger.info("Extracting product cards from Asda page")
            
            raw_products = await self.asda_tool.extract_product_cards()
            
            # Convert to AsdaProduct objects
            asda_products = []
            for product_dict in raw_products:
                asda_product = self._convert_to_asda_product(product_dict)
                if asda_product:
                    asda_products.append(asda_product)
            
            # Log extraction event
            self._log_memory_event("asda_product_extraction", details={
                "products_count": len(asda_products),
                "raw_products_count": len(raw_products)
            })
            
            return asda_products
            
        except Exception as e:
            self.logger.error(f"Failed to extract product cards: {e}")
            return []
    
    async def scrape_category(self, category: Union[str, AsdaCategory], 
                            max_pages: int = 5) -> List[AsdaProduct]:
        """
        Scrape all products from an Asda category.
        
        Args:
            category: Category name or AsdaCategory enum
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of AsdaProduct objects
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
                has_next = await self.asda_tool._has_next_page()
                if not has_next:
                    break
                
                # Navigate to next page
                await self.asda_tool._navigate_to_next_page()
                page_count += 1
            
            # Log scraping event
            self._log_memory_event("asda_category_scrape", details={
                "category": category,
                "products_count": len(all_products),
                "pages_scraped": page_count + 1
            })
            
            return all_products
            
        except Exception as e:
            self.logger.error(f"Failed to scrape category {category}: {e}")
            return []
    
    async def scrape_search_results(self, query: str, max_pages: int = 3) -> List[AsdaProduct]:
        """
        Scrape products from search results.
        
        Args:
            query: Search query
            max_pages: Maximum number of pages to scrape
            
        Returns:
            List of AsdaProduct objects
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
                has_next = await self.asda_tool._has_next_page()
                if not has_next:
                    break
                
                # Navigate to next page
                await self.asda_tool._navigate_to_next_page()
                page_count += 1
            
            # Log scraping event
            self._log_memory_event("asda_search_scrape", details={
                "query": query,
                "products_count": len(all_products),
                "pages_scraped": page_count + 1
            })
            
            return all_products
            
        except Exception as e:
            self.logger.error(f"Failed to scrape search results for {query}: {e}")
            return []
    
    async def get_detailed_products(self, products: List[AsdaProduct]) -> List[AsdaProduct]:
        """
        Get detailed information for a list of products by visiting each product page.
        
        Args:
            products: List of AsdaProduct objects with basic information
            
        Returns:
            List of AsdaProduct objects with detailed information
        """
        try:
            self.logger.info(f"Getting detailed information for {len(products)} products")
            
            detailed_products = []
            
            for product in products:
                if product.url:
                    try:
                        # Navigate to product page
                        await self.asda_tool.browser_operations.navigate(
                            url=product.url,
                            wait_until="networkidle"
                        )
                        
                        # Extract detailed information
                        detailed_product = await self._extract_product_details(product)
                        if detailed_product:
                            detailed_products.append(detailed_product)
                        
                        # Apply anti-bot delays
                        await self.asda_tool._apply_vendor_specific_delays('extraction')
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to get details for product {product.product_id}: {e}")
                        detailed_products.append(product)  # Keep original product
            
            # Log detailed extraction event
            self._log_memory_event("asda_detailed_extraction", details={
                "total_products": len(products),
                "successful_extractions": len(detailed_products)
            })
            
            return detailed_products
            
        except Exception as e:
            self.logger.error(f"Failed to get detailed products: {e}")
            return products
    
    def _convert_to_asda_product(self, product_dict: Dict[str, Any]) -> Optional[AsdaProduct]:
        """
        Convert raw product dictionary to AsdaProduct object.
        
        Args:
            product_dict: Raw product data dictionary
            
        Returns:
            AsdaProduct object or None if conversion fails
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
            
            # Extract Asda-specific fields
            specifications = product_dict.get('specifications', {})
            availability = product_dict.get('availability', '')
            stock_status = product_dict.get('stock_status', '')
            unit_price = product_dict.get('unit_price', '')
            offer_type = product_dict.get('offer_type', '')
            asda_price = product_dict.get('asda_price', '')
            nutrition_info = product_dict.get('nutrition_info', {})
            allergens = product_dict.get('allergens', [])
            dietary_info = product_dict.get('dietary_info', [])
            product_code = product_dict.get('product_code', '')
            
            return AsdaProduct(
                vendor="Asda",
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
                asda_price=asda_price,
                nutrition_info=nutrition_info,
                allergens=allergens,
                dietary_info=dietary_info,
                product_code=product_code,
                scraped_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to convert product dict to AsdaProduct: {e}")
            return None
    
    async def _extract_product_details(self, product: AsdaProduct) -> Optional[AsdaProduct]:
        """
        Extract detailed information from a product page.
        
        Args:
            product: AsdaProduct object with basic information
            
        Returns:
            Updated AsdaProduct object with detailed information
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
                    specs_element = await self.asda_tool.browser_operations.find_element(selector)
                    if specs_element:
                        specs_text = await self.asda_tool.browser_operations.get_text(selector)
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
                    nutrition_element = await self.asda_tool.browser_operations.find_element(selector)
                    if nutrition_element:
                        nutrition_text = await self.asda_tool.browser_operations.get_text(selector)
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
                    allergen_element = await self.asda_tool.browser_operations.find_element(selector)
                    if allergen_element:
                        allergen_text = await self.asda_tool.browser_operations.get_text(selector)
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
                    dietary_element = await self.asda_tool.browser_operations.find_element(selector)
                    if dietary_element:
                        dietary_text = await self.asda_tool.browser_operations.get_text(selector)
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
                    code_element = await self.asda_tool.browser_operations.find_element(selector)
                    if code_element:
                        product.product_code = await self.asda_tool.browser_operations.get_text(selector)
                        break
                except:
                    continue
            
            return product
            
        except Exception as e:
            self.logger.error(f"Failed to extract product details: {e}")
            return product
    
    def create_crewai_agent(self) -> Agent:
        """
        Create a CrewAI agent specifically for Asda scraping.
        
        Returns:
            CrewAI Agent configured for Asda scraping
        """
        return Agent(
            role="Asda Product Scraper",
            goal="Extract comprehensive product data from Asda's grocery platform with high accuracy",
            backstory="""You are a specialized Asda web scraper with deep expertise in Asda's website structure, 
            navigation patterns, and product data extraction. You excel at handling Asda's breadcrumb navigation, 
            faceted filtering, and anti-bot measures. You ensure data quality and completeness while 
            maintaining respectful scraping practices. You have extensive experience with Asda's product categories, 
            pricing structures, and product specifications.""",
            verbose=True,
            allow_delegation=True,
            tools=[self.asda_tool]
        )


def create_asda_agent(browser_manager: BrowserbaseManager,
                     storage_manager: Optional[StorageManager] = None,
                     anti_bot_config: Optional[Dict[str, Any]] = None) -> AsdaAgent:
    """
    Factory function to create an AsdaAgent instance.
    
    Args:
        browser_manager: BrowserbaseManager for session management
        storage_manager: Optional StorageManager for data persistence
        anti_bot_config: Optional anti-bot configuration
        
    Returns:
        Configured AsdaAgent instance
    """
    return AsdaAgent(
        browser_manager=browser_manager,
        storage_manager=storage_manager,
        anti_bot_config=anti_bot_config
    ) 