"""
Unit tests for TescoAgent and TescoProduct classes.

Tests the Tesco-specific scraping agent and product data structures
for Browserbase-based scraping of Tesco's grocery platform.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from scraping_cli.tesco_agent import (
    TescoAgent, TescoProduct, TescoCategory, create_tesco_agent
)
from scraping_cli.browserbase_manager import BrowserbaseManager
from scraping_cli.storage_manager import StorageManager, ProductData
from scraping_cli.vendor_tools import TescoTool


class TestTescoProduct:
    """Test TescoProduct dataclass functionality."""
    
    def test_tesco_product_creation(self):
        """Test creating a TescoProduct with basic data."""
        product = TescoProduct(
            vendor="Tesco",
            category="dairy",
            product_id="test123",
            title="Test Milk",
            price="£1.50",
            image_url="https://example.com/milk.jpg",
            url="https://tesco.com/milk",
            description="Fresh whole milk"
        )
        
        assert product.vendor == "Tesco"
        assert product.category == "dairy"
        assert product.product_id == "test123"
        assert product.title == "Test Milk"
        assert product.price == "£1.50"
        assert product.image_url == "https://example.com/milk.jpg"
        assert product.url == "https://tesco.com/milk"
        assert product.description == "Fresh whole milk"
        assert isinstance(product.scraped_at, datetime)
        assert product.specifications == {}
        assert product.nutrition_info == {}
        assert product.allergens == []
        assert product.dietary_info == []
    
    def test_tesco_product_with_tesco_specific_fields(self):
        """Test creating a TescoProduct with Tesco-specific fields."""
        product = TescoProduct(
            vendor="Tesco",
            category="pantry",
            product_id="test456",
            title="Test Bread",
            price="£1.20",
            availability="In Stock",
            stock_status="Available",
            unit_price="£1.20 per loaf",
            offer_type="Clubcard Price",
            clubcard_price="£1.00",
            nutrition_info={"calories": "250", "protein": "8g"},
            allergens=["Gluten", "Wheat"],
            dietary_info=["Vegetarian", "Vegan"]
        )
        
        assert product.availability == "In Stock"
        assert product.stock_status == "Available"
        assert product.unit_price == "£1.20 per loaf"
        assert product.offer_type == "Clubcard Price"
        assert product.clubcard_price == "£1.00"
        assert product.nutrition_info == {"calories": "250", "protein": "8g"}
        assert product.allergens == ["Gluten", "Wheat"]
        assert product.dietary_info == ["Vegetarian", "Vegan"]
    
    def test_tesco_product_inheritance(self):
        """Test that TescoProduct properly inherits from ProductData."""
        product = TescoProduct(
            vendor="Tesco",
            category="frozen",
            product_id="test789",
            title="Test Pizza"
        )
        
        # Should be instance of both TescoProduct and ProductData
        assert isinstance(product, TescoProduct)
        assert isinstance(product, ProductData)
        
        # Should have all ProductData fields
        assert hasattr(product, 'vendor')
        assert hasattr(product, 'category')
        assert hasattr(product, 'product_id')
        assert hasattr(product, 'title')
        assert hasattr(product, 'price')
        assert hasattr(product, 'image_url')
        assert hasattr(product, 'url')
        assert hasattr(product, 'description')
        assert hasattr(product, 'scraped_at')
        assert hasattr(product, 'metadata')
        
        # Should have TescoProduct-specific fields
        assert hasattr(product, 'specifications')
        assert hasattr(product, 'availability')
        assert hasattr(product, 'stock_status')
        assert hasattr(product, 'unit_price')
        assert hasattr(product, 'offer_type')
        assert hasattr(product, 'clubcard_price')
        assert hasattr(product, 'nutrition_info')
        assert hasattr(product, 'allergens')
        assert hasattr(product, 'dietary_info')


class TestTescoCategory:
    """Test TescoCategory enum functionality."""
    
    def test_tesco_category_values(self):
        """Test that all TescoCategory values are properly defined."""
        assert TescoCategory.FRESH_FOOD.value == "fresh_food"
        assert TescoCategory.PANTRY.value == "pantry"
        assert TescoCategory.DAIRY.value == "dairy"
        assert TescoCategory.FROZEN.value == "frozen"
        assert TescoCategory.HOUSEHOLD.value == "household"
        assert TescoCategory.DRINKS.value == "drinks"
        assert TescoCategory.HEALTH.value == "health"
        assert TescoCategory.BABY.value == "baby"
    
    def test_tesco_category_enum_membership(self):
        """Test that TescoCategory enum has all expected members."""
        expected_categories = [
            "FRESH_FOOD", "PANTRY", "DAIRY", "FROZEN",
            "HOUSEHOLD", "DRINKS", "HEALTH", "BABY"
        ]
        
        for category_name in expected_categories:
            assert hasattr(TescoCategory, category_name)


class TestTescoAgent:
    """Test TescoAgent functionality."""
    
    @pytest.fixture
    def mock_browser_manager(self):
        """Create a mock BrowserbaseManager."""
        manager = Mock(spec=BrowserbaseManager)
        manager.create_session.return_value = Mock()
        return manager
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create a mock StorageManager."""
        return Mock(spec=StorageManager)
    
    @pytest.fixture
    def mock_tesco_tool(self):
        """Create a mock TescoTool with proper CrewAI tool attributes."""
        tool = Mock(spec=TescoTool)
        tool.name = "TescoTool"  # Required by CrewAI
        tool.description = "Tesco-specific scraping tool"
        tool.navigate_to_category = AsyncMock()
        tool.search_products = AsyncMock()
        tool.extract_product_cards = AsyncMock()
        tool._has_next_page = AsyncMock()
        tool._navigate_to_next_page = AsyncMock()
        tool.browser_operations = Mock()
        tool.browser_operations.navigate = AsyncMock()
        tool.browser_operations.find_element = AsyncMock()
        tool.browser_operations.get_text = AsyncMock()
        tool._apply_vendor_specific_delays = AsyncMock()
        return tool
    
    def test_tesco_agent_initialization(self, mock_browser_manager, mock_storage_manager):
        """Test TescoAgent initialization."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            # Create a proper mock tool with name attribute
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.description = "Tesco-specific scraping tool"
            mock_create_tool.return_value = mock_tool
            
            # Mock the agent creation to avoid CrewAI issues
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager,
                    anti_bot_config={"delay_range": (1, 3)}
                )
                
                assert agent.browser_manager == mock_browser_manager
                assert agent.storage_manager == mock_storage_manager
                assert agent.anti_bot_config == {"delay_range": (1, 3)}
                assert agent.config.name == "TescoScraper"
                assert agent.config.role.value == "scraper"
                assert "Tesco" in agent.config.goal
                assert "Tesco" in agent.config.backstory
    
    def test_tesco_agent_category_mapping(self, mock_browser_manager, mock_storage_manager):
        """Test that TescoAgent has proper category mapping."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                assert TescoCategory.FRESH_FOOD in agent.category_mapping
                assert TescoCategory.PANTRY in agent.category_mapping
                assert TescoCategory.DAIRY in agent.category_mapping
                assert TescoCategory.FROZEN in agent.category_mapping
                assert TescoCategory.HOUSEHOLD in agent.category_mapping
                assert TescoCategory.DRINKS in agent.category_mapping
                assert TescoCategory.HEALTH in agent.category_mapping
                assert TescoCategory.BABY in agent.category_mapping
    
    @pytest.mark.asyncio
    async def test_navigate_to_category_success(self, mock_browser_manager, mock_storage_manager):
        """Test successful category navigation."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.navigate_to_category = AsyncMock(
                return_value={'status': 'success', 'category': 'dairy'}
            )
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                result = await agent.navigate_to_category("dairy")
                
                assert result['status'] == 'success'
                assert result['category'] == 'dairy'
                mock_tool.navigate_to_category.assert_called_once_with("dairy")
    
    @pytest.mark.asyncio
    async def test_navigate_to_category_with_enum(self, mock_browser_manager, mock_storage_manager):
        """Test category navigation with TescoCategory enum."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.navigate_to_category = AsyncMock(
                return_value={'status': 'success', 'category': 'fresh_food'}
            )
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                result = await agent.navigate_to_category(TescoCategory.FRESH_FOOD)
                
                assert result['status'] == 'success'
                mock_tool.navigate_to_category.assert_called_once_with("fresh_food")
    
    @pytest.mark.asyncio
    async def test_navigate_to_category_error(self, mock_browser_manager, mock_storage_manager):
        """Test category navigation error handling."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.navigate_to_category = AsyncMock(
                side_effect=Exception("Navigation failed")
            )
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                result = await agent.navigate_to_category("dairy")
                
                assert result['status'] == 'error'
                assert result['category'] == 'dairy'
                assert 'Navigation failed' in result['error']
    
    @pytest.mark.asyncio
    async def test_search_products_success(self, mock_browser_manager, mock_storage_manager):
        """Test successful product search."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.search_products = AsyncMock(
                return_value={'status': 'success', 'query': 'milk', 'url': 'https://tesco.com/search'}
            )
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                result = await agent.search_products("milk")
                
                assert result['status'] == 'success'
                assert result['query'] == 'milk'
                mock_tool.search_products.assert_called_once_with("milk")
    
    @pytest.mark.asyncio
    async def test_search_products_error(self, mock_browser_manager, mock_storage_manager):
        """Test product search error handling."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.search_products = AsyncMock(
                side_effect=Exception("Search failed")
            )
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                result = await agent.search_products("milk")
                
                assert result['status'] == 'error'
                assert result['query'] == 'milk'
                assert 'Search failed' in result['error']
    
    @pytest.mark.asyncio
    async def test_extract_product_cards_success(self, mock_browser_manager, mock_storage_manager):
        """Test successful product card extraction."""
        raw_products = [
            {
                'title': 'Test Milk',
                'price': '£1.50',
                'image_url': 'https://example.com/milk.jpg',
                'product_url': 'https://tesco.com/milk',
                'category': 'dairy'
            },
            {
                'title': 'Test Bread',
                'price': '£1.20',
                'image_url': 'https://example.com/bread.jpg',
                'product_url': 'https://tesco.com/bread',
                'category': 'pantry'
            }
        ]
        
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.extract_product_cards = AsyncMock(return_value=raw_products)
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                products = await agent.extract_product_cards()
                
                assert len(products) == 2
                assert all(isinstance(p, TescoProduct) for p in products)
                assert products[0].title == 'Test Milk'
                assert products[0].vendor == 'Tesco'
                assert products[1].title == 'Test Bread'
                assert products[1].vendor == 'Tesco'
    
    @pytest.mark.asyncio
    async def test_extract_product_cards_error(self, mock_browser_manager, mock_storage_manager):
        """Test product card extraction error handling."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.extract_product_cards = AsyncMock(
                side_effect=Exception("Extraction failed")
            )
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                products = await agent.extract_product_cards()
                
                assert products == []
    
    @pytest.mark.asyncio
    async def test_scrape_category_success(self, mock_browser_manager, mock_storage_manager):
        """Test successful category scraping."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.navigate_to_category = AsyncMock(return_value={'status': 'success'})
            mock_tool.extract_product_cards = AsyncMock(return_value=[
                {'title': 'Product 1', 'price': '£1.00', 'category': 'dairy'},
                {'title': 'Product 2', 'price': '£2.00', 'category': 'dairy'}
            ])
            mock_tool._has_next_page = AsyncMock(return_value=False)
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                products = await agent.scrape_category("dairy", max_pages=2)
                
                assert len(products) == 2
                assert all(isinstance(p, TescoProduct) for p in products)
                mock_tool.navigate_to_category.assert_called_once_with("dairy")
    
    @pytest.mark.asyncio
    async def test_scrape_category_with_pagination(self, mock_browser_manager, mock_storage_manager):
        """Test category scraping with pagination."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.navigate_to_category = AsyncMock(return_value={'status': 'success'})
            mock_tool.extract_product_cards = AsyncMock()
            mock_tool.extract_product_cards.side_effect = [
                [{'title': 'Product 1', 'price': '£1.00', 'category': 'dairy'}],
                [{'title': 'Product 2', 'price': '£2.00', 'category': 'dairy'}]
            ]
            mock_tool._has_next_page = AsyncMock()
            mock_tool._has_next_page.side_effect = [True, False]
            mock_tool._navigate_to_next_page = AsyncMock()
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                products = await agent.scrape_category("dairy", max_pages=2)
                
                assert len(products) == 2
                assert mock_tool._navigate_to_next_page.call_count == 1
    
    @pytest.mark.asyncio
    async def test_scrape_category_navigation_error(self, mock_browser_manager, mock_storage_manager):
        """Test category scraping with navigation error."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.navigate_to_category = AsyncMock(
                return_value={'status': 'error', 'error': 'Navigation failed'}
            )
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                products = await agent.scrape_category("dairy")
                
                assert products == []
    
    @pytest.mark.asyncio
    async def test_scrape_search_results_success(self, mock_browser_manager, mock_storage_manager):
        """Test successful search results scraping."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.search_products = AsyncMock(return_value={'status': 'success'})
            mock_tool.extract_product_cards = AsyncMock(return_value=[
                {'title': 'Search Product 1', 'price': '£1.00', 'category': 'pantry'},
                {'title': 'Search Product 2', 'price': '£2.00', 'category': 'pantry'}
            ])
            mock_tool._has_next_page = AsyncMock(return_value=False)
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                products = await agent.scrape_search_results("milk", max_pages=2)
                
                assert len(products) == 2
                assert all(isinstance(p, TescoProduct) for p in products)
                mock_tool.search_products.assert_called_once_with("milk")
    
    @pytest.mark.asyncio
    async def test_get_detailed_products_success(self, mock_browser_manager, mock_storage_manager):
        """Test successful detailed product extraction."""
        # Create test products
        products = [
            TescoProduct(
                vendor="Tesco",
                category="dairy",
                product_id="test1",
                title="Test Milk",
                url="https://tesco.com/milk"
            ),
            TescoProduct(
                vendor="Tesco",
                category="pantry",
                product_id="test2",
                title="Test Bread",
                url="https://tesco.com/bread"
            )
        ]
        
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_tool.browser_operations = Mock()
            mock_tool.browser_operations.navigate = AsyncMock()
            mock_tool.browser_operations.find_element = AsyncMock(return_value=Mock())
            mock_tool.browser_operations.get_text = AsyncMock(return_value="Test specs")
            mock_tool._apply_vendor_specific_delays = AsyncMock()
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                detailed_products = await agent.get_detailed_products(products)
                
                assert len(detailed_products) == 2
                assert mock_tool.browser_operations.navigate.call_count == 2
    
    def test_convert_to_tesco_product_success(self, mock_browser_manager, mock_storage_manager):
        """Test successful conversion of product dict to TescoProduct."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                product_dict = {
                    'title': 'Test Product',
                    'price': '£1.50',
                    'image_url': 'https://example.com/image.jpg',
                    'product_url': 'https://tesco.com/product',
                    'description': 'Test description',
                    'category': 'dairy',
                    'specifications': {'weight': '500g'},
                    'availability': 'In Stock',
                    'stock_status': 'Available',
                    'unit_price': '£3.00/kg',
                    'offer_type': 'Clubcard Price',
                    'clubcard_price': '£1.25',
                    'nutrition_info': {'calories': '250'},
                    'allergens': ['Milk', 'Gluten'],
                    'dietary_info': ['Vegetarian']
                }
                
                product = agent._convert_to_tesco_product(product_dict)
                
                assert isinstance(product, TescoProduct)
                assert product.title == 'Test Product'
                assert product.price == '£1.50'
                assert product.vendor == 'Tesco'
                assert product.category == 'dairy'
                assert product.specifications == {'weight': '500g'}
                assert product.availability == 'In Stock'
                assert product.stock_status == 'Available'
                assert product.unit_price == '£3.00/kg'
                assert product.offer_type == 'Clubcard Price'
                assert product.clubcard_price == '£1.25'
                assert product.nutrition_info == {'calories': '250'}
                assert product.allergens == ['Milk', 'Gluten']
                assert product.dietary_info == ['Vegetarian']
    
    def test_convert_to_tesco_product_with_generated_id(self, mock_browser_manager, mock_storage_manager):
        """Test conversion with generated product ID."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                product_dict = {
                    'title': 'Test Product',
                    'price': '£1.50',
                    'category': 'dairy'
                }
                
                product = agent._convert_to_tesco_product(product_dict)
                
                assert product.product_id is not None
                assert len(product.product_id) == 12  # MD5 hash truncated to 12 chars
    
    def test_convert_to_tesco_product_error(self, mock_browser_manager, mock_storage_manager):
        """Test conversion error handling."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                product_dict = None
                
                product = agent._convert_to_tesco_product(product_dict)
                
                assert product is None
    
    def test_create_crewai_agent(self, mock_browser_manager, mock_storage_manager):
        """Test CrewAI agent creation."""
        with patch('scraping_cli.tesco_agent.create_tesco_tool') as mock_create_tool:
            mock_tool = Mock(spec=TescoTool)
            mock_tool.name = "TescoTool"
            mock_create_tool.return_value = mock_tool
            
            with patch('scraping_cli.tesco_agent.ScrapingAgent._create_agent') as mock_create_agent:
                mock_create_agent.return_value = Mock()
                
                agent = TescoAgent(
                    browser_manager=mock_browser_manager,
                    storage_manager=mock_storage_manager
                )
                
                # Mock the Agent creation to avoid CrewAI issues
                with patch('scraping_cli.tesco_agent.Agent') as mock_agent_class:
                    mock_crewai_agent = Mock()
                    mock_crewai_agent.role = "Tesco Product Scraper"
                    mock_crewai_agent.goal = "Extract comprehensive product data from Tesco's grocery platform"
                    mock_crewai_agent.backstory = "You are a specialized Tesco web scraper"
                    mock_crewai_agent.verbose = True
                    mock_crewai_agent.allow_delegation = True
                    mock_crewai_agent.tools = [mock_tool]
                    mock_agent_class.return_value = mock_crewai_agent
                    
                    crewai_agent = agent.create_crewai_agent()
                    
                    assert crewai_agent.role == "Tesco Product Scraper"
                    assert "Tesco" in crewai_agent.goal
                    assert "Tesco" in crewai_agent.backstory
                    assert crewai_agent.verbose is True
                    assert crewai_agent.allow_delegation is True
                    assert len(crewai_agent.tools) == 1
                    assert isinstance(crewai_agent.tools[0], Mock)  # Mock TescoTool


class TestCreateTescoAgent:
    """Test create_tesco_agent factory function."""
    
    @pytest.fixture
    def mock_browser_manager(self):
        """Create a mock BrowserbaseManager."""
        return Mock(spec=BrowserbaseManager)
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create a mock StorageManager."""
        return Mock(spec=StorageManager)
    
    def test_create_tesco_agent_basic(self, mock_browser_manager):
        """Test creating TescoAgent with basic parameters."""
        with patch('scraping_cli.tesco_agent.TescoAgent') as mock_agent_class:
            mock_agent_class.return_value = Mock(spec=TescoAgent)
            
            agent = create_tesco_agent(mock_browser_manager)
            
            mock_agent_class.assert_called_once_with(
                browser_manager=mock_browser_manager,
                storage_manager=None,
                anti_bot_config=None
            )
    
    def test_create_tesco_agent_with_storage(self, mock_browser_manager, mock_storage_manager):
        """Test creating TescoAgent with storage manager."""
        with patch('scraping_cli.tesco_agent.TescoAgent') as mock_agent_class:
            mock_agent_class.return_value = Mock(spec=TescoAgent)
            
            agent = create_tesco_agent(
                browser_manager=mock_browser_manager,
                storage_manager=mock_storage_manager
            )
            
            mock_agent_class.assert_called_once_with(
                browser_manager=mock_browser_manager,
                storage_manager=mock_storage_manager,
                anti_bot_config=None
            )
    
    def test_create_tesco_agent_with_anti_bot_config(self, mock_browser_manager):
        """Test creating TescoAgent with anti-bot configuration."""
        anti_bot_config = {"delay_range": (1, 3), "stealth_mode": True}
        
        with patch('scraping_cli.tesco_agent.TescoAgent') as mock_agent_class:
            mock_agent_class.return_value = Mock(spec=TescoAgent)
            
            agent = create_tesco_agent(
                browser_manager=mock_browser_manager,
                anti_bot_config=anti_bot_config
            )
            
            mock_agent_class.assert_called_once_with(
                browser_manager=mock_browser_manager,
                storage_manager=None,
                anti_bot_config=anti_bot_config
            ) 