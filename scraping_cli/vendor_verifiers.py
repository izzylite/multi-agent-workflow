"""
Vendor-Specific Verification Modules

Provides dedicated verification modules for Tesco, Asda, and Costco,
each encapsulating selectors, interaction logic, and test cases unique
to the respective vendor's web interface.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from .vendor_verifier import (
    VendorToolsVerifier, VerificationResult, VerificationStatus,
    VerificationLevel, create_vendor_verifier
)
from .vendor_tools import TescoTool, AsdaTool, CostcoTool
from .browserbase_manager import BrowserbaseManager
from .browser_operations import BrowserOperations
from .exceptions import (
    ElementNotFoundError, ElementInteractionError, NavigationError,
    TimeoutError, SessionConnectionError
)


@dataclass
class VendorTestScenario:
    """Test scenario for vendor verification."""
    name: str
    description: str
    test_function: Callable
    timeout: int = 30
    level: VerificationLevel = VerificationLevel.BASIC
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class BaseVendorVerifier:
    """Base class for vendor-specific verifiers."""
    
    def __init__(self, verifier: VendorToolsVerifier, vendor_name: str):
        """
        Initialize base vendor verifier.
        
        Args:
            verifier: VendorToolsVerifier instance
            vendor_name: Name of the vendor
        """
        self.verifier = verifier
        self.vendor_name = vendor_name
        self.logger = logging.getLogger(f"{__name__}.{vendor_name}Verifier")
        self.test_scenarios: List[VendorTestScenario] = []
        
    async def register_test_scenarios(self):
        """Register test scenarios for this vendor. Override in subclasses."""
        pass
    
    async def run_basic_verification(self) -> List[VerificationResult]:
        """Run basic verification tests."""
        basic_tests = [scenario for scenario in self.test_scenarios 
                      if scenario.level == VerificationLevel.BASIC]
        return await self._run_scenarios(basic_tests)
    
    async def run_comprehensive_verification(self) -> List[VerificationResult]:
        """Run comprehensive verification tests."""
        comprehensive_tests = [scenario for scenario in self.test_scenarios 
                             if scenario.level in [VerificationLevel.BASIC, VerificationLevel.COMPREHENSIVE]]
        return await self._run_scenarios(comprehensive_tests)
    
    async def run_stress_verification(self) -> List[VerificationResult]:
        """Run stress verification tests."""
        stress_tests = [scenario for scenario in self.test_scenarios 
                       if scenario.level == VerificationLevel.STRESS]
        return await self._run_scenarios(stress_tests)
    
    async def _run_scenarios(self, scenarios: List[VendorTestScenario]) -> List[VerificationResult]:
        """Run a list of test scenarios."""
        results = []
        
        for scenario in scenarios:
            try:
                result = await self.verifier.run_verification_test(
                    test_name=scenario.name,
                    test_func=scenario.test_function,
                    timeout=scenario.timeout
                )
                results.append(result)
                
                # Log result
                status_emoji = "✅" if result.status == VerificationStatus.PASSED else "❌"
                self.logger.info(f"{status_emoji} {scenario.name}: {result.status.value} ({result.duration:.2f}s)")
                
            except Exception as e:
                self.logger.error(f"Failed to run scenario {scenario.name}: {e}")
                result = VerificationResult(
                    test_name=scenario.name,
                    vendor=self.vendor_name,
                    status=VerificationStatus.ERROR,
                    duration=0.0,
                    error_message=str(e)
                )
                results.append(result)
        
        return results


class TescoVerifier(BaseVendorVerifier):
    """Tesco-specific verification module."""
    
    def __init__(self, verifier: VendorToolsVerifier):
        super().__init__(verifier, "Tesco")
        self.tesco_tool: Optional[TescoTool] = None
    
    async def register_test_scenarios(self):
        """Register Tesco-specific test scenarios."""
        self.test_scenarios = [
            VendorTestScenario(
                name="tesco_homepage_navigation",
                description="Navigate to Tesco homepage and verify basic elements",
                test_function=self._test_homepage_navigation,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="tesco_category_navigation",
                description="Navigate to a product category and verify category page",
                test_function=self._test_category_navigation,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="tesco_product_search",
                description="Search for products and verify search results",
                test_function=self._test_product_search,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="tesco_product_extraction",
                description="Extract product cards from category/search pages",
                test_function=self._test_product_extraction,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="tesco_challenge_handling",
                description="Test handling of common challenges (cookies, login prompts)",
                test_function=self._test_challenge_handling,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="tesco_pagination",
                description="Test pagination functionality",
                test_function=self._test_pagination,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="tesco_product_details",
                description="Navigate to product detail pages and extract information",
                test_function=self._test_product_details,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="tesco_stress_test",
                description="Run multiple operations in sequence to test stability",
                test_function=self._test_stress_operations,
                level=VerificationLevel.STRESS,
                timeout=60
            )
        ]
    
    async def _test_homepage_navigation(self) -> None:
        """Test navigation to Tesco homepage."""
        if not self.tesco_tool:
            raise ValueError("Tesco tool not initialized")
        
        # Navigate to Tesco homepage
        result = await self.tesco_tool.navigate_to_homepage()
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to Tesco homepage: {result.get('error')}")
        
        # Verify basic elements are present
        await self._verify_homepage_elements()
    
    async def _test_category_navigation(self) -> None:
        """Test navigation to a product category."""
        if not self.tesco_tool:
            raise ValueError("Tesco tool not initialized")
        
        # Navigate to a test category (e.g., "dairy")
        result = await self.tesco_tool.navigate_to_category("dairy")
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to category: {result.get('error')}")
        
        # Verify category page elements
        await self._verify_category_page_elements()
    
    async def _test_product_search(self) -> None:
        """Test product search functionality."""
        if not self.tesco_tool:
            raise ValueError("Tesco tool not initialized")
        
        # Search for a test product
        result = await self.tesco_tool.search_products("milk")
        if not result.get('success'):
            raise NavigationError(f"Failed to search products: {result.get('error')}")
        
        # Verify search results
        await self._verify_search_results()
    
    async def _test_product_extraction(self) -> None:
        """Test product card extraction."""
        if not self.tesco_tool:
            raise ValueError("Tesco tool not initialized")
        
        # Extract product cards
        products = await self.tesco_tool.extract_product_cards()
        
        if not products:
            raise ElementNotFoundError("No product cards found")
        
        # Verify product data structure
        for product in products[:5]:  # Check first 5 products
            required_fields = ['title', 'price', 'url']
            for field in required_fields:
                if field not in product or not product[field]:
                    raise ValueError(f"Product missing required field: {field}")
    
    async def _test_challenge_handling(self) -> None:
        """Test handling of common challenges."""
        if not self.tesco_tool:
            raise ValueError("Tesco tool not initialized")
        
        # Test challenge handling
        result = await self.tesco_tool.handle_tesco_challenges()
        if not result.get('success'):
            raise ElementInteractionError(f"Failed to handle challenges: {result.get('error')}")
    
    async def _test_pagination(self) -> None:
        """Test pagination functionality."""
        if not self.tesco_tool:
            raise ValueError("Tesco tool not initialized")
        
        # Check if pagination exists
        has_next = await self.tesco_tool._has_next_page()
        
        if has_next:
            # Test navigation to next page
            success = await self.tesco_tool._navigate_to_next_page()
            if not success:
                raise NavigationError("Failed to navigate to next page")
    
    async def _test_product_details(self) -> None:
        """Test product detail page navigation."""
        if not self.tesco_tool:
            raise ValueError("Tesco tool not initialized")
        
        # Get a product URL from search results
        products = await self.tesco_tool.extract_product_cards()
        if not products:
            raise ElementNotFoundError("No products found for detail testing")
        
        # Navigate to first product detail page
        product_url = products[0].get('url')
        if not product_url:
            raise ValueError("Product URL not found")
        
        result = await self.tesco_tool.navigate_to_product_detail(product_url)
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to product detail: {result.get('error')}")
    
    async def _test_stress_operations(self) -> None:
        """Test multiple operations in sequence."""
        if not self.tesco_tool:
            raise ValueError("Tesco tool not initialized")
        
        # Run multiple operations to test stability
        operations = [
            self._test_homepage_navigation,
            self._test_category_navigation,
            self._test_product_search,
            self._test_product_extraction
        ]
        
        for operation in operations:
            await operation()
            await asyncio.sleep(1)  # Brief pause between operations
    
    async def _verify_homepage_elements(self) -> None:
        """Verify basic homepage elements are present."""
        # This would check for common homepage elements
        # Implementation depends on specific Tesco homepage structure
        pass
    
    async def _verify_category_page_elements(self) -> None:
        """Verify category page elements are present."""
        # This would check for category-specific elements
        pass
    
    async def _verify_search_results(self) -> None:
        """Verify search results are displayed correctly."""
        # This would check for search result elements
        pass


class AsdaVerifier(BaseVendorVerifier):
    """Asda-specific verification module."""
    
    def __init__(self, verifier: VendorToolsVerifier):
        super().__init__(verifier, "Asda")
        self.asda_tool: Optional[AsdaTool] = None
    
    async def register_test_scenarios(self):
        """Register Asda-specific test scenarios."""
        self.test_scenarios = [
            VendorTestScenario(
                name="asda_homepage_navigation",
                description="Navigate to Asda homepage and verify basic elements",
                test_function=self._test_homepage_navigation,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="asda_category_navigation",
                description="Navigate to a product category and verify category page",
                test_function=self._test_category_navigation,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="asda_product_search",
                description="Search for products and verify search results",
                test_function=self._test_product_search,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="asda_product_extraction",
                description="Extract product cards from category/search pages",
                test_function=self._test_product_extraction,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="asda_challenge_handling",
                description="Test handling of common challenges (cookies, login prompts)",
                test_function=self._test_challenge_handling,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="asda_pagination",
                description="Test pagination functionality",
                test_function=self._test_pagination,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="asda_product_details",
                description="Navigate to product detail pages and extract information",
                test_function=self._test_product_details,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="asda_stress_test",
                description="Run multiple operations in sequence to test stability",
                test_function=self._test_stress_operations,
                level=VerificationLevel.STRESS,
                timeout=60
            )
        ]
    
    async def _test_homepage_navigation(self) -> None:
        """Test navigation to Asda homepage."""
        if not self.asda_tool:
            raise ValueError("Asda tool not initialized")
        
        # Navigate to Asda homepage
        result = await self.asda_tool.navigate_to_homepage()
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to Asda homepage: {result.get('error')}")
        
        # Verify basic elements are present
        await self._verify_homepage_elements()
    
    async def _test_category_navigation(self) -> None:
        """Test navigation to a product category."""
        if not self.asda_tool:
            raise ValueError("Asda tool not initialized")
        
        # Navigate to a test category (e.g., "dairy")
        result = await self.asda_tool.navigate_to_category("dairy")
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to category: {result.get('error')}")
        
        # Verify category page elements
        await self._verify_category_page_elements()
    
    async def _test_product_search(self) -> None:
        """Test product search functionality."""
        if not self.asda_tool:
            raise ValueError("Asda tool not initialized")
        
        # Search for a test product
        result = await self.asda_tool.search_products("milk")
        if not result.get('success'):
            raise NavigationError(f"Failed to search products: {result.get('error')}")
        
        # Verify search results
        await self._verify_search_results()
    
    async def _test_product_extraction(self) -> None:
        """Test product card extraction."""
        if not self.asda_tool:
            raise ValueError("Asda tool not initialized")
        
        # Extract product cards
        products = await self.asda_tool.extract_product_cards()
        
        if not products:
            raise ElementNotFoundError("No product cards found")
        
        # Verify product data structure
        for product in products[:5]:  # Check first 5 products
            required_fields = ['title', 'price', 'url']
            for field in required_fields:
                if field not in product or not product[field]:
                    raise ValueError(f"Product missing required field: {field}")
    
    async def _test_challenge_handling(self) -> None:
        """Test handling of common challenges."""
        if not self.asda_tool:
            raise ValueError("Asda tool not initialized")
        
        # Test challenge handling
        result = await self.asda_tool.handle_asda_challenges()
        if not result.get('success'):
            raise ElementInteractionError(f"Failed to handle challenges: {result.get('error')}")
    
    async def _test_pagination(self) -> None:
        """Test pagination functionality."""
        if not self.asda_tool:
            raise ValueError("Asda tool not initialized")
        
        # Check if pagination exists
        has_next = await self.asda_tool._has_next_page()
        
        if has_next:
            # Test navigation to next page
            success = await self.asda_tool._navigate_to_next_page()
            if not success:
                raise NavigationError("Failed to navigate to next page")
    
    async def _test_product_details(self) -> None:
        """Test product detail page navigation."""
        if not self.asda_tool:
            raise ValueError("Asda tool not initialized")
        
        # Get a product URL from search results
        products = await self.asda_tool.extract_product_cards()
        if not products:
            raise ElementNotFoundError("No products found for detail testing")
        
        # Navigate to first product detail page
        product_url = products[0].get('url')
        if not product_url:
            raise ValueError("Product URL not found")
        
        result = await self.asda_tool.navigate_to_product_detail(product_url)
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to product detail: {result.get('error')}")
    
    async def _test_stress_operations(self) -> None:
        """Test multiple operations in sequence."""
        if not self.asda_tool:
            raise ValueError("Asda tool not initialized")
        
        # Run multiple operations to test stability
        operations = [
            self._test_homepage_navigation,
            self._test_category_navigation,
            self._test_product_search,
            self._test_product_extraction
        ]
        
        for operation in operations:
            await operation()
            await asyncio.sleep(1)  # Brief pause between operations
    
    async def _verify_homepage_elements(self) -> None:
        """Verify basic homepage elements are present."""
        # This would check for common homepage elements
        # Implementation depends on specific Asda homepage structure
        pass
    
    async def _verify_category_page_elements(self) -> None:
        """Verify category page elements are present."""
        # This would check for category-specific elements
        pass
    
    async def _verify_search_results(self) -> None:
        """Verify search results are displayed correctly."""
        # This would check for search result elements
        pass


class CostcoVerifier(BaseVendorVerifier):
    """Costco-specific verification module."""
    
    def __init__(self, verifier: VendorToolsVerifier):
        super().__init__(verifier, "Costco")
        self.costco_tool: Optional[CostcoTool] = None
    
    async def register_test_scenarios(self):
        """Register Costco-specific test scenarios."""
        self.test_scenarios = [
            VendorTestScenario(
                name="costco_homepage_navigation",
                description="Navigate to Costco homepage and verify basic elements",
                test_function=self._test_homepage_navigation,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="costco_category_navigation",
                description="Navigate to a product category and verify category page",
                test_function=self._test_category_navigation,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="costco_product_search",
                description="Search for products and verify search results",
                test_function=self._test_product_search,
                level=VerificationLevel.BASIC
            ),
            VendorTestScenario(
                name="costco_product_extraction",
                description="Extract product cards from category/search pages",
                test_function=self._test_product_extraction,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="costco_challenge_handling",
                description="Test handling of common challenges (cookies, login prompts)",
                test_function=self._test_challenge_handling,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="costco_pagination",
                description="Test pagination functionality",
                test_function=self._test_pagination,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="costco_product_details",
                description="Navigate to product detail pages and extract information",
                test_function=self._test_product_details,
                level=VerificationLevel.COMPREHENSIVE
            ),
            VendorTestScenario(
                name="costco_stress_test",
                description="Run multiple operations in sequence to test stability",
                test_function=self._test_stress_operations,
                level=VerificationLevel.STRESS,
                timeout=60
            )
        ]
    
    async def _test_homepage_navigation(self) -> None:
        """Test navigation to Costco homepage."""
        if not self.costco_tool:
            raise ValueError("Costco tool not initialized")
        
        # Navigate to Costco homepage
        result = await self.costco_tool.navigate_to_homepage()
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to Costco homepage: {result.get('error')}")
        
        # Verify basic elements are present
        await self._verify_homepage_elements()
    
    async def _test_category_navigation(self) -> None:
        """Test navigation to a product category."""
        if not self.costco_tool:
            raise ValueError("Costco tool not initialized")
        
        # Navigate to a test category (e.g., "dairy")
        result = await self.costco_tool.navigate_to_category("dairy")
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to category: {result.get('error')}")
        
        # Verify category page elements
        await self._verify_category_page_elements()
    
    async def _test_product_search(self) -> None:
        """Test product search functionality."""
        if not self.costco_tool:
            raise ValueError("Costco tool not initialized")
        
        # Search for a test product
        result = await self.costco_tool.search_products("milk")
        if not result.get('success'):
            raise NavigationError(f"Failed to search products: {result.get('error')}")
        
        # Verify search results
        await self._verify_search_results()
    
    async def _test_product_extraction(self) -> None:
        """Test product card extraction."""
        if not self.costco_tool:
            raise ValueError("Costco tool not initialized")
        
        # Extract product cards
        products = await self.costco_tool.extract_product_cards()
        
        if not products:
            raise ElementNotFoundError("No product cards found")
        
        # Verify product data structure
        for product in products[:5]:  # Check first 5 products
            required_fields = ['title', 'price', 'url']
            for field in required_fields:
                if field not in product or not product[field]:
                    raise ValueError(f"Product missing required field: {field}")
    
    async def _test_challenge_handling(self) -> None:
        """Test handling of common challenges."""
        if not self.costco_tool:
            raise ValueError("Costco tool not initialized")
        
        # Test challenge handling
        result = await self.costco_tool.handle_costco_challenges()
        if not result.get('success'):
            raise ElementInteractionError(f"Failed to handle challenges: {result.get('error')}")
    
    async def _test_pagination(self) -> None:
        """Test pagination functionality."""
        if not self.costco_tool:
            raise ValueError("Costco tool not initialized")
        
        # Check if pagination exists
        has_next = await self.costco_tool._has_next_page()
        
        if has_next:
            # Test navigation to next page
            success = await self.costco_tool._navigate_to_next_page()
            if not success:
                raise NavigationError("Failed to navigate to next page")
    
    async def _test_product_details(self) -> None:
        """Test product detail page navigation."""
        if not self.costco_tool:
            raise ValueError("Costco tool not initialized")
        
        # Get a product URL from search results
        products = await self.costco_tool.extract_product_cards()
        if not products:
            raise ElementNotFoundError("No products found for detail testing")
        
        # Navigate to first product detail page
        product_url = products[0].get('url')
        if not product_url:
            raise ValueError("Product URL not found")
        
        result = await self.costco_tool.navigate_to_product_detail(product_url)
        if not result.get('success'):
            raise NavigationError(f"Failed to navigate to product detail: {result.get('error')}")
    
    async def _test_stress_operations(self) -> None:
        """Test multiple operations in sequence."""
        if not self.costco_tool:
            raise ValueError("Costco tool not initialized")
        
        # Run multiple operations to test stability
        operations = [
            self._test_homepage_navigation,
            self._test_category_navigation,
            self._test_product_search,
            self._test_product_extraction
        ]
        
        for operation in operations:
            await operation()
            await asyncio.sleep(1)  # Brief pause between operations
    
    async def _verify_homepage_elements(self) -> None:
        """Verify basic homepage elements are present."""
        # This would check for common homepage elements
        # Implementation depends on specific Costco homepage structure
        pass
    
    async def _verify_category_page_elements(self) -> None:
        """Verify category page elements are present."""
        # This would check for category-specific elements
        pass
    
    async def _verify_search_results(self) -> None:
        """Verify search results are displayed correctly."""
        # This would check for search result elements
        pass


def create_tesco_verifier(verifier: VendorToolsVerifier) -> TescoVerifier:
    """Create a TescoVerifier instance."""
    return TescoVerifier(verifier)


def create_asda_verifier(verifier: VendorToolsVerifier) -> AsdaVerifier:
    """Create an AsdaVerifier instance."""
    return AsdaVerifier(verifier)


def create_costco_verifier(verifier: VendorToolsVerifier) -> CostcoVerifier:
    """Create a CostcoVerifier instance."""
    return CostcoVerifier(verifier)


def create_all_vendor_verifiers(verifier: VendorToolsVerifier) -> Dict[str, BaseVendorVerifier]:
    """Create all vendor verifier instances."""
    return {
        "tesco": create_tesco_verifier(verifier),
        "asda": create_asda_verifier(verifier),
        "costco": create_costco_verifier(verifier)
    } 