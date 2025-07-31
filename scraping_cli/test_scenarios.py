"""
Comprehensive Test Scenarios for Vendor Verification

Provides detailed test scenarios for each vendor (Tesco, Asda, Costco)
covering various aspects of scraping functionality and edge cases.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .vendor_verifier import VendorToolsVerifier, VerificationLevel, VerificationResult
from .vendor_verification_modules import BaseVendorVerifier, VendorTestScenario
from .challenge_handling_verification import ChallengeHandler, ChallengeType


class TestScenarioType(Enum):
    """Types of test scenarios."""
    FUNCTIONAL = "functional"
    EDGE_CASE = "edge_case"
    STRESS = "stress"
    REGRESSION = "regression"
    INTEGRATION = "integration"


@dataclass
class TestScenario:
    """Represents a comprehensive test scenario."""
    name: str
    description: str
    scenario_type: TestScenarioType
    vendor: str
    test_function: Callable
    expected_duration: float = 30.0
    required_level: VerificationLevel = VerificationLevel.BASIC
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class TestScenarioManager:
    """Manages comprehensive test scenarios for vendor verification."""

    def __init__(self, verifier: VendorToolsVerifier):
        self.verifier = verifier
        self.logger = logging.getLogger(__name__)
        self.scenarios: Dict[str, TestScenario] = {}
        self.challenge_handler = ChallengeHandler()

    def register_scenario(self, scenario: TestScenario) -> None:
        """Register a test scenario."""
        self.scenarios[scenario.name] = scenario
        self.logger.debug(f"Registered scenario: {scenario.name}")

    async def run_scenario(self, scenario_name: str, **kwargs) -> VerificationResult:
        """Run a specific test scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        scenario = self.scenarios[scenario_name]
        self.logger.info(f"Running scenario: {scenario.name}")

        try:
            start_time = datetime.now()
            result = await scenario.test_function(self.verifier, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()

            return VerificationResult(
                test_name=scenario.name,
                status=result.status,
                duration=duration,
                details=result.details,
                error_message=result.error_message,
                screenshots=result.screenshots
            )

        except Exception as e:
            self.logger.error(f"Scenario '{scenario.name}' failed: {e}")
            return VerificationResult(
                test_name=scenario.name,
                status="ERROR",
                details=f"Scenario execution failed: {str(e)}",
                error_message=str(e)
            )

    async def run_vendor_scenarios(self, vendor: str, level: VerificationLevel = VerificationLevel.BASIC) -> List[VerificationResult]:
        """Run all scenarios for a specific vendor."""
        vendor_scenarios = [
            s for s in self.scenarios.values()
            if str(s.vendor).lower() == str(vendor).lower() and s.required_level.value <= level.value
        ]

        results = []
        for scenario in vendor_scenarios:
            result = await self.run_scenario(scenario.name)
            results.append(result)

        return results


# Tesco Test Scenarios
async def tesco_homepage_navigation(verifier: VendorToolsVerifier, **kwargs) -> VerificationResult:
    """Test Tesco homepage navigation and basic functionality."""
    try:
        # Navigate to homepage
        await verifier.navigate_to_url("https://www.tesco.com")
        
        # Verify page loaded
        title = await verifier.get_page_title()
        if "Tesco" not in title:
            return VerificationResult(
                test_name="tesco_homepage_navigation",
                status="FAILED",
                details="Homepage title does not contain 'Tesco'"
            )

        # Check for essential elements
        elements = await verifier.find_elements([
            "header", "nav", "main", "footer"
        ])
        
        if len(elements) < 3:
            return VerificationResult(
                test_name="tesco_homepage_navigation",
                status="FAILED",
                details="Essential page elements not found"
            )

        return VerificationResult(
            test_name="tesco_homepage_navigation",
            status="PASSED",
            details="Homepage navigation successful"
        )

    except Exception as e:
        return VerificationResult(
            test_name="tesco_homepage_navigation",
            status="ERROR",
            details=f"Homepage navigation failed: {str(e)}",
            error_message=str(e)
        )


async def tesco_search_functionality(verifier: VendorToolsVerifier, **kwargs) -> VerificationResult:
    """Test Tesco search functionality."""
    try:
        # Navigate to homepage
        await verifier.navigate_to_url("https://www.tesco.com")
        
        # Find and interact with search box
        search_box = await verifier.find_element("input[type='search'], input[name='search'], .search-input")
        if not search_box:
            return VerificationResult(
                test_name="tesco_search_functionality",
                status="FAILED",
                details="Search box not found"
            )

        # Enter search term
        await verifier.type_text(search_box, "milk")
        
        # Submit search
        search_button = await verifier.find_element("button[type='submit'], .search-button")
        if search_button:
            await verifier.click_element(search_button)
        else:
            # Try pressing Enter
            await verifier.press_key(search_box, "Enter")

        # Wait for results
        await verifier.wait_for_element(".product-list, .search-results, [data-testid='product']", timeout=10)

        # Verify search results
        results = await verifier.find_elements(".product-item, .product-card, [data-testid='product']")
        
        if len(results) == 0:
            return VerificationResult(
                test_name="tesco_search_functionality",
                status="FAILED",
                details="No search results found"
            )

        return VerificationResult(
            test_name="tesco_search_functionality",
            status="PASSED",
            details=f"Search successful, found {len(results)} results"
        )

    except Exception as e:
        return VerificationResult(
            test_name="tesco_search_functionality",
            status="ERROR",
            details=f"Search functionality failed: {str(e)}",
            error_message=str(e)
        )


async def tesco_category_navigation(verifier: VendorToolsVerifier, **kwargs) -> VerificationResult:
    """Test Tesco category navigation."""
    try:
        # Navigate to homepage
        await verifier.navigate_to_url("https://www.tesco.com")
        
        # Find category menu
        category_menu = await verifier.find_element("nav, .category-menu, .department-menu")
        if not category_menu:
            return VerificationResult(
                test_name="tesco_category_navigation",
                status="FAILED",
                details="Category menu not found"
            )

        # Find a category link
        category_links = await verifier.find_elements("a[href*='category'], a[href*='department'], .category-link")
        if not category_links:
            return VerificationResult(
                test_name="tesco_category_navigation",
                status="FAILED",
                details="No category links found"
            )

        # Click first category
        await verifier.click_element(category_links[0])
        
        # Wait for category page to load
        await verifier.wait_for_element(".category-page, .department-page, .product-grid", timeout=10)

        # Verify category page loaded
        title = await verifier.get_page_title()
        if not title:
            return VerificationResult(
                test_name="tesco_category_navigation",
                status="FAILED",
                details="Category page title not found"
            )

        return VerificationResult(
            test_name="tesco_category_navigation",
            status="PASSED",
            details="Category navigation successful"
        )

    except Exception as e:
        return VerificationResult(
            test_name="tesco_category_navigation",
            status="ERROR",
            details=f"Category navigation failed: {str(e)}",
            error_message=str(e)
        )


# Asda Test Scenarios
async def asda_homepage_navigation(verifier: VendorToolsVerifier, **kwargs) -> VerificationResult:
    """Test Asda homepage navigation and basic functionality."""
    try:
        # Navigate to homepage
        await verifier.navigate_to_url("https://groceries.asda.com")
        
        # Verify page loaded
        title = await verifier.get_page_title()
        if "ASDA" not in title.upper():
            return VerificationResult(
                test_name="asda_homepage_navigation",
                status="FAILED",
                details="Homepage title does not contain 'ASDA'"
            )

        # Check for essential elements
        elements = await verifier.find_elements([
            "header", "nav", "main", "footer"
        ])
        
        if len(elements) < 3:
            return VerificationResult(
                test_name="asda_homepage_navigation",
                status="FAILED",
                details="Essential page elements not found"
            )

        return VerificationResult(
            test_name="asda_homepage_navigation",
            status="PASSED",
            details="Homepage navigation successful"
        )

    except Exception as e:
        return VerificationResult(
            test_name="asda_homepage_navigation",
            status="ERROR",
            details=f"Homepage navigation failed: {str(e)}",
            error_message=str(e)
        )


async def asda_search_functionality(verifier: VendorToolsVerifier, **kwargs) -> VerificationResult:
    """Test Asda search functionality."""
    try:
        # Navigate to homepage
        await verifier.navigate_to_url("https://groceries.asda.com")
        
        # Find and interact with search box
        search_box = await verifier.find_element("input[type='search'], input[name='search'], .search-input")
        if not search_box:
            return VerificationResult(
                test_name="asda_search_functionality",
                status="FAILED",
                details="Search box not found"
            )

        # Enter search term
        await verifier.type_text(search_box, "bread")
        
        # Submit search
        search_button = await verifier.find_element("button[type='submit'], .search-button")
        if search_button:
            await verifier.click_element(search_button)
        else:
            # Try pressing Enter
            await verifier.press_key(search_box, "Enter")

        # Wait for results
        await verifier.wait_for_element(".product-list, .search-results, [data-testid='product']", timeout=10)

        # Verify search results
        results = await verifier.find_elements(".product-item, .product-card, [data-testid='product']")
        
        if len(results) == 0:
            return VerificationResult(
                test_name="asda_search_functionality",
                status="FAILED",
                details="No search results found"
            )

        return VerificationResult(
            test_name="asda_search_functionality",
            status="PASSED",
            details=f"Search successful, found {len(results)} results"
        )

    except Exception as e:
        return VerificationResult(
            test_name="asda_search_functionality",
            status="ERROR",
            details=f"Search functionality failed: {str(e)}",
            error_message=str(e)
        )


# Costco Test Scenarios
async def costco_homepage_navigation(verifier: VendorToolsVerifier, **kwargs) -> VerificationResult:
    """Test Costco homepage navigation and basic functionality."""
    try:
        # Navigate to homepage
        await verifier.navigate_to_url("https://www.costco.com")
        
        # Verify page loaded
        title = await verifier.get_page_title()
        if "Costco" not in title:
            return VerificationResult(
                test_name="costco_homepage_navigation",
                status="FAILED",
                details="Homepage title does not contain 'Costco'"
            )

        # Check for essential elements
        elements = await verifier.find_elements([
            "header", "nav", "main", "footer"
        ])
        
        if len(elements) < 3:
            return VerificationResult(
                test_name="costco_homepage_navigation",
                status="FAILED",
                details="Essential page elements not found"
            )

        return VerificationResult(
            test_name="costco_homepage_navigation",
            status="PASSED",
            details="Homepage navigation successful"
        )

    except Exception as e:
        return VerificationResult(
            test_name="costco_homepage_navigation",
            status="ERROR",
            details=f"Homepage navigation failed: {str(e)}",
            error_message=str(e)
        )


async def costco_search_functionality(verifier: VendorToolsVerifier, **kwargs) -> VerificationResult:
    """Test Costco search functionality."""
    try:
        # Navigate to homepage
        await verifier.navigate_to_url("https://www.costco.com")
        
        # Find and interact with search box
        search_box = await verifier.find_element("input[type='search'], input[name='search'], .search-input")
        if not search_box:
            return VerificationResult(
                test_name="costco_search_functionality",
                status="FAILED",
                details="Search box not found"
            )

        # Enter search term
        await verifier.type_text(search_box, "electronics")
        
        # Submit search
        search_button = await verifier.find_element("button[type='submit'], .search-button")
        if search_button:
            await verifier.click_element(search_button)
        else:
            # Try pressing Enter
            await verifier.press_key(search_box, "Enter")

        # Wait for results
        await verifier.wait_for_element(".product-list, .search-results, [data-testid='product']", timeout=10)

        # Verify search results
        results = await verifier.find_elements(".product-item, .product-card, [data-testid='product']")
        
        if len(results) == 0:
            return VerificationResult(
                test_name="costco_search_functionality",
                status="FAILED",
                details="No search results found"
            )

        return VerificationResult(
            test_name="costco_search_functionality",
            status="PASSED",
            details=f"Search successful, found {len(results)} results"
        )

    except Exception as e:
        return VerificationResult(
            test_name="costco_search_functionality",
            status="ERROR",
            details=f"Search functionality failed: {str(e)}",
            error_message=str(e)
        )


def create_test_scenario_manager(verifier: VendorToolsVerifier) -> TestScenarioManager:
    """Create and configure a test scenario manager with all scenarios."""
    manager = TestScenarioManager(verifier)

    # Register Tesco scenarios
    manager.register_scenario(TestScenario(
        name="tesco_homepage_navigation",
        description="Test Tesco homepage navigation and basic functionality",
        scenario_type=TestScenarioType.FUNCTIONAL,
        vendor="tesco",
        test_function=tesco_homepage_navigation,
        required_level=VerificationLevel.BASIC
    ))

    manager.register_scenario(TestScenario(
        name="tesco_search_functionality",
        description="Test Tesco search functionality",
        scenario_type=TestScenarioType.FUNCTIONAL,
        vendor="tesco",
        test_function=tesco_search_functionality,
        required_level=VerificationLevel.BASIC
    ))

    manager.register_scenario(TestScenario(
        name="tesco_category_navigation",
        description="Test Tesco category navigation",
        scenario_type=TestScenarioType.FUNCTIONAL,
        vendor="tesco",
        test_function=tesco_category_navigation,
        required_level=VerificationLevel.BASIC
    ))

    # Register Asda scenarios
    manager.register_scenario(TestScenario(
        name="asda_homepage_navigation",
        description="Test Asda homepage navigation and basic functionality",
        scenario_type=TestScenarioType.FUNCTIONAL,
        vendor="asda",
        test_function=asda_homepage_navigation,
        required_level=VerificationLevel.BASIC
    ))

    manager.register_scenario(TestScenario(
        name="asda_search_functionality",
        description="Test Asda search functionality",
        scenario_type=TestScenarioType.FUNCTIONAL,
        vendor="asda",
        test_function=asda_search_functionality,
        required_level=VerificationLevel.BASIC
    ))

    # Register Costco scenarios
    manager.register_scenario(TestScenario(
        name="costco_homepage_navigation",
        description="Test Costco homepage navigation and basic functionality",
        scenario_type=TestScenarioType.FUNCTIONAL,
        vendor="costco",
        test_function=costco_homepage_navigation,
        required_level=VerificationLevel.BASIC
    ))

    manager.register_scenario(TestScenario(
        name="costco_search_functionality",
        description="Test Costco search functionality",
        scenario_type=TestScenarioType.FUNCTIONAL,
        vendor="costco",
        test_function=costco_search_functionality,
        required_level=VerificationLevel.BASIC
    ))

    return manager 