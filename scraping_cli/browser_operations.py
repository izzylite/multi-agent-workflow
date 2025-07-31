"""
Browser Operations Module

Provides high-level wrapper methods for browser operations using Browserbase SDK.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, Union, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from playwright.sync_api import sync_playwright

from .browserbase_manager import BrowserbaseManager, SessionInfo, SessionConfig
from .exceptions import (
    ElementNotFoundError, ElementInteractionError, ScreenshotError,
    NavigationError, TimeoutError, SessionConnectionError,
    classify_error, is_retryable_error, get_error_severity
)


class BrowserOperationType(Enum):
    """Types of browser operations."""
    NAVIGATION = "navigation"
    INTERACTION = "interaction"
    EXTRACTION = "extraction"
    SCREENSHOT = "screenshot"
    WAIT = "wait"


@dataclass
class BrowserOperationResult:
    """Result of a browser operation."""
    success: bool
    operation_type: BrowserOperationType
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = None
    duration: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BrowserOperations:
    """High-level wrapper for browser operations using Browserbase."""
    
    def __init__(self, browser_manager: BrowserbaseManager, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize browser operations wrapper.
        
        Args:
            browser_manager: BrowserbaseManager instance
            max_retries: Maximum number of retries for failed operations
            retry_delay: Base delay between retries in seconds
        """
        self.browser_manager = browser_manager
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        self.current_session: Optional[SessionInfo] = None
        self.current_page: Optional[Page] = None
        self.current_browser: Optional[Browser] = None
        
        # Operation statistics
        self.stats = {
            'operations_performed': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_duration': 0.0,
            'retries': 0,
            'errors_by_type': {}
        }
    
    async def connect_session(self, session_info: SessionInfo) -> bool:
        """
        Connect to a browser session using Playwright.
        
        Args:
            session_info: Session to connect to
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to session {session_info.session_id}")
            
            # Connect to browser using Playwright
            self.current_browser = await async_playwright().start()
            browser = await self.current_browser.chromium.connect_over_cdp(session_info.connect_url)
            
            # Get the first page
            context = browser.contexts[0]
            self.current_page = context.pages[0] if context.pages else await context.new_page()
            
            self.current_session = session_info
            self.logger.info(f"Successfully connected to session {session_info.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to session {session_info.session_id}: {e}")
            return False
    
    async def disconnect_session(self) -> None:
        """Disconnect from the current browser session."""
        if self.current_browser:
            await self.current_browser.close()
            self.current_browser = None
            self.current_page = None
            self.current_session = None
            self.logger.info("Disconnected from browser session")
    
    async def _retry_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: Async function to retry
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts are exhausted
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.stats['retries'] += 1
                
                # Track error by type
                error_type = classify_error(e)
                self.stats['errors_by_type'][error_type] = self.stats['errors_by_type'].get(error_type, 0) + 1
                
                # Check if error is retryable
                if not is_retryable_error(e):
                    self.logger.error(f"Non-retryable error: {e}")
                    raise e
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = (2 ** attempt) * self.retry_delay
                    self.logger.warning(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All retry attempts exhausted: {e}")
                    raise last_error
    
    async def navigate(self, url: str, wait_until: str = "networkidle", timeout: int = 30000) -> BrowserOperationResult:
        """
        Navigate to a URL with retry logic and error handling.
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete ('load', 'domcontentloaded', 'networkidle')
            timeout: Navigation timeout in milliseconds
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        if not self.current_page:
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.NAVIGATION,
                error="No active page",
                duration=time.time() - start_time
            )
        
        async def _navigate_operation():
            """Inner function for navigation that can be retried."""
            self.logger.info(f"Navigating to {url}")
            response = await self.current_page.goto(url, wait_until=wait_until, timeout=timeout)
            return response
        
        try:
            response = await self._retry_operation(_navigate_operation)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.NAVIGATION,
                data={
                    'url': url,
                    'status': response.status if response else None,
                    'final_url': self.current_page.url
                },
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            error_msg = str(e)
            if "timeout" in str(error_msg).lower():
                raise TimeoutError(f"Navigation timeout: {error_msg}", timeout_duration=timeout, operation="navigation")
            elif "net::" in str(error_msg).lower():
                raise NavigationError(f"Network error during navigation: {error_msg}", url=url)
            else:
                raise NavigationError(f"Navigation failed: {error_msg}", url=url)
    
    async def click(self, selector: str, timeout: int = 5000) -> BrowserOperationResult:
        """
        Click an element.
        
        Args:
            selector: CSS selector or XPath
            timeout: Click timeout in milliseconds
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.INTERACTION,
                    error="No active page"
                )
            
            self.logger.info(f"Clicking element: {selector}")
            
            await self.current_page.click(selector, timeout=timeout)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.INTERACTION,
                data={'selector': selector, 'action': 'click'},
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.INTERACTION,
                error=str(e),
                duration=duration
            )
    
    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> BrowserOperationResult:
        """
        Type text into an element.
        
        Args:
            selector: CSS selector or XPath
            text: Text to type
            timeout: Type timeout in milliseconds
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.INTERACTION,
                    error="No active page"
                )
            
            self.logger.info(f"Typing text into element: {selector}")
            
            await self.current_page.fill(selector, text, timeout=timeout)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.INTERACTION,
                data={'selector': selector, 'action': 'type', 'text': text},
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.INTERACTION,
                error=str(e),
                duration=duration
            )
    
    async def select_option(self, selector: str, value: str, timeout: int = 5000) -> BrowserOperationResult:
        """
        Select an option from a dropdown.
        
        Args:
            selector: CSS selector for the select element
            value: Value to select
            timeout: Selection timeout in milliseconds
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.INTERACTION,
                    error="No active page"
                )
            
            self.logger.info(f"Selecting option {value} from {selector}")
            
            await self.current_page.select_option(selector, value, timeout=timeout)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.INTERACTION,
                data={'selector': selector, 'action': 'select', 'value': value},
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.INTERACTION,
                error=str(e),
                duration=duration
            )
    
    async def extract_text(self, selector: str) -> BrowserOperationResult:
        """
        Extract text content from an element.
        
        Args:
            selector: CSS selector or XPath
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.EXTRACTION,
                    error="No active page"
                )
            
            self.logger.info(f"Extracting text from: {selector}")
            
            text = await self.current_page.text_content(selector)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.EXTRACTION,
                data={'selector': selector, 'text': text},
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.EXTRACTION,
                error=str(e),
                duration=duration
            )
    
    async def extract_html(self, selector: str) -> BrowserOperationResult:
        """
        Extract HTML content from an element.
        
        Args:
            selector: CSS selector or XPath
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.EXTRACTION,
                    error="No active page"
                )
            
            self.logger.info(f"Extracting HTML from: {selector}")
            
            html = await self.current_page.inner_html(selector)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.EXTRACTION,
                data={'selector': selector, 'html': html},
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.EXTRACTION,
                error=str(e),
                duration=duration
            )
    
    async def extract_attributes(self, selector: str, attributes: List[str]) -> BrowserOperationResult:
        """
        Extract specific attributes from an element.
        
        Args:
            selector: CSS selector or XPath
            attributes: List of attribute names to extract
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.EXTRACTION,
                    error="No active page"
                )
            
            self.logger.info(f"Extracting attributes {attributes} from: {selector}")
            
            element = await self.current_page.query_selector(selector)
            if not element:
                raise Exception(f"Element not found: {selector}")
            
            extracted_attrs = {}
            for attr in attributes:
                value = await element.get_attribute(attr)
                extracted_attrs[attr] = value
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.EXTRACTION,
                data={'selector': selector, 'attributes': extracted_attrs},
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.EXTRACTION,
                error=str(e),
                duration=duration
            )
    
    async def take_screenshot(self, path: Optional[str] = None, full_page: bool = False) -> BrowserOperationResult:
        """
        Take a screenshot of the current page.
        
        Args:
            path: File path to save screenshot (optional)
            full_page: Whether to capture full page or viewport only
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.SCREENSHOT,
                    error="No active page"
                )
            
            self.logger.info("Taking screenshot")
            
            screenshot_bytes = await self.current_page.screenshot(full_page=full_page)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            result_data = {
                'screenshot_bytes': screenshot_bytes,
                'full_page': full_page,
                'size_bytes': len(screenshot_bytes)
            }
            
            if path:
                with open(path, 'wb') as f:
                    f.write(screenshot_bytes)
                result_data['saved_path'] = path
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.SCREENSHOT,
                data=result_data,
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.SCREENSHOT,
                error=str(e),
                duration=duration
            )
    
    async def wait_for_element(self, selector: str, timeout: int = 10000) -> BrowserOperationResult:
        """
        Wait for an element to appear on the page.
        
        Args:
            selector: CSS selector or XPath
            timeout: Wait timeout in milliseconds
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.WAIT,
                    error="No active page"
                )
            
            self.logger.info(f"Waiting for element: {selector}")
            
            await self.current_page.wait_for_selector(selector, timeout=timeout)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.WAIT,
                data={'selector': selector, 'action': 'wait_for_element'},
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.WAIT,
                error=str(e),
                duration=duration
            )
    
    async def wait_for_navigation(self, timeout: int = 10000) -> BrowserOperationResult:
        """
        Wait for page navigation to complete.
        
        Args:
            timeout: Wait timeout in milliseconds
            
        Returns:
            BrowserOperationResult
        """
        start_time = time.time()
        
        try:
            if not self.current_page:
                return BrowserOperationResult(
                    success=False,
                    operation_type=BrowserOperationType.WAIT,
                    error="No active page"
                )
            
            self.logger.info("Waiting for navigation to complete")
            
            await self.current_page.wait_for_load_state('networkidle', timeout=timeout)
            
            duration = time.time() - start_time
            self._update_stats(True, duration)
            
            return BrowserOperationResult(
                success=True,
                operation_type=BrowserOperationType.WAIT,
                data={'action': 'wait_for_navigation'},
                duration=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            
            return BrowserOperationResult(
                success=False,
                operation_type=BrowserOperationType.WAIT,
                error=str(e),
                duration=duration
            )
    
    def _update_stats(self, success: bool, duration: float) -> None:
        """Update operation statistics."""
        self.stats['operations_performed'] += 1
        self.stats['total_duration'] += duration
        
        if success:
            self.stats['successful_operations'] += 1
        else:
            self.stats['failed_operations'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        total_ops = self.stats['operations_performed']
        success_rate = (self.stats['successful_operations'] / total_ops * 100) if total_ops > 0 else 0
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'average_duration': self.stats['total_duration'] / total_ops if total_ops > 0 else 0
        }


# Factory function for creating BrowserOperations instances
def create_browser_operations(browser_manager: BrowserbaseManager) -> BrowserOperations:
    """
    Create and return a new BrowserOperations instance.
    
    Args:
        browser_manager: BrowserbaseManager instance
        
    Returns:
        BrowserOperations instance
    """
    return BrowserOperations(browser_manager) 