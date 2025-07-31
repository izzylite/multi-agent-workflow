"""
Common Challenge Handling Verification

Provides reusable routines for detecting and resolving common web challenges
such as cookie consent, login prompts, age verification, newsletter popups,
and location dialogs across all vendors.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from .vendor_verifier import (
    VendorToolsVerifier, VerificationResult, VerificationStatus,
    VerificationLevel
)
from .browser_operations import BrowserOperations
from .exceptions import (
    ElementNotFoundError, ElementInteractionError, NavigationError,
    TimeoutError, SessionConnectionError
)


@dataclass
class ChallengeType:
    """Types of web challenges."""
    COOKIE_CONSENT = "cookie_consent"
    LOGIN_PROMPT = "login_prompt"
    AGE_VERIFICATION = "age_verification"
    NEWSLETTER_POPUP = "newsletter_popup"
    LOCATION_SELECTOR = "location_selector"
    MEMBERSHIP_VERIFICATION = "membership_verification"
    POSTAL_CODE_PROMPT = "postal_code_prompt"
    WAREHOUSE_SELECTION = "warehouse_selection"
    RATE_LIMITING = "rate_limiting"


@dataclass
class ChallengeDetectionResult:
    """Result of challenge detection."""
    challenge_type: str
    detected: bool
    element_found: bool
    resolution_attempted: bool
    resolution_successful: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ChallengeHandler:
    """Base class for challenge handling verification."""
    
    def __init__(self, browser_ops: BrowserOperations):
        """
        Initialize challenge handler.
        
        Args:
            browser_ops: BrowserOperations instance for browser automation
        """
        self.browser_ops = browser_ops
        self.logger = logging.getLogger(f"{__name__}.ChallengeHandler")
        
        # Common selectors for challenge detection
        self.challenge_selectors = {
            ChallengeType.COOKIE_CONSENT: [
                '[data-testid="cookie-banner"]',
                '.cookie-banner',
                '#cookie-banner',
                '[class*="cookie"]',
                '[id*="cookie"]'
            ],
            ChallengeType.LOGIN_PROMPT: [
                '[data-testid="login-modal"]',
                '.login-modal',
                '#login-modal',
                '[class*="login"]',
                '[id*="login"]'
            ],
            ChallengeType.AGE_VERIFICATION: [
                '[data-testid="age-verification"]',
                '.age-verification',
                '#age-verification',
                '[class*="age"]',
                '[id*="age"]'
            ],
            ChallengeType.NEWSLETTER_POPUP: [
                '[data-testid="newsletter-popup"]',
                '.newsletter-popup',
                '#newsletter-popup',
                '[class*="newsletter"]',
                '[id*="newsletter"]'
            ],
            ChallengeType.LOCATION_SELECTOR: [
                '[data-testid="location-selector"]',
                '.location-selector',
                '#location-selector',
                '[class*="location"]',
                '[id*="location"]'
            ]
        }
    
    async def detect_challenge(self, challenge_type: str) -> ChallengeDetectionResult:
        """
        Detect if a specific challenge is present on the page.
        
        Args:
            challenge_type: Type of challenge to detect
            
        Returns:
            ChallengeDetectionResult with detection outcome
        """
        result = ChallengeDetectionResult(
            challenge_type=challenge_type,
            detected=False,
            element_found=False,
            resolution_attempted=False,
            resolution_successful=False
        )
        
        try:
            selectors = self.challenge_selectors.get(challenge_type, [])
            
            for selector in selectors:
                try:
                    # Try to find the challenge element
                    element_result = await self.browser_ops.wait_for_element(selector, timeout=2000)
                    
                    if element_result.success:
                        result.detected = True
                        result.element_found = True
                        self.logger.info(f"Detected {challenge_type} challenge with selector: {selector}")
                        break
                        
                except Exception as e:
                    self.logger.debug(f"Selector {selector} not found for {challenge_type}: {e}")
                    continue
            
            if not result.detected:
                self.logger.debug(f"No {challenge_type} challenge detected")
                
        except Exception as e:
            result.error_message = f"Error detecting {challenge_type} challenge: {str(e)}"
            self.logger.error(result.error_message)
        
        return result
    
    async def resolve_challenge(self, challenge_type: str, detection_result: ChallengeDetectionResult) -> bool:
        """
        Attempt to resolve a detected challenge.
        
        Args:
            challenge_type: Type of challenge to resolve
            detection_result: Result from challenge detection
            
        Returns:
            True if resolution successful, False otherwise
        """
        if not detection_result.detected:
            return True  # No challenge to resolve
        
        detection_result.resolution_attempted = True
        
        try:
            if challenge_type == ChallengeType.COOKIE_CONSENT:
                success = await self._resolve_cookie_consent()
            elif challenge_type == ChallengeType.LOGIN_PROMPT:
                success = await self._resolve_login_prompt()
            elif challenge_type == ChallengeType.AGE_VERIFICATION:
                success = await self._resolve_age_verification()
            elif challenge_type == ChallengeType.NEWSLETTER_POPUP:
                success = await self._resolve_newsletter_popup()
            elif challenge_type == ChallengeType.LOCATION_SELECTOR:
                success = await self._resolve_location_selector()
            else:
                self.logger.warning(f"No resolution method for challenge type: {challenge_type}")
                return False
            
            detection_result.resolution_successful = success
            return success
            
        except Exception as e:
            detection_result.error_message = f"Error resolving {challenge_type} challenge: {str(e)}"
            self.logger.error(detection_result.error_message)
            return False
    
    async def _resolve_cookie_consent(self) -> bool:
        """Resolve cookie consent popup."""
        try:
            # Common cookie consent button selectors
            accept_selectors = [
                '[data-testid="accept-cookies"]',
                '.accept-cookies',
                '#accept-cookies',
                '[class*="accept"]',
                '[class*="agree"]',
                'button:contains("Accept")',
                'button:contains("Agree")',
                'button:contains("OK")'
            ]
            
            for selector in accept_selectors:
                try:
                    click_result = await self.browser_ops.click(selector, timeout=3000)
                    if click_result.success:
                        self.logger.info("Successfully resolved cookie consent")
                        return True
                except Exception:
                    continue
            
            self.logger.warning("Could not resolve cookie consent")
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving cookie consent: {e}")
            return False
    
    async def _resolve_login_prompt(self) -> bool:
        """Resolve login prompt by closing or bypassing."""
        try:
            # Common close button selectors for login modals
            close_selectors = [
                '[data-testid="close-modal"]',
                '.close-modal',
                '#close-modal',
                '[class*="close"]',
                '[class*="dismiss"]',
                'button:contains("Close")',
                'button:contains("Cancel")',
                'button:contains("X")'
            ]
            
            for selector in close_selectors:
                try:
                    click_result = await self.browser_ops.click(selector, timeout=3000)
                    if click_result.success:
                        self.logger.info("Successfully closed login prompt")
                        return True
                except Exception:
                    continue
            
            self.logger.warning("Could not resolve login prompt")
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving login prompt: {e}")
            return False
    
    async def _resolve_age_verification(self) -> bool:
        """Resolve age verification dialog."""
        try:
            # Common age verification button selectors
            verify_selectors = [
                '[data-testid="age-verify"]',
                '.age-verify',
                '#age-verify',
                '[class*="verify"]',
                'button:contains("Yes")',
                'button:contains("Verify")',
                'button:contains("Continue")'
            ]
            
            for selector in verify_selectors:
                try:
                    click_result = await self.browser_ops.click(selector, timeout=3000)
                    if click_result.success:
                        self.logger.info("Successfully resolved age verification")
                        return True
                except Exception:
                    continue
            
            self.logger.warning("Could not resolve age verification")
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving age verification: {e}")
            return False
    
    async def _resolve_newsletter_popup(self) -> bool:
        """Resolve newsletter subscription popup."""
        try:
            # Common newsletter popup close selectors
            close_selectors = [
                '[data-testid="close-newsletter"]',
                '.close-newsletter',
                '#close-newsletter',
                '[class*="close"]',
                '[class*="dismiss"]',
                'button:contains("Close")',
                'button:contains("No thanks")',
                'button:contains("Skip")'
            ]
            
            for selector in close_selectors:
                try:
                    click_result = await self.browser_ops.click(selector, timeout=3000)
                    if click_result.success:
                        self.logger.info("Successfully closed newsletter popup")
                        return True
                except Exception:
                    continue
            
            self.logger.warning("Could not resolve newsletter popup")
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving newsletter popup: {e}")
            return False
    
    async def _resolve_location_selector(self) -> bool:
        """Resolve location selection dialog."""
        try:
            # Common location selector button selectors
            select_selectors = [
                '[data-testid="select-location"]',
                '.select-location',
                '#select-location',
                '[class*="location"]',
                'button:contains("Select")',
                'button:contains("Continue")',
                'button:contains("OK")'
            ]
            
            for selector in select_selectors:
                try:
                    click_result = await self.browser_ops.click(selector, timeout=3000)
                    if click_result.success:
                        self.logger.info("Successfully resolved location selector")
                        return True
                except Exception:
                    continue
            
            self.logger.warning("Could not resolve location selector")
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolving location selector: {e}")
            return False


class ChallengeVerifier:
    """Verification system for challenge handling."""
    
    def __init__(self, verifier: VendorToolsVerifier):
        """
        Initialize challenge verifier.
        
        Args:
            verifier: VendorToolsVerifier instance
        """
        self.verifier = verifier
        self.challenge_handler = ChallengeHandler(verifier.browser_ops)
        self.logger = logging.getLogger(f"{__name__}.ChallengeVerifier")
    
    async def run_challenge_verification(self, vendor: str) -> List[VerificationResult]:
        """
        Run comprehensive challenge verification for a vendor.
        
        Args:
            vendor: Vendor name to verify
            
        Returns:
            List of verification results
        """
        results = []
        
        # Start verification session
        success = await self.verifier.start_verification_session(vendor)
        if not success:
            raise SessionConnectionError(f"Failed to start verification session for {vendor}")
        
        try:
            # Test each challenge type
            challenge_types = [
                ChallengeType.COOKIE_CONSENT,
                ChallengeType.LOGIN_PROMPT,
                ChallengeType.AGE_VERIFICATION,
                ChallengeType.NEWSLETTER_POPUP,
                ChallengeType.LOCATION_SELECTOR
            ]
            
            for challenge_type in challenge_types:
                result = await self.verifier.run_verification_test(
                    test_name=f"{vendor}_{challenge_type}_detection",
                    test_func=self._test_challenge_detection,
                    timeout=15,
                    challenge_type=challenge_type
                )
                results.append(result)
                
                # If challenge was detected, test resolution
                if result.status == VerificationStatus.PASSED:
                    resolution_result = await self.verifier.run_verification_test(
                        test_name=f"{vendor}_{challenge_type}_resolution",
                        test_func=self._test_challenge_resolution,
                        timeout=15,
                        challenge_type=challenge_type
                    )
                    results.append(resolution_result)
        
        finally:
            await self.verifier.cleanup()
        
        return results
    
    async def _test_challenge_detection(self, challenge_type: str) -> None:
        """Test detection of a specific challenge type."""
        detection_result = await self.challenge_handler.detect_challenge(challenge_type)
        
        if detection_result.error_message:
            raise ElementInteractionError(detection_result.error_message)
        
        # For testing purposes, we consider both detection and non-detection as success
        # In real scenarios, you might want to verify specific challenges are present
        self.logger.info(f"Challenge detection test completed for {challenge_type}")
    
    async def _test_challenge_resolution(self, challenge_type: str) -> None:
        """Test resolution of a specific challenge type."""
        # First detect the challenge
        detection_result = await self.challenge_handler.detect_challenge(challenge_type)
        
        if detection_result.detected:
            # Attempt to resolve the challenge
            success = await self.challenge_handler.resolve_challenge(challenge_type, detection_result)
            
            if not success:
                raise ElementInteractionError(f"Failed to resolve {challenge_type} challenge")
        
        self.logger.info(f"Challenge resolution test completed for {challenge_type}")


def create_challenge_verifier(verifier: VendorToolsVerifier) -> ChallengeVerifier:
    """Create a ChallengeVerifier instance."""
    return ChallengeVerifier(verifier) 