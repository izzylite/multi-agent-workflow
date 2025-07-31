"""
Vendor Tools Verification System

Provides a comprehensive testing framework to verify vendor-specific tools functionality
across Tesco, Asda, and Costco using Browserbase MCP tools.
"""

import asyncio
import logging
import time
import json
import os
from typing import List, Optional, Dict, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from .browserbase_manager import BrowserbaseManager, SessionInfo
from .vendor_tools import VendorTool, TescoTool, AsdaTool, CostcoTool
from .browser_operations import BrowserOperations, BrowserOperationResult
from .exceptions import (
    ElementNotFoundError, ElementInteractionError, NavigationError,
    TimeoutError, SessionConnectionError
)


class VerificationStatus(Enum):
    """Status of verification tests."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    TIMEOUT = "timeout"


class VerificationLevel(Enum):
    """Levels of verification testing."""
    BASIC = "basic"           # Core functionality tests
    COMPREHENSIVE = "comprehensive"  # Full feature tests
    STRESS = "stress"         # Performance and edge case tests
    REGRESSION = "regression" # Historical comparison tests


@dataclass
class VerificationResult:
    """Result of a verification test."""
    test_name: str
    vendor: str
    status: VerificationStatus
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Optional[str] = None
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Comprehensive report of verification results."""
    vendor: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration: float
    timestamp: datetime = field(default_factory=datetime.now)
    results: List[VerificationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class VendorToolsVerifier:
    """
    Core framework for vendor-specific tools verification.
    
    Provides a unified interface for automated test execution, standardized
    verification procedures, and integrated reporting/logging for all vendor modules.
    """
    
    def __init__(self, 
                 browser_manager: BrowserbaseManager,
                 output_dir: str = "verification_reports",
                 log_level: str = "INFO"):
        """
        Initialize the vendor tools verifier.
        
        Args:
            browser_manager: BrowserbaseManager instance for browser automation
            output_dir: Directory to store verification reports and screenshots
            log_level: Logging level for verification operations
        """
        self.browser_manager = browser_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.VendorToolsVerifier")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize browser operations
        self.browser_ops = BrowserOperations(browser_manager)
        
        # Verification state
        self.current_session: Optional[SessionInfo] = None
        self.current_vendor_tool: Optional[VendorTool] = None
        self.verification_results: List[VerificationResult] = []
        
        # Statistics
        self.stats = {
            'total_verifications': 0,
            'successful_verifications': 0,
            'failed_verifications': 0,
            'total_duration': 0.0,
            'vendor_stats': {}
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.current_session:
            try:
                await self.browser_ops.disconnect_session()
                self.current_session = None
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")
    
    def create_vendor_tool(self, vendor: str, **kwargs) -> VendorTool:
        """
        Create a vendor-specific tool for verification.
        
        Args:
            vendor: Vendor name (tesco, asda, costco)
            **kwargs: Additional arguments for tool creation
            
        Returns:
            VendorTool instance
        """
        vendor_lower = str(vendor).lower()
        
        if vendor_lower == "tesco":
            return TescoTool(self.browser_manager, **kwargs)
        elif vendor_lower == "asda":
            return AsdaTool(self.browser_manager, **kwargs)
        elif vendor_lower == "costco":
            return CostcoTool(self.browser_manager, **kwargs)
        else:
            raise ValueError(f"Unsupported vendor: {vendor}")
    
    async def start_verification_session(self, vendor: str, **kwargs) -> bool:
        """
        Start a verification session for a specific vendor.
        
        Args:
            vendor: Vendor name to verify
            **kwargs: Additional arguments for tool creation
            
        Returns:
            True if session started successfully
        """
        try:
            self.logger.info(f"Starting verification session for {vendor}")
            
            # Create vendor tool
            self.current_vendor_tool = self.create_vendor_tool(vendor, **kwargs)
            
            # Create browser session
            session_config = {
                'session_timeout': 60000,  # 60 seconds for verification
                'enable_stealth': True,
                'viewport': {'width': 1920, 'height': 1080}
            }
            
            self.current_session = await self.browser_manager.create_session(session_config)
            
            # Connect to session
            success = await self.browser_ops.connect_session(self.current_session)
            if not success:
                raise SessionConnectionError(f"Failed to connect to session for {vendor}")
            
            self.logger.info(f"Verification session started for {vendor}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start verification session for {vendor}: {e}")
            return False
    
    async def run_verification_test(self, 
                                  test_name: str,
                                  test_func: Callable,
                                  timeout: int = 30,
                                  **kwargs) -> VerificationResult:
        """
        Run a single verification test.
        
        Args:
            test_name: Name of the test
            test_func: Async function to execute the test
            timeout: Test timeout in seconds
            **kwargs: Additional arguments for the test function
            
        Returns:
            VerificationResult with test outcome
        """
        start_time = time.time()
        result = VerificationResult(
            test_name=test_name,
            vendor=self.current_vendor_tool.vendor_config.name if self.current_vendor_tool else "unknown",
            status=VerificationStatus.ERROR,
            duration=0.0
        )
        
        try:
            self.logger.info(f"Running verification test: {test_name}")
            
            # Run test with timeout
            test_task = asyncio.create_task(test_func(**kwargs))
            await asyncio.wait_for(test_task, timeout=timeout)
            
            # Test completed successfully
            result.status = VerificationStatus.PASSED
            result.details = f"Test {test_name} completed successfully"
            
        except asyncio.TimeoutError:
            result.status = VerificationStatus.TIMEOUT
            result.error_message = f"Test {test_name} timed out after {timeout} seconds"
            self.logger.error(result.error_message)
            
        except Exception as e:
            result.status = VerificationStatus.FAILED
            result.error_message = f"Test {test_name} failed: {str(e)}"
            self.logger.error(result.error_message)
            
            # Take screenshot on failure
            try:
                screenshot_path = await self._take_failure_screenshot(test_name)
                result.screenshot_path = screenshot_path
            except Exception as screenshot_error:
                self.logger.warning(f"Failed to take failure screenshot: {screenshot_error}")
        
        finally:
            result.duration = time.time() - start_time
            self.verification_results.append(result)
            self._update_stats(result)
        
        return result
    
    async def _take_failure_screenshot(self, test_name: str) -> Optional[str]:
        """Take a screenshot for failed tests."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{test_name}_{timestamp}.png"
            screenshot_path = self.output_dir / filename
            
            screenshot_result = await self.browser_ops.take_screenshot(
                path=str(screenshot_path),
                full_page=True
            )
            
            if screenshot_result.success:
                return str(screenshot_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to take screenshot: {e}")
        
        return None
    
    def _update_stats(self, result: VerificationResult):
        """Update verification statistics."""
        self.stats['total_verifications'] += 1
        self.stats['total_duration'] += result.duration
        
        if result.status == VerificationStatus.PASSED:
            self.stats['successful_verifications'] += 1
        elif result.status in [VerificationStatus.FAILED, VerificationStatus.ERROR]:
            self.stats['failed_verifications'] += 1
        
        # Update vendor-specific stats
        vendor = result.vendor
        if vendor not in self.stats['vendor_stats']:
            self.stats['vendor_stats'][vendor] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'duration': 0.0
            }
        
        self.stats['vendor_stats'][vendor]['total'] += 1
        self.stats['vendor_stats'][vendor]['duration'] += result.duration
        
        if result.status == VerificationStatus.PASSED:
            self.stats['vendor_stats'][vendor]['passed'] += 1
        else:
            self.stats['vendor_stats'][vendor]['failed'] += 1
    
    def generate_verification_report(self, vendor: str) -> VerificationReport:
        """
        Generate a comprehensive verification report for a vendor.
        
        Args:
            vendor: Vendor name
            
        Returns:
            VerificationReport with all results and statistics
        """
        vendor_results = [r for r in self.verification_results if str(r.vendor).lower() == str(vendor).lower()]
        
        total_tests = len(vendor_results)
        passed_tests = len([r for r in vendor_results if r.status == VerificationStatus.PASSED])
        failed_tests = len([r for r in vendor_results if r.status == VerificationStatus.FAILED])
        skipped_tests = len([r for r in vendor_results if r.status == VerificationStatus.SKIPPED])
        error_tests = len([r for r in vendor_results if r.status == VerificationStatus.ERROR])
        total_duration = sum(r.duration for r in vendor_results)
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate summary
        summary = {
            'success_rate': success_rate,
            'average_duration': total_duration / total_tests if total_tests > 0 else 0,
            'failure_reasons': self._analyze_failures(vendor_results),
            'performance_metrics': self._calculate_performance_metrics(vendor_results)
        }
        
        return VerificationReport(
            vendor=vendor,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_duration=total_duration,
            results=vendor_results,
            summary=summary
        )
    
    def _analyze_failures(self, results: List[VerificationResult]) -> Dict[str, int]:
        """Analyze failure reasons from verification results."""
        failure_reasons = {}
        for result in results:
            if result.status in [VerificationStatus.FAILED, VerificationStatus.ERROR]:
                error_type = type(result.error_message).__name__ if result.error_message else "Unknown"
                failure_reasons[error_type] = failure_reasons.get(error_type, 0) + 1
        return failure_reasons
    
    def _calculate_performance_metrics(self, results: List[VerificationResult]) -> Dict[str, float]:
        """Calculate performance metrics from verification results."""
        if not results:
            return {}
        
        durations = [r.duration for r in results]
        return {
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_duration': sum(durations) / len(durations),
            'median_duration': sorted(durations)[len(durations) // 2]
        }
    
    async def save_verification_report(self, report: VerificationReport) -> str:
        """
        Save verification report to file.
        
        Args:
            report: VerificationReport to save
            
        Returns:
            Path to saved report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.vendor}_verification_report_{timestamp}.json"
        report_path = self.output_dir / filename
        
        # Convert report to JSON-serializable format
        report_data = {
            'vendor': report.vendor,
            'total_tests': report.total_tests,
            'passed_tests': report.passed_tests,
            'failed_tests': report.failed_tests,
            'skipped_tests': report.skipped_tests,
            'error_tests': report.error_tests,
            'total_duration': report.total_duration,
            'timestamp': report.timestamp.isoformat(),
            'summary': report.summary,
            'results': [
                {
                    'test_name': r.test_name,
                    'vendor': r.vendor,
                    'status': r.status.value,
                    'duration': r.duration,
                    'timestamp': r.timestamp.isoformat(),
                    'details': r.details,
                    'error_message': r.error_message,
                    'screenshot_path': r.screenshot_path,
                    'metadata': r.metadata
                }
                for r in report.results
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Verification report saved to: {report_path}")
        return str(report_path)
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get current verification statistics."""
        return {
            'total_verifications': self.stats['total_verifications'],
            'successful_verifications': self.stats['successful_verifications'],
            'failed_verifications': self.stats['failed_verifications'],
            'success_rate': (
                self.stats['successful_verifications'] / self.stats['total_verifications'] * 100
                if self.stats['total_verifications'] > 0 else 0
            ),
            'total_duration': self.stats['total_duration'],
            'vendor_stats': self.stats['vendor_stats']
        }


def create_vendor_verifier(browser_manager: BrowserbaseManager,
                          output_dir: str = "verification_reports") -> VendorToolsVerifier:
    """
    Create a VendorToolsVerifier instance.
    
    Args:
        browser_manager: BrowserbaseManager instance
        output_dir: Directory for verification reports
        
    Returns:
        VendorToolsVerifier instance
    """
    return VendorToolsVerifier(browser_manager, output_dir) 