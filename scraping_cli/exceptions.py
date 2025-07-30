"""
Custom Exception Classes for Browser Automation

This module defines custom exception classes for handling different types
of errors that can occur during browser automation operations.
"""

from typing import Optional, Dict, Any


class BrowserAutomationError(Exception):
    """Base exception class for browser automation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class SessionError(BrowserAutomationError):
    """Exception raised for session-related errors."""
    
    def __init__(self, message: str, session_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.session_id = session_id
        super().__init__(message, details)


class SessionCreationError(SessionError):
    """Exception raised when session creation fails."""
    pass


class SessionConnectionError(SessionError):
    """Exception raised when session connection fails."""
    pass


class SessionTimeoutError(SessionError):
    """Exception raised when session times out."""
    pass


class SessionHealthError(SessionError):
    """Exception raised when session health check fails."""
    pass


class NetworkError(BrowserAutomationError):
    """Exception raised for network-related errors."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.url = url
        self.status_code = status_code
        super().__init__(message, details)


class NavigationError(NetworkError):
    """Exception raised when navigation fails."""
    pass


class ElementNotFoundError(BrowserAutomationError):
    """Exception raised when an element is not found."""
    
    def __init__(self, message: str, selector: Optional[str] = None, timeout: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.selector = selector
        self.timeout = timeout
        super().__init__(message, details)


class ElementInteractionError(BrowserAutomationError):
    """Exception raised when element interaction fails."""
    
    def __init__(self, message: str, selector: Optional[str] = None, action: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.selector = selector
        self.action = action
        super().__init__(message, details)


class ScreenshotError(BrowserAutomationError):
    """Exception raised when screenshot capture fails."""
    
    def __init__(self, message: str, path: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.path = path
        super().__init__(message, details)


class ConfigurationError(BrowserAutomationError):
    """Exception raised for configuration-related errors."""
    pass


class RetryExhaustedError(BrowserAutomationError):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str, max_retries: int, last_error: Optional[Exception] = None, details: Optional[Dict[str, Any]] = None):
        self.max_retries = max_retries
        self.last_error = last_error
        super().__init__(message, details)


class HealthCheckError(BrowserAutomationError):
    """Exception raised when health check fails."""
    
    def __init__(self, message: str, component: Optional[str] = None, health_status: Optional[Dict[str, Any]] = None, details: Optional[Dict[str, Any]] = None):
        self.component = component
        self.health_status = health_status
        super().__init__(message, details)


class TimeoutError(BrowserAutomationError):
    """Exception raised when an operation times out."""
    
    def __init__(self, message: str, timeout_duration: Optional[int] = None, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.timeout_duration = timeout_duration
        self.operation = operation
        super().__init__(message, details)


# Error classification helpers
def classify_error(error: Exception) -> str:
    """Classify an error based on its type and message."""
    if isinstance(error, SessionError):
        return "session"
    elif isinstance(error, NetworkError):
        return "network"
    elif isinstance(error, ElementNotFoundError):
        return "element_not_found"
    elif isinstance(error, ElementInteractionError):
        return "element_interaction"
    elif isinstance(error, TimeoutError):
        return "timeout"
    elif isinstance(error, RetryExhaustedError):
        return "retry_exhausted"
    else:
        return "unknown"


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    retryable_types = (
        NetworkError,
        SessionConnectionError,
        ElementNotFoundError,
        TimeoutError
    )
    
    # Check if it's a retryable error type
    if isinstance(error, retryable_types):
        return True
    
    # Check for specific error messages that indicate retryable conditions
    error_message = str(error).lower()
    retryable_keywords = [
        "timeout",
        "connection",
        "network",
        "temporary",
        "temporarily",
        "rate limit",
        "too many requests",
        "service unavailable",
        "gateway timeout"
    ]
    
    return any(keyword in error_message for keyword in retryable_keywords)


def get_error_severity(error: Exception) -> str:
    """Get the severity level of an error."""
    if isinstance(error, (ConfigurationError, SessionCreationError)):
        return "critical"
    elif isinstance(error, (SessionHealthError, RetryExhaustedError)):
        return "high"
    elif isinstance(error, (NetworkError, ElementInteractionError)):
        return "medium"
    elif isinstance(error, (ElementNotFoundError, TimeoutError)):
        return "low"
    else:
        return "unknown" 