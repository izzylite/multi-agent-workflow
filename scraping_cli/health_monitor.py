"""
Session Health Monitoring Module

This module provides session health monitoring, automatic recovery,
and health check functionality for browser sessions.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .exceptions import (
    SessionHealthError, SessionTimeoutError, HealthCheckError,
    SessionConnectionError, BrowserAutomationError
)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    response_time: Optional[float] = None


@dataclass
class HealthConfig:
    """Configuration for health monitoring."""
    check_interval: int = 30  # seconds
    timeout: int = 10  # seconds
    max_failures: int = 3
    recovery_timeout: int = 60  # seconds
    enable_auto_recovery: bool = True
    health_check_endpoints: List[str] = field(default_factory=list)


class SessionHealthMonitor:
    """Monitor session health and provide automatic recovery."""
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.logger = logging.getLogger(__name__)
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_check: Dict[str, datetime] = {}
        self.recovery_callbacks: List[Callable] = []
        
    def add_recovery_callback(self, callback: Callable) -> None:
        """Add a callback to be called when recovery is needed."""
        self.recovery_callbacks.append(callback)
    
    async def check_session_health(self, session_id: str, session_info: Any) -> HealthCheckResult:
        """Check the health of a specific session."""
        start_time = time.time()
        
        try:
            # Basic session validation
            if not session_info or not hasattr(session_info, 'session_id'):
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    timestamp=datetime.now(),
                    details={'error': 'Invalid session info'},
                    response_time=time.time() - start_time
                )
            
            # Check if session is expired
            if hasattr(session_info, 'created_at'):
                age = datetime.now() - session_info.created_at
                if age.total_seconds() > self.config.recovery_timeout:
                    return HealthCheckResult(
                        status=HealthStatus.UNHEALTHY,
                        timestamp=datetime.now(),
                        details={'error': 'Session expired', 'age_seconds': age.total_seconds()},
                        response_time=time.time() - start_time
                    )
            
            # Check if session has been inactive too long
            if hasattr(session_info, 'last_used'):
                idle_time = datetime.now() - session_info.last_used
                if idle_time.total_seconds() > self.config.timeout * 2:
                    return HealthCheckResult(
                        status=HealthStatus.DEGRADED,
                        timestamp=datetime.now(),
                        details={'warning': 'Session idle', 'idle_seconds': idle_time.total_seconds()},
                        response_time=time.time() - start_time
                    )
            
            # Perform connectivity test if connect_url is available
            if hasattr(session_info, 'connect_url') and session_info.connect_url:
                # This would typically involve a lightweight connectivity test
                # For now, we'll assume the session is healthy if it has a valid connect_url
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    timestamp=datetime.now(),
                    details={'connect_url': session_info.connect_url},
                    response_time=time.time() - start_time
                )
            
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                details={'warning': 'Unable to perform full health check'},
                response_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed for session {session_id}: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                details={'error': str(e)},
                error=e,
                response_time=time.time() - start_time
            )
    
    async def monitor_sessions(self, sessions: Dict[str, Any]) -> Dict[str, HealthCheckResult]:
        """Monitor health of multiple sessions."""
        results = {}
        
        for session_id, session_info in sessions.items():
            # Check if we need to perform a health check
            last_check = self.last_check.get(session_id)
            if last_check and (datetime.now() - last_check).total_seconds() < self.config.check_interval:
                continue
            
            result = await self.check_session_health(session_id, session_info)
            results[session_id] = result
            self.last_check[session_id] = datetime.now()
            
            # Update health history
            if session_id not in self.health_history:
                self.health_history[session_id] = []
            self.health_history[session_id].append(result)
            
            # Keep only recent history (last 10 checks)
            if len(self.health_history[session_id]) > 10:
                self.health_history[session_id] = self.health_history[session_id][-10:]
            
            # Handle unhealthy sessions
            if result.status == HealthStatus.UNHEALTHY:
                await self._handle_unhealthy_session(session_id, session_info, result)
            elif result.status == HealthStatus.DEGRADED:
                await self._handle_degraded_session(session_id, session_info, result)
        
        return results
    
    async def _handle_unhealthy_session(self, session_id: str, session_info: Any, result: HealthCheckResult) -> None:
        """Handle an unhealthy session."""
        self.failure_counts[session_id] = self.failure_counts.get(session_id, 0) + 1
        
        self.logger.warning(f"Session {session_id} is unhealthy (failure #{self.failure_counts[session_id]}): {result.details}")
        
        if self.failure_counts[session_id] >= self.config.max_failures:
            self.logger.error(f"Session {session_id} exceeded max failures, triggering recovery")
            
            if self.config.enable_auto_recovery:
                await self._trigger_recovery(session_id, session_info, result)
    
    async def _handle_degraded_session(self, session_id: str, session_info: Any, result: HealthCheckResult) -> None:
        """Handle a degraded session."""
        self.logger.info(f"Session {session_id} is degraded: {result.details}")
        
        # For degraded sessions, we might want to schedule a recovery
        # but not immediately
        if self.config.enable_auto_recovery and self.failure_counts.get(session_id, 0) >= 1:
            await self._trigger_recovery(session_id, session_info, result)
    
    async def _trigger_recovery(self, session_id: str, session_info: Any, result: HealthCheckResult) -> None:
        """Trigger recovery for a session."""
        self.logger.info(f"Triggering recovery for session {session_id}")
        
        # Call all recovery callbacks
        for callback in self.recovery_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(session_id, session_info, result)
                else:
                    callback(session_id, session_info, result)
            except Exception as e:
                self.logger.error(f"Recovery callback failed: {e}")
        
        # Reset failure count after recovery attempt
        self.failure_counts[session_id] = 0
    
    def get_session_health_summary(self, session_id: str) -> Dict[str, Any]:
        """Get health summary for a session."""
        if session_id not in self.health_history:
            return {'status': 'unknown', 'message': 'No health data available'}
        
        history = self.health_history[session_id]
        if not history:
            return {'status': 'unknown', 'message': 'No health data available'}
        
        latest = history[-1]
        recent_history = history[-5:]  # Last 5 checks
        
        healthy_count = sum(1 for check in recent_history if check.status == HealthStatus.HEALTHY)
        total_count = len(recent_history)
        health_rate = (healthy_count / total_count) * 100 if total_count > 0 else 0
        
        return {
            'current_status': latest.status.value,
            'health_rate': health_rate,
            'failure_count': self.failure_counts.get(session_id, 0),
            'last_check': latest.timestamp.isoformat(),
            'response_time': latest.response_time,
            'details': latest.details
        }
    
    def get_overall_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary for all monitored sessions."""
        if not self.health_history:
            return {'status': 'unknown', 'message': 'No sessions monitored'}
        
        total_sessions = len(self.health_history)
        healthy_sessions = 0
        degraded_sessions = 0
        unhealthy_sessions = 0
        
        for session_id in self.health_history:
            summary = self.get_session_health_summary(session_id)
            if summary['current_status'] == 'healthy':
                healthy_sessions += 1
            elif summary['current_status'] == 'degraded':
                degraded_sessions += 1
            elif summary['current_status'] == 'unhealthy':
                unhealthy_sessions += 1
        
        return {
            'total_sessions': total_sessions,
            'healthy_sessions': healthy_sessions,
            'degraded_sessions': degraded_sessions,
            'unhealthy_sessions': unhealthy_sessions,
            'overall_health_rate': (healthy_sessions / total_sessions) * 100 if total_sessions > 0 else 0
        }


class HealthCheckRunner:
    """Runs periodic health checks."""
    
    def __init__(self, monitor: SessionHealthMonitor, sessions_provider: Callable):
        self.monitor = monitor
        self.sessions_provider = sessions_provider
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.check_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the health check runner."""
        if self.running:
            return
        
        self.running = True
        self.check_task = asyncio.create_task(self._run_health_checks())
        self.logger.info("Health check runner started")
    
    async def stop(self) -> None:
        """Stop the health check runner."""
        self.running = False
        if self.check_task:
            self.check_task.cancel()
            try:
                await self.check_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health check runner stopped")
    
    async def _run_health_checks(self) -> None:
        """Run periodic health checks."""
        while self.running:
            try:
                sessions = self.sessions_provider()
                results = await self.monitor.monitor_sessions(sessions)
                
                # Log summary
                if results:
                    unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
                    if unhealthy_count > 0:
                        self.logger.warning(f"Found {unhealthy_count} unhealthy sessions")
                
                # Wait for next check interval
                await asyncio.sleep(self.monitor.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health check runner error: {e}")
                await asyncio.sleep(5)  # Short delay before retry


def create_health_monitor(config: Optional[HealthConfig] = None) -> SessionHealthMonitor:
    """Create a health monitor instance."""
    return SessionHealthMonitor(config)


def create_health_runner(monitor: SessionHealthMonitor, sessions_provider: Callable) -> HealthCheckRunner:
    """Create a health check runner instance."""
    return HealthCheckRunner(monitor, sessions_provider) 