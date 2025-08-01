"""
Async Agent Deployment System

Provides advanced async agent deployment and orchestration capabilities
for concurrent scraping operations using CrewAI 0.150.0's async features.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager

from .crewai_integration import (
    ScrapingAgent, ScrapingTask, ScrapingCrew, AgentRole, TaskType,
    AgentConfig, TaskConfig, AgentFactory, TaskFactory
)
from .browserbase_manager import BrowserbaseManager, SessionConfig, SessionInfo
from .exceptions import (
    SessionCreationError, SessionConnectionError, SessionTimeoutError,
    ConfigurationError, RetryExhaustedError
)


class DeploymentStatus(Enum):
    """Status of agent deployment."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SCALING = "scaling"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentTaskStatus(Enum):
    """Status of agent tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency controls."""
    max_agents_per_vendor: int = 3
    max_total_agents: int = 10
    max_concurrent_sessions: int = 10
    rate_limit_per_domain: float = 1.0  # requests per second
    rate_limit_window: int = 60  # seconds
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5  # failures before opening
    circuit_breaker_timeout: int = 300  # seconds to wait before retry


@dataclass
class ScalingConfig:
    """Configuration for dynamic scaling."""
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # utilization percentage
    scale_down_threshold: float = 0.3  # utilization percentage
    min_agents: int = 1
    max_agents: int = 20
    scale_check_interval: int = 30  # seconds
    resource_monitor_enabled: bool = True


@dataclass
class AgentTaskInfo:
    """Information about an agent task."""
    task_id: str
    agent_id: str
    vendor: str
    task_type: TaskType
    status: AgentTaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    session_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    task_data: Optional[Dict[str, Any]] = None


@dataclass
class VendorMetrics:
    """Metrics for a specific vendor."""
    vendor: str
    active_agents: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_requests: int = 0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    circuit_breaker_open: bool = False
    circuit_breaker_until: Optional[datetime] = None


class CircuitBreaker:
    """Circuit breaker for vendor-specific error handling."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and (datetime.now() - self.last_failure_time).total_seconds() > self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self) -> None:
        """Record a successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class RateLimiter:
    """Rate limiter for domain-specific request throttling."""
    
    def __init__(self, requests_per_second: float = 1.0, window_size: int = 60):
        self.requests_per_second = requests_per_second
        self.window_size = window_size
        self.requests: List[datetime] = []
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        now = datetime.now()
        
        # Remove old requests outside the window
        cutoff = now - timedelta(seconds=self.window_size)
        self.requests = [req_time for req_time in self.requests if req_time > cutoff]
        
        # Check if we're within rate limits
        if len(self.requests) >= self.requests_per_second * self.window_size:
            # Calculate delay needed
            oldest_request = min(self.requests)
            delay = self.window_size - (now - oldest_request).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
        
        # Record this request
        self.requests.append(now)


class AgentDeployer:
    """
    Advanced async agent deployment system for concurrent scraping operations.
    
    Features:
    - Async agent orchestration using CrewAI 0.150.0
    - Session pool management integration
    - Dynamic scaling based on workload
    - Concurrency controls and rate limiting
    - Circuit breakers for fault tolerance
    - Health monitoring and recovery
    - Graceful shutdown handling
    """
    
    def __init__(self, 
                 browserbase_manager: BrowserbaseManager,
                 concurrency_config: Optional[ConcurrencyConfig] = None,
                 scaling_config: Optional[ScalingConfig] = None):
        """
        Initialize the agent deployer.
        
        Args:
            browserbase_manager: Manager for Browserbase sessions
            concurrency_config: Configuration for concurrency controls
            scaling_config: Configuration for dynamic scaling
        """
        self.browserbase_manager = browserbase_manager
        self.concurrency_config = concurrency_config or ConcurrencyConfig()
        self.scaling_config = scaling_config or ScalingConfig()
        
        # Register browser manager globally for tools
        from .browser_tools import browser_registry
        browser_registry.register_manager(browserbase_manager)
        
        # Deployment state
        self.status = DeploymentStatus.INITIALIZING
        self.start_time: Optional[datetime] = None
        
        # Agent and task management
        self.active_agents: Dict[str, ScrapingAgent] = {}
        self.agent_tasks: Dict[str, AgentTaskInfo] = {}
        self.pending_tasks: List[AgentTaskInfo] = []
        self.running_tasks: Dict[str, AgentTaskInfo] = {}
        self.completed_tasks: List[AgentTaskInfo] = []
        self.failed_tasks: List[AgentTaskInfo] = []
        
        # Vendor-specific management
        self.vendor_metrics: Dict[str, VendorMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        # Concurrency controls
        self.agent_semaphores: Dict[str, asyncio.Semaphore] = {}
        self.global_semaphore = asyncio.Semaphore(self.concurrency_config.max_total_agents)
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()
        
        # Monitoring and logging
        self.memory_events: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.stats = {
            'agents_deployed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'sessions_created': 0,
            'scaling_events': 0,
            'circuit_breaker_activations': 0,
            'recovery_attempts': 0
        }
        
        # Initialize factories
        self.agent_factory = AgentFactory()
        self.task_factory = TaskFactory()
        
        self._initialize_vendor_controls()
        self._log_memory_event("deployer_initialized", details={
            'max_total_agents': self.concurrency_config.max_total_agents,
            'max_concurrent_sessions': self.concurrency_config.max_concurrent_sessions,
            'auto_scaling_enabled': self.scaling_config.enable_auto_scaling
        })
    
    def _initialize_vendor_controls(self) -> None:
        """Initialize vendor-specific controls and metrics."""
        # Common vendors
        vendors = ['tesco', 'asda', 'costco']
        
        for vendor in vendors:
            # Initialize metrics
            self.vendor_metrics[vendor] = VendorMetrics(vendor=vendor)
            
            # Initialize circuit breaker
            if self.concurrency_config.enable_circuit_breaker:
                self.circuit_breakers[vendor] = CircuitBreaker(
                    failure_threshold=self.concurrency_config.circuit_breaker_threshold,
                    timeout=self.concurrency_config.circuit_breaker_timeout
                )
            
            # Initialize rate limiter
            self.rate_limiters[vendor] = RateLimiter(
                requests_per_second=self.concurrency_config.rate_limit_per_domain,
                window_size=self.concurrency_config.rate_limit_window
            )
            
            # Initialize semaphore for max agents per vendor
            self.agent_semaphores[vendor] = asyncio.Semaphore(
                self.concurrency_config.max_agents_per_vendor
            )
    
    def _log_memory_event(self, event_type: str, agent_id: Optional[str] = None,
                          task_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log a memory event for observability."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'deployer_id': 'agent_deployer',
            'agent_id': agent_id,
            'task_id': task_id,
            'details': details or {}
        }
        self.memory_events.append(event)
        self.logger.info(f"DeployerMemoryEvent: {event_type} - Agent: {agent_id}, Task: {task_id}, Details: {details}")
    
    async def start(self) -> None:
        """Start the agent deployment system."""
        if self.status != DeploymentStatus.INITIALIZING:
            raise RuntimeError(f"Cannot start deployer in {self.status} state")
        
        self.status = DeploymentStatus.RUNNING
        self.start_time = datetime.now()
        
        self._log_memory_event("deployment_started", details={
            'start_time': self.start_time.isoformat()
        })
        
        # Start background tasks
        if self.scaling_config.enable_auto_scaling:
            scaling_task = asyncio.create_task(self._scaling_monitor())
            self.background_tasks.add(scaling_task)
            scaling_task.add_done_callback(self.background_tasks.discard)
        
        # Start health monitor
        health_task = asyncio.create_task(self._health_monitor())
        self.background_tasks.add(health_task)
        health_task.add_done_callback(self.background_tasks.discard)
        
        # Start task processor
        processor_task = asyncio.create_task(self._task_processor())
        self.background_tasks.add(processor_task)
        processor_task.add_done_callback(self.background_tasks.discard)
        
        self.logger.info("Agent deployment system started successfully")
    
    async def stop(self, timeout: int = 30, force: bool = False) -> None:
        """
        Stop the agent deployment system gracefully with enhanced shutdown procedures.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown in seconds
            force: If True, force immediate shutdown without waiting
        """
        if self.status == DeploymentStatus.STOPPED:
            self.logger.info("Deployment system already stopped")
            return
            
        self.status = DeploymentStatus.STOPPING
        shutdown_start = datetime.now()
        
        self._log_memory_event("deployment_stopping", details={
            'timeout': timeout,
            'force': force,
            'running_tasks': len(self.running_tasks),
            'pending_tasks': len(self.pending_tasks)
        })
        
        try:
            # Signal shutdown to background tasks
            self.shutdown_event.set()
            
            if not force:
                # Graceful shutdown sequence
                await self._graceful_shutdown_sequence(timeout)
            else:
                # Force immediate shutdown
                await self._force_shutdown()
            
            # Cleanup and final statistics
            await self._final_cleanup()
            
            self.status = DeploymentStatus.STOPPED
            total_runtime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            shutdown_duration = (datetime.now() - shutdown_start).total_seconds()
            
            self._log_memory_event("deployment_stopped", details={
                'total_runtime': total_runtime,
                'shutdown_duration': shutdown_duration,
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'final_session_count': len(self.browserbase_manager.active_sessions),
                'graceful': not force
            })
            
            self.logger.info(f"Agent deployment system stopped {'gracefully' if not force else 'forcefully'} "
                           f"in {shutdown_duration:.2f}s")
        
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            # Emergency shutdown
            await self._emergency_shutdown()
            self.status = DeploymentStatus.ERROR
            
    async def _graceful_shutdown_sequence(self, timeout: int) -> None:
        """Execute graceful shutdown sequence with timeout."""
        timeout_time = datetime.now() + timedelta(seconds=timeout)
        
        # Phase 1: Stop accepting new tasks
        self.logger.info("Phase 1: Stopping new task acceptance")
        self.pending_tasks.clear()  # Clear pending tasks
        
        # Phase 2: Wait for running tasks to complete
        if self.running_tasks:
            self.logger.info(f"Phase 2: Waiting for {len(self.running_tasks)} running tasks to complete...")
            
            while self.running_tasks and datetime.now() < timeout_time:
                await asyncio.sleep(1)
                
                # Log progress every 5 seconds
                remaining_time = (timeout_time - datetime.now()).total_seconds()
                if int(remaining_time) % 5 == 0:
                    self.logger.info(f"Still waiting for {len(self.running_tasks)} tasks, "
                                   f"{remaining_time:.0f}s remaining")
            
            # If tasks are still running after timeout, mark them as cancelled
            if self.running_tasks:
                self.logger.warning(f"Timeout reached, cancelling {len(self.running_tasks)} remaining tasks")
                await self._cancel_remaining_tasks()
        
        # Phase 3: Cancel background monitoring tasks
        self.logger.info("Phase 3: Cancelling background tasks")
        await self._cancel_background_tasks()
        
        # Phase 4: Close all sessions
        self.logger.info("Phase 4: Closing all browser sessions")
        await self._close_all_sessions_gracefully()
    
    async def _force_shutdown(self) -> None:
        """Execute immediate force shutdown."""
        self.logger.warning("Executing force shutdown")
        
        # Cancel all running tasks immediately
        await self._cancel_remaining_tasks()
        
        # Cancel background tasks
        await self._cancel_background_tasks()
        
        # Force close all sessions
        self.browserbase_manager.close_all_sessions()
    
    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown for critical errors."""
        self.logger.critical("Executing emergency shutdown")
        
        try:
            # Force cancel all tasks
            for task in list(self.background_tasks):
                task.cancel()
            
            # Clear all data structures
            self.running_tasks.clear()
            self.pending_tasks.clear()
            self.active_agents.clear()
            
            # Emergency session cleanup
            try:
                self.browserbase_manager.close_all_sessions()
            except:
                pass  # Best effort
                
        except Exception as e:
            self.logger.critical(f"Emergency shutdown failed: {e}")
    
    async def _cancel_remaining_tasks(self) -> None:
        """Cancel all remaining running tasks."""
        cancelled_count = 0
        for task_id, task_info in list(self.running_tasks.items()):
            try:
                task_info.status = AgentTaskStatus.FAILED
                task_info.last_error = "Cancelled during shutdown"
                self.failed_tasks.append(task_info)
                del self.running_tasks[task_id]
                cancelled_count += 1
            except Exception as e:
                self.logger.error(f"Error cancelling task {task_id}: {e}")
        
        if cancelled_count > 0:
            self.logger.info(f"Cancelled {cancelled_count} running tasks")
    
    async def _cancel_background_tasks(self) -> None:
        """Cancel all background monitoring tasks."""
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        if self.background_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.background_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Background tasks didn't shut down cleanly within timeout")
    
    async def _close_all_sessions_gracefully(self) -> None:
        """Close all sessions with proper cleanup."""
        try:
            # Release sessions that are still in use
            for session_info in list(self.browserbase_manager.active_sessions.values()):
                try:
                    # Create a dummy task info for release
                    dummy_task = AgentTaskInfo(
                        task_id="shutdown_cleanup",
                        agent_id="shutdown",
                        vendor="system",
                        task_type=TaskType.COORDINATE_SCRAPING,
                        status=AgentTaskStatus.COMPLETED,
                        created_at=datetime.now()
                    )
                    await self._release_session(session_info, dummy_task)
                except Exception as e:
                    self.logger.warning(f"Error releasing session {session_info.session_id}: {e}")
            
            # Final cleanup through browserbase manager
            self.browserbase_manager.close_all_sessions()
            
        except Exception as e:
            self.logger.error(f"Error during session cleanup: {e}")
    
    async def _final_cleanup(self) -> None:
        """Perform final cleanup operations."""
        try:
            # Clear background tasks set
            self.background_tasks.clear()
            
            # Log final statistics
            self._log_final_statistics()
            
            # Export final memory events if needed
            final_events = self.get_memory_events()
            self.logger.info(f"Generated {len(final_events)} memory events during session")
            
        except Exception as e:
            self.logger.error(f"Error during final cleanup: {e}")
    
    def _log_final_statistics(self) -> None:
        """Log comprehensive final statistics."""
        try:
            total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
            success_rate = (len(self.completed_tasks) / total_tasks * 100) if total_tasks > 0 else 0
            
            final_stats = {
                'total_tasks_processed': total_tasks,
                'success_rate': success_rate,
                'agents_deployed': self.stats.get('agents_deployed', 0),
                'sessions_created': self.stats.get('sessions_created', 0),
                'scaling_events': self.stats.get('scaling_events', 0),
                'circuit_breaker_activations': self.stats.get('circuit_breaker_activations', 0),
                'recovery_attempts': self.stats.get('recovery_attempts', 0)
            }
            
            self._log_memory_event("final_statistics", details=final_stats)
            
            self.logger.info("Final Statistics:")
            for key, value in final_stats.items():
                self.logger.info(f"  {key}: {value}")
                
        except Exception as e:
            self.logger.error(f"Error logging final statistics: {e}")
    
    async def _scaling_monitor(self) -> None:
        """Monitor system load and scale agents dynamically."""
        while not self.shutdown_event.is_set():
            try:
                # Check system utilization
                utilization = self._calculate_system_utilization()
                
                if utilization > self.scaling_config.scale_up_threshold:
                    await self._scale_up()
                elif utilization < self.scaling_config.scale_down_threshold:
                    await self._scale_down()
                
                await asyncio.sleep(self.scaling_config.scale_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scaling monitor: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _health_monitor(self) -> None:
        """Monitor health of agents and sessions."""
        while not self.shutdown_event.is_set():
            try:
                # Check session health and recover failed sessions
                await self._check_session_health()
                
                # Clean up expired sessions
                await self._cleanup_expired_sessions()
                
                # Update vendor metrics
                self._update_vendor_metrics()
                
                # Check circuit breakers
                self._update_circuit_breakers()
                
                # Log health summary
                session_metrics = self.get_session_metrics()
                self._log_memory_event("health_monitor_cycle", details={
                    'session_metrics': session_metrics,
                    'running_tasks': len(self.running_tasks),
                    'pending_tasks': len(self.pending_tasks)
                })
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _task_processor(self) -> None:
        """Process pending tasks and assign them to agents."""
        while not self.shutdown_event.is_set():
            try:
                if self.pending_tasks:
                    task_info = self.pending_tasks.pop(0)
                    
                    # Check if vendor is available (circuit breaker)
                    if not self._is_vendor_available(task_info.vendor):
                        # Put task back and try later
                        self.pending_tasks.append(task_info)
                        await asyncio.sleep(5)
                        continue
                    
                    # Try to execute the task
                    await self._execute_task(task_info)
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
            except Exception as e:
                self.logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    def _calculate_system_utilization(self) -> float:
        """Calculate current system utilization."""
        active_count = len(self.running_tasks)
        max_capacity = self.concurrency_config.max_total_agents
        return active_count / max_capacity if max_capacity > 0 else 0.0
    
    async def _scale_up(self) -> None:
        """Scale up the number of agents based on demand."""
        try:
            current_agents = len(self.active_agents)
            pending_tasks = len(self.pending_tasks)
            
            # Only scale up if we have pending tasks and haven't reached max
            if pending_tasks > 0 and current_agents < self.scaling_config.max_agents:
                # Calculate how many agents to add (at most 2 at a time for gradual scaling)
                agents_to_add = min(2, 
                                   pending_tasks // 2,  # One agent per 2 pending tasks
                                   self.scaling_config.max_agents - current_agents)
                
                if agents_to_add > 0:
                    self._log_memory_event("scaling_up", details={
                        'current_agents': current_agents,
                        'pending_tasks': pending_tasks,
                        'agents_to_add': agents_to_add
                    })
                    
                    # Pre-create agents for the most demanded vendors
                    vendor_demand = self._calculate_vendor_demand()
                    
                    for _ in range(agents_to_add):
                        if vendor_demand:
                            # Create agent for vendor with highest demand
                            vendor = max(vendor_demand.items(), key=lambda x: x[1])[0]
                            await self._pre_create_agent(vendor)
                            
                            # Update demand
                            vendor_demand[vendor] = max(0, vendor_demand[vendor] - 1)
                    
                    self.stats['scaling_events'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error during scale up: {e}")
    
    async def _scale_down(self) -> None:
        """Scale down the number of agents when demand is low."""
        try:
            current_agents = len(self.active_agents)
            running_tasks = len(self.running_tasks)
            pending_tasks = len(self.pending_tasks)
            
            # Only scale down if we have more agents than needed
            if (current_agents > self.scaling_config.min_agents and 
                running_tasks + pending_tasks < current_agents * 0.5):  # Less than 50% utilization
                
                # Calculate how many agents to remove (at most 1 at a time for safety)
                max_removable = current_agents - self.scaling_config.min_agents
                agents_to_remove = min(1, max_removable)
                
                if agents_to_remove > 0:
                    self._log_memory_event("scaling_down", details={
                        'current_agents': current_agents,
                        'running_tasks': running_tasks,
                        'pending_tasks': pending_tasks,
                        'agents_to_remove': agents_to_remove
                    })
                    
                    # Remove least utilized agents
                    await self._remove_idle_agents(agents_to_remove)
                    
                    self.stats['scaling_events'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error during scale down: {e}")
    
    def _calculate_vendor_demand(self) -> Dict[str, int]:
        """Calculate demand per vendor based on pending tasks."""
        vendor_demand = {}
        
        for task_info in self.pending_tasks:
            vendor = task_info.vendor
            vendor_demand[vendor] = vendor_demand.get(vendor, 0) + 1
        
        return vendor_demand
    
    async def _pre_create_agent(self, vendor: str) -> None:
        """Pre-create an agent for a specific vendor to improve response time."""
        try:
            # Create a coordinator agent for the vendor
            agent_key = f"{vendor}_coordinator_precreated"
            
            if agent_key not in self.active_agents:
                agent = self.agent_factory.create_coordinator_agent()
                
                # Customize for vendor
                agent.config.name = f"{vendor.title()}Coordinator"
                agent.config.goal = f"Coordinate and manage {vendor} scraping operations efficiently"
                
                # Add browser tools
                await self._add_browser_tools_to_agent(agent, vendor)
                
                self.active_agents[agent_key] = agent
                self.stats['agents_deployed'] += 1
                
                self._log_memory_event("agent_pre_created", 
                                     agent_id=agent_key,
                                     details={'vendor': vendor})
                
        except Exception as e:
            self.logger.error(f"Failed to pre-create agent for {vendor}: {e}")
    
    async def _remove_idle_agents(self, count: int) -> None:
        """Remove idle agents that aren't currently processing tasks."""
        try:
            removed_count = 0
            agents_to_remove = []
            
            # Find agents not currently running tasks
            active_agent_ids = set()
            for task_info in self.running_tasks.values():
                if task_info.agent_id:
                    active_agent_ids.add(task_info.agent_id)
            
            # Look for idle agents (prefer pre-created ones)
            for agent_key in list(self.active_agents.keys()):
                if removed_count >= count:
                    break
                    
                if agent_key not in active_agent_ids:
                    agents_to_remove.append(agent_key)
                    removed_count += 1
            
            # Remove the idle agents
            for agent_key in agents_to_remove:
                del self.active_agents[agent_key]
                
                self._log_memory_event("agent_removed", 
                                     agent_id=agent_key,
                                     details={'reason': 'scaling_down'})
            
            self.logger.info(f"Removed {removed_count} idle agents during scale down")
            
        except Exception as e:
            self.logger.error(f"Error removing idle agents: {e}")
    
    async def _acquire_session(self, vendor: str, task_info: AgentTaskInfo) -> Optional[SessionInfo]:
        """
        Acquire a session from the pool for task execution.
        
        Args:
            vendor: Vendor name for session configuration
            task_info: Task information for session customization
            
        Returns:
            SessionInfo if successful, None if failed
        """
        try:
            # Create vendor-specific session configuration
            session_config = self._create_vendor_session_config(vendor)
            
            # Get session from browserbase manager (which manages the pool)
            session_info = self.browserbase_manager.get_session(session_config)
            
            # Update session last used time
            session_info.last_used = datetime.now()
            
            # Update statistics
            self.stats['sessions_created'] += 1
            
            self._log_memory_event("session_acquired",
                                 task_id=task_info.task_id,
                                 details={
                                     'session_id': session_info.session_id,
                                     'vendor': vendor,
                                     'pool_stats': self.browserbase_manager.session_pool.get_stats()
                                 })
            
            return session_info
            
        except Exception as e:
            self.logger.error(f"Failed to acquire session for vendor {vendor}: {e}")
            self._log_memory_event("session_acquisition_failed",
                                 task_id=task_info.task_id,
                                 details={'vendor': vendor, 'error': str(e)})
            return None
    
    async def _release_session(self, session_info: SessionInfo, task_info: AgentTaskInfo) -> None:
        """
        Release a session back to the pool.
        
        Args:
            session_info: Session to release
            task_info: Task information for logging
        """
        try:
            # Update session usage
            session_info.last_used = datetime.now()
            
            # Check if session had errors during task execution
            if task_info.error_count > 0:
                session_info.error_count += task_info.error_count
                session_info.last_error = task_info.last_error
            
            # Release session back to pool
            self.browserbase_manager.release_session(session_info)
            
            self._log_memory_event("session_released",
                                 task_id=task_info.task_id,
                                 details={
                                     'session_id': session_info.session_id,
                                     'session_errors': session_info.error_count,
                                     'pool_stats': self.browserbase_manager.session_pool.get_stats()
                                 })
            
        except Exception as e:
            self.logger.error(f"Failed to release session {session_info.session_id}: {e}")
            # Force close session if release fails
            try:
                self.browserbase_manager.close_session(session_info.session_id)
            except:
                pass  # Best effort cleanup
    
    def _create_vendor_session_config(self, vendor: str) -> SessionConfig:
        """
        Create a vendor-specific session configuration.
        
        Args:
            vendor: Vendor name
            
        Returns:
            SessionConfig optimized for the vendor
        """
        # Import SessionConfig if not already available
        from .browserbase_manager import SessionConfig
        
        # Base configuration
        config = SessionConfig()
        
        # Vendor-specific optimizations
        if vendor == 'tesco':
            config.stealth_mode = True
            config.viewport_width = 1920
            config.viewport_height = 1080
            config.timeout = 45000  # Tesco can be slow
        elif vendor == 'asda':
            config.stealth_mode = True
            config.viewport_width = 1366
            config.viewport_height = 768
            config.timeout = 30000
        elif vendor == 'costco':
            config.stealth_mode = False  # Costco is less strict
            config.viewport_width = 1440
            config.viewport_height = 900
            config.timeout = 35000
        else:
            # Default configuration for unknown vendors
            config.stealth_mode = True
            config.timeout = 30000
        
        return config
    
    async def _check_session_health(self) -> None:
        """Check health of all active sessions and recover if needed."""
        try:
            # Get all active sessions from running tasks
            active_session_ids = []
            for task_info in self.running_tasks.values():
                if task_info.session_id:
                    active_session_ids.append(task_info.session_id)
            
            # Check health of each active session
            for session_id in active_session_ids:
                try:
                    health_result = await self.browserbase_manager.check_session_health(session_id)
                    
                    if health_result.get('status') == 'error':
                        # Session is unhealthy, attempt recovery
                        await self._recover_session(session_id)
                        
                except Exception as e:
                    self.logger.warning(f"Health check failed for session {session_id}: {e}")
            
            # Monitor overall session pool health
            pool_stats = self.browserbase_manager.session_pool.get_stats()
            utilization = pool_stats.get('pool_utilization', 0)
            
            # Log health summary
            self._log_memory_event("health_check_completed", details={
                'active_sessions': len(active_session_ids),
                'pool_utilization': utilization,
                'pool_stats': pool_stats
            })
            
        except Exception as e:
            self.logger.error(f"Session health check failed: {e}")
    
    async def _recover_session(self, session_id: str) -> None:
        """
        Attempt to recover a failed session.
        
        Args:
            session_id: ID of the session to recover
        """
        try:
            self.logger.info(f"Attempting to recover session {session_id}")
            
            # Find tasks using this session
            affected_tasks = []
            for task_info in self.running_tasks.values():
                if task_info.session_id == session_id:
                    affected_tasks.append(task_info)
            
            # Close the failed session
            self.browserbase_manager.close_session(session_id)
            
            # For each affected task, try to get a new session
            for task_info in affected_tasks:
                try:
                    # Create new session for the task
                    new_session = await self._acquire_session(task_info.vendor, task_info)
                    if new_session:
                        task_info.session_id = new_session.session_id
                        self.logger.info(f"Task {task_info.task_id} recovered with new session {new_session.session_id}")
                    else:
                        # If we can't get a new session, mark task for retry
                        await self._handle_task_failure(task_info, Exception("Session recovery failed"))
                        
                except Exception as e:
                    self.logger.error(f"Failed to recover task {task_info.task_id}: {e}")
                    await self._handle_task_failure(task_info, e)
            
            # Update statistics
            self.stats['recovery_attempts'] += 1
            
            self._log_memory_event("session_recovery_completed", details={
                'recovered_session_id': session_id,
                'affected_tasks': len(affected_tasks)
            })
            
        except Exception as e:
            self.logger.error(f"Session recovery failed for {session_id}: {e}")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions and update metrics."""
        try:
            # Use browserbase manager's cleanup
            cleaned_count = self.browserbase_manager.cleanup_expired_sessions()
            
            if cleaned_count > 0:
                self._log_memory_event("sessions_cleaned_up", details={
                    'cleaned_sessions': cleaned_count
                })
            
        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}")
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive session metrics.
        
        Returns:
            Dictionary with session pool and usage metrics
        """
        try:
            # Get pool statistics
            pool_stats = self.browserbase_manager.session_pool.get_stats()
            
            # Get session limits information
            session_limits = self.browserbase_manager.get_session_limits()
            
            # Count sessions by vendor
            vendor_sessions = {}
            for task_info in self.running_tasks.values():
                if task_info.session_id:
                    vendor = task_info.vendor
                    vendor_sessions[vendor] = vendor_sessions.get(vendor, 0) + 1
            
            return {
                'pool_stats': pool_stats,
                'session_limits': session_limits,
                'vendor_distribution': vendor_sessions,
                'total_sessions_created': self.stats['sessions_created'],
                'recovery_attempts': self.stats['recovery_attempts']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get session metrics: {e}")
            return {'error': str(e)}
    
    def _update_vendor_metrics(self) -> None:
        """Update metrics for all vendors based on current task states."""
        try:
            current_time = datetime.now()
            
            for vendor in self.vendor_metrics:
                metrics = self.vendor_metrics[vendor]
                
                # Count active agents for this vendor
                active_agents = 0
                for agent_key in self.active_agents:
                    if vendor in agent_key:
                        active_agents += 1
                metrics.active_agents = active_agents
                
                # Count tasks by status for this vendor
                pending_tasks = 0
                running_tasks = 0
                for task_info in self.pending_tasks:
                    if task_info.vendor == vendor:
                        pending_tasks += 1
                metrics.pending_tasks = pending_tasks
                
                for task_info in self.running_tasks.values():
                    if task_info.vendor == vendor:
                        running_tasks += 1
                
                # Count completed and failed tasks
                vendor_completed = len([t for t in self.completed_tasks if t.vendor == vendor])
                vendor_failed = len([t for t in self.failed_tasks if t.vendor == vendor])
                
                metrics.completed_tasks = vendor_completed
                metrics.failed_tasks = vendor_failed
                
                # Calculate error rate
                total_tasks = vendor_completed + vendor_failed
                if total_tasks > 0:
                    metrics.error_rate = vendor_failed / total_tasks
                else:
                    metrics.error_rate = 0.0
                
                # Update circuit breaker status
                if vendor in self.circuit_breakers:
                    cb = self.circuit_breakers[vendor]
                    metrics.circuit_breaker_open = (cb.state == "open")
                    if cb.state == "open" and cb.last_failure_time:
                        metrics.circuit_breaker_until = cb.last_failure_time + timedelta(seconds=cb.timeout)
                
                # Update request count (approximate based on tasks)
                metrics.total_requests = vendor_completed + vendor_failed + running_tasks
                
                # Set last request time if there are active tasks
                if running_tasks > 0 or pending_tasks > 0:
                    metrics.last_request_time = current_time
                
                # Calculate average response time (simplified)
                if vendor_completed > 0:
                    total_duration = 0
                    count = 0
                    for task_info in self.completed_tasks[-20:]:  # Last 20 tasks
                        if (task_info.vendor == vendor and 
                            task_info.started_at and task_info.completed_at):
                            duration = (task_info.completed_at - task_info.started_at).total_seconds()
                            total_duration += duration
                            count += 1
                    
                    if count > 0:
                        metrics.avg_response_time = total_duration / count
            
            # Log metrics summary
            self._log_memory_event("vendor_metrics_updated", details={
                vendor: {
                    'active_agents': metrics.active_agents,
                    'pending_tasks': metrics.pending_tasks,
                    'error_rate': metrics.error_rate,
                    'circuit_breaker_open': metrics.circuit_breaker_open
                } for vendor, metrics in self.vendor_metrics.items()
            })
            
        except Exception as e:
            self.logger.error(f"Error updating vendor metrics: {e}")
    
    def _update_circuit_breakers(self) -> None:
        """Update circuit breaker states based on recent failures."""
        try:
            current_time = datetime.now()
            
            for vendor, circuit_breaker in self.circuit_breakers.items():
                metrics = self.vendor_metrics.get(vendor)
                if not metrics:
                    continue
                
                # Check if circuit breaker should transition from open to half-open
                if (circuit_breaker.state == "open" and 
                    circuit_breaker.last_failure_time and
                    (current_time - circuit_breaker.last_failure_time).total_seconds() >= circuit_breaker.timeout):
                    
                    circuit_breaker.state = "half-open"
                    circuit_breaker.failure_count = 0
                    
                    self._log_memory_event("circuit_breaker_half_open", details={
                        'vendor': vendor,
                        'previous_failures': circuit_breaker.failure_count
                    })
                
                # Update metrics with circuit breaker state
                metrics.circuit_breaker_open = (circuit_breaker.state == "open")
                
                # Log circuit breaker activations
                if circuit_breaker.state == "open" and not metrics.circuit_breaker_open:
                    self.stats['circuit_breaker_activations'] += 1
                    
                    self._log_memory_event("circuit_breaker_opened", details={
                        'vendor': vendor,
                        'failure_count': circuit_breaker.failure_count,
                        'error_rate': metrics.error_rate
                    })
                
        except Exception as e:
            self.logger.error(f"Error updating circuit breakers: {e}")
    
    def get_vendor_status(self) -> Dict[str, Any]:
        """
        Get comprehensive vendor status information.
        
        Returns:
            Dictionary with vendor metrics and circuit breaker states
        """
        try:
            vendor_status = {}
            
            for vendor, metrics in self.vendor_metrics.items():
                circuit_breaker = self.circuit_breakers.get(vendor)
                
                vendor_status[vendor] = {
                    'active_agents': metrics.active_agents,
                    'pending_tasks': metrics.pending_tasks,
                    'completed_tasks': metrics.completed_tasks,
                    'failed_tasks': metrics.failed_tasks,
                    'error_rate': round(metrics.error_rate * 100, 2),  # As percentage
                    'avg_response_time': round(metrics.avg_response_time, 2),
                    'total_requests': metrics.total_requests,
                    'last_request_time': metrics.last_request_time.isoformat() if metrics.last_request_time else None,
                    'circuit_breaker': {
                        'open': metrics.circuit_breaker_open,
                        'state': circuit_breaker.state if circuit_breaker else 'unknown',
                        'failure_count': circuit_breaker.failure_count if circuit_breaker else 0,
                        'until': metrics.circuit_breaker_until.isoformat() if metrics.circuit_breaker_until else None
                    } if circuit_breaker else None
                }
            
            return vendor_status
            
        except Exception as e:
            self.logger.error(f"Error getting vendor status: {e}")
            return {'error': str(e)}
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """
        Get scaling and performance metrics.
        
        Returns:
            Dictionary with scaling information
        """
        try:
            return {
                'scaling_config': {
                    'auto_scaling_enabled': self.scaling_config.enable_auto_scaling,
                    'min_agents': self.scaling_config.min_agents,
                    'max_agents': self.scaling_config.max_agents,
                    'scale_up_threshold': self.scaling_config.scale_up_threshold,
                    'scale_down_threshold': self.scaling_config.scale_down_threshold
                },
                'current_metrics': {
                    'total_agents': len(self.active_agents),
                    'system_utilization': self._calculate_system_utilization(),
                    'pending_tasks': len(self.pending_tasks),
                    'running_tasks': len(self.running_tasks),
                    'scaling_events': self.stats['scaling_events']
                },
                'concurrency_limits': {
                    'max_total_agents': self.concurrency_config.max_total_agents,
                    'max_agents_per_vendor': self.concurrency_config.max_agents_per_vendor,
                    'max_concurrent_sessions': self.concurrency_config.max_concurrent_sessions,
                    'rate_limit_per_domain': self.concurrency_config.rate_limit_per_domain
                },
                'vendor_demand': self._calculate_vendor_demand()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting scaling metrics: {e}")
            return {'error': str(e)}
    
    def _is_vendor_available(self, vendor: str) -> bool:
        """Check if vendor is available for new tasks."""
        if vendor in self.circuit_breakers:
            return self.circuit_breakers[vendor].can_execute()
        return True
    
    async def _execute_task(self, task_info: AgentTaskInfo) -> None:
        """
        Execute a specific task with an agent, including session management.
        
        Args:
            task_info: Information about the task to execute
        """
        vendor = task_info.vendor
        session_info: Optional[SessionInfo] = None
        
        try:
            # Acquire vendor-specific semaphore
            async with self.agent_semaphores[vendor]:
                # Acquire global semaphore
                async with self.global_semaphore:
                    # Apply rate limiting
                    await self.rate_limiters[vendor].acquire()
                    
                    # Acquire session from pool
                    session_info = await self._acquire_session(vendor, task_info)
                    task_info.session_id = session_info.session_id if session_info else None
                    
                    # Update task status
                    task_info.status = AgentTaskStatus.RUNNING
                    task_info.started_at = datetime.now()
                    self.running_tasks[task_info.task_id] = task_info
                    
                    self._log_memory_event("task_execution_started", 
                                         agent_id=task_info.agent_id,
                                         task_id=task_info.task_id,
                                         details={
                                             'vendor': vendor, 
                                             'task_type': task_info.task_type.value,
                                             'session_id': task_info.session_id
                                         })
                    
                    # Get or create agent for this task
                    agent = await self._get_or_create_agent(task_info)
                    
                    # Create scraping task
                    scraping_task = await self._create_scraping_task(task_info, agent)
                    
                    # Create crew and execute asynchronously
                    crew = ScrapingCrew([agent], [scraping_task])
                    
                    # Execute the crew asynchronously using CrewAI 0.150.0
                    result = await crew.execute_async()
                    
                    # Task completed successfully
                    task_info.status = AgentTaskStatus.COMPLETED
                    task_info.completed_at = datetime.now()
                    
                    # Remove from running tasks and add to completed
                    if task_info.task_id in self.running_tasks:
                        del self.running_tasks[task_info.task_id]
                    self.completed_tasks.append(task_info)
                    
                    # Update statistics
                    self.stats['tasks_completed'] += 1
                    
                    # Record success for circuit breaker
                    if vendor in self.circuit_breakers:
                        self.circuit_breakers[vendor].record_success()
                    
                    self._log_memory_event("task_execution_completed",
                                         agent_id=task_info.agent_id,
                                         task_id=task_info.task_id,
                                         details={
                                             'duration': (task_info.completed_at - task_info.started_at).total_seconds(),
                                             'result_length': len(result) if result else 0,
                                             'session_id': task_info.session_id
                                         })
        
        except Exception as e:
            # Handle task failure
            await self._handle_task_failure(task_info, e)
        
        finally:
            # Always release session back to pool
            if session_info:
                await self._release_session(session_info, task_info)
    
    async def _get_or_create_agent(self, task_info: AgentTaskInfo) -> ScrapingAgent:
        """
        Get existing agent or create a new one for the task.
        
        Args:
            task_info: Information about the task
            
        Returns:
            ScrapingAgent instance
        """
        # Check if we have an existing agent for this vendor/type
        agent_key = f"{task_info.vendor}_{task_info.task_type.value}"
        
        if agent_key not in self.active_agents:
            # Create new agent using factory
            if task_info.task_type in [TaskType.SCRAPE_PRODUCTS, TaskType.SCRAPE_CATEGORIES]:
                agent = self.agent_factory.create_scraper_agent(task_info.vendor)
            elif task_info.task_type == TaskType.ANALYZE_DATA:
                agent = self.agent_factory.create_analyzer_agent()
            elif task_info.task_type == TaskType.VALIDATE_RESULTS:
                agent = self.agent_factory.create_validator_agent()
            else:
                agent = self.agent_factory.create_coordinator_agent()
            
            # Add browser tools to agent
            await self._add_browser_tools_to_agent(agent, task_info.vendor)
            
            self.active_agents[agent_key] = agent
            self.stats['agents_deployed'] += 1
            
            self._log_memory_event("agent_created", 
                                 agent_id=agent_key,
                                 details={'vendor': task_info.vendor, 'task_type': task_info.task_type.value})
        
        return self.active_agents[agent_key]
    
    async def _add_browser_tools_to_agent(self, agent: ScrapingAgent, vendor: str) -> None:
        """
        Add browser tools to an agent with vendor-specific configuration.
        
        Args:
            agent: Agent to add tools to
            vendor: Vendor name for tool configuration
        """
        # Import browser tools
        from .browser_tools import (
            NavigationTool, InteractionTool, ExtractionTool, 
            ScreenshotTool, AntiBotConfig
        )
        
        # Create anti-bot configuration for vendor
        anti_bot_config = AntiBotConfig(
            enable_random_delays=True,
            enable_human_mouse=True,
            enable_user_agent_rotation=True,
            enable_stealth_mode=True
        )
        
        # Create browser tools with manager and anti-bot config
        tools = [
            NavigationTool(
                self.browserbase_manager,
                anti_bot_config=anti_bot_config
            ),
            InteractionTool(
                self.browserbase_manager,
                anti_bot_config=anti_bot_config
            ),
            ExtractionTool(
                self.browserbase_manager,
                anti_bot_config=anti_bot_config
            ),
            ScreenshotTool(
                self.browserbase_manager,
                anti_bot_config=anti_bot_config
            )
        ]
        
        # Add tools to agent's configuration
        if agent.config.tools is None:
            agent.config.tools = []
        agent.config.tools.extend(tools)
        
        # Recreate the agent with new tools
        agent.agent = agent._create_agent()
    
    async def _create_scraping_task(self, task_info: AgentTaskInfo, agent: ScrapingAgent) -> ScrapingTask:
        """
        Create a scraping task based on task information.
        
        Args:
            task_info: Information about the task
            agent: Agent to assign the task to
            
        Returns:
            ScrapingTask instance
        """
        # Get task data from the task info
        task_data = getattr(task_info, 'task_data', {}) or {}
        
        # Create task using factory based on task type
        if task_info.task_type == TaskType.SCRAPE_PRODUCTS:
            # Use actual URL from task data
            url = task_data.get('url', f"https://{task_info.vendor}.com/products")
            urls = [url] if isinstance(url, str) else url if isinstance(url, list) else [str(url)]
            scraping_task = self.task_factory.create_scrape_products_task(
                agent.get_agent(), urls, task_info.vendor
            )
        elif task_info.task_type == TaskType.ANALYZE_DATA:
            scraping_task = self.task_factory.create_analyze_data_task(
                agent.get_agent(), task_data
            )
        elif task_info.task_type == TaskType.VALIDATE_RESULTS:
            scraping_task = self.task_factory.create_validate_results_task(
                agent.get_agent(), task_data
            )
        else:
            # Create a generic task for coordination
            config = TaskConfig(
                task_type=task_info.task_type,
                description=f"Coordinate {task_info.vendor} scraping operations",
                expected_output="Coordination results and status updates",
                agent=agent.get_agent(),
                async_execution=True
            )
            scraping_task = ScrapingTask(config)
        
        return scraping_task
    
    async def _handle_task_failure(self, task_info: AgentTaskInfo, error: Exception) -> None:
        """
        Handle task execution failure with retry logic.
        
        Args:
            task_info: Information about the failed task
            error: Exception that caused the failure
        """
        task_info.error_count += 1
        task_info.last_error = str(error)
        task_info.status = AgentTaskStatus.FAILED
        
        # Remove from running tasks
        if task_info.task_id in self.running_tasks:
            del self.running_tasks[task_info.task_id]
        
        # Record failure for circuit breaker
        vendor = task_info.vendor
        if vendor in self.circuit_breakers:
            self.circuit_breakers[vendor].record_failure()
        
        # Update statistics
        self.stats['tasks_failed'] += 1
        
        # Check if we should retry
        if task_info.retry_count < task_info.max_retries:
            task_info.retry_count += 1
            task_info.status = AgentTaskStatus.RETRYING
            
            # Add back to pending tasks for retry
            self.pending_tasks.append(task_info)
            
            self._log_memory_event("task_retry_scheduled",
                                 agent_id=task_info.agent_id,
                                 task_id=task_info.task_id,
                                 details={
                                     'retry_count': task_info.retry_count,
                                     'max_retries': task_info.max_retries,
                                     'error': str(error)
                                 })
        else:
            # Max retries exceeded, mark as permanently failed
            self.failed_tasks.append(task_info)
            
            self._log_memory_event("task_execution_failed",
                                 agent_id=task_info.agent_id,
                                 task_id=task_info.task_id,
                                 details={
                                     'error': str(error),
                                     'retry_count': task_info.retry_count,
                                     'final_failure': True
                                 })
    
    async def add_task(self, vendor: str, task_type: TaskType, 
                       task_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new task to the deployment queue.
        
        Args:
            vendor: Vendor name (e.g., 'tesco', 'asda', 'costco')
            task_type: Type of task to perform
            task_data: Optional data for the task
            
        Returns:
            Task ID of the created task
        """
        task_id = f"{vendor}_{task_type.value}_{int(time.time() * 1000)}"
        agent_id = f"{vendor}_{task_type.value}_agent"
        
        task_info = AgentTaskInfo(
            task_id=task_id,
            agent_id=agent_id,
            vendor=vendor,
            task_type=task_type,
            status=AgentTaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        # Store task data if provided
        if task_data:
            task_info.task_data = task_data
        
        self.pending_tasks.append(task_info)
        self.agent_tasks[task_id] = task_info
        
        self._log_memory_event("task_added",
                             task_id=task_id,
                             details={
                                 'vendor': vendor,
                                 'task_type': task_type.value,
                                 'queue_size': len(self.pending_tasks)
                             })
        
        return task_id
    
    async def add_task_batch(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple tasks to the deployment queue.
        
        Args:
            tasks: List of task dictionaries with 'vendor', 'task_type', and optional 'task_data'
            
        Returns:
            List of task IDs created
        """
        task_ids = []
        
        for task_dict in tasks:
            vendor = task_dict['vendor']
            task_type = TaskType(task_dict['task_type'])
            task_data = task_dict.get('task_data')
            
            task_id = await self.add_task(vendor, task_type, task_data)
            task_ids.append(task_id)
        
        self._log_memory_event("batch_tasks_added", details={
            'batch_size': len(tasks),
            'task_ids': task_ids
        })
        
        return task_ids
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a specific task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            Task status information or None if not found
        """
        if task_id not in self.agent_tasks:
            return None
        
        task_info = self.agent_tasks[task_id]
        return {
            'task_id': task_info.task_id,
            'agent_id': task_info.agent_id,
            'vendor': task_info.vendor,
            'task_type': task_info.task_type.value,
            'status': task_info.status.value,
            'created_at': task_info.created_at.isoformat(),
            'started_at': task_info.started_at.isoformat() if task_info.started_at else None,
            'completed_at': task_info.completed_at.isoformat() if task_info.completed_at else None,
            'error_count': task_info.error_count,
            'last_error': task_info.last_error,
            'retry_count': task_info.retry_count,
            'session_id': task_info.session_id
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        if task_id not in self.agent_tasks:
            return False
        
        task_info = self.agent_tasks[task_id]
        
        # Can only cancel pending or running tasks
        if task_info.status in [AgentTaskStatus.COMPLETED, AgentTaskStatus.FAILED]:
            return False
        
        # Remove from pending tasks if present
        self.pending_tasks = [t for t in self.pending_tasks if t.task_id != task_id]
        
        # Remove from running tasks if present
        if task_id in self.running_tasks:
            del self.running_tasks[task_id]
        
        # Mark as cancelled (using FAILED status with specific error)
        task_info.status = AgentTaskStatus.FAILED
        task_info.last_error = "Task cancelled by user"
        self.failed_tasks.append(task_info)
        
        self._log_memory_event("task_cancelled", task_id=task_id)
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status with all metrics."""
        return {
            'deployment': {
                'status': self.status.value,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'system_utilization': self._calculate_system_utilization()
            },
            'tasks': {
                'pending': len(self.pending_tasks),
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks),
                'total_processed': len(self.completed_tasks) + len(self.failed_tasks)
            },
            'agents': {
                'active_count': len(self.active_agents),
                'by_vendor': self.get_vendor_status()
            },
            'sessions': self.get_session_metrics(),
            'scaling': self.get_scaling_metrics(),
            'performance': {
                'success_rate': (len(self.completed_tasks) / max(1, len(self.completed_tasks) + len(self.failed_tasks))) * 100,
                'avg_task_duration': self._calculate_avg_task_duration(),
                'tasks_per_minute': self._calculate_tasks_per_minute()
            },
            'stats': self.stats
        }
    
    def _calculate_avg_task_duration(self) -> float:
        """Calculate average task duration for completed tasks."""
        try:
            if not self.completed_tasks:
                return 0.0
            
            total_duration = 0
            count = 0
            
            for task_info in self.completed_tasks[-50:]:  # Last 50 tasks
                if task_info.started_at and task_info.completed_at:
                    duration = (task_info.completed_at - task_info.started_at).total_seconds()
                    total_duration += duration
                    count += 1
            
            return total_duration / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating average task duration: {e}")
            return 0.0
    
    def _calculate_tasks_per_minute(self) -> float:
        """Calculate tasks per minute based on recent completion rate."""
        try:
            if not self.completed_tasks:
                return 0.0
            
            now = datetime.now()
            one_hour_ago = now - timedelta(hours=1)
            
            # Count tasks completed in the last hour
            recent_tasks = [
                task for task in self.completed_tasks 
                if task.completed_at and task.completed_at > one_hour_ago
            ]
            
            if not recent_tasks:
                return 0.0
            
            # Calculate rate based on actual time span
            earliest_time = min(task.completed_at for task in recent_tasks)
            time_span_minutes = (now - earliest_time).total_seconds() / 60
            
            return len(recent_tasks) / time_span_minutes if time_span_minutes > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating tasks per minute: {e}")
            return 0.0
    
    def get_memory_events(self) -> List[Dict[str, Any]]:
        """Get all memory events for observability."""
        return self.memory_events
    
    @asynccontextmanager
    async def deployment_context(self):
        """Context manager for automatic deployment lifecycle management."""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


def create_agent_deployer(browserbase_manager: BrowserbaseManager,
                         concurrency_config: Optional[ConcurrencyConfig] = None,
                         scaling_config: Optional[ScalingConfig] = None) -> AgentDeployer:
    """
    Create and return a new AgentDeployer instance.
    
    Args:
        browserbase_manager: Manager for Browserbase sessions
        concurrency_config: Configuration for concurrency controls
        scaling_config: Configuration for dynamic scaling
        
    Returns:
        AgentDeployer instance
    """
    return AgentDeployer(
        browserbase_manager=browserbase_manager,
        concurrency_config=concurrency_config,
        scaling_config=scaling_config
    )