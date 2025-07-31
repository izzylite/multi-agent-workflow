"""
Intelligent Load Balancing System

Provides intelligent load balancing capabilities for distributing tasks across agents
based on capabilities, performance, and availability.
"""

import asyncio
import logging
import time
import heapq
from typing import List, Optional, Dict, Any, Set, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import threading

from .agent_deployer import AgentTaskInfo, AgentTaskStatus, TaskType
from .crewai_integration import ScrapingAgent, TaskType as CrewAITaskType
from .browserbase_manager import SessionInfo
from .exceptions import ConfigurationError


class LoadBalancingStrategy(Enum):
    """Available load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"
    AVAILABILITY_BASED = "availability_based"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class LoadBalancerConfig:
    """Configuration for the load balancer."""
    default_strategy: LoadBalancingStrategy = LoadBalancingStrategy.HYBRID
    enable_priority_queue: bool = True
    max_queue_size: int = 1000
    priority_adjustment_interval: int = 30  # seconds
    performance_history_window: int = 300  # seconds
    enable_preemption: bool = True
    preemption_threshold: float = 0.8  # priority difference threshold
    enable_adaptive_rate_limiting: bool = True
    feedback_loop_interval: int = 60  # seconds
    enable_ml_optimization: bool = False  # for future ML-based optimization


@dataclass
class AgentCapability:
    """Agent capability information."""
    agent_id: str
    vendor: str
    task_types: Set[TaskType]
    performance_score: float = 1.0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    session_available: bool = True
    last_activity: Optional[datetime] = None
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0


@dataclass
class TaskInfo:
    """Enhanced task information for load balancing."""
    task_id: str
    vendor: str
    task_type: TaskType
    priority: TaskPriority
    created_at: datetime
    assigned_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    estimated_duration: Optional[float] = None
    importance_score: float = 1.0
    vendor_response_time: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for load balancing decisions."""
    agent_id: str
    vendor: str
    avg_task_duration: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    throughput: float = 0.0  # tasks per minute
    last_updated: datetime = field(default_factory=datetime.now)
    session_availability: float = 1.0
    response_time_variance: float = 0.0


class PriorityQueue:
    """Thread-safe priority queue for task management."""
    
    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
        self._task_map = {}  # task_id -> (priority, task_info)
    
    def put(self, task_info: TaskInfo) -> None:
        """Add a task to the priority queue."""
        with self._lock:
            # Priority is (priority_level, creation_time) for FIFO within same priority
            priority_tuple = (task_info.priority.value, task_info.created_at.timestamp())
            heapq.heappush(self._queue, (priority_tuple, task_info))
            self._task_map[task_info.task_id] = (priority_tuple, task_info)
    
    def get(self) -> Optional[TaskInfo]:
        """Get the highest priority task from the queue."""
        with self._lock:
            if not self._queue:
                return None
            
            priority_tuple, task_info = heapq.heappop(self._queue)
            if task_info.task_id in self._task_map:
                del self._task_map[task_info.task_id]
            return task_info
    
    def peek(self) -> Optional[TaskInfo]:
        """Peek at the highest priority task without removing it."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0][1]
    
    def remove(self, task_id: str) -> bool:
        """Remove a specific task from the queue."""
        with self._lock:
            if task_id not in self._task_map:
                return False
            
            # Remove from map
            del self._task_map[task_id]
            
            # Rebuild queue without the removed task
            new_queue = []
            for priority_tuple, task_info in self._queue:
                if task_info.task_id != task_id:
                    heapq.heappush(new_queue, (priority_tuple, task_info))
            
            self._queue = new_queue
            return True
    
    def update_priority(self, task_id: str, new_priority: TaskPriority) -> bool:
        """Update the priority of a task in the queue."""
        with self._lock:
            if task_id not in self._task_map:
                return False
            
            # Remove old entry
            old_priority_tuple, task_info = self._task_map[task_id]
            del self._task_map[task_id]
            
            # Update priority
            task_info.priority = new_priority
            new_priority_tuple = (new_priority.value, task_info.created_at.timestamp())
            
            # Rebuild queue with updated priority
            new_queue = []
            for priority_tuple, existing_task in self._queue:
                if existing_task.task_id != task_id:
                    heapq.heappush(new_queue, (priority_tuple, existing_task))
            
            # Add back with new priority
            heapq.heappush(new_queue, (new_priority_tuple, task_info))
            self._task_map[task_id] = (new_priority_tuple, task_info)
            self._queue = new_queue
            return True
    
    def size(self) -> int:
        """Get the current size of the queue."""
        with self._lock:
            return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0
    
    def get_all_tasks(self) -> List[TaskInfo]:
        """Get all tasks in the queue (for monitoring/debugging)."""
        with self._lock:
            return [task_info for _, task_info in self._queue]


class LoadBalancer:
    """Intelligent load balancer for distributing tasks across agents."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.task_queue = PriorityQueue()
        self.agent_capabilities: Dict[str, AgentCapability] = {}
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.vendor_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # State tracking
        self.current_strategy = self.config.default_strategy
        self.agent_assignments: Dict[str, List[str]] = defaultdict(list)  # agent_id -> [task_ids]
        self.task_history: List[TaskInfo] = []
        self.feedback_data: List[Dict[str, Any]] = []
        
        # Background tasks
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        
        # Performance tracking
        self._last_priority_adjustment = datetime.now()
        self._last_feedback_update = datetime.now()
        self._total_tasks_processed = 0
        self._total_tasks_failed = 0
        
        self.logger.info(f"LoadBalancer initialized with strategy: {self.current_strategy.value}")
    
    async def start(self) -> None:
        """Start the load balancer and background tasks."""
        if self._running:
            return
        
        self._running = True
        self.logger.info("Starting LoadBalancer...")
        
        # Start background tasks
        self._background_tasks.add(
            asyncio.create_task(self._priority_adjustment_loop())
        )
        self._background_tasks.add(
            asyncio.create_task(self._feedback_loop())
        )
        self._background_tasks.add(
            asyncio.create_task(self._performance_monitoring_loop())
        )
        
        self.logger.info("LoadBalancer started successfully")
    
    async def stop(self) -> None:
        """Stop the load balancer and clean up resources."""
        if not self._running:
            return
        
        self._running = False
        self.logger.info("Stopping LoadBalancer...")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        self.logger.info("LoadBalancer stopped")
    
    def register_agent(self, agent_id: str, vendor: str, 
                      task_types: Set[TaskType], 
                      initial_performance_score: float = 1.0) -> None:
        """Register an agent with the load balancer."""
        capability = AgentCapability(
            agent_id=agent_id,
            vendor=vendor,
            task_types=task_types,
            performance_score=initial_performance_score
        )
        
        self.agent_capabilities[agent_id] = capability
        
        # Initialize performance metrics
        self.performance_metrics[agent_id] = PerformanceMetrics(
            agent_id=agent_id,
            vendor=vendor
        )
        
        self.logger.info(f"Registered agent {agent_id} for vendor {vendor}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the load balancer."""
        if agent_id in self.agent_capabilities:
            del self.agent_capabilities[agent_id]
        
        if agent_id in self.performance_metrics:
            del self.performance_metrics[agent_id]
        
        # Remove from assignments
        if agent_id in self.agent_assignments:
            del self.agent_assignments[agent_id]
        
        self.logger.info(f"Unregistered agent {agent_id}")
    
    def add_task(self, task_info: TaskInfo) -> None:
        """Add a task to the load balancer queue."""
        if self.task_queue.size() >= self.config.max_queue_size:
            self.logger.warning(f"Task queue full, rejecting task {task_info.task_id}")
            return
        
        self.task_queue.put(task_info)
        self.logger.info(f"Added task {task_info.task_id} with priority {task_info.priority.value}")
    
    def get_next_task(self, available_agents: List[str]) -> Optional[Tuple[TaskInfo, str]]:
        """
        Get the next task and assign it to the best available agent.
        
        Args:
            available_agents: List of agent IDs that are available for assignment
            
        Returns:
            Tuple of (TaskInfo, agent_id) or None if no suitable task/agent
        """
        if self.task_queue.is_empty() or not available_agents:
            return None
        
        # Get the highest priority task
        task_info = self.task_queue.get()
        if not task_info:
            return None
        
        # Select the best agent based on current strategy
        selected_agent = self._select_agent(task_info, available_agents)
        if not selected_agent:
            # No suitable agent found, put task back in queue
            self.task_queue.put(task_info)
            return None
        
        # Update task assignment
        task_info.assigned_at = datetime.now()
        task_info.assigned_agent = selected_agent
        
        # Track assignment
        self.agent_assignments[selected_agent].append(task_info.task_id)
        
        self.logger.info(f"Assigned task {task_info.task_id} to agent {selected_agent}")
        return task_info, selected_agent
    
    def preempt_task(self, task_id: str, reason: str = "High priority task") -> bool:
        """
        Preempt a running task to make room for higher priority tasks.
        
        Args:
            task_id: ID of the task to preempt
            reason: Reason for preemption
            
        Returns:
            True if task was preempted, False if not found or not running
        """
        # Find the task in the queue
        all_tasks = self.task_queue.get_all_tasks()
        task_to_preempt = None
        
        for task_info in all_tasks:
            if task_info.task_id == task_id:
                task_to_preempt = task_info
                break
        
        if not task_to_preempt:
            return False
        
        # Remove task from queue
        if self.task_queue.remove(task_id):
            # Add back to queue with lower priority
            task_to_preempt.priority = self._demote_priority(task_to_preempt.priority)
            self.task_queue.put(task_to_preempt)
            
            self.logger.info(f"Preempted task {task_id}: {reason}")
            return True
        
        return False
    
    def handle_task_failure(self, task_id: str, error_message: str, 
                          agent_id: Optional[str] = None) -> bool:
        """
        Handle task failure with retry logic and error tracking.
        
        Args:
            task_id: ID of the failed task
            error_message: Error message describing the failure
            agent_id: ID of the agent that failed (optional)
            
        Returns:
            True if task was requeued for retry, False if max retries exceeded
        """
        # Find the task in the queue
        all_tasks = self.task_queue.get_all_tasks()
        failed_task = None
        
        for task_info in all_tasks:
            if task_info.task_id == task_id:
                failed_task = task_info
                break
        
        if not failed_task:
            self.logger.warning(f"Failed task {task_id} not found in queue")
            return False
        
        # Update retry count
        failed_task.retry_count += 1
        
        # Check if max retries exceeded
        if failed_task.retry_count >= failed_task.max_retries:
            self.logger.error(f"Task {task_id} failed permanently after {failed_task.retry_count} retries")
            self._total_tasks_failed += 1
            return False
        
        # Update agent performance if agent_id provided
        if agent_id:
            self.update_agent_performance(agent_id, 0.0, False, error_message)
        
        # Requeue with higher priority (failed tasks get priority)
        if self.task_queue.remove(task_id):
            failed_task.priority = self._promote_priority(failed_task.priority)
            self.task_queue.put(failed_task)
            
            self.logger.info(f"Requeued failed task {task_id} for retry {failed_task.retry_count}")
            return True
        
        return False
    
    def reassign_task(self, task_id: str, new_agent_id: str, reason: str = "Agent failure") -> bool:
        """
        Reassign a task to a different agent.
        
        Args:
            task_id: ID of the task to reassign
            new_agent_id: ID of the new agent
            reason: Reason for reassignment
            
        Returns:
            True if task was reassigned, False if not found
        """
        # Find the task in the queue
        all_tasks = self.task_queue.get_all_tasks()
        task_to_reassign = None
        
        for task_info in all_tasks:
            if task_info.task_id == task_id:
                task_to_reassign = task_info
                break
        
        if not task_to_reassign:
            return False
        
        # Update assignment
        old_agent = task_to_reassign.assigned_agent
        task_to_reassign.assigned_agent = new_agent_id
        task_to_reassign.assigned_at = datetime.now()
        
        # Update agent assignments
        if old_agent and old_agent in self.agent_assignments:
            self.agent_assignments[old_agent] = [
                tid for tid in self.agent_assignments[old_agent] 
                if tid != task_id
            ]
        
        if new_agent_id not in self.agent_assignments:
            self.agent_assignments[new_agent_id] = []
        self.agent_assignments[new_agent_id].append(task_id)
        
        self.logger.info(f"Reassigned task {task_id} from {old_agent} to {new_agent_id}: {reason}")
        return True
    
    def get_failed_tasks_summary(self) -> Dict[str, Any]:
        """Get a summary of failed tasks and retry statistics."""
        all_tasks = self.task_queue.get_all_tasks()
        failed_tasks = [t for t in all_tasks if t.retry_count > 0]
        
        return {
            'total_failed_tasks': len(failed_tasks),
            'total_retries': sum(t.retry_count for t in failed_tasks),
            'avg_retries_per_task': sum(t.retry_count for t in failed_tasks) / max(len(failed_tasks), 1),
            'tasks_at_max_retries': sum(1 for t in failed_tasks if t.retry_count >= t.max_retries),
            'failed_tasks_by_vendor': defaultdict(int, {
                t.vendor: sum(1 for ft in failed_tasks if ft.vendor == t.vendor)
                for t in failed_tasks
            })
        }
    
    def emergency_shutdown(self) -> Dict[str, Any]:
        """
        Emergency shutdown - clear all tasks and return summary.
        
        Returns:
            Summary of cleared tasks
        """
        all_tasks = self.task_queue.get_all_tasks()
        task_summary = {
            'total_tasks_cleared': len(all_tasks),
            'tasks_by_priority': defaultdict(int),
            'tasks_by_vendor': defaultdict(int),
            'tasks_by_status': {
                'pending': len(all_tasks),
                'assigned': sum(1 for t in all_tasks if t.assigned_agent),
                'retry': sum(1 for t in all_tasks if t.retry_count > 0)
            }
        }
        
        # Count tasks by priority and vendor
        for task_info in all_tasks:
            task_summary['tasks_by_priority'][task_info.priority.value] += 1
            task_summary['tasks_by_vendor'][task_info.vendor] += 1
        
        # Clear all tasks
        while not self.task_queue.is_empty():
            self.task_queue.get()
        
        # Clear agent assignments
        self.agent_assignments.clear()
        
        self.logger.warning(f"Emergency shutdown completed. Cleared {len(all_tasks)} tasks.")
        return task_summary
    
    def bulk_task_operation(self, operation: str, task_ids: List[str], 
                          **kwargs) -> Dict[str, Any]:
        """
        Perform bulk operations on multiple tasks.
        
        Args:
            operation: Operation to perform ('preempt', 'reassign', 'update_priority')
            task_ids: List of task IDs to operate on
            **kwargs: Additional arguments for the operation
            
        Returns:
            Summary of operation results
        """
        results = {
            'operation': operation,
            'total_tasks': len(task_ids),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for task_id in task_ids:
            try:
                if operation == 'preempt':
                    success = self.preempt_task(task_id, kwargs.get('reason', 'Bulk preemption'))
                elif operation == 'reassign':
                    new_agent = kwargs.get('new_agent_id')
                    if not new_agent:
                        results['errors'].append(f"Missing new_agent_id for task {task_id}")
                        results['failed'] += 1
                        continue
                    success = self.reassign_task(task_id, new_agent, kwargs.get('reason', 'Bulk reassignment'))
                elif operation == 'update_priority':
                    new_priority = kwargs.get('new_priority')
                    if not new_priority:
                        results['errors'].append(f"Missing new_priority for task {task_id}")
                        results['failed'] += 1
                        continue
                    success = self.task_queue.update_priority(task_id, new_priority)
                else:
                    results['errors'].append(f"Unknown operation: {operation}")
                    results['failed'] += 1
                    continue
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                results['errors'].append(f"Error processing task {task_id}: {str(e)}")
                results['failed'] += 1
        
        self.logger.info(f"Bulk operation '{operation}' completed: "
                        f"{results['successful']} successful, {results['failed']} failed")
        return results
    
    def update_agent_performance(self, agent_id: str, 
                               task_duration: float, 
                               success: bool, 
                               error_message: Optional[str] = None) -> None:
        """Update performance metrics for an agent."""
        if agent_id not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[agent_id]
        
        # Update basic metrics
        if success:
            metrics.avg_task_duration = (
                (metrics.avg_task_duration * metrics.total_tasks_completed + task_duration) /
                (metrics.total_tasks_completed + 1)
            )
            metrics.success_rate = (
                (metrics.success_rate * metrics.total_tasks_completed + 1) /
                (metrics.total_tasks_completed + 1)
            )
            metrics.total_tasks_completed += 1
        else:
            metrics.error_rate = (
                (metrics.error_rate * metrics.total_tasks_failed + 1) /
                (metrics.total_tasks_failed + 1)
            )
            metrics.total_tasks_failed += 1
        
        # Update throughput (tasks per minute)
        metrics.throughput = metrics.total_tasks_completed / max(1, 
            (datetime.now() - metrics.last_updated).total_seconds() / 60)
        
        metrics.last_updated = datetime.now()
        
        # Update agent capability
        if agent_id in self.agent_capabilities:
            capability = self.agent_capabilities[agent_id]
            capability.performance_score = metrics.success_rate
            capability.error_rate = metrics.error_rate
            capability.avg_response_time = metrics.avg_task_duration
            capability.last_activity = datetime.now()
        
        self.logger.debug(f"Updated performance for agent {agent_id}: "
                         f"success_rate={metrics.success_rate:.3f}, "
                         f"avg_duration={metrics.avg_task_duration:.3f}")
    
    def update_session_availability(self, agent_id: str, session_available: bool) -> None:
        """Update session availability for an agent."""
        if agent_id in self.agent_capabilities:
            self.agent_capabilities[agent_id].session_available = session_available
        
        if agent_id in self.performance_metrics:
            self.performance_metrics[agent_id].session_availability = 1.0 if session_available else 0.0
    
    def update_task_importance(self, task_id: str, importance_score: float) -> bool:
        """Update the importance score of a task in the queue."""
        all_tasks = self.task_queue.get_all_tasks()
        for task_info in all_tasks:
            if task_info.task_id == task_id:
                task_info.importance_score = importance_score
                self.logger.info(f"Updated importance score for task {task_id}: {importance_score}")
                return True
        return False
    
    def update_vendor_response_time(self, task_id: str, response_time: float) -> bool:
        """Update the vendor response time for a task."""
        all_tasks = self.task_queue.get_all_tasks()
        for task_info in all_tasks:
            if task_info.task_id == task_id:
                task_info.vendor_response_time = response_time
                self.logger.debug(f"Updated vendor response time for task {task_id}: {response_time}")
                return True
        return False
    
    def get_priority_statistics(self) -> Dict[str, Any]:
        """Get statistics about task priorities in the queue."""
        all_tasks = self.task_queue.get_all_tasks()
        priority_counts = defaultdict(int)
        
        for task_info in all_tasks:
            priority_counts[task_info.priority.value] += 1
        
        return {
            'total_tasks': len(all_tasks),
            'priority_distribution': dict(priority_counts),
            'avg_importance_score': sum(t.importance_score for t in all_tasks) / max(len(all_tasks), 1),
            'tasks_with_dependencies': sum(1 for t in all_tasks if t.dependencies),
            'retry_tasks': sum(1 for t in all_tasks if t.retry_count > 0)
        }
    
    def set_strategy(self, strategy: LoadBalancingStrategy) -> None:
        """Change the load balancing strategy."""
        self.current_strategy = strategy
        self.logger.info(f"Load balancing strategy changed to: {strategy.value}")
    
    def get_available_strategies(self) -> List[LoadBalancingStrategy]:
        """Get list of available load balancing strategies."""
        return list(LoadBalancingStrategy)
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """Get performance metrics for the current strategy."""
        return {
            'current_strategy': self.current_strategy.value,
            'total_tasks_processed': self._total_tasks_processed,
            'success_rate': (
                (self._total_tasks_processed - self._total_tasks_failed) / 
                max(1, self._total_tasks_processed)
            ),
            'queue_size': self.task_queue.size(),
            'registered_agents': len(self.agent_capabilities)
        }
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of agent performance across all agents."""
        if not self.performance_metrics:
            return {}
        
        total_agents = len(self.performance_metrics)
        avg_success_rate = sum(m.success_rate for m in self.performance_metrics.values()) / total_agents
        avg_task_duration = sum(m.avg_task_duration for m in self.performance_metrics.values()) / total_agents
        avg_error_rate = sum(m.error_rate for m in self.performance_metrics.values()) / total_agents
        
        return {
            'total_agents': total_agents,
            'avg_success_rate': avg_success_rate,
            'avg_task_duration': avg_task_duration,
            'avg_error_rate': avg_error_rate,
            'agents_with_sessions': sum(1 for m in self.performance_metrics.values() 
                                      if m.session_availability > 0)
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including feedback analysis."""
        return {
            'load_balancer_status': self.get_status(),
            'agent_performance': self.get_agent_performance_summary(),
            'priority_statistics': self.get_priority_statistics(),
            'failed_tasks_summary': self.get_failed_tasks_summary(),
            'vendor_metrics': self.vendor_metrics,
            'feedback_data_points': len(self.feedback_data),
            'last_priority_adjustment': self._last_priority_adjustment.isoformat(),
            'last_feedback_update': self._last_feedback_update.isoformat(),
            'total_tasks_processed': self._total_tasks_processed,
            'total_tasks_failed': self._total_tasks_failed
        }
    
    def get_feedback_analysis_report(self) -> Dict[str, Any]:
        """Get a detailed feedback analysis report."""
        if len(self.feedback_data) < 3:
            return {'error': 'Insufficient feedback data for analysis'}
        
        # Calculate trends over time
        recent_data = self.feedback_data[-10:]  # Last 10 data points
        older_data = self.feedback_data[:-10] if len(self.feedback_data) > 10 else self.feedback_data
        
        recent_avg_queue_size = sum(d.get('queue_size', 0) for d in recent_data) / len(recent_data)
        older_avg_queue_size = sum(d.get('queue_size', 0) for d in older_data) / len(older_data) if older_data else 0
        
        return {
            'trends': {
                'queue_size_trend': recent_avg_queue_size - older_avg_queue_size,
                'recent_avg_queue_size': recent_avg_queue_size,
                'older_avg_queue_size': older_avg_queue_size
            },
            'strategy_performance': self._analyze_strategy_performance(),
            'agent_trends': self._analyze_agent_performance_trends(),
            'queue_performance': self._analyze_queue_performance(),
            'data_points_analyzed': len(self.feedback_data),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in the specified format."""
        import json
        
        metrics = self.get_comprehensive_metrics()
        
        if str(format).lower() == 'json':
            return json.dumps(metrics, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get current optimization recommendations based on feedback analysis."""
        if len(self.feedback_data) < 5:
            return []
        
        strategy_performance = self._analyze_strategy_performance()
        agent_trends = self._analyze_agent_performance_trends()
        queue_performance = self._analyze_queue_performance()
        
        return self._generate_optimization_recommendations(
            strategy_performance, agent_trends, queue_performance
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status."""
        return {
            'queue_size': self.task_queue.size(),
            'registered_agents': len(self.agent_capabilities),
            'current_strategy': self.current_strategy.value,
            'total_tasks_processed': self._total_tasks_processed,
            'total_tasks_failed': self._total_tasks_failed,
            'success_rate': (
                (self._total_tasks_processed - self._total_tasks_failed) / 
                max(1, self._total_tasks_processed)
            ),
            'agent_assignments': dict(self.agent_assignments),
            'performance_metrics': {
                agent_id: {
                    'avg_task_duration': metrics.avg_task_duration,
                    'success_rate': metrics.success_rate,
                    'error_rate': metrics.error_rate,
                    'throughput': metrics.throughput,
                    'session_availability': metrics.session_availability
                }
                for agent_id, metrics in self.performance_metrics.items()
            }
        }
    
    def _select_agent(self, task_info: TaskInfo, available_agents: List[str]) -> Optional[str]:
        """Select the best agent for a task based on current strategy."""
        if not available_agents:
            return None
        
        if self.current_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_agents)
        elif self.current_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(task_info, available_agents)
        elif self.current_strategy == LoadBalancingStrategy.CAPABILITY_BASED:
            return self._capability_based_select(task_info, available_agents)
        elif self.current_strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_select(task_info, available_agents)
        elif self.current_strategy == LoadBalancingStrategy.AVAILABILITY_BASED:
            return self._availability_based_select(task_info, available_agents)
        elif self.current_strategy == LoadBalancingStrategy.HYBRID:
            return self._hybrid_select(task_info, available_agents)
        else:
            # Fallback to round-robin
            return self._round_robin_select(available_agents)
    
    def _round_robin_select(self, available_agents: List[str]) -> str:
        """Enhanced round-robin selection with per-vendor tracking."""
        if not available_agents:
            return None
        
        # Group agents by vendor for better distribution
        vendor_agents = defaultdict(list)
        for agent_id in available_agents:
            if agent_id in self.agent_capabilities:
                vendor = self.agent_capabilities[agent_id].vendor
                vendor_agents[vendor].append(agent_id)
        
        # Find the vendor with the least recent assignment
        vendor_last_assignment = {}
        for vendor, agents in vendor_agents.items():
            last_assignment = max(
                (self.agent_capabilities[agent_id].last_activity or datetime.min 
                 for agent_id in agents),
                default=datetime.min
            )
            vendor_last_assignment[vendor] = last_assignment
        
        if not vendor_last_assignment:
            return available_agents[0]
        
        # Select vendor with oldest last assignment
        selected_vendor = min(vendor_last_assignment.items(), key=lambda x: x[1])[0]
        
        # Within the selected vendor, choose the agent with oldest last activity
        vendor_agent_list = vendor_agents[selected_vendor]
        return min(vendor_agent_list, 
                  key=lambda agent_id: self.agent_capabilities[agent_id].last_activity or datetime.min)
    
    def _weighted_round_robin_select(self, task_info: TaskInfo, available_agents: List[str]) -> Optional[str]:
        """Weighted round-robin selection based on agent performance and capabilities."""
        if not available_agents:
            return None
        
        # Calculate weights for each agent
        agent_weights = []
        for agent_id in available_agents:
            if agent_id not in self.agent_capabilities:
                continue
            
            capability = self.agent_capabilities[agent_id]
            metrics = self.performance_metrics.get(agent_id)
            
            # Calculate weight based on multiple factors
            weight = 1.0  # Base weight
            
            # Performance-based weight adjustment
            if metrics:
                # Higher success rate = higher weight
                weight *= (1.0 + metrics.success_rate)
                
                # Faster agents get higher weight
                speed_factor = 1.0 / max(metrics.avg_task_duration, 0.1)
                weight *= (1.0 + speed_factor * 0.1)
                
                # Lower error rate = higher weight
                weight *= (1.0 - metrics.error_rate)
            
            # Capability-based weight adjustment
            if task_info.task_type in capability.task_types:
                weight *= 1.5  # Bonus for capability match
            
            # Vendor match bonus
            if capability.vendor == task_info.vendor:
                weight *= 1.3
            
            # Session availability bonus
            if capability.session_available:
                weight *= 1.2
            
            # Current load penalty (prefer less loaded agents)
            current_load = len(self.agent_assignments.get(agent_id, []))
            weight *= (1.0 / max(current_load + 1, 1))
            
            agent_weights.append((agent_id, weight))
        
        if not agent_weights:
            return None
        
        # Sort by weight (highest first)
        agent_weights.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top 50% of agents (to maintain some randomness)
        top_agents = agent_weights[:max(1, len(agent_weights) // 2)]
        
        # Return the highest weighted agent
        return top_agents[0][0]
    
    def _capability_based_select(self, task_info: TaskInfo, available_agents: List[str]) -> Optional[str]:
        """Enhanced capability-based selection with vendor matching and performance scoring."""
        suitable_agents = []
        
        for agent_id in available_agents:
            if agent_id not in self.agent_capabilities:
                continue
            
            capability = self.agent_capabilities[agent_id]
            
            # Check if agent can handle this task type
            if task_info.task_type not in capability.task_types:
                continue
            
            # Check if agent is for the same vendor (prefer vendor-specific agents)
            vendor_match = capability.vendor == task_info.vendor
            
            # Calculate capability score
            capability_score = 0.0
            
            # Base score for having the required task type
            capability_score += 1.0
            
            # Bonus for vendor match
            if vendor_match:
                capability_score += 2.0
            
            # Bonus for session availability
            if capability.session_available:
                capability_score += 1.0
            
            # Bonus for recent activity (indicates agent is healthy)
            if capability.last_activity:
                time_since_activity = (datetime.now() - capability.last_activity).total_seconds()
                if time_since_activity < 300:  # 5 minutes
                    capability_score += 0.5
            
            # Penalty for high error rate
            capability_score -= capability.error_rate * 2.0
            
            suitable_agents.append((agent_id, capability_score))
        
        if not suitable_agents:
            return None
        
        # Return agent with highest capability score
        return max(suitable_agents, key=lambda x: x[1])[0]
    
    def _performance_based_select(self, task_info: TaskInfo, available_agents: List[str]) -> Optional[str]:
        """Enhanced performance-based selection with adaptive scoring and vendor consideration."""
        if not available_agents:
            return None
        
        # Score agents based on performance metrics
        agent_scores = []
        for agent_id in available_agents:
            if agent_id not in self.performance_metrics:
                continue
            
            metrics = self.performance_metrics[agent_id]
            capability = self.agent_capabilities.get(agent_id)
            
            # Base performance score
            performance_score = 0.0
            
            # Success rate component (40% weight)
            performance_score += metrics.success_rate * 0.4
            
            # Speed component (30% weight) - inverse of average duration
            speed_score = 1.0 / max(metrics.avg_task_duration, 0.1)
            performance_score += speed_score * 0.3
            
            # Session availability component (20% weight)
            performance_score += metrics.session_availability * 0.2
            
            # Throughput component (10% weight) - normalized to 0-1
            throughput_score = min(metrics.throughput / 10.0, 1.0)  # Normalize to max 10 tasks/min
            performance_score += throughput_score * 0.1
            
            # Vendor-specific bonus (if agent matches task vendor)
            if capability and capability.vendor == task_info.vendor:
                performance_score += 0.2
            
            # Penalty for high error rate
            performance_score -= metrics.error_rate * 0.5
            
            # Penalty for high response time variance (indicates instability)
            performance_score -= metrics.response_time_variance * 0.1
            
            # Ensure score is non-negative
            performance_score = max(0.0, performance_score)
            
            agent_scores.append((agent_id, performance_score))
        
        if not agent_scores:
            return None
        
        # Return agent with highest score
        return max(agent_scores, key=lambda x: x[1])[0]
    
    def _availability_based_select(self, task_info: TaskInfo, available_agents: List[str]) -> Optional[str]:
        """Enhanced availability-based selection with load balancing and health checks."""
        available_with_sessions = []
        
        for agent_id in available_agents:
            if agent_id not in self.agent_capabilities:
                continue
            
            capability = self.agent_capabilities[agent_id]
            metrics = self.performance_metrics.get(agent_id)
            
            # Check session availability
            if not capability.session_available:
                continue
            
            # Calculate availability score
            availability_score = 0.0
            
            # Base score for having a session
            availability_score += 1.0
            
            # Bonus for session availability metric
            if metrics:
                availability_score += metrics.session_availability
            
            # Penalty for current load (prefer less loaded agents)
            current_load = len(self.agent_assignments.get(agent_id, []))
            load_penalty = current_load * 0.2
            availability_score -= load_penalty
            
            # Penalty for recent failures
            if metrics and metrics.error_rate > 0.1:  # More than 10% error rate
                availability_score -= metrics.error_rate * 0.5
            
            # Bonus for vendor match
            if capability.vendor == task_info.vendor:
                availability_score += 0.5
            
            # Penalty for long inactivity (indicates potential issues)
            if capability.last_activity:
                time_since_activity = (datetime.now() - capability.last_activity).total_seconds()
                if time_since_activity > 600:  # 10 minutes
                    availability_score -= 0.3
            
            # Ensure score is non-negative
            availability_score = max(0.0, availability_score)
            
            available_with_sessions.append((agent_id, availability_score))
        
        if not available_with_sessions:
            return None
        
        # Return agent with highest availability score
        return max(available_with_sessions, key=lambda x: x[1])[0]
    
    def _hybrid_select(self, task_info: TaskInfo, available_agents: List[str]) -> Optional[str]:
        """Enhanced hybrid selection combining all strategies with adaptive weighting."""
        if not available_agents:
            return None
        
        # First, filter by capability
        capable_agents = []
        for agent_id in available_agents:
            if agent_id not in self.agent_capabilities:
                continue
            
            capability = self.agent_capabilities[agent_id]
            if task_info.task_type in capability.task_types:
                capable_agents.append(agent_id)
        
        if not capable_agents:
            # Fallback to all available agents
            capable_agents = available_agents
        
        # Calculate comprehensive scores for capable agents
        agent_scores = []
        for agent_id in capable_agents:
            if agent_id not in self.performance_metrics:
                continue
            
            metrics = self.performance_metrics[agent_id]
            capability = self.agent_capabilities.get(agent_id)
            
            if not capability:
                continue
            
            # Calculate composite score with adaptive weighting
            composite_score = 0.0
            
            # 1. Capability score (25% weight)
            capability_score = 1.0  # Base score for being capable
            if capability.vendor == task_info.vendor:
                capability_score += 1.0  # Vendor match bonus
            if capability.session_available:
                capability_score += 0.5  # Session available bonus
            composite_score += capability_score * 0.25
            
            # 2. Performance score (30% weight)
            performance_score = metrics.success_rate * 0.6
            speed_score = 1.0 / max(metrics.avg_task_duration, 0.1)
            performance_score += speed_score * 0.4
            composite_score += performance_score * 0.30
            
            # 3. Availability score (25% weight)
            availability_score = metrics.session_availability
            current_load = len(self.agent_assignments.get(agent_id, []))
            load_penalty = min(current_load * 0.1, 0.5)  # Cap load penalty
            availability_score -= load_penalty
            composite_score += availability_score * 0.25
            
            # 4. Health score (20% weight)
            health_score = 1.0
            if metrics.error_rate > 0.1:
                health_score -= metrics.error_rate * 0.5
            if capability.last_activity:
                time_since_activity = (datetime.now() - capability.last_activity).total_seconds()
                if time_since_activity > 300:  # 5 minutes
                    health_score -= 0.2
            composite_score += health_score * 0.20
            
            # Ensure score is non-negative
            composite_score = max(0.0, composite_score)
            
            agent_scores.append((agent_id, composite_score))
        
        if not agent_scores:
            return None
        
        # Return agent with highest composite score
        return max(agent_scores, key=lambda x: x[1])[0]
    
    async def _priority_adjustment_loop(self) -> None:
        """Background loop for adjusting task priorities."""
        while self._running:
            try:
                await asyncio.sleep(self.config.priority_adjustment_interval)
                
                if not self._running:
                    break
                
                await self._adjust_task_priorities()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in priority adjustment loop: {e}")
    
    async def _feedback_loop(self) -> None:
        """Background loop for feedback and optimization."""
        while self._running:
            try:
                await asyncio.sleep(self.config.feedback_loop_interval)
                
                if not self._running:
                    break
                
                await self._update_feedback_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in feedback loop: {e}")
    
    async def _performance_monitoring_loop(self) -> None:
        """Background loop for performance monitoring."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                if not self._running:
                    break
                
                await self._update_performance_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
    
    async def _adjust_task_priorities(self) -> None:
        """Adjust task priorities based on various factors."""
        if self.task_queue.is_empty():
            return
        
        self.logger.debug("Starting priority adjustment cycle")
        
        # Get all tasks in queue for analysis
        all_tasks = self.task_queue.get_all_tasks()
        if not all_tasks:
            return
        
        # Calculate priority adjustments for each task
        for task_info in all_tasks:
            new_priority = self._calculate_adjusted_priority(task_info)
            
            if new_priority != task_info.priority:
                self.logger.info(f"Adjusting priority for task {task_info.task_id}: "
                               f"{task_info.priority.value} -> {new_priority.value}")
                
                # Update priority in queue
                self.task_queue.update_priority(task_info.task_id, new_priority)
                task_info.priority = new_priority
        
        self._last_priority_adjustment = datetime.now()
        self.logger.debug("Priority adjustment cycle completed")
    
    def _calculate_adjusted_priority(self, task_info: TaskInfo) -> TaskPriority:
        """Calculate adjusted priority based on various factors."""
        current_priority = task_info.priority
        adjusted_priority = current_priority
        
        # Factor 1: Task age - older tasks get higher priority
        age_seconds = (datetime.now() - task_info.created_at).total_seconds()
        if age_seconds > 300:  # 5 minutes
            adjusted_priority = self._promote_priority(adjusted_priority)
        if age_seconds > 600:  # 10 minutes
            adjusted_priority = self._promote_priority(adjusted_priority)
        if age_seconds > 1800:  # 30 minutes
            adjusted_priority = self._promote_priority(adjusted_priority)
        
        # Factor 2: Retry count - failed tasks get higher priority
        if task_info.retry_count > 0:
            adjusted_priority = self._promote_priority(adjusted_priority)
        if task_info.retry_count > 1:
            adjusted_priority = self._promote_priority(adjusted_priority)
        
        # Factor 3: Vendor response time - slow vendors get higher priority
        if task_info.vendor_response_time and task_info.vendor_response_time > 10.0:
            adjusted_priority = self._promote_priority(adjusted_priority)
        
        # Factor 4: Task importance score
        if task_info.importance_score > 1.5:
            adjusted_priority = self._promote_priority(adjusted_priority)
        elif task_info.importance_score < 0.5:
            adjusted_priority = self._demote_priority(adjusted_priority)
        
        # Factor 5: Vendor error rate - high error rate vendors get higher priority
        vendor_error_rate = self._get_vendor_error_rate(task_info.vendor)
        if vendor_error_rate > 0.2:  # 20% error rate
            adjusted_priority = self._promote_priority(adjusted_priority)
        
        # Factor 6: Dependencies - tasks with dependencies get lower priority
        if task_info.dependencies:
            adjusted_priority = self._demote_priority(adjusted_priority)
        
        # Factor 7: Time-based urgency (business hours, etc.)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            # Slightly higher priority during business hours
            pass  # Could implement business hour adjustments
        
        # Ensure priority doesn't go below BACKGROUND or above CRITICAL
        adjusted_priority = max(TaskPriority.BACKGROUND, adjusted_priority)
        adjusted_priority = min(TaskPriority.CRITICAL, adjusted_priority)
        
        return adjusted_priority
    
    def _promote_priority(self, priority: TaskPriority) -> TaskPriority:
        """Promote a priority level (lower number = higher priority)."""
        priority_values = [p.value for p in TaskPriority]
        current_index = priority_values.index(priority.value)
        new_index = max(0, current_index - 1)  # Don't go below 0
        return TaskPriority(priority_values[new_index])
    
    def _demote_priority(self, priority: TaskPriority) -> TaskPriority:
        """Demote a priority level (higher number = lower priority)."""
        priority_values = [p.value for p in TaskPriority]
        current_index = priority_values.index(priority.value)
        new_index = min(len(priority_values) - 1, current_index + 1)  # Don't go above max
        return TaskPriority(priority_values[new_index])
    
    def _get_vendor_error_rate(self, vendor: str) -> float:
        """Get the current error rate for a vendor."""
        vendor_agents = [
            agent_id for agent_id, capability in self.agent_capabilities.items()
            if capability.vendor == vendor
        ]
        
        if not vendor_agents:
            return 0.0
        
        total_error_rate = 0.0
        count = 0
        
        for agent_id in vendor_agents:
            if agent_id in self.performance_metrics:
                total_error_rate += self.performance_metrics[agent_id].error_rate
                count += 1
        
        return total_error_rate / count if count > 0 else 0.0
    
    async def _update_feedback_data(self) -> None:
        """Update feedback data for optimization."""
        self.logger.debug("Starting feedback data update cycle")
        
        # Collect current system state
        current_state = {
            'timestamp': datetime.now().isoformat(),
            'queue_size': self.task_queue.size(),
            'registered_agents': len(self.agent_capabilities),
            'current_strategy': self.current_strategy.value,
            'performance_metrics': {},
            'priority_statistics': self.get_priority_statistics(),
            'failed_tasks_summary': self.get_failed_tasks_summary()
        }
        
        # Collect performance metrics for each agent
        for agent_id, metrics in self.performance_metrics.items():
            current_state['performance_metrics'][agent_id] = {
                'success_rate': metrics.success_rate,
                'avg_task_duration': metrics.avg_task_duration,
                'error_rate': metrics.error_rate,
                'throughput': metrics.throughput,
                'session_availability': metrics.session_availability,
                'response_time_variance': metrics.response_time_variance
            }
        
        # Add to feedback data
        self.feedback_data.append(current_state)
        
        # Keep only recent feedback data (last 100 entries)
        if len(self.feedback_data) > 100:
            self.feedback_data = self.feedback_data[-100:]
        
        # Analyze feedback for optimization opportunities
        await self._analyze_feedback_for_optimization()
        
        self._last_feedback_update = datetime.now()
        self.logger.debug("Feedback data update cycle completed")
    
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics for all agents."""
        self.logger.debug("Starting performance metrics update cycle")
        
        for agent_id, metrics in self.performance_metrics.items():
            # Update throughput calculation
            if metrics.last_updated:
                time_diff = (datetime.now() - metrics.last_updated).total_seconds() / 60
                if time_diff > 0:
                    # Recalculate throughput based on recent activity
                    recent_tasks = metrics.total_tasks_completed
                    metrics.throughput = recent_tasks / time_diff
            
            # Update response time variance
            if agent_id in self.agent_capabilities:
                capability = self.agent_capabilities[agent_id]
                if capability.avg_response_time > 0:
                    # Simple variance calculation (could be enhanced)
                    metrics.response_time_variance = capability.avg_response_time * 0.1
            
            # Update session availability based on capability
            if agent_id in self.agent_capabilities:
                capability = self.agent_capabilities[agent_id]
                metrics.session_availability = 1.0 if capability.session_available else 0.0
        
        # Update vendor-level metrics
        self._update_vendor_level_metrics()
        
        self.logger.debug("Performance metrics update cycle completed")
    
    async def _analyze_feedback_for_optimization(self) -> None:
        """Analyze feedback data to identify optimization opportunities."""
        if len(self.feedback_data) < 5:  # Need minimum data points
            return
        
        # Analyze strategy performance
        strategy_performance = self._analyze_strategy_performance()
        
        # Analyze agent performance trends
        agent_trends = self._analyze_agent_performance_trends()
        
        # Analyze queue performance
        queue_performance = self._analyze_queue_performance()
        
        # Make optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            strategy_performance, agent_trends, queue_performance
        )
        
        # Apply automatic optimizations if enabled
        if self.config.enable_ml_optimization:
            await self._apply_automatic_optimizations(recommendations)
        
        self.logger.info(f"Feedback analysis completed. Generated {len(recommendations)} recommendations")
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance of different load balancing strategies."""
        strategy_data = defaultdict(list)
        
        for entry in self.feedback_data:
            strategy = entry.get('current_strategy', 'unknown')
            success_rate = entry.get('success_rate', 0.0)
            queue_size = entry.get('queue_size', 0)
            
            strategy_data[strategy].append({
                'success_rate': success_rate,
                'queue_size': queue_size,
                'timestamp': entry.get('timestamp')
            })
        
        analysis = {}
        for strategy, data in strategy_data.items():
            if len(data) >= 3:  # Minimum data points for analysis
                avg_success_rate = sum(d['success_rate'] for d in data) / len(data)
                avg_queue_size = sum(d['queue_size'] for d in data) / len(data)
                
                analysis[strategy] = {
                    'avg_success_rate': avg_success_rate,
                    'avg_queue_size': avg_queue_size,
                    'data_points': len(data)
                }
        
        return analysis
    
    def _analyze_agent_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends for individual agents."""
        agent_trends = {}
        
        for agent_id in self.performance_metrics:
            agent_data = []
            for entry in self.feedback_data:
                if agent_id in entry.get('performance_metrics', {}):
                    agent_metrics = entry['performance_metrics'][agent_id]
                    agent_data.append({
                        'success_rate': agent_metrics.get('success_rate', 0.0),
                        'avg_task_duration': agent_metrics.get('avg_task_duration', 0.0),
                        'error_rate': agent_metrics.get('error_rate', 0.0),
                        'timestamp': entry.get('timestamp')
                    })
            
            if len(agent_data) >= 3:
                # Calculate trends
                recent_data = agent_data[-5:]  # Last 5 data points
                older_data = agent_data[:-5] if len(agent_data) > 5 else agent_data
                
                if older_data:
                    recent_avg_success = sum(d['success_rate'] for d in recent_data) / len(recent_data)
                    older_avg_success = sum(d['success_rate'] for d in older_data) / len(older_data)
                    
                    agent_trends[agent_id] = {
                        'success_rate_trend': recent_avg_success - older_avg_success,
                        'recent_performance': recent_avg_success,
                        'data_points': len(agent_data)
                    }
        
        return agent_trends
    
    def _analyze_queue_performance(self) -> Dict[str, Any]:
        """Analyze queue performance metrics."""
        queue_sizes = [entry.get('queue_size', 0) for entry in self.feedback_data]
        priority_stats = [entry.get('priority_statistics', {}) for entry in self.feedback_data]
        
        if not queue_sizes:
            return {}
        
        return {
            'avg_queue_size': sum(queue_sizes) / len(queue_sizes),
            'max_queue_size': max(queue_sizes),
            'queue_size_variance': self._calculate_variance(queue_sizes),
            'priority_distribution': self._aggregate_priority_distribution(priority_stats)
        }
    
    def _generate_optimization_recommendations(self, strategy_performance: Dict[str, Any],
                                            agent_trends: Dict[str, Any],
                                            queue_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Strategy optimization recommendations
        if strategy_performance:
            best_strategy = max(strategy_performance.items(), 
                              key=lambda x: x[1].get('avg_success_rate', 0.0))
            
            if best_strategy[0] != self.current_strategy.value:
                recommendations.append({
                    'type': 'strategy_change',
                    'current_strategy': self.current_strategy.value,
                    'recommended_strategy': best_strategy[0],
                    'expected_improvement': best_strategy[1].get('avg_success_rate', 0.0),
                    'priority': 'medium'
                })
        
        # Agent performance recommendations
        for agent_id, trend in agent_trends.items():
            if trend['success_rate_trend'] < -0.1:  # 10% decline
                recommendations.append({
                    'type': 'agent_issue',
                    'agent_id': agent_id,
                    'issue': 'declining_performance',
                    'trend': trend['success_rate_trend'],
                    'priority': 'high'
                })
        
        # Queue performance recommendations
        if queue_performance.get('avg_queue_size', 0) > 50:
            recommendations.append({
                'type': 'queue_optimization',
                'issue': 'high_queue_size',
                'avg_queue_size': queue_performance['avg_queue_size'],
                'recommendation': 'Consider scaling up agents or adjusting strategy',
                'priority': 'high'
            })
        
        return recommendations
    
    async def _apply_automatic_optimizations(self, recommendations: List[Dict[str, Any]]) -> None:
        """Apply automatic optimizations based on recommendations."""
        for rec in recommendations:
            if rec['priority'] == 'high':
                if rec['type'] == 'strategy_change':
                    new_strategy = LoadBalancingStrategy(rec['recommended_strategy'])
                    self.set_strategy(new_strategy)
                    self.logger.info(f"Automatically changed strategy to {rec['recommended_strategy']}")
                
                elif rec['type'] == 'agent_issue':
                    # Could implement automatic agent replacement or restart
                    self.logger.warning(f"Agent {rec['agent_id']} showing declining performance")
                
                elif rec['type'] == 'queue_optimization':
                    # Could implement automatic scaling
                    self.logger.warning(f"Queue size optimization needed: {rec['avg_queue_size']} tasks")
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _aggregate_priority_distribution(self, priority_stats: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate priority distribution across multiple data points."""
        aggregated = defaultdict(int)
        
        for stats in priority_stats:
            priority_dist = stats.get('priority_distribution', {})
            for priority, count in priority_dist.items():
                aggregated[priority] += count
        
        return dict(aggregated)
    
    def _update_vendor_level_metrics(self) -> None:
        """Update vendor-level performance metrics."""
        vendor_metrics = defaultdict(lambda: {
            'total_agents': 0,
            'avg_success_rate': 0.0,
            'avg_error_rate': 0.0,
            'total_tasks': 0
        })
        
        for agent_id, metrics in self.performance_metrics.items():
            if agent_id in self.agent_capabilities:
                vendor = self.agent_capabilities[agent_id].vendor
                vendor_metrics[vendor]['total_agents'] += 1
                vendor_metrics[vendor]['avg_success_rate'] += metrics.success_rate
                vendor_metrics[vendor]['avg_error_rate'] += metrics.error_rate
                vendor_metrics[vendor]['total_tasks'] += metrics.total_tasks_completed
        
        # Calculate averages
        for vendor, metrics in vendor_metrics.items():
            if metrics['total_agents'] > 0:
                metrics['avg_success_rate'] /= metrics['total_agents']
                metrics['avg_error_rate'] /= metrics['total_agents']
        
        self.vendor_metrics = dict(vendor_metrics)


def create_load_balancer(config: Optional[LoadBalancerConfig] = None) -> LoadBalancer:
    """Factory function to create a LoadBalancer instance."""
    return LoadBalancer(config) 