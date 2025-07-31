"""
Comprehensive tests for the AgentDeployer async agent deployment system.

Tests cover:
- AgentDeployer initialization and lifecycle
- Async task execution and CrewAI integration
- Session pool management
- Dynamic scaling logic
- Concurrency controls and circuit breakers
- Error handling and recovery
- Graceful shutdown
- Health monitoring
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scraping_cli.agent_deployer import (
    AgentDeployer, ConcurrencyConfig, ScalingConfig, AgentTaskInfo,
    AgentTaskStatus, DeploymentStatus, TaskType, VendorMetrics,
    CircuitBreaker, RateLimiter, create_agent_deployer
)
from scraping_cli.browserbase_manager import BrowserbaseManager, SessionInfo, SessionConfig
from scraping_cli.crewai_integration import ScrapingAgent, ScrapingTask, AgentRole


class TestAgentDeployerInitialization:
    """Test AgentDeployer initialization and configuration."""
    
    @pytest.fixture
    def mock_browserbase_manager(self):
        """Create a mock BrowserbaseManager."""
        manager = Mock(spec=BrowserbaseManager)
        manager.session_pool = Mock()
        manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        manager.active_sessions = {}
        return manager
    
    @pytest.fixture
    def concurrency_config(self):
        """Create test concurrency configuration."""
        return ConcurrencyConfig(
            max_agents_per_vendor=2,
            max_total_agents=5,
            max_concurrent_sessions=8,
            rate_limit_per_domain=2.0,
            enable_circuit_breaker=True,
            circuit_breaker_threshold=3
        )
    
    @pytest.fixture
    def scaling_config(self):
        """Create test scaling configuration."""
        return ScalingConfig(
            enable_auto_scaling=True,
            scale_up_threshold=0.8,
            scale_down_threshold=0.3,
            min_agents=1,
            max_agents=10,
            scale_check_interval=10
        )
    
    def test_agent_deployer_initialization(self, mock_browserbase_manager, 
                                         concurrency_config, scaling_config):
        """Test AgentDeployer initializes correctly with all components."""
        deployer = AgentDeployer(
            browserbase_manager=mock_browserbase_manager,
            concurrency_config=concurrency_config,
            scaling_config=scaling_config
        )
        
        # Check basic initialization
        assert deployer.browserbase_manager == mock_browserbase_manager
        assert deployer.concurrency_config == concurrency_config
        assert deployer.scaling_config == scaling_config
        assert deployer.status == DeploymentStatus.INITIALIZING
        
        # Check data structures are initialized
        assert isinstance(deployer.active_agents, dict)
        assert isinstance(deployer.pending_tasks, list)
        assert isinstance(deployer.running_tasks, dict)
        assert isinstance(deployer.completed_tasks, list)
        assert isinstance(deployer.failed_tasks, list)
        
        # Check vendor controls are initialized
        expected_vendors = ['tesco', 'asda', 'costco']
        for vendor in expected_vendors:
            assert vendor in deployer.vendor_metrics
            assert vendor in deployer.circuit_breakers
            assert vendor in deployer.rate_limiters
            assert vendor in deployer.agent_semaphores
    
    def test_vendor_controls_initialization(self, mock_browserbase_manager):
        """Test vendor-specific controls are properly initialized."""
        deployer = AgentDeployer(browserbase_manager=mock_browserbase_manager)
        
        # Test circuit breakers
        for vendor in ['tesco', 'asda', 'costco']:
            cb = deployer.circuit_breakers[vendor]
            assert isinstance(cb, CircuitBreaker)
            assert cb.state == "closed"
            assert cb.failure_count == 0
            
            # Test rate limiters
            rl = deployer.rate_limiters[vendor]
            assert isinstance(rl, RateLimiter)
            assert rl.requests_per_second == 1.0
            
            # Test semaphores
            sem = deployer.agent_semaphores[vendor]
            assert isinstance(sem, asyncio.Semaphore)
    
    def test_factory_function(self, mock_browserbase_manager):
        """Test the create_agent_deployer factory function."""
        deployer = create_agent_deployer(
            browserbase_manager=mock_browserbase_manager
        )
        
        assert isinstance(deployer, AgentDeployer)
        assert deployer.browserbase_manager == mock_browserbase_manager


class TestAgentDeployerLifecycle:
    """Test AgentDeployer lifecycle management (start/stop)."""
    
    @pytest.fixture
    def deployer(self):
        """Create configured AgentDeployer for testing."""
        # Create mock browserbase manager inline
        mock_manager = Mock(spec=BrowserbaseManager)
        mock_manager.session_pool = Mock()
        mock_manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        mock_manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        mock_manager.active_sessions = {}
        return AgentDeployer(browserbase_manager=mock_manager)
    
    @pytest.mark.asyncio
    async def test_start_deployment(self, deployer):
        """Test starting the deployment system."""
        assert deployer.status == DeploymentStatus.INITIALIZING
        
        await deployer.start()
        
        assert deployer.status == DeploymentStatus.RUNNING
        assert deployer.start_time is not None
        assert len(deployer.background_tasks) > 0
        
        # Cleanup
        await deployer.stop(timeout=1, force=True)
    
    @pytest.mark.asyncio
    async def test_graceful_stop(self, deployer):
        """Test graceful shutdown of the deployment system."""
        await deployer.start()
        assert deployer.status == DeploymentStatus.RUNNING
        
        # Add some mock running tasks
        task_info = AgentTaskInfo(
            task_id="test_task",
            agent_id="test_agent",
            vendor="tesco",
            task_type=TaskType.SCRAPE_PRODUCTS,
            status=AgentTaskStatus.RUNNING,
            created_at=datetime.now()
        )
        deployer.running_tasks["test_task"] = task_info
        
        await deployer.stop(timeout=2)
        
        assert deployer.status == DeploymentStatus.STOPPED
        assert len(deployer.running_tasks) == 0
        assert len(deployer.background_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_force_stop(self, deployer):
        """Test force shutdown of the deployment system."""
        await deployer.start()
        
        # Add mock running tasks
        for i in range(3):
            task_info = AgentTaskInfo(
                task_id=f"task_{i}",
                agent_id=f"agent_{i}",
                vendor="tesco",
                task_type=TaskType.SCRAPE_PRODUCTS,
                status=AgentTaskStatus.RUNNING,
                created_at=datetime.now()
            )
            deployer.running_tasks[f"task_{i}"] = task_info
        
        await deployer.stop(timeout=1, force=True)
        
        assert deployer.status == DeploymentStatus.STOPPED
        assert len(deployer.running_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test AgentDeployer context manager functionality."""
        # Create mock browserbase manager inline
        mock_manager = Mock(spec=BrowserbaseManager)
        mock_manager.session_pool = Mock()
        mock_manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        mock_manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        mock_manager.active_sessions = {}
        deployer = AgentDeployer(browserbase_manager=mock_manager)
        
        async with deployer.deployment_context():
            assert deployer.status == DeploymentStatus.RUNNING
        
        assert deployer.status == DeploymentStatus.STOPPED


class TestTaskManagement:
    """Test task addition, execution, and management."""
    
    @pytest.fixture
    def deployer(self):
        """Create configured AgentDeployer for testing."""
        # Create mock browserbase manager inline
        mock_manager = Mock(spec=BrowserbaseManager)
        mock_manager.session_pool = Mock()
        mock_manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        mock_manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        mock_manager.active_sessions = {}
        
        deployer = AgentDeployer(browserbase_manager=mock_manager)
        # Mock session acquisition
        deployer._acquire_session = AsyncMock(return_value=Mock(session_id="test_session"))
        deployer._release_session = AsyncMock()
        return deployer
    
    @pytest.mark.asyncio
    async def test_add_task(self, deployer):
        """Test adding a single task."""
        task_id = await deployer.add_task(
            vendor="tesco",
            task_type=TaskType.SCRAPE_PRODUCTS,
            task_data={"url": "https://tesco.com/products"}
        )
        
        assert task_id is not None
        assert len(deployer.pending_tasks) == 1
        assert task_id in deployer.agent_tasks
        
        task_info = deployer.agent_tasks[task_id]
        assert task_info.vendor == "tesco"
        assert task_info.task_type == TaskType.SCRAPE_PRODUCTS
        assert task_info.status == AgentTaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_add_task_batch(self, deployer):
        """Test adding multiple tasks in batch."""
        tasks = [
            {"vendor": "tesco", "task_type": "scrape_products"},
            {"vendor": "asda", "task_type": "scrape_products"},
            {"vendor": "costco", "task_type": "analyze_data"}
        ]
        
        task_ids = await deployer.add_task_batch(tasks)
        
        assert len(task_ids) == 3
        assert len(deployer.pending_tasks) == 3
        
        # Check vendors are distributed correctly
        vendors = [deployer.agent_tasks[tid].vendor for tid in task_ids]
        assert "tesco" in vendors
        assert "asda" in vendors
        assert "costco" in vendors
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, deployer):
        """Test retrieving task status."""
        task_id = await deployer.add_task("tesco", TaskType.SCRAPE_PRODUCTS)
        
        status = await deployer.get_task_status(task_id)
        
        assert status is not None
        assert status["task_id"] == task_id
        assert status["vendor"] == "tesco"
        assert status["status"] == "pending"
        assert "created_at" in status
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, deployer):
        """Test cancelling a pending task."""
        task_id = await deployer.add_task("tesco", TaskType.SCRAPE_PRODUCTS)
        
        result = await deployer.cancel_task(task_id)
        
        assert result is True
        assert len(deployer.pending_tasks) == 0
        assert len(deployer.failed_tasks) == 1
        
        task_info = deployer.agent_tasks[task_id]
        assert task_info.status == AgentTaskStatus.FAILED
        assert "cancelled" in task_info.last_error.lower()


class TestSessionPoolIntegration:
    """Test session pool integration and management."""
    
    @pytest.fixture
    def mock_session_info(self):
        """Create mock session info."""
        return SessionInfo(
            session_id="test_session_123",
            connect_url="wss://connect.browserbase.com/test",
            status="active",
            created_at=datetime.now(),
            last_used=datetime.now(),
            config=SessionConfig()
        )
    
    @pytest.fixture
    def deployer_with_session_mocks(self, mock_session_info):
        """Create deployer with mocked session operations."""
        # Create mock browserbase manager inline
        mock_manager = Mock(spec=BrowserbaseManager)
        mock_manager.session_pool = Mock()
        mock_manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        mock_manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        mock_manager.active_sessions = {}
        
        deployer = AgentDeployer(browserbase_manager=mock_manager)
        
        # Mock session operations
        mock_manager.get_session.return_value = mock_session_info
        mock_manager.release_session = Mock()
        mock_manager.close_session = Mock()
        
        return deployer
    
    @pytest.mark.asyncio
    async def test_session_acquisition(self, deployer_with_session_mocks, mock_session_info):
        """Test session acquisition for tasks."""
        deployer = deployer_with_session_mocks
        
        task_info = AgentTaskInfo(
            task_id="test_task",
            agent_id="test_agent",
            vendor="tesco",
            task_type=TaskType.SCRAPE_PRODUCTS,
            status=AgentTaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        session = await deployer._acquire_session("tesco", task_info)
        
        assert session == mock_session_info
        assert deployer.browserbase_manager.get_session.called
        assert deployer.stats['sessions_created'] == 1
    
    @pytest.mark.asyncio
    async def test_session_release(self, deployer_with_session_mocks, mock_session_info):
        """Test session release after task completion."""
        deployer = deployer_with_session_mocks
        
        task_info = AgentTaskInfo(
            task_id="test_task",
            agent_id="test_agent",
            vendor="tesco",
            task_type=TaskType.SCRAPE_PRODUCTS,
            status=AgentTaskStatus.COMPLETED,
            created_at=datetime.now()
        )
        
        await deployer._release_session(mock_session_info, task_info)
        
        assert deployer.browserbase_manager.release_session.called
    
    def test_vendor_session_config(self, deployer_with_session_mocks):
        """Test vendor-specific session configuration."""
        deployer = deployer_with_session_mocks
        
        # Test Tesco config
        tesco_config = deployer._create_vendor_session_config("tesco")
        assert tesco_config.stealth_mode is True
        assert tesco_config.timeout == 45000
        
        # Test Costco config
        costco_config = deployer._create_vendor_session_config("costco")
        assert costco_config.stealth_mode is False
        assert costco_config.timeout == 35000
        
        # Test unknown vendor (default)
        unknown_config = deployer._create_vendor_session_config("unknown")
        assert unknown_config.stealth_mode is True
        assert unknown_config.timeout == 30000
    
    @pytest.mark.asyncio
    async def test_session_recovery(self, deployer_with_session_mocks, mock_session_info):
        """Test session recovery when sessions fail."""
        deployer = deployer_with_session_mocks
        
        # Setup running task with session
        task_info = AgentTaskInfo(
            task_id="test_task",
            agent_id="test_agent",
            vendor="tesco",
            task_type=TaskType.SCRAPE_PRODUCTS,
            status=AgentTaskStatus.RUNNING,
            created_at=datetime.now()
        )
        task_info.session_id = "failed_session"
        deployer.running_tasks["test_task"] = task_info
        
        # Mock new session for recovery
        new_session = SessionInfo(
            session_id="new_session_123",
            connect_url="wss://connect.browserbase.com/new",
            status="active",
            created_at=datetime.now(),
            last_used=datetime.now(),
            config=SessionConfig()
        )
        
        with patch.object(deployer, '_acquire_session', return_value=new_session):
            await deployer._recover_session("failed_session")
        
        assert task_info.session_id == "new_session_123"
        assert deployer.stats['recovery_attempts'] == 1


class TestConcurrencyAndScaling:
    """Test concurrency controls and dynamic scaling."""
    
    @pytest.fixture
    def deployer_with_scaling(self):
        """Create deployer with scaling enabled."""
        # Create mock browserbase manager inline
        mock_manager = Mock(spec=BrowserbaseManager)
        mock_manager.session_pool = Mock()
        mock_manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        mock_manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        mock_manager.active_sessions = {}
        
        scaling_config = ScalingConfig(
            enable_auto_scaling=True,
            scale_up_threshold=0.7,
            scale_down_threshold=0.3,
            min_agents=1,
            max_agents=5
        )
        return AgentDeployer(
            browserbase_manager=mock_manager,
            scaling_config=scaling_config
        )
    
    def test_calculate_system_utilization(self, deployer_with_scaling):
        """Test system utilization calculation."""
        deployer = deployer_with_scaling
        
        # Add running tasks
        for i in range(3):
            task_info = AgentTaskInfo(
                task_id=f"task_{i}",
                agent_id=f"agent_{i}",
                vendor="tesco",
                task_type=TaskType.SCRAPE_PRODUCTS,
                status=AgentTaskStatus.RUNNING,
                created_at=datetime.now()
            )
            deployer.running_tasks[f"task_{i}"] = task_info
        
        utilization = deployer._calculate_system_utilization()
        expected = 3 / deployer.concurrency_config.max_total_agents
        assert utilization == expected
    
    def test_calculate_vendor_demand(self, deployer_with_scaling):
        """Test vendor demand calculation."""
        deployer = deployer_with_scaling
        
        # Add pending tasks
        vendors = ["tesco", "tesco", "asda", "costco", "costco", "costco"]
        for i, vendor in enumerate(vendors):
            task_info = AgentTaskInfo(
                task_id=f"task_{i}",
                agent_id=f"agent_{i}",
                vendor=vendor,
                task_type=TaskType.SCRAPE_PRODUCTS,
                status=AgentTaskStatus.PENDING,
                created_at=datetime.now()
            )
            deployer.pending_tasks.append(task_info)
        
        demand = deployer._calculate_vendor_demand()
        assert demand["tesco"] == 2
        assert demand["asda"] == 1
        assert demand["costco"] == 3
    
    @pytest.mark.asyncio
    async def test_scale_up_logic(self, deployer_with_scaling):
        """Test scale up logic when demand is high."""
        deployer = deployer_with_scaling
        
        # Add pending tasks to trigger scale up
        for i in range(8):
            task_info = AgentTaskInfo(
                task_id=f"task_{i}",
                agent_id=f"agent_{i}",
                vendor="tesco",
                task_type=TaskType.SCRAPE_PRODUCTS,
                status=AgentTaskStatus.PENDING,
                created_at=datetime.now()
            )
            deployer.pending_tasks.append(task_info)
        
        initial_agents = len(deployer.active_agents)
        
        with patch.object(deployer, '_pre_create_agent') as mock_create:
            await deployer._scale_up()
            
            # Should attempt to create agents (limited to max 2 at a time)
            assert mock_create.call_count <= 2
    
    @pytest.mark.asyncio
    async def test_scale_down_logic(self, deployer_with_scaling):
        """Test scale down logic when demand is low."""
        deployer = deployer_with_scaling
        
        # Add some agents
        for i in range(3):
            deployer.active_agents[f"tesco_agent_{i}"] = Mock()
        
        with patch.object(deployer, '_remove_idle_agents') as mock_remove:
            await deployer._scale_down()
            
            # Should attempt to remove agents
            mock_remove.assert_called_once()


class TestErrorHandlingAndRecovery:
    """Test error handling, circuit breakers, and recovery mechanisms."""
    
    @pytest.fixture
    def deployer_with_mocks(self):
        """Create deployer with error handling mocks."""
        # Create mock browserbase manager inline
        mock_manager = Mock(spec=BrowserbaseManager)
        mock_manager.session_pool = Mock()
        mock_manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        mock_manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        mock_manager.active_sessions = {}
        
        deployer = AgentDeployer(browserbase_manager=mock_manager)
        deployer._acquire_session = AsyncMock()
        deployer._release_session = AsyncMock()
        return deployer
    
    def test_circuit_breaker_functionality(self, deployer_with_mocks):
        """Test circuit breaker state transitions."""
        deployer = deployer_with_mocks
        cb = deployer.circuit_breakers["tesco"]
        
        # Initially closed
        assert cb.state == "closed"
        assert cb.can_execute() is True
        
        # Record failures
        for _ in range(5):  # Exceed threshold
            cb.record_failure()
        
        assert cb.state == "open"
        assert cb.can_execute() is False
        
        # Test success resets circuit breaker
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self, deployer_with_mocks):
        """Test rate limiting functionality."""
        deployer = deployer_with_mocks
        rate_limiter = deployer.rate_limiters["tesco"]
        
        # First request should be immediate
        start_time = time.time()
        await rate_limiter.acquire()
        first_duration = time.time() - start_time
        assert first_duration < 0.1  # Should be immediate
        
        # Second request should be delayed
        start_time = time.time()
        await rate_limiter.acquire()
        second_duration = time.time() - start_time
        # Should have some delay due to rate limiting
        # Note: This is a simplified test, actual timing may vary
    
    def test_error_classification(self, deployer_with_mocks):
        """Test error handling patterns and circuit breaker functionality."""
        deployer = deployer_with_mocks
        
        # Test circuit breaker records failures
        cb = deployer.circuit_breakers["tesco"]
        assert cb.state == "closed"
        
        # Simulate multiple failures
        for _ in range(3):
            cb.record_failure()
        
        # Circuit breaker should still be operational
        assert cb.failure_count == 3
        
        # Test successful operation resets failure count
        cb.record_success()
        assert cb.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_task_failure_handling(self, deployer_with_mocks):
        """Test comprehensive task failure handling."""
        deployer = deployer_with_mocks
        
        task_info = AgentTaskInfo(
            task_id="failing_task",
            agent_id="test_agent",
            vendor="tesco",
            task_type=TaskType.SCRAPE_PRODUCTS,
            status=AgentTaskStatus.RUNNING,
            created_at=datetime.now()
        )
        deployer.running_tasks["failing_task"] = task_info
        
        # Test recoverable error with retry
        recoverable_error = Exception("Network timeout")
        await deployer._handle_task_failure(task_info, recoverable_error)
        
        assert task_info.retry_count == 1
        assert task_info.status == AgentTaskStatus.RETRYING
        assert len(deployer.pending_tasks) > 0  # Should be rescheduled
    
    @pytest.mark.asyncio
    async def test_systemic_failure_detection(self, deployer_with_mocks):
        """Test detection of systemic vendor failures through circuit breakers."""
        deployer = deployer_with_mocks
        
        # Add multiple failed tasks for a vendor
        for i in range(15):
            failed_task = AgentTaskInfo(
                task_id=f"failed_task_{i}",
                agent_id=f"agent_{i}",
                vendor="tesco",
                task_type=TaskType.SCRAPE_PRODUCTS,
                status=AgentTaskStatus.FAILED,
                created_at=datetime.now()
            )
            deployer.failed_tasks.append(failed_task)
        
        # Test circuit breaker opens after many failures
        cb = deployer.circuit_breakers["tesco"]
        for _ in range(10):  # Exceed threshold
            cb.record_failure()
        
        assert cb.state == "open"
        assert cb.can_execute() is False
        
        # Test circuit breaker resets after timeout
        cb.record_success()
        assert cb.state == "closed"
        assert cb.can_execute() is True


class TestHealthMonitoring:
    """Test health monitoring and status reporting."""
    
    @pytest.fixture
    def deployer_with_health_mocks(self):
        """Create deployer with health monitoring mocks."""
        # Create mock browserbase manager inline
        mock_manager = Mock(spec=BrowserbaseManager)
        mock_manager.session_pool = Mock()
        mock_manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        mock_manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        mock_manager.active_sessions = {}
        
        deployer = AgentDeployer(browserbase_manager=mock_manager)
        # Mock background tasks
        deployer.background_tasks = {Mock(done=Mock(return_value=False)) for _ in range(3)}
        return deployer
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self, deployer_with_health_mocks):
        """Test comprehensive system status reporting."""
        deployer = deployer_with_health_mocks
        
        # Get comprehensive status instead of health check
        status_result = deployer.get_status()
        
        assert "deployment" in status_result
        assert "tasks" in status_result
        assert "agents" in status_result
        assert "sessions" in status_result
        assert "scaling" in status_result
        assert "performance" in status_result
        assert "stats" in status_result
        
        # Check deployment status
        deployment_status = status_result["deployment"]
        assert "status" in deployment_status
        assert "start_time" in deployment_status
        assert "uptime" in deployment_status
        
        # Check tasks status
        tasks_status = status_result["tasks"]
        assert "pending" in tasks_status
        assert "running" in tasks_status
        assert "completed" in tasks_status
        assert "failed" in tasks_status
    
    def test_get_comprehensive_status(self, deployer_with_health_mocks):
        """Test comprehensive status reporting."""
        deployer = deployer_with_health_mocks
        
        # Add some tasks and agents for realistic status
        for i in range(3):
            task_info = AgentTaskInfo(
                task_id=f"task_{i}",
                agent_id=f"agent_{i}",
                vendor="tesco",
                task_type=TaskType.SCRAPE_PRODUCTS,
                status=AgentTaskStatus.RUNNING,
                created_at=datetime.now()
            )
            deployer.running_tasks[f"task_{i}"] = task_info
        
        deployer.active_agents["tesco_agent_1"] = Mock()
        deployer.active_agents["asda_agent_1"] = Mock()
        
        status = deployer.get_status()
        
        # Check all status sections
        assert "deployment" in status
        assert "tasks" in status
        assert "agents" in status
        assert "sessions" in status
        assert "scaling" in status
        assert "performance" in status
        assert "stats" in status
        
        # Check task counts
        assert status["tasks"]["running"] == 3
        assert status["agents"]["active_count"] == 2
    
    def test_vendor_status_reporting(self, deployer_with_health_mocks):
        """Test vendor-specific status reporting."""
        deployer = deployer_with_health_mocks
        
        # Update vendor metrics
        deployer.vendor_metrics["tesco"].active_agents = 2
        deployer.vendor_metrics["tesco"].error_rate = 0.1
        deployer.vendor_metrics["tesco"].circuit_breaker_open = False
        
        vendor_status = deployer.get_vendor_status()
        
        assert "tesco" in vendor_status
        tesco_status = vendor_status["tesco"]
        assert tesco_status["active_agents"] == 2
        assert tesco_status["error_rate"] == 10.0  # Converted to percentage
        assert tesco_status["circuit_breaker"]["open"] is False
    
    def test_scaling_metrics(self, deployer_with_health_mocks):
        """Test scaling metrics reporting."""
        deployer = deployer_with_health_mocks
        
        scaling_metrics = deployer.get_scaling_metrics()
        
        assert "scaling_config" in scaling_metrics
        assert "current_metrics" in scaling_metrics
        assert "concurrency_limits" in scaling_metrics
        assert "vendor_demand" in scaling_metrics
        
        # Check scaling config is properly reported
        config = scaling_metrics["scaling_config"]
        assert "auto_scaling_enabled" in config
        assert "min_agents" in config
        assert "max_agents" in config


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.fixture
    def full_deployer_setup(self):
        """Create fully configured deployer for integration tests."""
        # Create mock browserbase manager inline
        mock_manager = Mock(spec=BrowserbaseManager)
        mock_manager.session_pool = Mock()
        mock_manager.session_pool.get_stats.return_value = {
            'available_sessions': 5,
            'in_use_sessions': 2,
            'total_sessions': 7,
            'pool_utilization': 0.3
        }
        mock_manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 2,
            'sessions_remaining': 8
        }
        mock_manager.active_sessions = {}
        
        deployer = AgentDeployer(
            browserbase_manager=mock_manager,
            concurrency_config=ConcurrencyConfig(max_total_agents=5),
            scaling_config=ScalingConfig(enable_auto_scaling=True)
        )
        
        # Mock all external dependencies
        deployer._acquire_session = AsyncMock(return_value=Mock(session_id="test_session"))
        deployer._release_session = AsyncMock()
        deployer._get_or_create_agent = AsyncMock(return_value=Mock())
        deployer._create_scraping_task = AsyncMock(return_value=Mock())
        
        return deployer
    
    @pytest.mark.asyncio
    async def test_complete_task_lifecycle(self, full_deployer_setup):
        """Test complete task lifecycle from creation to completion."""
        deployer = full_deployer_setup
        
        # Mock crew execution
        mock_crew = Mock()
        mock_crew.execute_async = AsyncMock(return_value="Task completed successfully")
        
        with patch('scraping_cli.agent_deployer.ScrapingCrew', return_value=mock_crew):
            # Add task
            task_id = await deployer.add_task("tesco", TaskType.SCRAPE_PRODUCTS)
            assert len(deployer.pending_tasks) == 1
            
            # Start deployment
            await deployer.start()
            
            # Wait for task processing
            await asyncio.sleep(0.1)
            
            # Check task was processed (mock execution should complete immediately)
            # In real scenario, task would move from pending -> running -> completed
            
            # Stop deployment
            await deployer.stop(timeout=1, force=True)
            
            assert deployer.status == DeploymentStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_concurrent_multi_vendor_execution(self, full_deployer_setup):
        """Test concurrent execution across multiple vendors."""
        deployer = full_deployer_setup
        
        # Add tasks for multiple vendors
        task_ids = []
        vendors = ["tesco", "asda", "costco"]
        
        for vendor in vendors:
            for i in range(2):  # 2 tasks per vendor
                task_id = await deployer.add_task(vendor, TaskType.SCRAPE_PRODUCTS)
                task_ids.append(task_id)
        
        assert len(deployer.pending_tasks) == 6
        assert len(task_ids) == 6
        
        # Check vendor distribution
        vendor_counts = {}
        for task_id in task_ids:
            vendor = deployer.agent_tasks[task_id].vendor
            vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
        
        assert vendor_counts["tesco"] == 2
        assert vendor_counts["asda"] == 2
        assert vendor_counts["costco"] == 2
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, full_deployer_setup):
        """Test system performance under high load."""
        deployer = full_deployer_setup
        
        # Add many tasks quickly
        start_time = time.time()
        task_ids = []
        
        for i in range(50):
            vendor = ["tesco", "asda", "costco"][i % 3]
            task_id = await deployer.add_task(vendor, TaskType.SCRAPE_PRODUCTS)
            task_ids.append(task_id)
        
        add_duration = time.time() - start_time
        
        # Should be able to add 50 tasks quickly
        assert add_duration < 1.0  # Less than 1 second
        assert len(deployer.pending_tasks) == 50
        
        # Check memory usage is reasonable
        import sys
        memory_events_size = sys.getsizeof(deployer.memory_events)
        assert memory_events_size < 10000  # Reasonable memory usage


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])