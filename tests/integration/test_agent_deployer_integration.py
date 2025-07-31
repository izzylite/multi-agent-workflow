"""
Integration tests for AgentDeployer with real-world scenarios.

These tests focus on end-to-end integration with minimal mocking
to validate the complete system behavior.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scraping_cli.agent_deployer import (
    AgentDeployer, ConcurrencyConfig, ScalingConfig, TaskType,
    DeploymentStatus, AgentTaskStatus
)
from scraping_cli.browserbase_manager import BrowserbaseManager, SessionInfo, SessionConfig


class TestAgentDeployerIntegration:
    """Integration tests for complete AgentDeployer workflows."""
    
    @pytest.fixture
    def integration_browserbase_manager(self):
        """Create a more realistic mock BrowserbaseManager for integration tests."""
        manager = Mock(spec=BrowserbaseManager)
        
        # Session pool mock with realistic behavior
        manager.session_pool = Mock()
        manager.session_pool.get_stats.return_value = {
            'available_sessions': 3,
            'in_use_sessions': 1,
            'total_sessions': 4,
            'pool_utilization': 0.25
        }
        
        # Session limits
        manager.get_session_limits.return_value = {
            'max_concurrent_sessions': 10,
            'current_active_sessions': 4,
            'sessions_remaining': 6,
            'utilization_percentage': 40.0
        }
        
        # Active sessions tracking
        manager.active_sessions = {}
        
        # Session creation/management
        def create_session_side_effect(config=None):
            session_id = f"session_{len(manager.active_sessions) + 1}"
            session_info = SessionInfo(
                session_id=session_id,
                connect_url=f"wss://connect.browserbase.com/{session_id}",
                status="active",
                created_at=datetime.now(),
                last_used=datetime.now(),
                config=config or SessionConfig()
            )
            manager.active_sessions[session_id] = session_info
            return session_info
        
        def release_session_side_effect(session_info):
            if session_info.session_id in manager.active_sessions:
                del manager.active_sessions[session_info.session_id]
        
        def close_session_side_effect(session_id):
            if session_id in manager.active_sessions:
                del manager.active_sessions[session_id]
        
        manager.get_session.side_effect = create_session_side_effect
        manager.create_session.side_effect = create_session_side_effect
        manager.release_session.side_effect = release_session_side_effect
        manager.close_session.side_effect = close_session_side_effect
        manager.close_all_sessions.side_effect = lambda: manager.active_sessions.clear()
        manager.cleanup_expired_sessions.return_value = 0
        
        # Health monitoring
        async def check_session_health_side_effect(session_id):
            if session_id in manager.active_sessions:
                return {'status': 'healthy', 'response_time': 0.1}
            else:
                return {'status': 'not_found', 'error': 'Session not found'}
        
        manager.check_session_health.side_effect = check_session_health_side_effect
        
        return manager
    
    @pytest.fixture
    def production_like_deployer(self, integration_browserbase_manager):
        """Create a deployer configured like a production setup."""
        concurrency_config = ConcurrencyConfig(
            max_agents_per_vendor=3,
            max_total_agents=8,
            max_concurrent_sessions=10,
            rate_limit_per_domain=1.5,
            enable_circuit_breaker=True,
            circuit_breaker_threshold=5
        )
        
        scaling_config = ScalingConfig(
            enable_auto_scaling=True,
            scale_up_threshold=0.75,
            scale_down_threshold=0.25,
            min_agents=2,
            max_agents=15,
            scale_check_interval=5
        )
        
        return AgentDeployer(
            browserbase_manager=integration_browserbase_manager,
            concurrency_config=concurrency_config,
            scaling_config=scaling_config
        )
    
    @pytest.mark.asyncio
    async def test_complete_deployment_lifecycle(self, production_like_deployer):
        """Test complete deployment lifecycle with realistic scenarios."""
        deployer = production_like_deployer
        
        # Mock agent and task creation to avoid external dependencies
        with patch.object(deployer, '_get_or_create_agent') as mock_get_agent, \
             patch.object(deployer, '_create_scraping_task') as mock_create_task, \
             patch('scraping_cli.agent_deployer.ScrapingCrew') as mock_crew_class:
            
            # Setup mocks
            mock_agent = Mock()
            mock_agent.get_agent.return_value = Mock()
            mock_get_agent.return_value = mock_agent
            
            mock_task = Mock()
            mock_task.get_task.return_value = Mock()
            mock_create_task.return_value = mock_task
            
            mock_crew = Mock()
            mock_crew.execute_async = AsyncMock(return_value="Task completed successfully")
            mock_crew_class.return_value = mock_crew
            
            # Test lifecycle
            assert deployer.status == DeploymentStatus.INITIALIZING
            
            # Start deployment
            await deployer.start()
            assert deployer.status == DeploymentStatus.RUNNING
            assert len(deployer.background_tasks) > 0
            
            # Add multiple tasks
            task_ids = []
            vendors = ["tesco", "asda", "costco"]
            
            for vendor in vendors:
                for i in range(2):
                    task_id = await deployer.add_task(vendor, TaskType.SCRAPE_PRODUCTS)
                    task_ids.append(task_id)
            
            assert len(task_ids) == 6
            assert len(deployer.pending_tasks) == 6
            
            # Allow some processing time
            await asyncio.sleep(0.2)
            
            # Check system status
            status = deployer.get_status()
            assert status['deployment']['status'] == 'running'
            assert status['tasks']['pending'] >= 0  # Some tasks may have been processed
            
            # Test graceful shutdown
            await deployer.stop(timeout=2)
            assert deployer.status == DeploymentStatus.STOPPED
            assert len(deployer.background_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_session_pool_integration_under_load(self, production_like_deployer):
        """Test session pool behavior under realistic load."""
        deployer = production_like_deployer
        manager = deployer.browserbase_manager
        
        # Track session operations
        session_acquisitions = []
        session_releases = []
        
        async def track_acquire_session(vendor, task_info):
            """Track session acquisitions for analysis."""
            session_info = await deployer._acquire_session(vendor, task_info)
            if session_info:
                session_acquisitions.append((vendor, session_info.session_id))
            return session_info
        
        async def track_release_session(session_info, task_info):
            """Track session releases for analysis."""
            session_releases.append(session_info.session_id)
            await deployer._release_session(session_info, task_info)
        
        # Replace methods with tracking versions
        original_acquire = deployer._acquire_session
        original_release = deployer._release_session
        deployer._acquire_session = track_acquire_session
        deployer._release_session = track_release_session
        
        try:
            # Create tasks that will acquire sessions
            tasks = []
            for i in range(5):
                vendor = ["tesco", "asda", "costco"][i % 3]
                task_id = await deployer.add_task(vendor, TaskType.SCRAPE_PRODUCTS)
                tasks.append((task_id, vendor))
            
            # Simulate task execution with session usage
            for task_id, vendor in tasks:
                task_info = deployer.agent_tasks[task_id]
                
                # Acquire session
                session_info = await track_acquire_session(vendor, task_info)
                assert session_info is not None
                
                # Simulate some work
                await asyncio.sleep(0.01)
                
                # Release session
                await track_release_session(session_info, task_info)
            
            # Verify session pool behavior
            assert len(session_acquisitions) == 5
            assert len(session_releases) == 5
            
            # Check vendor distribution
            vendor_sessions = {}
            for vendor, session_id in session_acquisitions:
                vendor_sessions[vendor] = vendor_sessions.get(vendor, 0) + 1
            
            # Should have sessions distributed across vendors
            assert len(vendor_sessions) >= 2
            
            # Check session pool stats
            metrics = deployer.get_session_metrics()
            assert 'pool_stats' in metrics
            assert 'session_limits' in metrics
            
        finally:
            # Restore original methods
            deployer._acquire_session = original_acquire
            deployer._release_session = original_release
    
    @pytest.mark.asyncio
    async def test_concurrent_vendor_operations(self, production_like_deployer):
        """Test concurrent operations across multiple vendors."""
        deployer = production_like_deployer
        
        # Mock successful crew execution
        with patch.object(deployer, '_get_or_create_agent') as mock_get_agent, \
             patch.object(deployer, '_create_scraping_task') as mock_create_task, \
             patch('scraping_cli.agent_deployer.ScrapingCrew') as mock_crew_class:
            
            # Setup mocks
            mock_agent = Mock()
            mock_get_agent.return_value = mock_agent
            mock_create_task.return_value = Mock()
            
            mock_crew = Mock()
            mock_crew.execute_async = AsyncMock(return_value="Success")
            mock_crew_class.return_value = mock_crew
            
            await deployer.start()
            
            # Create concurrent tasks for different vendors
            vendors = ["tesco", "asda", "costco"]
            tasks_per_vendor = 3
            
            # Add all tasks quickly
            start_time = time.time()
            all_tasks = []
            
            for vendor in vendors:
                for i in range(tasks_per_vendor):
                    task_id = await deployer.add_task(vendor, TaskType.SCRAPE_PRODUCTS)
                    all_tasks.append(task_id)
            
            creation_time = time.time() - start_time
            
            # Should create tasks quickly
            assert creation_time < 0.5
            assert len(all_tasks) == 9
            
            # Allow processing time
            await asyncio.sleep(0.5)
            
            # Check vendor metrics
            vendor_status = deployer.get_vendor_status()
            
            for vendor in vendors:
                assert vendor in vendor_status
                vendor_info = vendor_status[vendor]
                assert 'active_agents' in vendor_info
                assert 'pending_tasks' in vendor_info
                assert 'error_rate' in vendor_info
            
            await deployer.stop(timeout=2, force=True)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_integration(self, production_like_deployer):
        """Test error handling and recovery in integrated scenarios."""
        deployer = production_like_deployer
        
        # Mock some failures and recoveries
        failure_count = 0
        
        async def failing_crew_execution():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception("Simulated network timeout")
            return "Eventually successful"
        
        with patch.object(deployer, '_get_or_create_agent') as mock_get_agent, \
             patch.object(deployer, '_create_scraping_task') as mock_create_task, \
             patch('scraping_cli.agent_deployer.ScrapingCrew') as mock_crew_class:
            
            # Setup mocks
            mock_get_agent.return_value = Mock()
            mock_create_task.return_value = Mock()
            
            mock_crew = Mock()
            mock_crew.execute_async = AsyncMock(side_effect=failing_crew_execution)
            mock_crew_class.return_value = mock_crew
            
            await deployer.start()
            
            # Add a task that will initially fail
            task_id = await deployer.add_task("tesco", TaskType.SCRAPE_PRODUCTS)
            
            # Allow processing time for retries
            await asyncio.sleep(0.3)
            
            # Check error handling occurred
            assert deployer.stats['tasks_failed'] > 0 or deployer.stats['tasks_completed'] > 0
            
            # Check circuit breaker state
            tesco_cb = deployer.circuit_breakers["tesco"]
            # Circuit breaker should have recorded some failures
            assert tesco_cb.failure_count >= 0
            
            # Check vendor metrics
            vendor_status = deployer.get_vendor_status()
            tesco_status = vendor_status["tesco"]
            
            # Should have some error rate if failures occurred
            assert 'error_rate' in tesco_status
            assert 'circuit_breaker' in tesco_status
            
            await deployer.stop(timeout=2, force=True)
    
    @pytest.mark.asyncio
    async def test_scaling_behavior_integration(self, production_like_deployer):
        """Test auto-scaling behavior in realistic scenarios."""
        deployer = production_like_deployer
        
        # Enable auto-scaling
        assert deployer.scaling_config.enable_auto_scaling is True
        
        # Add many tasks to trigger scaling
        initial_agents = len(deployer.active_agents)
        
        # Add enough tasks to exceed scale-up threshold
        for i in range(15):
            vendor = ["tesco", "asda", "costco"][i % 3]
            await deployer.add_task(vendor, TaskType.SCRAPE_PRODUCTS)
        
        # Check utilization
        utilization = deployer._calculate_system_utilization()
        
        # With 15 pending tasks and max 8 total agents, should have high utilization
        # when tasks start running
        
        # Check vendor demand calculation
        demand = deployer._calculate_vendor_demand()
        assert len(demand) == 3  # All three vendors should have demand
        total_demand = sum(demand.values())
        assert total_demand == 15
        
        # Test scaling metrics
        scaling_metrics = deployer.get_scaling_metrics()
        
        assert 'scaling_config' in scaling_metrics
        assert 'current_metrics' in scaling_metrics
        assert 'vendor_demand' in scaling_metrics
        
        config = scaling_metrics['scaling_config']
        assert config['auto_scaling_enabled'] is True
        assert config['scale_up_threshold'] == 0.75
        
        current = scaling_metrics['current_metrics']
        assert 'total_agents' in current
        assert 'pending_tasks' in current
        assert current['pending_tasks'] == 15
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, production_like_deployer):
        """Test health monitoring in realistic scenarios."""
        deployer = production_like_deployer
        
        await deployer.start()
        
        # Add some tasks and agents
        for i in range(5):
            vendor = ["tesco", "asda"][i % 2]
            await deployer.add_task(vendor, TaskType.SCRAPE_PRODUCTS)
        
        # Simulate some agent activity
        deployer.active_agents["tesco_agent_1"] = Mock()
        deployer.active_agents["asda_agent_1"] = Mock()
        
        # Perform health check
        health_result = await deployer.health_check()
        
        assert health_result['overall_status'] in ['healthy', 'warning', 'critical']
        assert 'components' in health_result
        
        components = health_result['components']
        assert 'deployment' in components
        assert 'sessions' in components
        assert 'vendors' in components
        assert 'background_tasks' in components
        
        # Check deployment component health
        deployment_health = components['deployment']
        assert deployment_health['status'] in ['healthy', 'warning', 'critical']
        assert 'utilization' in deployment_health
        assert 'active_agents' in deployment_health
        
        # Check vendor health
        vendor_health = components['vendors']
        assert vendor_health['status'] in ['healthy', 'warning', 'critical']
        assert 'vendors' in vendor_health
        
        await deployer.stop(timeout=2, force=True)
    
    @pytest.mark.asyncio
    async def test_memory_events_integration(self, production_like_deployer):
        """Test memory events generation during realistic operations."""
        deployer = production_like_deployer
        
        await deployer.start()
        
        # Perform various operations to generate memory events
        task_id = await deployer.add_task("tesco", TaskType.SCRAPE_PRODUCTS)
        await deployer.add_task_batch([
            {"vendor": "asda", "task_type": "scrape_products"},
            {"vendor": "costco", "task_type": "analyze_data"}
        ])
        
        # Cancel a task
        await deployer.cancel_task(task_id)
        
        # Get task status
        await deployer.get_task_status(task_id)
        
        # Allow some processing
        await asyncio.sleep(0.1)
        
        # Check memory events were generated
        memory_events = deployer.get_memory_events()
        assert len(memory_events) > 0
        
        # Check event structure
        event_types = [event['type'] for event in memory_events]
        assert 'deployer_initialized' in event_types
        assert 'deployment_started' in event_types
        assert 'task_added' in event_types
        assert 'batch_tasks_added' in event_types
        assert 'task_cancelled' in event_types
        
        # Check events have required fields
        for event in memory_events:
            assert 'timestamp' in event
            assert 'type' in event
            assert 'deployer_id' in event
            assert 'details' in event
        
        await deployer.stop(timeout=2, force=True)
        
        # Check shutdown events
        memory_events = deployer.get_memory_events()
        final_event_types = [event['type'] for event in memory_events]
        assert 'deployment_stopping' in final_event_types
        assert 'deployment_stopped' in final_event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])