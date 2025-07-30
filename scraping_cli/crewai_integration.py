"""
CrewAI Integration Module

Provides CrewAI integration for agent orchestration in the scraping CLI.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from crewai import Crew, Agent, Task
from crewai.tools import BaseTool


class AgentRole(Enum):
    """Available agent roles for scraping tasks."""
    SCRAPER = "scraper"
    ANALYZER = "analyzer"
    COORDINATOR = "coordinator"
    VALIDATOR = "validator"


class TaskType(Enum):
    """Types of tasks that can be performed."""
    SCRAPE_PRODUCTS = "scrape_products"
    SCRAPE_CATEGORIES = "scrape_categories"
    ANALYZE_DATA = "analyze_data"
    VALIDATE_RESULTS = "validate_results"
    COORDINATE_SCRAPING = "coordinate_scraping"


@dataclass
class AgentConfig:
    """Configuration for creating agents."""
    role: AgentRole
    name: str
    goal: str
    backstory: str
    verbose: bool = False
    allow_delegation: bool = True
    tools: Optional[List[BaseTool]] = None


@dataclass
class TaskConfig:
    """Configuration for creating tasks."""
    task_type: TaskType
    description: str
    expected_output: str
    agent: Optional[Agent] = None
    context: Optional[Dict[str, Any]] = None
    async_execution: bool = True


class ScrapingAgent:
    """Base class for scraping agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.memory_events = []
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create a CrewAI agent from configuration."""
        agent = Agent(
            role=self.config.role.value,
            goal=self.config.goal,
            backstory=self.config.backstory,
            verbose=self.config.verbose,
            allow_delegation=self.config.allow_delegation,
            tools=self.config.tools or []
        )
        
        # Log agent creation event
        self._log_memory_event("agent_created", details={
            "role": self.config.role.value,
            "name": self.config.name,
            "goal": self.config.goal,
            "tools_count": len(self.config.tools or [])
        })
        
        return agent
    
    def _log_memory_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        """Log a memory event for this agent."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'agent_id': self.config.name,
            'details': details or {}
        }
        self.memory_events.append(event)
        self.logger.info(f"Agent MemoryEvent: {event_type} - Agent: {self.config.name}, Details: {details}")
    
    def get_agent(self) -> Agent:
        """Get the underlying CrewAI agent."""
        return self.agent
    
    def get_memory_events(self) -> List[Dict[str, Any]]:
        """Get all memory events for this agent."""
        return self.memory_events


class ScrapingTask:
    """Base class for scraping tasks."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.memory_events = []
        self.task = self._create_task()
    
    def _create_task(self) -> Task:
        """Create a CrewAI task from configuration."""
        task = Task(
            description=self.config.description,
            expected_output=self.config.expected_output,
            agent=self.config.agent
        )
        
        # Log task creation event
        self._log_memory_event("task_created", details={
            "type": self.config.task_type.value,
            "description_length": len(self.config.description),
            "async_execution": self.config.async_execution
        })
        
        return task
    
    def _log_memory_event(self, event_type: str, details: Optional[Dict[str, Any]] = None):
        """Log a memory event for this task."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'task_id': f"task_{self.config.task_type.value}",
            'details': details or {}
        }
        self.memory_events.append(event)
        self.logger.info(f"Task MemoryEvent: {event_type} - Task: {self.config.task_type.value}, Details: {details}")
    
    def get_task(self) -> Task:
        """Get the underlying CrewAI task."""
        return self.task
    
    def get_memory_events(self) -> List[Dict[str, Any]]:
        """Get all memory events for this task."""
        return self.memory_events


class ScrapingCrew:
    """Manages a crew of agents for scraping tasks."""
    
    def __init__(self, agents: List[ScrapingAgent], tasks: List[ScrapingTask]):
        self.agents = agents
        self.tasks = tasks
        self.crew = self._create_crew()
        self.memory_events = []
        self.logger = logging.getLogger(__name__)
    
    def _create_crew(self) -> Crew:
        """Create a CrewAI crew from agents and tasks."""
        crew_agents = [agent.get_agent() for agent in self.agents]
        crew_tasks = [task.get_task() for task in self.tasks]
        
        return Crew(
            agents=crew_agents,
            tasks=crew_tasks,
            verbose=True
        )
    
    def _log_memory_event(self, event_type: str, agent_id: Optional[str] = None, 
                          task_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Log a memory event for observability."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'agent_id': agent_id,
            'task_id': task_id,
            'details': details or {}
        }
        self.memory_events.append(event)
        self.logger.info(f"MemoryEvent: {event_type} - Agent: {agent_id}, Task: {task_id}, Details: {details}")
    
    def get_memory_events(self) -> List[Dict[str, Any]]:
        """Get all memory events for observability."""
        return self.memory_events
    
    def export_memory_events(self, format: str = 'json') -> str:
        """Export memory events in specified format."""
        if format == 'json':
            import json
            return json.dumps(self.memory_events, indent=2)
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            if self.memory_events:
                # Get all unique fieldnames from all events
                fieldnames = set()
                for event in self.memory_events:
                    fieldnames.update(event.keys())
                fieldnames = sorted(list(fieldnames))
                
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.memory_events)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def execute_async(self) -> str:
        """Execute the crew asynchronously."""
        try:
            self._log_memory_event("crew_execution_start", details={"mode": "async"})
            
            # Log agent and task information
            for i, agent in enumerate(self.agents):
                self._log_memory_event("agent_ready", agent_id=f"agent_{i}", 
                                     details={"role": agent.config.role.value, "name": agent.config.name})
            
            for i, task in enumerate(self.tasks):
                self._log_memory_event("task_ready", task_id=f"task_{i}", 
                                     details={"type": task.config.task_type.value})
            
            result = await self.crew.kickoff_async()
            
            self._log_memory_event("crew_execution_complete", details={"mode": "async", "result_length": len(result)})
            return result
        except Exception as e:
            self._log_memory_event("crew_execution_error", details={"mode": "async", "error": str(e)})
            raise Exception(f"Crew execution failed: {e}")
    
    def execute_sync(self) -> str:
        """Execute the crew synchronously."""
        try:
            self._log_memory_event("crew_execution_start", details={"mode": "sync"})
            
            # Log agent and task information
            for i, agent in enumerate(self.agents):
                self._log_memory_event("agent_ready", agent_id=f"agent_{i}", 
                                     details={"role": agent.config.role.value, "name": agent.config.name})
            
            for i, task in enumerate(self.tasks):
                self._log_memory_event("task_ready", task_id=f"task_{i}", 
                                     details={"type": task.config.task_type.value})
            
            result = self.crew.kickoff()
            
            self._log_memory_event("crew_execution_complete", details={"mode": "sync", "result_length": len(result)})
            return result
        except Exception as e:
            self._log_memory_event("crew_execution_error", details={"mode": "sync", "error": str(e)})
            raise Exception(f"Crew execution failed: {e}")
    
    def get_crew(self) -> Crew:
        """Get the underlying CrewAI crew."""
        return self.crew


class AgentFactory:
    """Factory for creating different types of agents."""
    
    @staticmethod
    def create_scraper_agent(vendor: str, category: Optional[str] = None) -> ScrapingAgent:
        """Create a scraper agent for a specific vendor."""
        goal = f"Scrape product data from {vendor} website"
        if category:
            goal += f" focusing on {category} products"
        
        backstory = f"""You are an expert web scraper specializing in {vendor} e-commerce platforms. 
        You have extensive experience extracting product information, prices, and availability data 
        from {vendor} websites. You are thorough and ensure data accuracy."""
        
        config = AgentConfig(
            role=AgentRole.SCRAPER,
            name=f"{vendor.title()}Scraper",
            goal=goal,
            backstory=backstory,
            verbose=True
        )
        
        return ScrapingAgent(config)
    
    @staticmethod
    def create_analyzer_agent() -> ScrapingAgent:
        """Create an analyzer agent for data processing."""
        config = AgentConfig(
            role=AgentRole.ANALYZER,
            name="DataAnalyzer",
            goal="Analyze and process scraped product data to extract insights and validate information",
            backstory="""You are a data analyst expert specializing in e-commerce product data. 
            You excel at processing raw scraped data, identifying patterns, and ensuring data quality. 
            You can validate product information and extract meaningful insights from large datasets.""",
            verbose=True
        )
        
        return ScrapingAgent(config)
    
    @staticmethod
    def create_coordinator_agent() -> ScrapingAgent:
        """Create a coordinator agent for managing scraping workflows."""
        config = AgentConfig(
            role=AgentRole.COORDINATOR,
            name="ScrapingCoordinator",
            goal="Coordinate and manage scraping workflows across multiple agents and tasks",
            backstory="""You are a project coordinator specializing in web scraping operations. 
            You excel at managing multiple agents, coordinating tasks, and ensuring smooth workflow execution. 
            You can handle complex scraping scenarios and optimize resource allocation.""",
            verbose=True
        )
        
        return ScrapingAgent(config)
    
    @staticmethod
    def create_validator_agent() -> ScrapingAgent:
        """Create a validator agent for data validation."""
        config = AgentConfig(
            role=AgentRole.VALIDATOR,
            name="DataValidator",
            goal="Validate scraped data for accuracy, completeness, and consistency",
            backstory="""You are a data validation expert specializing in e-commerce product data. 
            You have a keen eye for detecting inconsistencies, missing information, and data quality issues. 
            You ensure that all scraped data meets quality standards before final output.""",
            verbose=True
        )
        
        return ScrapingAgent(config)


class TaskFactory:
    """Factory for creating different types of tasks."""
    
    @staticmethod
    def create_scrape_products_task(agent: Agent, urls: List[str], vendor: str) -> ScrapingTask:
        """Create a task for scraping product data."""
        description = f"""Scrape product information from the following {vendor} URLs:
        {chr(10).join(f"- {url}" for url in urls)}
        
        For each product, extract:
        - Product name
        - Price
        - Availability
        - Product description
        - Product images
        - Product specifications
        - Customer reviews (if available)
        """
        
        expected_output = """A comprehensive list of products with all requested information 
        in a structured format suitable for analysis and export."""
        
        config = TaskConfig(
            task_type=TaskType.SCRAPE_PRODUCTS,
            description=description,
            expected_output=expected_output,
            agent=agent,
            async_execution=True
        )
        
        return ScrapingTask(config)
    
    @staticmethod
    def create_analyze_data_task(agent: Agent, scraped_data: Dict[str, Any]) -> ScrapingTask:
        """Create a task for analyzing scraped data."""
        description = """Analyze the scraped product data to:
        1. Identify pricing trends
        2. Detect product availability patterns
        3. Extract key product features
        4. Generate summary statistics
        5. Identify data quality issues
        """
        
        expected_output = """A comprehensive analysis report including:
        - Pricing analysis and trends
        - Availability statistics
        - Product categorization
        - Data quality assessment
        - Key insights and recommendations
        """
        
        config = TaskConfig(
            task_type=TaskType.ANALYZE_DATA,
            description=description,
            expected_output=expected_output,
            agent=agent,
            async_execution=True
        )
        
        return ScrapingTask(config)
    
    @staticmethod
    def create_validate_results_task(agent: Agent, results: Dict[str, Any]) -> ScrapingTask:
        """Create a task for validating scraping results."""
        description = """Validate the scraping results for:
        1. Data completeness
        2. Information accuracy
        3. Format consistency
        4. Missing critical fields
        5. Duplicate entries
        """
        
        expected_output = """A validation report including:
        - Data quality score
        - List of issues found
        - Recommendations for improvement
        - Validation summary
        """
        
        config = TaskConfig(
            task_type=TaskType.VALIDATE_RESULTS,
            description=description,
            expected_output=expected_output,
            agent=agent,
            async_execution=True
        )
        
        return ScrapingTask(config)


def create_agent_factory() -> AgentFactory:
    """Create and return a new agent factory instance."""
    return AgentFactory()


def create_task_factory() -> TaskFactory:
    """Create and return a new task factory instance."""
    return TaskFactory()


def create_scraping_crew(agents: List[ScrapingAgent], tasks: List[ScrapingTask]) -> ScrapingCrew:
    """Create and return a new scraping crew instance."""
    return ScrapingCrew(agents, tasks) 