"""
Progress Monitoring System

Provides real-time progress output and status updates for the CLI interface.
Supports multiple display modes, ANSI colors, and terminal width detection.
"""

import sys
import shutil
from typing import Dict, Optional, Any, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for cross-platform ANSI color support
init(autoreset=True)


class DisplayMode(Enum):
    """Available display modes for progress monitoring."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    DEBUG = "debug"


class StatusLevel(Enum):
    """Status levels for logging."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_task: Optional[str] = None
    agent_status: Dict[str, str] = field(default_factory=dict)
    browser_sessions: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None


class ProgressMonitor:
    """
    Real-time progress monitoring system for CLI interface.
    
    Supports multiple display modes, ANSI colors, and terminal width detection.
    """
    
    def __init__(self, 
                 total_tasks: int = 0,
                 mode: DisplayMode = DisplayMode.SIMPLE,
                 disable: bool = None):
        """
        Initialize the progress monitor.
        
        Args:
            total_tasks: Total number of tasks to track
            mode: Display mode (simple, detailed, debug)
            disable: Whether to disable progress bars (auto-detected if None)
        """
        self.mode = mode
        self.stats = ProgressStats(total_tasks=total_tasks)
        self.stats.start_time = datetime.now()
        
        # Auto-detect if we should disable progress bars
        if disable is None:
            disable = not sys.stdout.isatty()
        
        self.disable = disable
        
        # Get terminal width for responsive display
        try:
            self.term_width = shutil.get_terminal_size((80, 20)).columns
        except (OSError, AttributeError):
            self.term_width = 80
        
        # Initialize progress bars based on mode
        self._init_progress_bars()
    
    def _init_progress_bars(self) -> None:
        """Initialize progress bars based on display mode."""
        self.bars = {}
        
        if self.disable:
            return
        
        # Main progress bar
        if self.mode in [DisplayMode.SIMPLE, DisplayMode.DETAILED, DisplayMode.DEBUG]:
            self.bars['main'] = tqdm(
                total=self.stats.total_tasks,
                desc=self._get_colored_desc("Overall Progress"),
                ncols=self.term_width,
                disable=self.disable,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        # Detailed mode: individual agent bars
        if self.mode == DisplayMode.DETAILED:
            # These will be created dynamically as agents are added
            pass
        
        # Debug mode: additional debug bar
        if self.mode == DisplayMode.DEBUG:
            self.bars['debug'] = tqdm(
                total=0,
                desc=self._get_colored_desc("Debug Info"),
                ncols=self.term_width,
                disable=self.disable,
                position=1,
                leave=True
            )
    
    def _get_colored_desc(self, text: str, color: str = Fore.CYAN) -> str:
        """Get colored description text."""
        if self.disable:
            return text
        return f"{color}{text}{Style.RESET_ALL}"
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status text."""
        status_lower = status.lower()
        if 'error' in status_lower or 'failed' in status_lower:
            return Fore.RED
        elif 'warning' in status_lower:
            return Fore.YELLOW
        elif 'success' in status_lower or 'completed' in status_lower:
            return Fore.GREEN
        elif 'working' in status_lower or 'running' in status_lower:
            return Fore.BLUE
        else:
            return Fore.WHITE
    
    def update_progress(self, 
                       n: int = 1,
                       task_name: Optional[str] = None,
                       agent_status: Optional[Dict[str, str]] = None,
                       browser_status: Optional[Dict[str, str]] = None,
                       errors: Optional[List[str]] = None) -> None:
        """
        Update progress and status information.
        
        Args:
            n: Number of tasks completed
            task_name: Name of current task
            agent_status: Agent status updates
            browser_status: Browser session status updates
            errors: New errors to add
        """
        # Update statistics
        self.stats.completed_tasks += n
        self.stats.current_task = task_name
        self.stats.last_update = datetime.now()
        
        if agent_status:
            self.stats.agent_status.update(agent_status)
        
        if browser_status:
            self.stats.browser_sessions.update(browser_status)
        
        if errors:
            self.stats.errors.extend(errors)
            self.stats.failed_tasks += len(errors)
        
        # Update progress bars
        if not self.disable:
            self._update_bars()
    
    def _update_bars(self) -> None:
        """Update all progress bars with current status."""
        if 'main' in self.bars:
            # Update main progress bar
            postfix = {}
            
            if self.stats.current_task:
                postfix['task'] = self.stats.current_task
            
            # Add agent status if available
            if self.stats.agent_status:
                active_agents = [k for k, v in self.stats.agent_status.items() 
                               if 'working' in v.lower() or 'running' in v.lower()]
                if active_agents:
                    postfix['agents'] = f"{len(active_agents)} active"
            
            # Add browser session status
            if self.stats.browser_sessions:
                active_sessions = [k for k, v in self.stats.browser_sessions.items() 
                                 if 'active' in v.lower()]
                if active_sessions:
                    postfix['sessions'] = f"{len(active_sessions)} active"
            
            # Add error count
            if self.stats.errors:
                postfix['errors'] = len(self.stats.errors)
            
            self.bars['main'].set_postfix(postfix)
            self.bars['main'].update(0)  # Update without incrementing
    
    def log(self, message: str, level: StatusLevel = StatusLevel.INFO) -> None:
        """
        Log a message with appropriate color and formatting.
        
        Args:
            message: Message to log
            level: Log level for color coding
        """
        if self.disable:
            print(message)
            return
        
        # Get color for level
        if level == StatusLevel.INFO:
            color = Fore.CYAN
        elif level == StatusLevel.WARNING:
            color = Fore.YELLOW
        elif level == StatusLevel.ERROR:
            color = Fore.RED
        elif level == StatusLevel.SUCCESS:
            color = Fore.GREEN
        else:
            color = Fore.WHITE
        
        # Format message with timestamp for debug mode
        if self.mode == DisplayMode.DEBUG:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}"
        else:
            formatted_message = message
        
        # Write message using tqdm to avoid interfering with progress bars
        tqdm.write(f"{color}{formatted_message}{Style.RESET_ALL}")
    
    def set_agent_status(self, agent_name: str, status: str) -> None:
        """Set status for a specific agent."""
        self.stats.agent_status[agent_name] = status
        self.log(f"Agent '{agent_name}': {status}", 
                StatusLevel.INFO if 'working' in status.lower() else StatusLevel.WARNING)
        self._update_bars()
    
    def set_browser_session_status(self, session_id: str, status: str) -> None:
        """Set status for a browser session."""
        self.stats.browser_sessions[session_id] = status
        self.log(f"Browser session '{session_id}': {status}", StatusLevel.INFO)
        self._update_bars()
    
    def add_error(self, error: str) -> None:
        """Add an error to the tracking."""
        self.stats.errors.append(error)
        self.stats.failed_tasks += 1
        self.log(f"Error: {error}", StatusLevel.ERROR)
        self._update_bars()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of progress statistics."""
        if self.stats.start_time and self.stats.last_update:
            duration = self.stats.last_update - self.stats.start_time
        else:
            duration = None
        
        return {
            'total_tasks': self.stats.total_tasks,
            'completed_tasks': self.stats.completed_tasks,
            'failed_tasks': self.stats.failed_tasks,
            'success_rate': (self.stats.completed_tasks / self.stats.total_tasks * 100) if self.stats.total_tasks > 0 else 0,
            'duration': duration,
            'errors': self.stats.errors,
            'agent_status': self.stats.agent_status,
            'browser_sessions': self.stats.browser_sessions
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of the progress."""
        summary = self.get_summary()
        
        if self.disable:
            self._print_summary_plain(summary)
        else:
            self._print_summary_colored(summary)
    
    def _print_summary_plain(self, summary: Dict[str, Any]) -> None:
        """Print summary without colors."""
        print("\n" + "="*50)
        print("PROGRESS SUMMARY")
        print("="*50)
        print(f"Total Tasks: {summary['total_tasks']}")
        print(f"Completed: {summary['completed_tasks']}")
        print(f"Failed: {summary['failed_tasks']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['duration']:
            print(f"Duration: {summary['duration']}")
        
        if summary['errors']:
            print(f"\nErrors ({len(summary['errors'])}):")
            for error in summary['errors']:
                print(f"  - {error}")
        
        if summary['agent_status']:
            print(f"\nAgent Status:")
            for agent, status in summary['agent_status'].items():
                print(f"  - {agent}: {status}")
    
    def _print_summary_colored(self, summary: Dict[str, Any]) -> None:
        """Print summary with colors."""
        # Close all progress bars first
        for bar in self.bars.values():
            bar.close()
        
        print("\n" + "="*50)
        print(f"{Fore.CYAN}PROGRESS SUMMARY{Style.RESET_ALL}")
        print("="*50)
        
        # Success rate color
        success_rate = summary['success_rate']
        if success_rate >= 90:
            rate_color = Fore.GREEN
        elif success_rate >= 70:
            rate_color = Fore.YELLOW
        else:
            rate_color = Fore.RED
        
        print(f"Total Tasks: {Fore.WHITE}{summary['total_tasks']}{Style.RESET_ALL}")
        print(f"Completed: {Fore.GREEN}{summary['completed_tasks']}{Style.RESET_ALL}")
        print(f"Failed: {Fore.RED}{summary['failed_tasks']}{Style.RESET_ALL}")
        print(f"Success Rate: {rate_color}{success_rate:.1f}%{Style.RESET_ALL}")
        
        if summary['duration']:
            print(f"Duration: {Fore.CYAN}{summary['duration']}{Style.RESET_ALL}")
        
        if summary['errors']:
            print(f"\n{Fore.RED}Errors ({len(summary['errors'])}):{Style.RESET_ALL}")
            for error in summary['errors']:
                print(f"  {Fore.RED}-{Style.RESET_ALL} {error}")
        
        if summary['agent_status']:
            print(f"\n{Fore.BLUE}Agent Status:{Style.RESET_ALL}")
            for agent, status in summary['agent_status'].items():
                status_color = self._get_status_color(status)
                print(f"  {Fore.BLUE}-{Style.RESET_ALL} {agent}: {status_color}{status}{Style.RESET_ALL}")
    
    def close(self) -> None:
        """Close all progress bars and cleanup."""
        for bar in self.bars.values():
            bar.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_progress_monitor(total_tasks: int = 0,
                          mode: str = "simple",
                          disable: bool = None) -> ProgressMonitor:
    """
    Factory function to create a progress monitor.
    
    Args:
        total_tasks: Total number of tasks to track
        mode: Display mode ("simple", "detailed", "debug")
        disable: Whether to disable progress bars
    
    Returns:
        ProgressMonitor instance
    """
    try:
        display_mode = DisplayMode(mode.lower())
    except ValueError:
        display_mode = DisplayMode.SIMPLE
    
    return ProgressMonitor(
        total_tasks=total_tasks,
        mode=display_mode,
        disable=disable
    ) 