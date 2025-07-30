#!/usr/bin/env python3
"""
Example script demonstrating BrowserbaseManager session limiting functionality.

This script shows how to:
1. Set session limits via constructor parameters
2. Set session limits via environment variables
3. Handle session limit errors gracefully
4. Monitor session usage and limits
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraping_cli.browserbase_manager import BrowserbaseManager, SessionConfig
from scraping_cli.exceptions import SessionCreationError


def create_manager_with_limits(max_concurrent_sessions: int = 2) -> BrowserbaseManager:
    """Create a BrowserbaseManager with session limits."""
    return BrowserbaseManager(
        api_key=os.getenv('BROWSERBASE_API_KEY'),
        project_id=os.getenv('BROWSERBASE_PROJECT_ID'),
        pool_size=5,
        max_concurrent_sessions=max_concurrent_sessions,
        max_retries=2,
        session_timeout=60
    )


def demonstrate_session_limits():
    """Demonstrate session limiting functionality."""
    print("🔧 BrowserbaseManager Session Limits Demo")
    print("=" * 50)
    
    # Check if credentials are available
    if not os.getenv('BROWSERBASE_API_KEY') or not os.getenv('BROWSERBASE_PROJECT_ID'):
        print("❌ Browserbase credentials not found.")
        print("Please set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables.")
        return
    
    # Create manager with 2 concurrent session limit
    manager = create_manager_with_limits(max_concurrent_sessions=2)
    
    print(f"✅ Created BrowserbaseManager with max_concurrent_sessions={manager.max_concurrent_sessions}")
    
    # Show initial session limits
    limits = manager.get_session_limits()
    print(f"\n📊 Initial Session Limits:")
    print(f"  Max Concurrent Sessions: {limits['max_concurrent_sessions']}")
    print(f"  Current Active Sessions: {limits['current_active_sessions']}")
    print(f"  Sessions Remaining: {limits['sessions_remaining']}")
    print(f"  Utilization: {limits['utilization_percentage']:.1f}%")
    
    # Try to create sessions up to the limit
    sessions = []
    print(f"\n🔄 Creating sessions (limit: {manager.max_concurrent_sessions})...")
    
    for i in range(manager.max_concurrent_sessions + 1):  # Try to exceed limit
        try:
            session = manager.get_session()
            sessions.append(session)
            print(f"  ✅ Created session {session.session_id} ({i + 1}/{manager.max_concurrent_sessions})")
            
            # Show updated limits
            limits = manager.get_session_limits()
            print(f"    Sessions remaining: {limits['sessions_remaining']}")
            
        except SessionCreationError as e:
            print(f"  ❌ Failed to create session {i + 1}: {e}")
            break
    
    print(f"\n📊 Final Session Status:")
    limits = manager.get_session_limits()
    print(f"  Current Active Sessions: {limits['current_active_sessions']}")
    print(f"  Sessions Remaining: {limits['sessions_remaining']}")
    print(f"  Utilization: {limits['utilization_percentage']:.1f}%")
    
    # Release all sessions
    print(f"\n🔄 Releasing sessions...")
    for session in sessions:
        manager.release_session(session)
        print(f"  ✅ Released session {session.session_id}")
    
    # Show final status
    limits = manager.get_session_limits()
    print(f"\n📊 After Release:")
    print(f"  Current Active Sessions: {limits['current_active_sessions']}")
    print(f"  Sessions Remaining: {limits['sessions_remaining']}")
    print(f"  Utilization: {limits['utilization_percentage']:.1f}%")
    
    # Close all sessions
    manager.close_all_sessions()
    print(f"\n✅ Demo completed successfully!")


def demonstrate_environment_variable_limits():
    """Demonstrate setting session limits via environment variables."""
    print("\n🔧 Environment Variable Session Limits Demo")
    print("=" * 50)
    
    # Set environment variable for session limit
    os.environ['BROWSERBASE_MAX_SESSIONS'] = '3'
    
    # Create manager without explicit max_concurrent_sessions
    manager = BrowserbaseManager(
        api_key=os.getenv('BROWSERBASE_API_KEY'),
        project_id=os.getenv('BROWSERBASE_PROJECT_ID'),
        pool_size=5
    )
    
    print(f"✅ Created BrowserbaseManager with max_concurrent_sessions={manager.max_concurrent_sessions}")
    print(f"  (Set via BROWSERBASE_MAX_SESSIONS environment variable)")
    
    # Show session limits
    limits = manager.get_session_limits()
    print(f"\n📊 Session Limits:")
    print(f"  Max Concurrent Sessions: {limits['max_concurrent_sessions']}")
    print(f"  Current Active Sessions: {limits['current_active_sessions']}")
    print(f"  Sessions Remaining: {limits['sessions_remaining']}")
    
    # Clean up
    manager.close_all_sessions()
    
    # Remove environment variable
    if 'BROWSERBASE_MAX_SESSIONS' in os.environ:
        del os.environ['BROWSERBASE_MAX_SESSIONS']


def demonstrate_concurrent_session_usage():
    """Demonstrate concurrent session usage with limits."""
    print("\n🔧 Concurrent Session Usage Demo")
    print("=" * 50)
    
    manager = create_manager_with_limits(max_concurrent_sessions=3)
    
    def worker(worker_id: int) -> str:
        """Worker function that uses a session."""
        try:
            session = manager.get_session()
            print(f"  Worker {worker_id}: Got session {session.session_id}")
            
            # Simulate some work
            time.sleep(1)
            
            # Release session
            manager.release_session(session)
            print(f"  Worker {worker_id}: Released session {session.session_id}")
            
            return f"Worker {worker_id} completed successfully"
            
        except SessionCreationError as e:
            return f"Worker {worker_id} failed: {e}"
    
    # Try to run 5 workers with only 3 session limit
    print(f"🔄 Running 5 workers with 3 session limit...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker, i) for i in range(5)]
        
        for future in as_completed(futures):
            result = future.result()
            print(f"  {result}")
    
    # Show final status
    limits = manager.get_session_limits()
    print(f"\n📊 Final Status:")
    print(f"  Current Active Sessions: {limits['current_active_sessions']}")
    print(f"  Sessions Remaining: {limits['sessions_remaining']}")
    print(f"  Utilization: {limits['utilization_percentage']:.1f}%")
    
    # Clean up
    manager.close_all_sessions()


def demonstrate_error_handling():
    """Demonstrate error handling for session limits."""
    print("\n🔧 Session Limit Error Handling Demo")
    print("=" * 50)
    
    # Create manager with very low limit
    manager = create_manager_with_limits(max_concurrent_sessions=1)
    
    # Create one session
    session1 = manager.get_session()
    print(f"✅ Created first session: {session1.session_id}")
    
    # Try to create another session - should fail
    try:
        session2 = manager.get_session()
        print(f"✅ Created second session: {session2.session_id}")
    except SessionCreationError as e:
        print(f"❌ Failed to create second session: {e}")
        
        # Show current limits
        limits = manager.get_session_limits()
        print(f"📊 Current limits:")
        print(f"  Max Concurrent Sessions: {limits['max_concurrent_sessions']}")
        print(f"  Current Active Sessions: {limits['current_active_sessions']}")
        print(f"  Sessions Remaining: {limits['sessions_remaining']}")
    
    # Release the first session
    manager.release_session(session1)
    print(f"✅ Released first session: {session1.session_id}")
    
    # Now try to create another session - should succeed
    try:
        session2 = manager.get_session()
        print(f"✅ Created second session after release: {session2.session_id}")
        manager.release_session(session2)
    except SessionCreationError as e:
        print(f"❌ Still failed to create second session: {e}")
    
    # Clean up
    manager.close_all_sessions()


def main():
    """Main function to run all demonstrations."""
    print("🚀 BrowserbaseManager Session Limits Demonstration")
    print("=" * 60)
    
    # Check if credentials are available
    if not os.getenv('BROWSERBASE_API_KEY') or not os.getenv('BROWSERBASE_PROJECT_ID'):
        print("❌ Browserbase credentials not found.")
        print("Please set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID environment variables.")
        print("\nExample:")
        print("  export BROWSERBASE_API_KEY='your-api-key'")
        print("  export BROWSERBASE_PROJECT_ID='your-project-id'")
        print("  python examples/session_limits_example.py")
        return
    
    try:
        # Run all demonstrations
        demonstrate_session_limits()
        demonstrate_environment_variable_limits()
        demonstrate_concurrent_session_usage()
        demonstrate_error_handling()
        
        print("\n🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 