#!/usr/bin/env python3
"""
Standalone script to cleanup Browserbase sessions.
Run this script to ensure all sessions are properly closed.

Usage:
    python scripts/cleanup_browserbase_sessions.py
"""

import os
import sys
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraping_cli.browserbase_manager import BrowserbaseManager


def setup_logging():
    """Setup logging for the cleanup script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def cleanup_sessions():
    """Close all running Browserbase sessions."""
    logger = logging.getLogger(__name__)
    
    try:
        # Get credentials from environment
        api_key = os.getenv('BROWSERBASE_API_KEY')
        project_id = os.getenv('BROWSERBASE_PROJECT_ID')
        
        if not api_key or not project_id:
            logger.error("❌ Browserbase credentials not found in environment variables.")
            logger.error("Please set BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID")
            return False
        
        logger.info(f"🔧 Initializing BrowserbaseManager for project: {project_id}")
        
        # Create manager
        manager = BrowserbaseManager(
            api_key=api_key,
            project_id=project_id,
            pool_size=1,  # Minimal pool for cleanup
            max_retries=2
        )
        
        logger.info("🔍 Checking for active sessions...")
        
        # Get all sessions from Browserbase API
        try:
            # List all sessions
            sessions = manager.bb.sessions.list()
            active_sessions = [s for s in sessions if s.status == 'running']
            
            if not active_sessions:
                logger.info("✅ No active sessions found.")
                return True
            
            logger.info(f"📊 Found {len(active_sessions)} active sessions:")
            
            # Close each session
            for i, session in enumerate(active_sessions, 1):
                try:
                    logger.info(f"  {i}. Closing session {session.id}...")
                    manager.bb.sessions.delete(session.id)
                    logger.info(f"     ✅ Session {session.id} closed successfully")
                except Exception as e:
                    logger.error(f"     ❌ Failed to close session {session.id}: {e}")
            
            logger.info(f"🎉 Cleanup completed. Closed {len(active_sessions)} sessions.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error listing sessions: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        return False


def main():
    """Main function for the cleanup script."""
    print("🧹 Browserbase Session Cleanup")
    print("=" * 40)
    
    setup_logging()
    success = cleanup_sessions()
    
    if success:
        print("\n✅ Cleanup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Cleanup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 