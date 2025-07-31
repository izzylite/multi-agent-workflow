"""
Test runner for AgentDeployer test suite.

Run this file to execute all AgentDeployer tests with proper configuration.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_agent_deployer_tests():
    """Run all AgentDeployer tests with optimal configuration."""
    
    # Test files to run
    test_files = [
        "tests/unit/test_agent_deployer.py",
        "tests/integration/test_agent_deployer_integration.py"
    ]
    
    # Pytest arguments for optimal test execution
    pytest_args = [
        "-v",  # Verbose output
        "--asyncio-mode=auto",  # Automatic async test detection
        "--tb=short",  # Short traceback format
        "--disable-warnings",  # Suppress warnings for cleaner output
        "--durations=10",  # Show 10 slowest tests
        "-x",  # Stop on first failure for faster debugging
    ] + test_files
    
    print("🧪 Running AgentDeployer Test Suite")
    print("=" * 50)
    print("Test files:")
    for test_file in test_files:
        print(f"  - {test_file}")
    print("=" * 50)
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All AgentDeployer tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    # Run the test suite
    exit_code = run_agent_deployer_tests()
    sys.exit(exit_code)