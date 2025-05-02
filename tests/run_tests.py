# tests/run_tests.py
"""
Main test runner for the crypto volatility analysis project.
"""

import pytest
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all unit tests."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pytest on all test files
    exit_code = pytest.main([
        test_dir,
        '-v',  # Verbose output
        '--tb=short',  # Shorter traceback format
        '-p', 'no:warnings',  # Disable warnings
        '--color=yes'  # Colored output
    ])
    
    return exit_code

def run_specific_test(test_file):
    """Run a specific test file."""
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_file)
    
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_file}")
        return 1
    
    exit_code = pytest.main([
        test_path,
        '-v',
        '--tb=short',
        '-p', 'no:warnings',
        '--color=yes'
    ])
    
    return exit_code

def main():
    """Main entry point for the test runner."""
    if len(sys.argv) > 1:
        # Run specific test file
        test_file = sys.argv[1]
        print(f"Running tests from: {test_file}")
        exit_code = run_specific_test(test_file)
    else:
        # Run all tests
        print("Running all tests...")
        exit_code = run_all_tests()
    
    # Print summary
    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed.")
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()