"""
Main entry point for Streamlit dashboard application.
"""

import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the main function from dashboard
from src.dashboard import main

if __name__ == "__main__":
    # Execute the main function from the dashboard module
    main()