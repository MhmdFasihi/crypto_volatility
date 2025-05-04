import os
import sys

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the Python path
sys.path.insert(0, current_dir)

# Try importing directly from the dashboard module
try:
    from src.dashboard import main
except ImportError:
    # If that fails, try adding src explicitly to the path
    sys.path.insert(0, os.path.join(current_dir, "src"))
    from dashboard import main

if __name__ == "__main__":
    main()