import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from src.dashboard
from src.dashboard import main

if __name__ == "__main__":
    main()