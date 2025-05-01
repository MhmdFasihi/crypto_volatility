# quick_start.py
"""
Quick start script for the crypto volatility analysis system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickStart:
    """Quick start setup for the system."""
    
    def __init__(self):
        self.root_dir = Path.cwd()
    
    def check_python_version(self):
        """Check Python version."""
        logger.info("Checking Python version...")
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
        logger.info(f"Python {sys.version} - OK")
        return True
    
    def create_directories(self):
        """Create required directories."""
        logger.info("Creating directories...")
        directories = ['data', 'models', 'logs']
        for dir_name in directories:
            path = self.root_dir / dir_name
            path.mkdir(exist_ok=True)
            logger.info(f"Created {dir_name}/")
    
    def install_dependencies(self):
        """Install dependencies."""
        logger.info("Installing dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def create_env_file(self):
        """Create environment file template."""
        logger.info("Creating .env file...")
        env_content = """# Environment Variables
DEBUG=False
LOG_LEVEL=INFO

# API Keys (add your keys here)
YFINANCE_API_KEY=
DERIBIT_API_KEY=

# Application Settings
DATA_DIR=data/
MODEL_DIR=models/
"""
        env_path = self.root_dir / '.env'
        if not env_path.exists():
            with open(env_path, 'w') as f:
                f.write(env_content)
            logger.info("Created .env file template")
        else:
            logger.info(".env file already exists")
    
    def run_initial_tests(self):
        """Run initial tests."""
        logger.info("Running initial tests...")
        try:
            result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-q'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Tests passed successfully")
                return True
            else:
                logger.warning("Some tests failed")
                return False
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def train_sample_model(self):
        """Train a sample model."""
        logger.info("Training sample model...")
        try:
            subprocess.run([sys.executable, 'src/train_models.py', 
                          '--ticker', 'BTC-USD', '--model', 'mlp'], check=True)
            logger.info("Sample model trained successfully")
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def launch_dashboard(self):
        """Launch the dashboard."""
        logger.info("Launching dashboard...")
        logger.info("Dashboard will be available at http://localhost:8501")
        logger.info("Press Ctrl+C to stop the dashboard")
        
        try:
            subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'src/dashboard.py'])
        except KeyboardInterrupt:
            logger.info("Dashboard stopped")
    
    def run(self):
        """Run quick start setup."""
        print("""
Crypto Volatility Analysis - Quick Start
=======================================
""")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Create environment file
        self.create_env_file()
        
        # Run tests
        test_success = self.run_initial_tests()
        
        # Train sample model
        model_success = self.train_sample_model()
        
        # Print summary
        print("""
Setup Complete!
--------------
""")
        print(f"✓ Python version: {sys.version.split()[0]}")
        print("✓ Directories created")
        print("✓ Dependencies installed")
        print("✓ Environment file created")
        print(f"{'✓' if test_success else '⚠'} Tests: {'Passed' if test_success else 'Some failures'}")
        print(f"{'✓' if model_success else '⚠'} Sample model: {'Trained' if model_success else 'Failed'}")
        print("""
Next Steps:
-----------
1. Edit .env file with your API keys
2. Launch the dashboard:
   python quick_start.py --dashboard
   
3. Or train more models:
   python src/train_models.py --ticker ETH-USD --model all
""")
        
        # Ask if user wants to launch dashboard
        response = input("\nLaunch dashboard now? (y/n): ")
        if response.lower() == 'y':
            self.launch_dashboard()
        
        return True

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start for crypto volatility analysis')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard directly')
    args = parser.parse_args()
    
    quick_start = QuickStart()
    
    if args.dashboard:
        quick_start.launch_dashboard()
    else:
        quick_start.run()

if __name__ == '__main__':
    main()