# validate_system.py
"""
System validation script for the crypto volatility analysis framework.
"""

import sys
import os
import subprocess
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SystemValidator:
    """System validation and health check."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def check_python_version(self):
        """Check Python version compatibility."""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.results['python_version'] = {
                'status': 'PASS',
                'version': f"{version.major}.{version.minor}.{version.micro}"
            }
        else:
            self.results['python_version'] = {
                'status': 'FAIL',
                'version': f"{version.major}.{version.minor}.{version.micro}",
                'error': 'Python 3.8+ required'
            }
    
    def check_dependencies(self):
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        required_packages = [
            'yfinance', 'pandas', 'numpy', 'scikit-learn', 
            'tensorflow', 'hmmlearn', 'dtw-python', 'streamlit', 
            'plotly', 'joblib', 'pytest', 'websockets'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            self.results['dependencies'] = {'status': 'PASS'}
        else:
            self.results['dependencies'] = {
                'status': 'FAIL',
                'missing': missing_packages
            }
    
    def check_directory_structure(self):
        """Verify directory structure."""
        logger.info("Checking directory structure...")
        
        required_dirs = ['src', 'tests', 'models', 'data', 'docs']
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
        
        if not missing_dirs:
            self.results['directory_structure'] = {'status': 'PASS'}
        else:
            self.results['directory_structure'] = {
                'status': 'FAIL',
                'missing': missing_dirs
            }
    
    def check_source_files(self):
        """Check if all source files exist."""
        logger.info("Checking source files...")
        
        required_files = [
            'src/data_acquisition.py',
            'src/preprocessing.py',
            'src/forecasting.py',
            'src/clustering.py',
            'src/classification.py',
            'src/anomaly_detection.py',
            'src/dashboard.py',
            'src/train_models.py',
            'src/config.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if not missing_files:
            self.results['source_files'] = {'status': 'PASS'}
        else:
            self.results['source_files'] = {
                'status': 'FAIL',
                'missing': missing_files
            }
    
    def run_unit_tests(self):
        """Run unit tests."""
        logger.info("Running unit tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/', '-q'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.results['unit_tests'] = {
                    'status': 'PASS',
                    'output': result.stdout
                }
            else:
                self.results['unit_tests'] = {
                    'status': 'FAIL',
                    'error': result.stderr
                }
        except Exception as e:
            self.results['unit_tests'] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def check_api_connectivity(self):
        """Check connectivity to external APIs."""
        logger.info("Checking API connectivity...")
        
        try:
            import yfinance as yf
            import websockets
            import asyncio
            
            # Test yfinance
            ticker = yf.Ticker("BTC-USD")
            info = ticker.info
            yfinance_status = 'PASS' if info else 'FAIL'
            
            # Test Deribit (websocket)
            async def test_deribit():
                try:
                    async with websockets.connect('wss://test.deribit.com/ws/api/v2') as ws:
                        return 'PASS'
                except:
                    return 'FAIL'
            
            deribit_status = asyncio.run(test_deribit())
            
            self.results['api_connectivity'] = {
                'status': 'PASS' if yfinance_status == 'PASS' and deribit_status == 'PASS' else 'FAIL',
                'yfinance': yfinance_status,
                'deribit': deribit_status
            }
            
        except Exception as e:
            self.results['api_connectivity'] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def check_model_training(self):
        """Test model training functionality."""
        logger.info("Testing model training...")
        
        try:
            from tests.test_config import generate_mock_data
            from src.preprocessing import calculate_returns, calculate_volatility
            from src.forecasting import train_mlp, prepare_data
            
            # Generate test data
            data = generate_mock_data('BTC-USD', days=30)
            data = calculate_returns(data)
            data = calculate_volatility(data)
            
            # Prepare data
            X, y = prepare_data(data)
            if len(X) > 10:
                X_train, y_train = X[:10], y[:10]
                model, scaler = train_mlp(X_train.values, y_train.values)
                
                self.results['model_training'] = {'status': 'PASS'}
            else:
                self.results['model_training'] = {
                    'status': 'FAIL',
                    'error': 'Insufficient data for training'
                }
                
        except Exception as e:
            self.results['model_training'] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def check_dashboard(self):
        """Check if dashboard can be launched."""
        logger.info("Checking dashboard...")
        
        try:
            # Test if streamlit is available
            result = subprocess.run(
                [sys.executable, '-m', 'streamlit', '--version'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.results['dashboard'] = {
                    'status': 'PASS',
                    'version': result.stdout.strip()
                }
            else:
                self.results['dashboard'] = {
                    'status': 'FAIL',
                    'error': result.stderr
                }
                
        except Exception as e:
            self.results['dashboard'] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def generate_report(self):
        """Generate validation report."""
        elapsed_time = time.time() - self.start_time
        
        # Count passes, fails, and errors
        passes = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        fails = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        errors = sum(1 for r in self.results.values() if r['status'] == 'ERROR')
        
        report = f"""
System Validation Report
========================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {elapsed_time:.2f} seconds

Summary
-------
Total Checks: {len(self.results)}
Passed: {passes}
Failed: {fails}
Errors: {errors}

Detailed Results
----------------
"""
        
        for check_name, result in self.results.items():
            status = result['status']
            report += f"\n{check_name}: {status}"
            
            if status == 'FAIL':
                if 'error' in result:
                    report += f"\n  Error: {result['error']}"
                if 'missing' in result:
                    report += f"\n  Missing: {', '.join(result['missing'])}"
            elif status == 'ERROR':
                report += f"\n  Error: {result['error']}"
        
        return report
    
    def run_validation(self):
        """Run all validation checks."""
        logger.info("Starting system validation...")
        
        # Run all checks
        self.check_python_version()
        self.check_dependencies()
        self.check_directory_structure()
        self.check_source_files()
        self.run_unit_tests()
        self.check_api_connectivity()
        self.check_model_training()
        self.check_dashboard()
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        with open('validation_report.txt', 'w') as f:
            f.write(report)
        
        # Print report
        print(report)
        
        # Return overall status
        all_passed = all(r['status'] == 'PASS' for r in self.results.values())
        return all_passed

def main():
    """Main entry point."""
    validator = SystemValidator()
    success = validator.run_validation()
    
    if success:
        logger.info("✅ System validation PASSED")
        sys.exit(0)
    else:
        logger.error("❌ System validation FAILED")
        sys.exit(1)

if __name__ == '__main__':
    main()