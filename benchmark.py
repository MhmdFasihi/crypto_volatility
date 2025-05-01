# benchmark.py
"""
Performance benchmark script for the crypto volatility analysis system.
"""

import time
import psutil
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any

from src.data_acquisition import get_data
from src.preprocessing import calculate_returns, calculate_volatility
from src.forecasting import train_mlp, train_rnn, prepare_data
from src.clustering import cluster_tickers
from src.classification import train_hmm
from src.anomaly_detection import ensemble_anomaly_detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceBenchmark:
    """Performance benchmarking for all components."""
    
    def __init__(self):
        self.results = {}
        self.memory_usage = {}
    
    def measure_performance(self, func, *args, **kwargs):
        """Measure execution time and memory usage of a function."""
        # Start memory monitoring
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # End memory monitoring
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'peak_memory': end_memory
        }
    
    def benchmark_data_acquisition(self):
        """Benchmark data acquisition performance."""
        logger.info("Benchmarking data acquisition...")
        
        tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD']
        periods = ['1y', '2y', '5y']
        
        results = {}
        for ticker in tickers:
            for period in periods:
                end_date = datetime.now().strftime('%Y-%m-%d')
                if period == '1y':
                    start_date = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
                elif period == '2y':
                    start_date = (datetime.now() - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
                else:
                    start_date = (datetime.now() - pd.Timedelta(days=1825)).strftime('%Y-%m-%d')
                
                perf = self.measure_performance(get_data, ticker, start_date, end_date)
                results[f"{ticker}_{period}"] = perf
        
        self.results['data_acquisition'] = results
    
    def benchmark_preprocessing(self):
        """Benchmark preprocessing performance."""
        logger.info("Benchmarking preprocessing...")
        
        # Generate test data
        data_sizes = [100, 1000, 5000, 10000]
        results = {}
        
        for size in data_sizes:
            # Create synthetic data
            dates = pd.date_range(end=datetime.now(), periods=size, freq='D')
            data = pd.DataFrame({
                'Close': np.random.randn(size).cumsum() + 100,
                'Open': np.random.randn(size).cumsum() + 100,
                'High': np.random.randn(size).cumsum() + 105,
                'Low': np.random.randn(size).cumsum() + 95
            }, index=dates)
            
            # Benchmark returns calculation
            returns_perf = self.measure_performance(calculate_returns, data)
            
            # Benchmark volatility calculation
            data_with_returns = returns_perf['result']
            vol_perf = self.measure_performance(calculate_volatility, data_with_returns)
            
            results[f"size_{size}"] = {
                'returns': returns_perf,
                'volatility': vol_perf
            }
        
        self.results['preprocessing'] = results
    
    def benchmark_model_training(self):
        """Benchmark model training performance."""
        logger.info("Benchmarking model training...")
        
        # Generate test data
        data_sizes = [100, 500, 1000]
        lag_values = [5, 10, 20]
        
        results = {}
        for size in data_sizes:
            for lags in lag_values:
                # Create synthetic data
                dates = pd.date_range(end=datetime.now(), periods=size, freq='D')
                data = pd.DataFrame({
                    'Volatility': np.random.randn(size) * 0.02 + 0.2
                }, index=dates)
                
                # Prepare data
                X, y = prepare_data(data, lags=lags)
                if len(X) > 50:
                    X_train, y_train = X[:50], y[:50]
                    
                    # Benchmark MLP
                    mlp_perf = self.measure_performance(
                        train_mlp, X_train.values, y_train.values
                    )
                    
                    # Benchmark RNN
                    X_rnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
                    rnn_perf = self.measure_performance(
                        train_rnn, X_rnn, y_train.values, epochs=10
                    )
                    
                    results[f"size_{size}_lags_{lags}"] = {
                        'mlp': mlp_perf,
                        'rnn': rnn_perf
                    }
        
        self.results['model_training'] = results
    
    def benchmark_clustering(self):
        """Benchmark clustering performance."""
        logger.info("Benchmarking clustering...")
        
        # Test with different numbers of tickers
        ticker_counts = [5, 10, 20]
        results = {}
        
        for count in ticker_counts:
            # Create mock data
            mock_data = {}
            for i in range(count):
                ticker = f"CRYPTO{i}-USD"
                dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                mock_data[ticker] = pd.DataFrame({
                    'Close': np.random.randn(100).cumsum() + 100
                }, index=dates)
            
            # Mock the get_data function
            def mock_get_data(ticker, start, end):
                return mock_data.get(ticker, pd.DataFrame())
            
            # Benchmark clustering
            from unittest.mock import patch
            with patch('src.clustering.get_data', mock_get_data):
                perf = self.measure_performance(
                    cluster_tickers, 
                    list(mock_data.keys()), 
                    n_clusters=3
                )
                results[f"tickers_{count}"] = perf
        
        self.results['clustering'] = results
    
    def generate_report(self):
        """Generate performance benchmark report."""
        
        # Add execution time summary
        total_time = 0
        for component, results in self.results.items():
            component_time = 0
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict) and 'execution_time' in value:
                        component_time += value['execution_time']
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict) and 'execution_time' in sub_value:
                                component_time += sub_value['execution_time']
            
            report += f"{component}: {component_time:.3f} seconds\n"
            total_time += component_time
        
        report += f"\nTotal Execution Time: {total_time:.3f} seconds\n"
        
        # Add detailed results
        report += "\nDetailed Results\n---------------\n"
        
        for component, results in self.results.items():
            report += f"\n{component.upper()}\n"
            report += "-" * len(component) + "\n"
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict) and 'execution_time' in value:
                        report += f"  {key}: {value['execution_time']:.3f}s, Memory: {value['memory_delta']:.1f}MB\n"
                    elif isinstance(value, dict):
                        report += f"  {key}:\n"
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, dict) and 'execution_time' in sub_value:
                                report += f"    {sub_key}: {sub_value['execution_time']:.3f}s, Memory: {sub_value['memory_delta']:.1f}MB\n"
        
        return report
    
    def create_visualization(self):
        """Create performance visualization charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Benchmark Results')
        
        # Data acquisition performance
        ax1 = axes[0, 0]
        if 'data_acquisition' in self.results:
            times = []
            labels = []
            for key, value in self.results['data_acquisition'].items():
                times.append(value['execution_time'])
                labels.append(key)
            
            ax1.bar(labels, times)
            ax1.set_title('Data Acquisition Performance')
            ax1.set_ylabel('Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
        
        # Preprocessing performance
        ax2 = axes[0, 1]
        if 'preprocessing' in self.results:
            sizes = []
            returns_times = []
            vol_times = []
            
            for key, value in self.results['preprocessing'].items():
                size = int(key.split('_')[1])
                sizes.append(size)
                returns_times.append(value['returns']['execution_time'])
                vol_times.append(value['volatility']['execution_time'])
            
            ax2.plot(sizes, returns_times, 'o-', label='Returns')
            ax2.plot(sizes, vol_times, 's-', label='Volatility')
            ax2.set_title('Preprocessing Performance')
            ax2.set_xlabel('Data Size')
            ax2.set_ylabel('Time (seconds)')
            ax2.legend()
        
        # Model training performance
        ax3 = axes[1, 0]
        if 'model_training' in self.results:
            mlp_times = []
            rnn_times = []
            labels = []
            
            for key, value in self.results['model_training'].items():
                labels.append(key)
                mlp_times.append(value['mlp']['execution_time'])
                rnn_times.append(value['rnn']['execution_time'])
            
            x = range(len(labels))
            width = 0.35
            ax3.bar([i - width/2 for i in x], mlp_times, width, label='MLP')
            ax3.bar([i + width/2 for i in x], rnn_times, width, label='RNN')
            ax3.set_title('Model Training Performance')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(labels, rotation=45)
            ax3.legend()
        
        # Memory usage
        ax4 = axes[1, 1]
        memory_data = []
        memory_labels = []
        
        for component, results in self.results.items():
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict) and 'memory_delta' in value:
                        memory_data.append(value['memory_delta'])
                        memory_labels.append(f"{component}_{key}")
        
        if memory_data:
            ax4.bar(range(len(memory_data)), memory_data)
            ax4.set_title('Memory Usage')
            ax4.set_ylabel('Memory (MB)')
            ax4.set_xticks(range(len(memory_labels)))
            ax4.set_xticklabels(memory_labels, rotation=90)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_benchmark(self):
        """Run complete performance benchmark."""
        logger.info("Starting performance benchmark...")
        
        # Run all benchmarks
        self.benchmark_preprocessing()
        self.benchmark_model_training()
        self.benchmark_clustering()
        
        # Note: Skip data acquisition in automated tests to avoid API calls
        # self.benchmark_data_acquisition()
        
        # Generate report
        report = self.generate_report()
        
        # Save report
        with open('benchmark_report.txt', 'w') as f:
            f.write(report)
        
        # Create visualization
        self.create_visualization()
        
        # Print summary
        print(report)
        
        return self.results

def main():
    """Main entry point."""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_benchmark()
    
    # Check if performance is acceptable
    acceptable = True
    for component, data in results.items():
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and 'execution_time' in value:
                    if value['execution_time'] > 30:  # 30 seconds threshold
                        logger.warning(f"{component}/{key} took too long: {value['execution_time']:.3f}s")
                        acceptable = False
    
    if acceptable:
        logger.info("✅ Performance benchmark PASSED")
    else:
        logger.warning("⚠️ Some components exceeded performance thresholds")

if __name__ == '__main__':
    main()