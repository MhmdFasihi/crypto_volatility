"""
Setup script for crypto_volatility package.
"""

from setuptools import setup, find_packages

setup(
    name="crypto_volatility",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.31.1",
        "numpy>=1.26.0",  # Updated for Python 3.12 compatibility
        "pandas>=2.0.0",
        "yfinance>=0.2.28",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "plotly==5.18.0",
        "hmmlearn==0.3.0",
        "dtw-python==1.3.0",
        "joblib==1.3.2",
        "python-dateutil==2.8.2",
        "pytz==2024.1",
        "requests>=2.31.0",
        "websockets==12.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-mock==3.12.0",
        "setuptools==69.0.3",
        "black==24.1.1",
        "flake8==7.0.0",
        "mypy==1.8.0",
        "isort==5.13.2",
        "sphinx==7.2.6",
        "sphinx-rtd-theme==2.0.0",
        "python-json-logger>=2.0.7",
        "prometheus-client==0.19.0",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "statsmodels>=0.14.0",
        "dtaidistance>=2.3.10",
        "fastdtw>=0.3.4"
    ],
    python_requires=">=3.8",
    author="Mohammad Fasihi",
    author_email="mhmdfasihi@gmail.com",
    description="Cryptocurrency volatility analysis and forecasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MhmdFasihi/crypto_volatility",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)