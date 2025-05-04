from setuptools import setup, find_packages

setup(
    name="crypto_volatility",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.36.0",
        "pandas>=2.2.2",  
        "numpy>=1.26.4",
        "yfinance>=0.2.40",
        "scikit-learn>=1.5.0",
        "tensorflow>=2.16.1",
        "plotly>=5.22.0",
        "hmmlearn>=0.3.2",
        "dtw-python>=1.5.1",
        "joblib>=1.4.2",
        "pytest>=8.2.0",
        "websockets>=12.0",
    ],
    python_requires=">=3.8",
)