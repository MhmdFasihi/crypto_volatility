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
    ],
    python_requires=">=3.8",
) 