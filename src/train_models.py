"""
Script to train and save models for cryptocurrency volatility analysis.
"""

import argparse
import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Use absolute imports
from src.data_acquisition import get_data, get_combined_volatility_data
from src.preprocessing import (
    calculate_returns, 
    calculate_volatility, 
    calculate_advanced_volatility_metrics,
    create_volatility_features
)
from src.forecasting import (
    prepare_data, 
    train_mlp, 
    train_rnn, 
    evaluate_model, 
    save_model,
    cross_validate_time_series
)
from src.classification import train_hmm, get_state_statistics
from src.anomaly_detection import (
    detect_anomalies_zscore,
    create_anomaly_features,
    train_random_forest_detector,
    evaluate_anomaly_detector
)
from src.config import (
    TICKERS, VOL_WINDOW, LAGS, HMM_STATES,
    DEFAULT_START_DATE, DEFAULT_END_DATE,
    RANDOM_SEED, TEST_SIZE, MODELS_DIR as MODEL_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_forecasting_model(ticker: str, 
                           model_type: str, 
                           data: pd.DataFrame,
                           use_cv: bool = False) -> dict:
    """
    Train a forecasting model for a given ticker.
    
    Args:
        ticker: Crypto ticker
        model_type: Model type ('mlp', 'rnn', 'lstm', 'gru')
        data: Preprocessed data
        use_cv: Whether to use cross-validation
    
    Returns:
        Dictionary with training results
    """
    logger.info(f"Training {model_type} model for {ticker}")
    
    # Ensure data is not None or empty
    if data is None or data.empty:
        logger.warning(f"No data available for {ticker}")
        return {'status': 'failed', 'reason': 'no_data'}
    
    try:
        # Prepare data
        X, y = prepare_data(data, lags=int(LAGS))
        
        if len(X) < 50:  # Minimum data requirement
            logger.warning(f"Insufficient data for {ticker}: {len(X)} samples")
            return {'status': 'failed', 'reason': 'insufficient_data'}
        
        if use_cv:
            # Cross-validation
            if model_type.lower() == 'mlp':
                cv_results, avg_metrics = cross_validate_time_series(train_mlp, X.values, y.values)
            else:
                cv_results, avg_metrics = cross_validate_time_series(
                    lambda x, y: train_rnn(x, y, model_type=model_type.upper()), 
                    X.values, y.values
                )
            
            logger.info(f"Cross-validation results for {ticker}: {avg_metrics}")
            return {'status': 'success', 'cv_metrics': avg_metrics}
        
        else:
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(TEST_SIZE), shuffle=False, random_state=int(RANDOM_SEED)
            )
            
            # Train model
            if model_type.lower() == 'mlp':
                model, scaler = train_mlp(X_train.values, y_train.values)
                is_rnn = False
            else:
                # Reshape for RNN
                X_train_seq = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
                model, scaler = train_rnn(X_train_seq, y_train.values, model_type=model_type.upper())
                is_rnn = True
            
            # Evaluate model
            try:
                if is_rnn:
                    X_test_seq = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
                    metrics = evaluate_model(model, X_test_seq, y_test.values, scaler, is_rnn=True)
                else:
                    metrics = evaluate_model(model, X_test.values, y_test.values, scaler, is_rnn=False)
            except Exception as e:
                logger.error(f"Error evaluating model: {str(e)}")
                metrics = {'error': str(e)}
            
            # Create models directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Save model
            model_filename = f"{model_type.lower()}_{ticker}"
            try:
                save_model(model, scaler, model_filename, is_rnn=is_rnn)
                logger.info(f"Model saved: {model_filename}, Metrics: {metrics}")
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                return {'status': 'partial_success', 'metrics': metrics, 'error': str(e)}
            
            return {'status': 'success', 'metrics': metrics, 'model_path': model_filename}
            
    except Exception as e:
        logger.error(f"Error training {model_type} for {ticker}: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

def train_classification_model(ticker: str, data: pd.DataFrame) -> dict:
    """
    Train HMM classification model for a given ticker.
    
    Args:
        ticker: Crypto ticker
        data: Preprocessed data
    
    Returns:
        Dictionary with training results
    """
    logger.info(f"Training HMM classification model for {ticker}")
    
    # Ensure data is not None or empty
    if data is None or data.empty:
        logger.warning(f"No data available for {ticker}")
        return {'status': 'failed', 'reason': 'no_data'}
    
    try:
        # Train HMM
        model, scaler = train_hmm(data, n_states=int(HMM_STATES))
        
        # Get state statistics
        state_stats = get_state_statistics(model, scaler)
        
        # Create models directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save model
        model_filename = f"hmm_{ticker}"
        save_model(model, scaler, model_filename, is_rnn=False)
        
        logger.info(f"HMM model saved: {model_filename}")
        
        return {
            'status': 'success',
            'model_path': model_filename,
            'state_stats': state_stats,
            'converged': model.monitor_.converged
        }
    except Exception as e:
        logger.error(f"Error training HMM for {ticker}: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

def train_anomaly_detector(ticker: str, data: pd.DataFrame) -> dict:
    """
    Train anomaly detection model for a given ticker.
    
    Args:
        ticker: Crypto ticker
        data: Preprocessed data
    
    Returns:
        Dictionary with training results
    """
    logger.info(f"Training anomaly detection model for {ticker}")
    
    # Ensure data is not None or empty
    if data is None or data.empty:
        logger.warning(f"No data available for {ticker}")
        return {'status': 'failed', 'reason': 'no_data'}
    
    try:
        # Create features
        data = create_anomaly_features(data)
        
        # Detect anomalies using Z-score for labels
        data = detect_anomalies_zscore(data)
        
        # Define features for Random Forest
        feature_cols = [col for col in data.columns if any(
            substr in col for substr in ['Vol_', 'MA_', 'Std_', 'Skew_', 'Kurt_', 'Distance_', 'RoC']
        )]
        
        # Train Random Forest
        if len(data.dropna()) > 100 and 'Anomaly' in data.columns:
            # Create models directory if it doesn't exist
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Train the model
            rf_model, scaler = train_random_forest_detector(data, feature_cols)
            
            # Evaluate model
            X = data[feature_cols].dropna()
            y = data.loc[X.index, 'Anomaly']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(TEST_SIZE), random_state=int(RANDOM_SEED), stratify=y
            )
            
            X_test_scaled = scaler.transform(X_test)
            metrics = evaluate_anomaly_detector(rf_model, X_test_scaled, y_test)
            
            # Save model
            model_filename = f"anomaly_rf_{ticker}"
            save_model(rf_model, scaler, model_filename, is_rnn=False)
            
            logger.info(f"Anomaly detector saved: {model_filename}, Metrics: {metrics}")
            
            return {'status': 'success', 'metrics': metrics, 'model_path': model_filename}
        else:
            logger.warning(f"Insufficient data for Random Forest training for {ticker}")
            return {'status': 'failed', 'reason': 'insufficient_data'}
    except Exception as e:
        logger.error(f"Error training anomaly detector for {ticker}: {str(e)}")
        return {'status': 'failed', 'error': str(e)}

def preprocess_ticker_data(ticker: str, 
                          start_date: str, 
                          end_date: str,
                          use_deribit: bool = False) -> pd.DataFrame:
    """
    Preprocess data for a given ticker.
    
    Args:
        ticker: Crypto ticker
        start_date: Start date
        end_date: End date
        use_deribit: Whether to include Deribit data
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Preprocessing data for {ticker}")
    
    try:
        # Get data
        if use_deribit:
            data = get_combined_volatility_data(ticker, start_date, end_date)
        else:
            data = get_data(ticker, start_date, end_date)
        
        if data is None or data.empty:
            logger.error(f"No data available for {ticker}")
            return pd.DataFrame()
        
        # Calculate returns and volatility
        data = calculate_returns(data)
        data = calculate_volatility(data, window=int(VOL_WINDOW))
        
        # Add advanced volatility metrics
        data = calculate_advanced_volatility_metrics(data, window=int(VOL_WINDOW))
        
        # Create features
        data = create_volatility_features(data, lags=int(LAGS))
        
        logger.info(f"Preprocessed {len(data)} data points for {ticker}")
        
        return data
    except Exception as e:
        logger.error(f"Error preprocessing data for {ticker}: {str(e)}")
        return pd.DataFrame()

def train_all_models(ticker: str, 
                    start_date: str, 
                    end_date: str,
                    models_to_train: list = None,
                    use_deribit: bool = False) -> dict:
    """
    Train all models for a given ticker.
    
    Args:
        ticker: Crypto ticker
        start_date: Start date
        end_date: End date
        models_to_train: List of models to train (default: all)
        use_deribit: Whether to include Deribit data
    
    Returns:
        Dictionary with training results
    """
    if models_to_train is None:
        models_to_train = ['mlp', 'rnn', 'lstm', 'gru', 'hmm', 'anomaly']
    
    results = {'ticker': ticker, 'timestamp': datetime.now().isoformat()}
    
    # Preprocess data
    data = preprocess_ticker_data(ticker, start_date, end_date, use_deribit)
    
    if data.empty:
        results['status'] = 'failed'
        results['reason'] = 'no_data'
        return results
    
    # Train forecasting models
    for model_type in ['mlp', 'rnn', 'lstm', 'gru']:
        if model_type in models_to_train:
            try:
                results[model_type] = train_forecasting_model(ticker, model_type, data)
            except Exception as e:
                logger.error(f"Error training {model_type} for {ticker}: {e}")
                results[model_type] = {'status': 'failed', 'error': str(e)}
    
    # Train classification model
    if 'hmm' in models_to_train:
        try:
            results['hmm'] = train_classification_model(ticker, data)
        except Exception as e:
            logger.error(f"Error training HMM for {ticker}: {e}")
            results['hmm'] = {'status': 'failed', 'error': str(e)}
    
    # Train anomaly detector
    if 'anomaly' in models_to_train:
        try:
            results['anomaly'] = train_anomaly_detector(ticker, data)
        except Exception as e:
            logger.error(f"Error training anomaly detector for {ticker}: {e}")
            results['anomaly'] = {'status': 'failed', 'error': str(e)}
    
    return results

def main():
    """Main function to parse arguments and train models."""
    parser = argparse.ArgumentParser(description='Train volatility analysis models')
    parser.add_argument('--ticker', type=str, help='Crypto ticker (default: all configured tickers)')
    parser.add_argument('--model', type=str, choices=['mlp', 'rnn', 'lstm', 'gru', 'hmm', 'anomaly', 'all'],
                       default='all', help='Model type to train')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE,
                       help='Start date for training data')
    parser.add_argument('--end-date', type=str, default=DEFAULT_END_DATE,
                       help='End date for training data')
    parser.add_argument('--use-deribit', action='store_true',
                       help='Include Deribit implied volatility data')
    parser.add_argument('--cv', action='store_true',
                       help='Use cross-validation for forecasting models')
    parser.add_argument('--output', type=str, default='training_results.json',
                       help='Output file for training results')
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Determine which models to train
    if args.model == 'all':
        models_to_train = ['mlp', 'rnn', 'lstm', 'gru', 'hmm', 'anomaly']
    else:
        models_to_train = [args.model]
    
    # Determine which tickers to process
    if args.ticker:
        tickers_to_process = [args.ticker]
    else:
        tickers_to_process = TICKERS
    
    # Train models for each ticker
    all_results = {}
    for ticker in tickers_to_process:
        logger.info(f"Starting training for {ticker}")
        try:
            results = train_all_models(
                ticker=ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                models_to_train=models_to_train,
                use_deribit=args.use_deribit
            )
            all_results[ticker] = results
        except Exception as e:
            logger.error(f"Failed to train models for {ticker}: {e}")
            all_results[ticker] = {'status': 'failed', 'error': str(e)}
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Training completed. Results saved to {args.output}")
    
    # Print summary
    successful_trainings = sum(1 for result in all_results.values() 
                             if result.get('status') != 'failed')
    print(f"\nTraining Summary:")
    print(f"Total tickers processed: {len(all_results)}")
    print(f"Successful trainings: {successful_trainings}")
    print(f"Failed trainings: {len(all_results) - successful_trainings}")

if __name__ == '__main__':
    main()