"""
Streamlit dashboard for cryptocurrency volatility analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

# Use absolute imports for better compatibility
from src.data_acquisition import get_data, get_combined_volatility_data, DeribitAPI
from src.preprocessing import (
    calculate_returns, 
    calculate_volatility,
    calculate_advanced_volatility_metrics,
    calculate_volatility_ratio,
    calculate_iv_rv_spread
)
from src.forecasting import (
    load_model, 
    prepare_data, 
    forecast_next_values,
    prepare_sequences
)
from src.anomaly_detection import (
    detect_anomalies_zscore,
    ensemble_anomaly_detection,
    get_anomaly_statistics
)
from src.classification import (
    train_hmm, 
    predict_states, 
    get_state_statistics,
    get_current_regime
)
from src.clustering import (
    cluster_tickers, 
    get_similar_tickers,
    get_cluster_characteristics
)
from src.config import (
    TICKERS, VOL_WINDOW, LAGS, N_CLUSTERS, HMM_STATES, 
    ANOMALY_THRESHOLD, DEFAULT_START_DATE, DEFAULT_END_DATE
)

def main():
    """Main entry point for the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title="Crypto Volatility Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 16px;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .insight-box {
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #0066cc;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = TICKERS[0]

    tab_labels = [
        "Overview", 
        "Forecasting", 
        "Regime Analysis",
        "Anomaly Detection",
        "Clustering",
        "Data Exploration"
    ]

    # Remove the check for len(tabs) == 6 and directly unpack
    # If the number of tabs does not match, Streamlit will raise an error, which is more informative for debugging.
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_labels)

    # Title and description
    st.title("🚀 Crypto Volatility Analysis Dashboard")
    st.markdown("""
    This dashboard provides comprehensive volatility analysis for cryptocurrencies including:
    - Historical volatility analysis
    - ML-based volatility forecasting
    - Volatility regime classification
    - Anomaly detection
    - Cryptocurrency clustering
    """)

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Ticker selection
    ticker = st.sidebar.selectbox(
        "Select Cryptocurrency",
        TICKERS,
        index=TICKERS.index(st.session_state.selected_ticker)
    )
    st.session_state.selected_ticker = ticker

    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            pd.to_datetime(DEFAULT_START_DATE)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            pd.to_datetime(DEFAULT_END_DATE)
        )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        use_deribit = st.checkbox("Include Deribit IV Data", value=False)
        vol_window = st.slider("Volatility Window", 5, 60, VOL_WINDOW)
        n_forecast_days = st.slider("Forecast Horizon (days)", 1, 30, 5)
        anomaly_threshold = st.slider("Anomaly Threshold", 1.0, 4.0, float(ANOMALY_THRESHOLD), 0.1)

    # Data refresh button
    if st.sidebar.button("Refresh Data", type="primary"):
        with st.spinner("Fetching data..."):
            try:
                # Convert date inputs to strings in the correct format
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                if use_deribit:
                    data = get_combined_volatility_data(ticker, start_date_str, end_date_str)
                else:
                    data = get_data(ticker, start_date_str, end_date_str)
                
                if data is not None and not data.empty:
                    # Preprocess data
                    data = calculate_returns(data)
                    data = calculate_volatility(data, window=int(vol_window))
                    data = calculate_advanced_volatility_metrics(data, window=int(vol_window))
                    
                    if use_deribit and 'ImpliedVolatility' in data.columns:
                        data = calculate_iv_rv_spread(data)
                    
                    st.session_state.data = data
                    st.success("Data loaded successfully!")
                else:
                    st.error("No data available for the selected ticker and date range.")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

    # Main content
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Tab 1: Overview
        with tab1:
            st.header("Market Overview")
            # Check for valid and non-empty Series for metrics
            try:
                close_col = data['Close']
                vol_col = data['Volatility']
                # If DataFrame, take first column
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                if isinstance(vol_col, pd.DataFrame):
                    vol_col = vol_col.iloc[:, 0]
                close_col = pd.to_numeric(close_col, errors='coerce')
                vol_col = pd.to_numeric(vol_col, errors='coerce')
                # Drop NaNs
                close_col = close_col.dropna()
                vol_col = vol_col.dropna()
                if len(close_col) == 0 or len(vol_col) == 0:
                    st.warning("No valid price or volatility data available for metrics.")
                else:
                    current_price = float(close_col.iloc[-1])
                    current_vol = float(vol_col.iloc[-1])
                    avg_vol = float(vol_col.mean())
                    vol_change = ((current_vol - avg_vol) / max(avg_vol, 0.0001)) * 100
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("Current Volatility", f"{current_vol:.2%}")
                    with col3:
                        st.metric("Average Volatility", f"{avg_vol:.2%}")
                    with col4:
                        st.metric("Vol. Change", f"{vol_change:.1f}%")
            except Exception as e:
                st.error(f"Error calculating metrics: {str(e)}")
            
            # Price chart
            if data is not None and not data.empty and 'Close' in data.columns:
                close_col = data['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                close_col = pd.to_numeric(close_col, errors='coerce').dropna()
                # Only plot if at least 2 data points
                if len(close_col) < 2:
                    st.warning("Not enough price data to plot price history.")
                else:
                    # Ensure index is DateTimeIndex and unique
                    if not isinstance(data.index, pd.DatetimeIndex):
                        try:
                            data.index = pd.to_datetime(data.index)
                        except Exception as e:
                            st.error(f"Error converting index to datetime: {str(e)}")
                    data = data[~data.index.duplicated(keep='first')]
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=data.index,
                        y=close_col,
                        name='Price',
                        line=dict(color='#1f77b4')
                    ))
                    fig_price.update_layout(
                        title=f"{ticker} Price History",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.warning("No valid price data available for the selected ticker and date range.")
            
            # Volatility chart
            try:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['Volatility'],
                    name='Realized Volatility',
                    line=dict(color='#ff7f0e')
                ))
                # Plot IV only if valid
                if 'ImpliedVolatility' in data.columns:
                    iv_col = pd.to_numeric(data['ImpliedVolatility'], errors='coerce').dropna()
                    if len(iv_col) >= 2:
                        fig_vol.add_trace(go.Scatter(
                            x=data.index,
                            y=iv_col,
                            name='Implied Volatility',
                            line=dict(color='#2ca02c')
                        ))
                    else:
                        st.warning("Not enough Implied Volatility data to plot IV line.")
                else:
                    st.info("Implied Volatility data not available for this ticker/date range.")
                fig_vol.update_layout(
                    title=f"{ticker} Volatility History",
                    xaxis_title="Date",
                    yaxis_title="Volatility",
                    template="plotly_white"
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering volatility chart: {str(e)}")
            
            # Volatility metrics comparison
            if all(col in data.columns for col in ['Parkinson_Vol', 'GK_Vol', 'RS_Vol']):
                try:
                    st.subheader("Volatility Metrics Comparison")
                    
                    fig_vol_comp = go.Figure()
                    vol_metrics = ['Volatility', 'Parkinson_Vol', 'GK_Vol', 'RS_Vol']
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    
                    for metric, color in zip(vol_metrics, colors):
                        fig_vol_comp.add_trace(go.Scatter(
                            x=data.index,
                            y=data[metric],
                            name=metric.replace('_', ' '),
                            line=dict(color=color, width=2)
                        ))
                    
                    fig_vol_comp.update_layout(
                        title="Comparison of Volatility Metrics",
                        xaxis_title="Date",
                        yaxis_title="Volatility",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_vol_comp, use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering volatility metrics comparison: {str(e)}")
        
        # Tab 2: Forecasting
        with tab2:
            st.header("Volatility Forecasting")
            model_type = st.selectbox(
                "Select Forecasting Model",
                ["MLP", "RNN", "LSTM", "GRU"],
                help="Choose the neural network architecture for forecasting"
            )
            model_filename = f"{model_type.lower()}_{ticker}"
            if st.button("Generate Forecast", key="generate_forecast"):
                try:
                    # Always train the model when button is clicked
                    with st.spinner(f"Training {model_type} model for {ticker}..."):
                        from src.train_models import train_forecasting_model
                        train_forecasting_model(ticker, model_type.lower(), data)
                    model, scaler = load_model(model_filename, is_rnn=(model_type != "MLP"))
                    X, y = prepare_data(data, lags=LAGS)
                    if len(X) > 0:
                        recent_data = X.iloc[-1].values
                        if model_type != "MLP":
                            recent_data_reshaped = recent_data.reshape(1, -1, 1)
                            forecast = forecast_next_values(
                                model, recent_data_reshaped, scaler, 
                                is_rnn=True, 
                                n_ahead=n_forecast_days
                            )
                        else:
                            forecast = forecast_next_values(
                                model, recent_data, scaler, 
                                is_rnn=False, 
                                n_ahead=n_forecast_days
                            )
                        last_date = data.index[-1]
                        forecast_dates = pd.date_range(
                            start=last_date + timedelta(days=1),
                            periods=n_forecast_days
                        )
                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(
                            x=data.index[-60:],
                            y=data['Volatility'][-60:],
                            name='Historical',
                            line=dict(color='#1f77b4')
                        ))
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast,
                            name='Forecast',
                            line=dict(color='#ff7f0e', dash='dash')
                        ))
                        fig_forecast.update_layout(
                            title=f"{model_type} Volatility Forecast",
                            xaxis_title="Date",
                            yaxis_title="Volatility",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        forecast_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecasted Volatility': forecast
                        })
                        st.dataframe(forecast_df.style.format({'Forecasted Volatility': '{:.2%}'}))
                    else:
                        st.warning("Insufficient data for forecasting")
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
        
        # Tab 3: Regime Analysis
        with tab3:
            st.header("Volatility Regime Classification")
            
            try:
                # Train HMM
                hmm_model, hmm_scaler = train_hmm(data, n_states=HMM_STATES)
                
                # Predict states
                states = predict_states(hmm_model, data, hmm_scaler)
                
                # Get state statistics
                state_stats = get_state_statistics(hmm_model, hmm_scaler)
                
                # Get current regime
                current_regime, current_state, probability = get_current_regime(
                    hmm_model, data, hmm_scaler, state_stats
                )
                
                # Map regime numbers to names
                regime_names = {0: "Low Volatility", 1: "Medium Volatility", 2: "High Volatility", 3: "Extreme Volatility"}
                regime_label = regime_names.get(current_regime, "Unknown")
                
                # Display current regime
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Regime", regime_label)
                with col2:
                    st.metric("State", f"State {current_state}")
                with col3:
                    st.metric("Confidence", f"{probability:.1%}")
                
                # State visualization
                fig_states = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                         vertical_spacing=0.05,
                                         row_heights=[0.7, 0.3])
                
                # Volatility with states
                colors = ['green', 'yellow', 'orange', 'red', 'darkred']
                for state in range(HMM_STATES):
                    mask = states == state
                    color = colors[state] if state < len(colors) else 'purple'
                    fig_states.add_trace(
                        go.Scatter(
                            x=data.index[mask],
                            y=data['Volatility'].iloc[mask],
                            mode='markers',
                            name=f'State {state}',
                            marker=dict(color=color, size=6)
                        ),
                        row=1, col=1
                    )
                
                # State sequence
                fig_states.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=states,
                        mode='lines',
                        name='State',
                        line=dict(color='blue', width=2)
                    ),
                    row=2, col=1
                )
                
                fig_states.update_layout(
                    title="Volatility Regimes",
                    height=600,
                    showlegend=True,
                    template="plotly_white"
                )
                fig_states.update_yaxes(title_text="Volatility", row=1, col=1)
                fig_states.update_yaxes(title_text="State", row=2, col=1)
                fig_states.update_xaxes(title_text="Date", row=2, col=1)
                
                st.plotly_chart(fig_states, use_container_width=True)
                
                # State statistics
                st.subheader("State Statistics")
                state_df = pd.DataFrame(state_stats).T
                state_df.index.name = 'State'
                st.dataframe(state_df.style.format({
                    'mean': '{:.2%}',
                    'std': '{:.2%}',
                    'stationary_prob': '{:.1%}'
                }))
                
            except Exception as e:
                st.error(f"Error in regime analysis: {str(e)}")
        
        # Tab 4: Anomaly Detection
        with tab4:
            st.header("Anomaly Detection")
            
            # Anomaly detection method
            detection_method = st.selectbox(
                "Detection Method",
                ["Z-Score", "Ensemble (Z-Score + Isolation Forest + Percentile)"],
                help="Choose the anomaly detection method"
            )
            
            try:
                if detection_method == "Z-Score":
                    anomaly_data = detect_anomalies_zscore(
                        data.copy(), 
                        window=int(vol_window), 
                        threshold=float(anomaly_threshold)
                    )
                    anomaly_col = 'Anomaly'
                else:
                    anomaly_data = ensemble_anomaly_detection(
                        data.copy(),
                        methods=['zscore', 'percentile'],
                        threshold=0.5
                    )
                    anomaly_col = 'Ensemble_Anomaly'
                
                # Get anomaly statistics
                anomaly_stats = get_anomaly_statistics(
                    anomaly_data, 
                    anomaly_col=anomaly_col
                )
                
                # Display statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", anomaly_stats['anomaly_count'])
                with col2:
                    st.metric("Anomaly Rate", f"{anomaly_stats['anomaly_percentage']:.1f}%")
                with col3:
                    st.metric("Max Consecutive", anomaly_stats['consecutive_anomalies'])
                
                # Anomaly visualization
                fig_anomaly = go.Figure()
                
                # Volatility
                fig_anomaly.add_trace(go.Scatter(
                    x=anomaly_data.index,
                    y=anomaly_data['Volatility'],
                    name='Volatility',
                    line=dict(color='#1f77b4')
                ))
                
                # Anomalies
                anomaly_mask = anomaly_data[anomaly_col]
                fig_anomaly.add_trace(go.Scatter(
                    x=anomaly_data.index[anomaly_mask],
                    y=anomaly_data['Volatility'][anomaly_mask],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10, symbol='x')
                ))
                
                fig_anomaly.update_layout(
                    title="Volatility Anomalies",
                    xaxis_title="Date",
                    yaxis_title="Volatility",
                    template="plotly_white"
                )
                st.plotly_chart(fig_anomaly, use_container_width=True)
                
                # Recent anomalies table
                st.subheader("Recent Anomalies")
                if anomaly_stats['anomaly_count'] > 0:
                    recent_anomalies = anomaly_data[anomaly_mask].tail(10)
                    st.dataframe(recent_anomalies[['Volatility', 'Z_Score']].style.format({
                        'Volatility': '{:.2%}',
                        'Z_Score': '{:.2f}'
                    }), use_container_width=True)
                else:
                    empty_df = pd.DataFrame(columns=['Volatility', 'Z_Score'])
                    st.dataframe(empty_df, use_container_width=True)
                    st.info("No anomalies detected in the selected period.")
                
            except Exception as e:
                st.error(f"Error in anomaly detection: {str(e)}")
        
        # Tab 5: Clustering
        with tab5:
            st.header("Cryptocurrency Clustering")
            
            # Clustering parameters
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.slider("Number of Clusters", 2, 10, N_CLUSTERS)
            with col2:
                clustering_method = st.selectbox(
                    "Clustering Method",
                    ["agglomerative", "kmeans"]
                )
            
            if st.button("Run Clustering Analysis"):
                with st.spinner("Performing clustering analysis..."):
                    try:
                        # Convert dates to string format for clustering
                        start_date_str = start_date.strftime('%Y-%m-%d')
                        end_date_str = end_date.strftime('%Y-%m-%d')
                        
                        # Perform clustering
                        cluster_mapping = cluster_tickers(
                            TICKERS,
                            start_date_str,
                            end_date_str,
                            n_clusters=int(n_clusters),
                            clustering_method=clustering_method
                        )
                        
                        if cluster_mapping:
                            # Show all tickers for each cluster as a styled DataFrame
                            st.subheader("Tickers in Each Cluster")
                            cluster_df = pd.DataFrame([
                                {"Cluster": f"Cluster {cluster_id}", "Tickers": ', '.join([ticker for ticker, cluster in cluster_mapping.items() if cluster == cluster_id])}
                                for cluster_id in sorted(set(cluster_mapping.values()))
                            ])
                            # Style: wrap tickers and bold cluster name
                            cluster_df_styled = cluster_df.style.set_properties(subset=["Tickers"], **{"white-space": "pre-wrap"}) \
                                .set_properties(subset=["Cluster"], **{"font-weight": "bold"})
                            st.dataframe(cluster_df_styled, use_container_width=True)

                            # Show cluster characteristics as a styled DataFrame
                            st.subheader("Cluster Characteristics")
                            cluster_chars = get_cluster_characteristics(TICKERS, cluster_mapping, start_date_str, end_date_str)
                            if isinstance(cluster_chars, dict):
                                # Flatten dict to DataFrame
                                chars_df = pd.DataFrame(cluster_chars).T.reset_index().rename(columns={"index": "Cluster"})
                                chars_df["Cluster"] = chars_df["Cluster"].apply(lambda x: f"Cluster {x}")
                                # Style: format numeric columns
                                chars_df_styled = chars_df.style.format({
                                    col: "{:.2%}" if "vol" in col else "{:.4f}" for col in chars_df.columns if col not in ["Cluster", "num_tickers"]
                                }).set_properties(subset=["Cluster"], **{"font-weight": "bold"})
                                st.dataframe(chars_df_styled, use_container_width=True)
                            elif isinstance(cluster_chars, pd.DataFrame):
                                st.dataframe(cluster_chars, use_container_width=True)
                            else:
                                st.write(cluster_chars)
                        else:
                            st.warning("Insufficient data for clustering analysis")
                        
                    except Exception as e:
                        st.error(f"Error in clustering analysis: {str(e)}")
        
        # Tab 6: Data Exploration
        with tab6:
            st.header("Data Exploration")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(data.tail(20))
            
            # Summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(data.describe())
            
            # Download data
            st.subheader("Download Data")
            csv = data.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker}_volatility_data.csv",
                mime="text/csv"
            )

    else:
        st.info("Please click 'Refresh Data' to load data for analysis.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Crypto Volatility Analysis Dashboard | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # When running this file directly, ensure proper imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()