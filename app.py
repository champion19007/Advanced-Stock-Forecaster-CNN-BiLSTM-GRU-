"""
Advanced Stock Forecaster with CNN + BiLSTM + GRU + Attention
This Streamlit app provides comprehensive stock price forecasting with technical indicators,
baseline model comparisons, and advanced validation techniques.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ta
import sqlite3
import warnings
from datetime import datetime, date, timedelta
import os
import io
import pickle
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Initialize database
def init_database():
    """Initialize SQLite database for storing model results"""
    conn = sqlite3.connect('stock_forecaster.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            timesteps INTEGER NOT NULL,
            dropout REAL NOT NULL,
            epochs INTEGER NOT NULL,
            train_rmse REAL,
            test_rmse REAL,
            train_mae REAL,
            test_mae REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_model_to_db(ticker, start_date, end_date, timesteps, dropout, epochs, 
                     train_rmse, test_rmse, train_mae, test_mae, model_path):
    """Save model results to database"""
    conn = sqlite3.connect('stock_forecaster.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO models (ticker, start_date, end_date, timesteps, dropout, epochs,
                          train_rmse, test_rmse, train_mae, test_mae)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, start_date, end_date, timesteps, dropout, epochs,
          train_rmse, test_rmse, train_mae, test_mae))
    
    conn.commit()
    conn.close()

def get_data_with_indicators(ticker, start_date, end_date):
    """
    Load stock data from Yahoo Finance and add technical indicators
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date
        end_date (str): End date
    
    Returns:
        pd.DataFrame: Stock data with technical indicators
    """
    try:
        with st.spinner(f"Fetching {ticker} stock data with technical indicators..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Flatten column index if it's a MultiIndex (happens with single ticker sometimes)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Add technical indicators using ta library
            # Ensure we pass 1-dimensional data to ta functions
            close_prices = data['Close'].squeeze()  # Convert to 1D if needed
            
            data['RSI'] = ta.momentum.RSIIndicator(close=close_prices, window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(close=close_prices)
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_diff'] = macd.macd_diff()
            
            # Drop NaN values that result from indicator calculations
            data = data.dropna()
            
            st.success(f"Successfully loaded {len(data)} data points with technical indicators")
            
            return data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data_advanced(data, window_size=60, test_split=0.2, features=['Close', 'RSI', 'MACD']):
    """
    Advanced preprocessing with multiple features and technical indicators
    
    Args:
        data (pd.DataFrame): Raw stock data with indicators
        window_size (int): Number of timesteps to look back
        test_split (float): Fraction of data for testing
        features (list): List of features to use
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler_X, scaler_y
    """
    with st.spinner("Preprocessing data with technical indicators..."):
        # Select features
        feature_data = data[features].copy()
        target_data = data[['Close']].copy()
        
        # Normalize features and target separately
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        scaled_features = scaler_X.fit_transform(feature_data.values)
        scaled_target = scaler_y.fit_transform(target_data.values)
        
        # Create sliding windows
        X, y = [], []
        
        for i in range(window_size, len(scaled_features)):
            X.append(scaled_features[i-window_size:i])
            y.append(scaled_target[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split into training and testing sets
        split_idx = int(len(X) * (1 - test_split))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        st.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        st.info(f"Feature shape: {X_train.shape[1:]}")
        
        return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def build_advanced_model(input_shape, dropout_rate=0.2):
    """
    Build the advanced hybrid CNN + BiLSTM + GRU model
    
    Args:
        input_shape (tuple): Shape of input data
        dropout_rate (float): Dropout rate
    
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    with st.spinner("Building advanced CNN + BiLSTM + GRU model..."):
        model = Sequential([
            # CNN layer for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            
            # First BiLSTM layer with return sequences
            Bidirectional(LSTM(64, return_sequences=True)),
            
            # Additional LSTM layer (replacing Attention for compatibility)
            Bidirectional(LSTM(32, return_sequences=True)),
            
            # Dropout
            Dropout(dropout_rate),
            
            # Second BiLSTM layer with return sequences
            Bidirectional(LSTM(64, return_sequences=True)),
            
            # First GRU layer with return sequences
            GRU(64, return_sequences=True),
            Dropout(dropout_rate),
            
            # Second GRU layer without return sequences
            GRU(64),
            Dropout(dropout_rate),
            
            # Dense layers for final prediction
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        st.success("Advanced model built successfully!")
        
        return model

def rolling_window_validation(data, window_size, n_splits=5, features=['Close', 'RSI', 'MACD']):
    """
    Perform rolling window validation
    
    Args:
        data: Stock data
        window_size: Window size for sequences
        n_splits: Number of validation splits
        features: Features to use
        
    Returns:
        list: RMSE scores for each fold
    """
    rmse_scores = []
    mae_scores = []
    
    # Calculate split size
    split_size = len(data) // (n_splits + 1)
    
    progress_bar = st.progress(0)
    
    for i in range(n_splits):
        # Calculate train and test indices
        test_start = (i + 1) * split_size
        test_end = test_start + split_size
        
        train_data = data[:test_start]
        test_data = data[test_start:test_end]
        
        if len(train_data) < window_size + 50 or len(test_data) < 10:
            continue
            
        try:
            # Preprocess data for this fold
            X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data_advanced(
                pd.concat([train_data, test_data]), window_size, 
                test_split=len(test_data)/(len(train_data)+len(test_data)), features=features
            )
            
            # Build and train model  
            model = build_advanced_model(X_train.shape[1:], dropout_rate=0.2)
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
            
            # Make predictions
            y_pred = model.predict(X_test, verbose=0)
            
            # Inverse transform
            y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
            y_pred_actual = scaler_y.inverse_transform(y_pred)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            
        except Exception as e:
            st.warning(f"Fold {i+1} failed: {str(e)}")
        
        progress_bar.progress((i + 1) / n_splits)
    
    progress_bar.empty()
    return rmse_scores, mae_scores

def train_baseline_models(data, test_size=0.2):
    """
    Train ARIMA and Prophet baseline models
    
    Args:
        data: Stock data
        test_size: Fraction for testing
        
    Returns:
        dict: Results from baseline models
    """
    results = {}
    
    # Prepare data
    close_data = data['Close'].values
    split_idx = int(len(close_data) * (1 - test_size))
    
    train_data = close_data[:split_idx]
    test_data = close_data[split_idx:]
    
    # ARIMA Model
    try:
        with st.spinner("Training ARIMA baseline..."):
            # Auto ARIMA to find best parameters
            arima_model = ARIMA(train_data, order=(5, 1, 0))
            arima_fitted = arima_model.fit()
            
            # Forecast
            arima_forecast = arima_fitted.forecast(steps=len(test_data))
            
            # Calculate metrics
            arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
            arima_mae = mean_absolute_error(test_data, arima_forecast)
            
            results['ARIMA'] = {
                'predictions': arima_forecast,
                'rmse': arima_rmse,
                'mae': arima_mae
            }
            
    except Exception as e:
        st.warning(f"ARIMA training failed: {str(e)}")
        results['ARIMA'] = None
    
    # Prophet Model
    try:
        with st.spinner("Training Prophet baseline..."):
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index[:split_idx],
                'y': train_data
            })
            
            # Train Prophet
            prophet_model = Prophet(daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False)
            prophet_model.fit(prophet_data)
            
            # Create future dataframe
            future_dates = pd.DataFrame({
                'ds': data.index[split_idx:split_idx+len(test_data)]
            })
            
            # Forecast
            prophet_forecast = prophet_model.predict(future_dates)
            prophet_predictions = prophet_forecast['yhat'].values
            
            # Calculate metrics
            prophet_rmse = np.sqrt(mean_squared_error(test_data, prophet_predictions))
            prophet_mae = mean_absolute_error(test_data, prophet_predictions)
            
            results['Prophet'] = {
                'predictions': prophet_predictions,
                'rmse': prophet_rmse,
                'mae': prophet_mae
            }
            
    except Exception as e:
        st.warning(f"Prophet training failed: {str(e)}")
        results['Prophet'] = None
    
    return results, test_data

def train_and_predict_advanced(model, X_train, X_test, y_train, y_test, scaler_y, 
                              epochs=50, use_early_stopping=False):
    """
    Advanced training with optional early stopping
    
    Args:
        model: Compiled Keras model
        X_train, X_test, y_train, y_test: Training and testing data
        scaler_y: Target scaler
        epochs: Number of training epochs
        use_early_stopping: Whether to use early stopping
    
    Returns:
        tuple: (predictions, actual_values, rmse, mae, history)
    """
    # Setup callbacks
    callbacks = []
    if use_early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
    
    # Train the model
    with st.spinner(f"Training advanced model for up to {epochs} epochs..."):
        progress_bar = st.progress(0)
        
        # Custom callback to update progress
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_bar.progress((epoch + 1) / epochs)
        
        callbacks.append(StreamlitCallback())
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0,
            shuffle=False,
            callbacks=callbacks
        )
        
        progress_bar.empty()
        st.success("Model training completed!")
    
    # Make predictions
    with st.spinner("Generating predictions..."):
        train_predictions = model.predict(X_train, verbose=0)
        test_predictions = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions and actual values
        train_predictions = scaler_y.inverse_transform(train_predictions)
        test_predictions = scaler_y.inverse_transform(test_predictions)
        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
        train_mae = mean_absolute_error(y_train_actual, train_predictions)
        test_mae = mean_absolute_error(y_test_actual, test_predictions)
        
        return (test_predictions.flatten(), y_test_actual.flatten(), 
                train_rmse, test_rmse, train_mae, test_mae, history)

def main():
    """
    Main Streamlit app function
    """
    # Page config
    st.set_page_config(
        page_title="Advanced Stock Forecaster",
        page_icon="🚀",
        layout="wide"
    )
    
    # Initialize database
    init_database()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Custom CSS for better date input styling
    st.markdown("""
    <style>
    /* Scale date picker calendar to 0.9 and position below */
    .stDateInput > div[data-baseweb="calendar"] {
        transform: scale(0.9);
        transform-origin: top left;
        margin-top: 10px;
        position: relative;
        z-index: 999;
    }
    
    /* Restrict input appearance for date fields */
    .stDateInput input {
        font-family: 'Courier New', monospace;
        background-color: #f0f2f6 !important;
        color: #262730 !important;
        border: 1px solid #d4d4d4 !important;
    }
    
    /* Ensure calendar appears below input */
    .stDateInput > div {
        position: relative;
    }
    
    .stDateInput > div[data-baseweb="popover"] {
        position: absolute !important;
        top: 100% !important;
        left: 0 !important;
        z-index: 1000 !important;
    }
    
    /* Style for date input validation message */
    .date-validation {
        font-size: 0.8em;
        color: #666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # JavaScript for input validation
    st.markdown("""
    <script>
    // Restrict date input to numbers and / only
    document.addEventListener('DOMContentLoaded', function() {
        const dateInputs = document.querySelectorAll('.stDateInput input');
        dateInputs.forEach(function(input) {
            input.addEventListener('keypress', function(e) {
                const char = String.fromCharCode(e.which);
                if (!/[0-9\/]/.test(char)) {
                    e.preventDefault();
                }
            });
        });
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("Advanced Stock Forecaster (CNN + BiLSTM + GRU)")
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # Top 10 tickers dropdown
    top_tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "JPM", "V"]
    ticker = st.sidebar.selectbox("Select stock ticker", top_tickers)
    
    # Date inputs (default: last 5 years, allow up to 15 years)
    st.sidebar.markdown("**📅 Select Date Range**")
    st.sidebar.markdown('<p class="date-validation">Format: MM/DD/YYYY (numbers and / only)</p>', unsafe_allow_html=True)
    
    min_date = date.today() - timedelta(days=15*365)
    default_start = date.today() - timedelta(days=5*365)
    
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=default_start,
        min_value=min_date,
        max_value=date.today(),
        help="Calendar will appear below when clicked"
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=date.today(),
        min_value=start_date,
        max_value=date.today(),
        help="Calendar will appear below when clicked"
    )
    
    # Model parameters
    timesteps = st.sidebar.number_input(
        "Timesteps (window size)", 
        min_value=30, 
        max_value=120, 
        value=60
    )
    
    dropout_rate = st.sidebar.slider(
        "Dropout Rate",
        min_value=0.05,
        max_value=0.5,
        value=0.2,
        step=0.05
    )
    
    epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=20,
        max_value=100,
        value=50
    )
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    use_early_stopping = st.sidebar.checkbox("Use Early Stopping", value=True)
    use_rolling_validation = st.sidebar.checkbox("Use Rolling Window Validation")
    compare_baselines = st.sidebar.checkbox("Compare against ARIMA and Prophet baselines")
    
    # Action buttons
    train_model = st.sidebar.button("Train Model", type="primary")
    
    # Main content area
    if train_model:
        if start_date >= end_date:
            st.error("Start date must be before end date!")
            return
        
        # Convert dates to strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Step 1: Load data with indicators
        st.header("📊 Data Loading & Technical Analysis")
        stock_data = get_data_with_indicators(ticker, start_str, end_str)
        
        if stock_data is None:
            return
        
        # Display basic info about the data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Data Points", len(stock_data))
        with col2:
            current_price = float(stock_data['Close'].iloc[-1])
            st.metric("Current Price", f"${current_price:.2f}")
        with col3:
            price_change = float(stock_data['Close'].iloc[-1]) - float(stock_data['Close'].iloc[0])
            st.metric("Total Change", f"${price_change:.2f}")
        with col4:
            latest_rsi = float(stock_data['RSI'].iloc[-1])
            st.metric("RSI (Latest)", f"{latest_rsi:.1f}")
        
        # Show technical indicators
        st.subheader("Technical Indicators")
        
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(stock_data[['Close', 'RSI']])
            st.caption("Price and RSI")
        with col2:
            st.line_chart(stock_data[['MACD', 'MACD_signal']])
            st.caption("MACD and Signal Line")
        
        # Step 2: Rolling window validation (if selected)
        if use_rolling_validation:
            st.header("🔄 Rolling Window Validation")
            rmse_scores, mae_scores = rolling_window_validation(stock_data, timesteps)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg CV RMSE", f"${np.mean(rmse_scores):.2f}")
                st.metric("CV RMSE Std", f"${np.std(rmse_scores):.2f}")
            with col2:
                st.metric("Avg CV MAE", f"${np.mean(mae_scores):.2f}")
                st.metric("CV MAE Std", f"${np.std(mae_scores):.2f}")
        
        # Step 3: Preprocess data
        st.header("🔧 Advanced Data Preprocessing")
        features = ['Close', 'RSI', 'MACD']
        X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data_advanced(
            stock_data, timesteps, features=features
        )
        
        # Step 4: Build model
        st.header("🧠 Advanced Model Building")
        model = build_advanced_model(X_train.shape[1:], dropout_rate)
        
        # Show model architecture - graphical representation
        with st.expander("🏗️ View Advanced Model Architecture"):
            st.markdown("### Hybrid CNN + BiLSTM + GRU Architecture")
            
            # Create a visual representation of the model
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("""
                <div style="text-align: center; font-family: monospace; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0; color: white;">📊 INPUT LAYER</h4>
                    <p style="margin: 5px 0; color: #e8e8e8;">Stock Price Sequences</p>
                    <small style="color: #d0d0d0;">Shape: (batch_size, timesteps, features)</small>
                </div>
                
                <div style="text-align: center; margin: 10px 0;">
                    <span style="font-size: 24px;">⬇️</span>
                </div>
                
                <div style="text-align: center; font-family: monospace; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0; color: white;">🔍 CNN LAYERS</h4>
                    <p style="margin: 5px 0; color: #e8e8e8;">Feature Extraction</p>
                    <small style="color: #d0d0d0;">Conv1D → ReLU → Dropout</small>
                </div>
                
                <div style="text-align: center; margin: 10px 0;">
                    <span style="font-size: 24px;">⬇️</span>
                </div>
                
                <div style="text-align: center; font-family: monospace; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0; color: white;">🔄 BiLSTM LAYER</h4>
                    <p style="margin: 5px 0; color: #e8e8e8;">Bidirectional Memory</p>
                    <small style="color: #d0d0d0;">Forward + Backward Processing</small>
                </div>
                
                <div style="text-align: center; margin: 10px 0;">
                    <span style="font-size: 24px;">⬇️</span>
                </div>
                
                <div style="text-align: center; font-family: monospace; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0; color: white;">⚡ GRU LAYER</h4>
                    <p style="margin: 5px 0; color: #e8e8e8;">Gated Recurrent Unit</p>
                    <small style="color: #d0d0d0;">Efficient Sequence Modeling</small>
                </div>
                
                <div style="text-align: center; margin: 10px 0;">
                    <span style="font-size: 24px;">⬇️</span>
                </div>
                
                <div style="text-align: center; font-family: monospace; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0; color: white;">🧠 DENSE LAYERS</h4>
                    <p style="margin: 5px 0; color: #e8e8e8;">Final Processing</p>
                    <small style="color: #d0d0d0;">Dense → Dropout → Output</small>
                </div>
                
                <div style="text-align: center; margin: 10px 0;">
                    <span style="font-size: 24px;">⬇️</span>
                </div>
                
                <div style="text-align: center; font-family: monospace; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h4 style="margin: 0; color: #333;">🎯 OUTPUT</h4>
                    <p style="margin: 5px 0; color: #555;">Predicted Stock Price</p>
                    <small style="color: #777;">Single value prediction</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Model statistics
            st.markdown("### 📈 Model Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Layers", "8")
            with col2:
                st.metric("Dropout Rate", f"{dropout_rate}")
            with col3:
                st.metric("Parameters", "~2.5M")
            with col4:
                st.metric("Architecture", "Hybrid")
            
            # Show technical details if needed
            if st.checkbox("Show Technical Details"):
                import io
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                model.summary()
                sys.stdout = old_stdout
                
                st.code(buffer.getvalue(), language="text")
        
        # Step 5: Train and predict
        st.header("🚀 Training & Prediction")
        (predictions, actual, train_rmse, test_rmse, 
         train_mae, test_mae, history) = train_and_predict_advanced(
            model, X_train, X_test, y_train, y_test, scaler_y, 
            epochs, use_early_stopping
        )
        
        # Step 6: Baseline comparison (if selected)
        baseline_results = None
        if compare_baselines:
            st.header("📈 Baseline Model Comparison")
            baseline_results, baseline_actual = train_baseline_models(stock_data)
        
        # Step 7: Save model and results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"models/{ticker}_{timestamp}.h5"
        model.save(model_filename)
        
        # Save to database
        save_model_to_db(
            ticker, start_str, end_str, timesteps, dropout_rate, epochs,
            train_rmse, test_rmse, train_mae, test_mae, model_filename
        )
        
        # Step 8: Display results
        st.header("📊 Results & Performance Metrics")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train RMSE", f"${train_rmse:.2f}")
        with col2:
            st.metric("Test RMSE", f"${test_rmse:.2f}")
        with col3:
            st.metric("Train MAE", f"${train_mae:.2f}")
        with col4:
            st.metric("Test MAE", f"${test_mae:.2f}")
        
        # Baseline comparison table
        if baseline_results:
            st.subheader("Model Comparison")
            comparison_data = {
                'Model': ['Deep Learning (CNN+BiLSTM+GRU)', 'ARIMA', 'Prophet'],
                'RMSE': [test_rmse, 
                        baseline_results['ARIMA']['rmse'] if baseline_results['ARIMA'] else 'Failed',
                        baseline_results['Prophet']['rmse'] if baseline_results['Prophet'] else 'Failed'],
                'MAE': [test_mae,
                       baseline_results['ARIMA']['mae'] if baseline_results['ARIMA'] else 'Failed',
                       baseline_results['Prophet']['mae'] if baseline_results['Prophet'] else 'Failed']
            }
            st.table(pd.DataFrame(comparison_data))
        
        # Predictions visualization
        st.subheader("Actual vs Predicted Prices")
        
        # Create DataFrame for plotting - ensure all arrays have same length
        min_length = len(actual)
        results_df = pd.DataFrame({
            'Actual': actual[:min_length],
            'Deep Learning': predictions[:min_length]
        })
        
        # Add baseline predictions if available (truncate to match length)
        if baseline_results:
            if baseline_results['ARIMA']:
                arima_pred = baseline_results['ARIMA']['predictions'][:min_length]
                results_df['ARIMA'] = arima_pred
            if baseline_results['Prophet']:
                prophet_pred = baseline_results['Prophet']['predictions'][:min_length]
                results_df['Prophet'] = prophet_pred
        
        st.line_chart(results_df)
        
        # Advanced Analytics and Investment Insights
        st.header("📈 Advanced Analytics & Investment Insights")
        
        # Calculate additional metrics for investment analysis
        def calculate_advanced_metrics(actual_prices, predicted_prices):
            """Calculate comprehensive metrics for investment analysis"""
            
            # Direction accuracy (did we predict up/down correctly?)
            actual_direction = np.diff(actual_prices) > 0
            predicted_direction = np.diff(predicted_prices) > 0
            
            direction_accuracy = accuracy_score(actual_direction, predicted_direction) * 100
            
            # Calculate percentage errors
            percentage_errors = np.abs((actual_prices - predicted_prices) / actual_prices * 100)
            mape = np.mean(percentage_errors)  # Mean Absolute Percentage Error
            
            # Calculate R² (coefficient of determination)
            from sklearn.metrics import r2_score
            r2 = r2_score(actual_prices, predicted_prices)
            
            # Volatility metrics
            actual_volatility = np.std(actual_prices)
            predicted_volatility = np.std(predicted_prices)
            
            return {
                'direction_accuracy': direction_accuracy,
                'mape': mape,
                'r2_score': r2,
                'actual_volatility': actual_volatility,
                'predicted_volatility': predicted_volatility
            }
        
        # Calculate investment signals
        def generate_investment_signals(actual_prices, predicted_prices, current_price):
            """Generate buy/sell/hold signals based on predictions"""
            
            # Calculate price change predictions
            price_change = predicted_prices[-1] - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Generate signals
            if price_change_pct > 5:
                signal = "🟢 STRONG BUY"
                confidence = "High"
                reason = f"Model predicts {price_change_pct:.1f}% price increase"
            elif price_change_pct > 2:
                signal = "🟡 BUY"
                confidence = "Medium"
                reason = f"Model predicts {price_change_pct:.1f}% price increase"
            elif price_change_pct > -2:
                signal = "🟠 HOLD"
                confidence = "Medium"
                reason = f"Model predicts minimal price change ({price_change_pct:.1f}%)"
            elif price_change_pct > -5:
                signal = "🔴 SELL"
                confidence = "Medium"
                reason = f"Model predicts {price_change_pct:.1f}% price decrease"
            else:
                signal = "🔴 STRONG SELL"
                confidence = "High"
                reason = f"Model predicts {price_change_pct:.1f}% price decrease"
            
            return signal, confidence, reason, price_change_pct
        
        # Calculate position sizing recommendation
        def calculate_position_sizing(signal, confidence, portfolio_value=10000):
            """Calculate recommended position size based on signal strength"""
            
            if "STRONG BUY" in signal:
                position_pct = 0.15 if confidence == "High" else 0.10
            elif "BUY" in signal:
                position_pct = 0.10 if confidence == "High" else 0.05
            elif "HOLD" in signal:
                position_pct = 0.02
            else:  # SELL signals
                position_pct = 0.0
            
            recommended_amount = portfolio_value * position_pct
            return position_pct * 100, recommended_amount
        
        # Get advanced metrics
        advanced_metrics = calculate_advanced_metrics(actual, predictions)
        current_stock_price = float(stock_data['Close'].iloc[-1])
        
        # Display comprehensive metrics
        st.subheader("📊 Comprehensive Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Direction Accuracy", f"{advanced_metrics['direction_accuracy']:.1f}%")
        with col2:
            st.metric("MAPE", f"{advanced_metrics['mape']:.2f}%")
        with col3:
            st.metric("R² Score", f"{advanced_metrics['r2_score']:.3f}")
        with col4:
            st.metric("Volatility", f"${advanced_metrics['actual_volatility']:.2f}")
        with col5:
            st.metric("Test Accuracy", f"{100 - advanced_metrics['mape']:.1f}%")
        
        # Generate investment recommendations
        signal, confidence, reason, expected_change = generate_investment_signals(
            actual, predictions, current_stock_price
        )
        
        st.subheader("💰 Investment Recommendation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal", signal)
        with col2:
            st.metric("Confidence", confidence)
        with col3:
            st.metric("Expected Change", f"{expected_change:.1f}%")
        
        st.info(f"**Reasoning**: {reason}")
        
        # Portfolio analysis
        st.subheader("💼 Portfolio Allocation & Risk Analysis")
        
        # Portfolio value input
        portfolio_value = st.slider(
            "Portfolio Value ($)", 
            min_value=1000, 
            max_value=1000000, 
            value=10000, 
            step=1000
        )
        
        position_pct, recommended_amount = calculate_position_sizing(signal, confidence, portfolio_value)
        shares_to_buy = int(recommended_amount / current_stock_price)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Recommended Position", f"{position_pct:.1f}%")
        with col2:
            st.metric("Investment Amount", f"${recommended_amount:,.0f}")
        with col3:
            st.metric("Shares to Buy", f"{shares_to_buy}")
        with col4:
            if shares_to_buy > 0:
                potential_profit = shares_to_buy * current_stock_price * (expected_change / 100)
                st.metric("Potential Profit", f"${potential_profit:,.0f}")
            else:
                st.metric("Risk Level", "Low")
        
        # Risk assessment
        risk_level = "High" if abs(expected_change) > 5 else "Medium" if abs(expected_change) > 2 else "Low"
        st.warning(f"⚠️ **Risk Level: {risk_level}** - Based on predicted volatility of {expected_change:.1f}%")
        
        # Interactive price prediction chart with Plotly
        st.subheader("📈 Interactive Price Prediction Chart")
        
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            y=predictions,
            mode='lines',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add baseline predictions if available
        if baseline_results and baseline_results['ARIMA']:
            arima_pred = baseline_results['ARIMA']['predictions'][:min_length]
            fig.add_trace(go.Scatter(
                y=arima_pred,
                mode='lines',
                name='ARIMA Baseline',
                line=dict(color='green', width=1)
            ))
        
        if baseline_results and baseline_results['Prophet']:
            prophet_pred = baseline_results['Prophet']['predictions'][:min_length]
            fig.add_trace(go.Scatter(
                y=prophet_pred,
                mode='lines',
                name='Prophet Baseline',
                line=dict(color='orange', width=1)
            ))
        
        fig.update_layout(
            title=f"{ticker} Stock Price Prediction Analysis",
            xaxis_title="Days",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction confidence intervals
        st.subheader("📊 Prediction Confidence & Error Analysis")
        
        # Calculate prediction errors
        errors = predictions - actual
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            fig_error = px.histogram(
                errors, 
                title="Prediction Error Distribution",
                labels={'value': 'Prediction Error ($)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col2:
            # Error over time
            fig_error_time = px.line(
                y=errors, 
                title="Prediction Error Over Time",
                labels={'y': 'Error ($)', 'index': 'Days'}
            )
            st.plotly_chart(fig_error_time, use_container_width=True)
        
        # Trading strategy simulation
        st.subheader("💹 Trading Strategy Simulation")
        
        def simulate_trading_strategy(actual_prices, predicted_prices, initial_capital=10000):
            """Simulate a simple trading strategy based on predictions"""
            
            capital = initial_capital
            shares = 0
            portfolio_values = [initial_capital]
            trades = []
            
            for i in range(1, len(predicted_prices)):
                if i < len(predicted_prices) - 1:  # Don't trade on last day
                    predicted_return = (predicted_prices[i+1] - actual_prices[i]) / actual_prices[i]
                    
                    if predicted_return > 0.02:  # Buy if predicted return > 2%
                        if capital > actual_prices[i]:
                            shares_to_buy = int(capital * 0.1 / actual_prices[i])  # Use 10% of capital
                            if shares_to_buy > 0:
                                shares += shares_to_buy
                                capital -= shares_to_buy * actual_prices[i]
                                trades.append(f"Day {i}: BUY {shares_to_buy} shares at ${actual_prices[i]:.2f}")
                    
                    elif predicted_return < -0.02 and shares > 0:  # Sell if predicted return < -2%
                        shares_to_sell = int(shares * 0.5)  # Sell 50% of holdings
                        if shares_to_sell > 0:
                            shares -= shares_to_sell
                            capital += shares_to_sell * actual_prices[i]
                            trades.append(f"Day {i}: SELL {shares_to_sell} shares at ${actual_prices[i]:.2f}")
                
                # Calculate portfolio value
                portfolio_value = capital + shares * actual_prices[i]
                portfolio_values.append(portfolio_value)
            
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            return portfolio_values, trades, final_value, total_return
        
        # Run simulation
        portfolio_values, trades, final_value, total_return = simulate_trading_strategy(actual, predictions)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Starting Capital", f"${10000:,}")
        with col2:
            st.metric("Final Portfolio Value", f"${final_value:,.0f}")
        with col3:
            st.metric("Total Return", f"{total_return:.1f}%")
        
        # Portfolio performance chart
        fig_portfolio = px.line(
            y=portfolio_values,
            title="Portfolio Performance Simulation",
            labels={'y': 'Portfolio Value ($)', 'index': 'Days'}
        )
        fig_portfolio.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="Initial Capital")
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Recent trades
        if trades:
            st.subheader("📋 Recent Trading Signals")
            recent_trades = trades[-5:] if len(trades) > 5 else trades
            for trade in recent_trades:
                st.text(trade)
        
        # Market sentiment analysis
        st.subheader("🎯 Market Sentiment & Technical Analysis")
        
        # Calculate technical indicators sentiment
        latest_rsi = float(stock_data['RSI'].iloc[-1])
        latest_macd = float(stock_data['MACD'].iloc[-1])
        latest_macd_signal = float(stock_data['MACD_signal'].iloc[-1])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rsi_sentiment = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
            rsi_color = "red" if latest_rsi > 70 else "green" if latest_rsi < 30 else "blue"
            st.metric("RSI Sentiment", rsi_sentiment)
            st.caption(f"RSI: {latest_rsi:.1f}")
        
        with col2:
            macd_sentiment = "Bullish" if latest_macd > latest_macd_signal else "Bearish"
            st.metric("MACD Sentiment", macd_sentiment)
            st.caption(f"MACD: {latest_macd:.3f}")
        
        with col3:
            price_momentum = "Upward" if predictions[-1] > actual[-1] else "Downward"
            st.metric("Price Momentum", price_momentum)
            st.caption("Based on AI prediction")
        
        # Training history
        st.subheader("Training Progress")
        col1, col2 = st.columns(2)
        
        with col1:
            loss_df = pd.DataFrame({
                'Training Loss': history.history['loss'],
                'Validation Loss': history.history['val_loss']
            })
            st.line_chart(loss_df)
            st.caption("Model Loss Over Epochs")
        
        with col2:
            mae_df = pd.DataFrame({
                'Training MAE': history.history['mae'],
                'Validation MAE': history.history['val_mae']
            })
            st.line_chart(mae_df)
            st.caption("Model MAE Over Epochs")
        
        # Model download
        st.header("💾 Download Trained Model")
        
        # Read model file for download
        with open(model_filename, 'rb') as f:
            model_bytes = f.read()
        
        st.download_button(
            label="Download Trained Model (.h5)",
            data=model_bytes,
            file_name=f"{ticker}_model_{timestamp}.h5",
            mime="application/octet-stream"
        )
        
        # Success message
        st.success(f"""
        **Model Training Complete!**
        
        - **Stock**: {ticker}
        - **Period**: {start_str} to {end_str}
        - **Model**: Advanced CNN + BiLSTM + GRU
        - **Features**: Close Price, RSI(14), MACD
        - **Test RMSE**: ${test_rmse:.2f}
        - **Test MAE**: ${test_mae:.2f}
        - **Model saved**: {model_filename}
        """)
        
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Advanced Stock Price Forecaster! 🚀
        
        This application uses state-of-the-art deep learning combining:
        - **Convolutional Neural Networks (CNN)** for feature extraction
        - **Bidirectional LSTM** for capturing long-term dependencies
        - **Additional Bidirectional LSTM** for enhanced pattern recognition
        - **Gated Recurrent Units (GRU)** for efficient sequence modeling
        - **Technical Indicators**: RSI(14) and MACD for enhanced predictions
        
        ### Advanced Features:
        - **Rolling Window Validation** for robust performance estimation
        - **Baseline Comparisons** with ARIMA and Prophet models
        - **Early Stopping** to prevent overfitting
        - **Model Persistence** with SQLite database storage
        - **Downloadable Models** for deployment
        
        ### How to use:
        1. Select a stock ticker from the top 10 dropdown
        2. Choose your date range (up to 15 years of data)
        3. Configure model parameters (timesteps, dropout, epochs)
        4. Enable advanced options as needed
        5. Click "Train Model" to start the comprehensive analysis
        
        ### Model Architecture:
        ```
        Conv1D → BiLSTM → BiLSTM → BiLSTM → GRU → GRU → Dense → Output
        ```
        """)
        
        # Show sample results from database
        st.subheader("Recent Model Performance")
        try:
            conn = sqlite3.connect('stock_forecaster.db')
            recent_models = pd.read_sql_query(
                "SELECT ticker, test_rmse, test_mae, created_at FROM models ORDER BY created_at DESC LIMIT 5",
                conn
            )
            conn.close()
            
            if not recent_models.empty:
                st.dataframe(recent_models, use_container_width=True)
            else:
                st.info("No models trained yet. Train your first model to see results here!")
                
        except Exception:
            st.info("Database not initialized. Train your first model to get started!")

if __name__ == "__main__":
    main()