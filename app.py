"""
Hybrid CNN + BiLSTM + GRU Stock Price Forecaster - Streamlit Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, GRU, Dense, Dropout
import warnings
from datetime import datetime, date

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_data(ticker, start_date, end_date):
    """
    Load stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date
        end_date (str): End date
    
    Returns:
        pd.DataFrame: Stock data with Close prices
    """
    try:
        with st.spinner(f"Fetching {ticker} stock data..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Use only the Close column
            close_data = data[['Close']].copy()
            st.success(f"Successfully loaded {len(close_data)} data points")
            
            return close_data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data, window_size=60, test_split=0.2):
    """
    Preprocess the stock data for training
    
    Args:
        data (pd.DataFrame): Raw stock data
        window_size (int): Number of timesteps to look back
        test_split (float): Fraction of data for testing
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    with st.spinner("Preprocessing data..."):
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)
        
        # Create sliding windows
        X, y = [], []
        
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for CNN input (samples, timesteps, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into training and testing sets
        split_idx = int(len(X) * (1 - test_split))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        st.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test, scaler

def build_model(input_shape=(60, 1)):
    """
    Build the hybrid CNN + BiLSTM + GRU model
    
    Args:
        input_shape (tuple): Shape of input data
    
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    with st.spinner("Building hybrid CNN + BiLSTM + GRU model..."):
        model = Sequential([
            # CNN layer for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            
            # First BiLSTM layer with return sequences
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            
            # Second BiLSTM layer with return sequences
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            
            # First GRU layer with return sequences
            GRU(64, return_sequences=True),
            Dropout(0.2),
            
            # Second GRU layer without return sequences
            GRU(64),
            Dropout(0.2),
            
            # Dense layers for final prediction
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        st.success("Model built successfully!")
        
        return model

def train_and_predict(model, X_train, X_test, y_train, y_test, scaler, epochs=15):
    """
    Train the model and generate predictions
    
    Args:
        model: Compiled Keras model
        X_train, X_test, y_train, y_test: Training and testing data
        scaler: MinMaxScaler used for data normalization
        epochs: Number of training epochs
    
    Returns:
        tuple: (predictions, actual_values, rmse, mae)
    """
    # Train the model
    with st.spinner(f"Training model for {epochs} epochs..."):
        progress_bar = st.progress(0)
        
        # Custom callback to update progress
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_bar.progress((epoch + 1) / epochs)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            verbose=0,  # Suppress verbose output for cleaner UI
            shuffle=False,  # Important for time series data
            callbacks=[StreamlitCallback()]
        )
        
        progress_bar.empty()
        st.success("Model training completed!")
    
    # Make predictions
    with st.spinner("Generating predictions..."):
        test_predictions = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions and actual values
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
        mae = mean_absolute_error(y_test_actual, test_predictions)
        
        return test_predictions.flatten(), y_test_actual.flatten(), rmse, mae, history

def main():
    """
    Main Streamlit app function
    """
    # Page config
    st.set_page_config(
        page_title="Stock Price Forecaster",
        page_icon="📈",
        layout="wide"
    )
    
    # Title
    st.title("Hybrid CNN + BiLSTM + GRU Stock Price Forecaster")
    st.markdown("---")
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # User inputs
    ticker = st.sidebar.text_input("Enter stock ticker", value="AAPL")
    
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=date(2018, 1, 1)
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=date.today()
    )
    
    window_size = st.sidebar.number_input(
        "Timesteps (window size)", 
        min_value=30, 
        max_value=120, 
        value=60
    )
    
    epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=10,
        max_value=30,
        value=15
    )
    
    run_forecast = st.sidebar.button("Run Forecast", type="primary")
    
    # Main content area
    if run_forecast:
        if start_date >= end_date:
            st.error("Start date must be before end date!")
            return
        
        # Convert dates to strings
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Step 1: Load data
        st.header("📊 Data Loading")
        stock_data = get_data(ticker, start_str, end_str)
        
        if stock_data is None:
            return
        
        # Display basic info about the data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Points", len(stock_data))
        with col2:
            st.metric("Current Price", f"${stock_data['Close'][-1]:.2f}")
        with col3:
            price_change = stock_data['Close'][-1] - stock_data['Close'][0]
            st.metric("Total Change", f"${price_change:.2f}")
        
        # Show raw data chart
        st.subheader("Raw Stock Price Data")
        st.line_chart(stock_data['Close'])
        
        # Step 2: Preprocess data
        st.header("🔧 Data Preprocessing")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data, window_size)
        
        # Step 3: Build model
        st.header("🧠 Model Building")
        model = build_model(input_shape=(window_size, 1))
        
        # Show model summary
        with st.expander("View Model Architecture"):
            # Capture model summary
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            model.summary()
            sys.stdout = old_stdout
            
            st.text(buffer.getvalue())
        
        # Step 4: Train and predict
        st.header("🚀 Training & Prediction")
        predictions, actual, rmse, mae, history = train_and_predict(
            model, X_train, X_test, y_train, y_test, scaler, epochs
        )
        
        # Step 5: Display results
        st.header("📈 Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"${rmse:.2f}")
        with col2:
            st.metric("MAE", f"${mae:.2f}")
        with col3:
            avg_actual = np.mean(actual)
            st.metric("Avg Actual Price", f"${avg_actual:.2f}")
        with col4:
            accuracy = 100 - (mae/avg_actual*100)
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        # Predictions chart
        st.subheader("Actual vs Predicted Prices")
        
        # Create DataFrame for plotting
        results_df = pd.DataFrame({
            'Actual': actual,
            'Predicted': predictions
        })
        
        st.line_chart(results_df)
        
        # Training history
        st.subheader("Training History")
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss chart
            loss_df = pd.DataFrame({
                'Training Loss': history.history['loss'],
                'Validation Loss': history.history['val_loss']
            })
            st.line_chart(loss_df)
            st.caption("Model Loss Over Epochs")
        
        with col2:
            # MAE chart
            mae_df = pd.DataFrame({
                'Training MAE': history.history['mae'],
                'Validation MAE': history.history['val_mae']
            })
            st.line_chart(mae_df)
            st.caption("Model MAE Over Epochs")
        
        # Summary
        st.header("📋 Summary")
        st.success(f"""
        **Forecasting Complete!**
        
        - **Stock**: {ticker}
        - **Period**: {start_str} to {end_str}
        - **Model**: Hybrid CNN + BiLSTM + GRU
        - **Test RMSE**: ${rmse:.2f}
        - **Test MAE**: ${mae:.2f}
        - **Prediction Accuracy**: {accuracy:.1f}%
        """)
        
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Stock Price Forecaster! 📈
        
        This application uses a hybrid deep learning model combining:
        - **Convolutional Neural Networks (CNN)** for feature extraction
        - **Bidirectional LSTM** for capturing long-term dependencies
        - **Gated Recurrent Units (GRU)** for efficient sequence modeling
        
        ### How to use:
        1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
        2. Select the date range for historical data
        3. Choose the number of timesteps for the sliding window
        4. Click "Run Forecast" to start the prediction process
        
        ### Features:
        - Real-time stock data fetching
        - Advanced deep learning model
        - Interactive visualizations
        - Performance metrics (RMSE, MAE)
        - Training progress tracking
        """)
        
        # Show example stocks
        st.subheader("Popular Stock Tickers")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("**Tech**\nAAPL, GOOGL, MSFT")
        with col2:
            st.info("**Finance**\nJPM, BAC, WFC")
        with col3:
            st.info("**Healthcare**\nJNJ, PFE, UNH")
        with col4:
            st.info("**Energy**\nXOM, CVX, COP")

if __name__ == "__main__":
    main()