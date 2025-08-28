"""
Stock Price Forecasting using Hybrid CNN + BiLSTM + GRU Model
This script implements a comprehensive deep learning approach for predicting AAPL stock prices.
"""

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
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(ticker="AAPL", start="2018-01-01", end="2023-01-01"):
    """
    Load stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol
        start (str): Start date in YYYY-MM-DD format
        end (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: Stock data with Close prices
    """
    try:
        print(f"Fetching {ticker} stock data from {start} to {end}...")
        data = yf.download(ticker, start=start, end=end, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Use only the Close column
        close_data = data[['Close']].copy()
        print(f"Successfully loaded {len(close_data)} data points")
        
        return close_data
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

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
    print("Preprocessing data...")
    
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
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Input shape: {X_train.shape[1:]}")
    
    return X_train, X_test, y_train, y_test, scaler

def build_model(input_shape=(60, 1)):
    """
    Build the hybrid CNN + BiLSTM + GRU model
    
    Args:
        input_shape (tuple): Shape of input data
    
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    print("Building hybrid CNN + BiLSTM + GRU model...")
    
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
    
    print("Model architecture:")
    model.summary()
    
    return model

def train_and_evaluate(model, X_train, X_test, y_train, y_test, scaler):
    """
    Train the model and evaluate its performance
    
    Args:
        model: Compiled Keras model
        X_train, X_test, y_train, y_test: Training and testing data
        scaler: MinMaxScaler used for data normalization
    
    Returns:
        tuple: (predictions, actual_values, rmse, mae)
    """
    print("Training the model...")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
        shuffle=False  # Important for time series data
    )
    
    print("\nEvaluating the model...")
    
    # Make predictions
    train_predictions = model.predict(X_train, verbose=0)
    test_predictions = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions and actual values
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    train_mae = mean_absolute_error(y_train_actual, train_predictions)
    test_mae = mean_absolute_error(y_test_actual, test_predictions)
    
    print(f"\nModel Performance:")
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Testing RMSE: ${test_rmse:.2f}")
    print(f"Training MAE: ${train_mae:.2f}")
    print(f"Testing MAE: ${test_mae:.2f}")
    
    return test_predictions.flatten(), y_test_actual.flatten(), test_rmse, test_mae, history

def plot_results(actual, predicted, title="AAPL Stock Price Prediction"):
    """
    Plot actual vs predicted stock prices
    
    Args:
        actual (array): Actual stock prices
        predicted (array): Predicted stock prices
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.plot(actual, label='Actual Price', color='blue', linewidth=2)
    plt.plot(predicted, label='Predicted Price', color='red', linewidth=2)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time (Days)', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    # Also save the plot
    plt.savefig('stock_prediction_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'stock_prediction_plot.png'")

def plot_training_history(history):
    """
    Plot training history (loss and validation loss)
    
    Args:
        history: Keras training history object
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")

def main():
    """
    Main function to run the complete stock forecasting pipeline
    """
    print("=" * 60)
    print("AAPL Stock Price Forecasting with Hybrid Deep Learning Model")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        stock_data = load_data()
        
        # Step 2: Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(stock_data)
        
        # Step 3: Build model
        model = build_model(input_shape=(60, 1))
        
        # Step 4: Train and evaluate
        predictions, actual, rmse, mae, history = train_and_evaluate(
            model, X_train, X_test, y_train, y_test, scaler
        )
        
        # Step 5: Plot results
        plot_results(actual, predictions)
        plot_training_history(history)
        
        # Step 6: Print final summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"Test Set RMSE: ${rmse:.2f}")
        print(f"Test Set MAE: ${mae:.2f}")
        print(f"Average actual price: ${np.mean(actual):.2f}")
        print(f"Average predicted price: ${np.mean(predictions):.2f}")
        print(f"Prediction accuracy: {100 - (mae/np.mean(actual)*100):.2f}%")
        print("=" * 60)
        
        # Optional: Save the model
        model.save('stock_forecasting_model.h5')
        print("Model saved as 'stock_forecasting_model.h5'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your internet connection and try again.")
        return False
    
    return True

if __name__ == "__main__":
    # Check if required packages are available
    required_packages = ['pandas', 'numpy', 'matplotlib', 'sklearn', 'yfinance', 'tensorflow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        # Run the main function
        success = main()
        
        if success:
            print("\nStock forecasting script completed successfully!")
        else:
            print("\nScript execution failed. Please check the error messages above.")
