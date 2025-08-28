# Stock Price Forecasting System

## Overview

This is a machine learning project that implements a hybrid deep learning model for stock price prediction, specifically targeting AAPL (Apple Inc.) stock forecasting. The system combines Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and Gated Recurrent Unit (GRU) architectures to create a sophisticated time series prediction model. The application fetches real-time stock data from Yahoo Finance and applies advanced preprocessing techniques before training the hybrid neural network model.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture Pattern
The system follows a pipeline-based architecture for time series forecasting, implementing a data ingestion → preprocessing → model training → prediction workflow. The architecture is designed as a monolithic Python script that handles the entire machine learning pipeline from data acquisition to model evaluation.

### Data Processing Layer
- **Data Source Integration**: Uses Yahoo Finance API (yfinance) for real-time stock data retrieval
- **Preprocessing Pipeline**: Implements MinMaxScaler for data normalization to improve model convergence
- **Time Series Windowing**: Creates sequential data windows for supervised learning from time series data

### Machine Learning Architecture
- **Hybrid Neural Network Design**: Combines three different neural network architectures:
  - CNN layers for feature extraction from sequential patterns
  - Bidirectional LSTM for capturing long-term temporal dependencies in both directions
  - GRU for efficient sequence modeling with reduced computational complexity
- **Sequential Model Structure**: Uses TensorFlow/Keras Sequential API for linear layer stacking
- **Regularization Strategy**: Implements Dropout layers to prevent overfitting

### Model Training and Evaluation
- **Reproducibility Framework**: Sets fixed random seeds for NumPy and TensorFlow to ensure consistent results
- **Performance Metrics**: Implements multiple evaluation metrics including MSE and MAE for comprehensive model assessment
- **Visualization Component**: Uses Matplotlib for data visualization and result analysis

### Error Handling and Logging
- **Graceful Degradation**: Implements try-catch blocks for data loading failures
- **Warning Suppression**: Configures TensorFlow logging to reduce noise during execution
- **Data Validation**: Checks for empty datasets and provides meaningful error messages

## External Dependencies

### Data Providers
- **Yahoo Finance API**: Primary data source for historical stock prices via yfinance library
- **Real-time Market Data**: Fetches OHLC (Open, High, Low, Close) data with configurable date ranges

### Machine Learning Framework
- **TensorFlow/Keras**: Core deep learning framework for model architecture and training
- **Scikit-learn**: Provides preprocessing utilities (MinMaxScaler) and evaluation metrics

### Data Processing and Visualization
- **Pandas**: Primary data manipulation and analysis library for handling time series data
- **NumPy**: Numerical computing foundation for array operations and mathematical functions
- **Matplotlib**: Visualization library for plotting stock price trends and model performance

### Development Tools
- **Python Standard Library**: Uses os module for system operations and warnings for output control
- **Jupyter Notebook Compatible**: Designed to work in interactive Python environments