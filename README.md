
# Advanced Stock Forecasting App 📈

A comprehensive deep learning application for stock price prediction using hybrid CNN-BiLSTM-GRU neural networks with advanced technical indicators.

## 🚀 Features

- **Hybrid Deep Learning Model**: Combines CNN, Bidirectional LSTM, and GRU layers
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and moving averages
- **Multiple Model Comparison**: Compare CNN-BiLSTM-GRU, Prophet, and ARIMA models
- **Interactive Streamlit Interface**: User-friendly web application
- **Real-time Data**: Fetches live stock data from Yahoo Finance
- **Model Persistence**: Save and load trained models
- **Database Integration**: SQLite database for storing model results

## 🛠️ Technologies Used

- **Python 3.11+**
- **TensorFlow/Keras** - Deep learning framework
- **Streamlit** - Web application framework
- **yfinance** - Stock data API
- **Prophet** - Time series forecasting
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Plotly** - Data visualization
- **Scikit-learn** - Machine learning utilities

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/champion19007/stock-forecasting-app.git
cd stock-forecasting-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Streamlit Web App
```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

### Command Line Script
```bash
python stock_forecasting.py
```

## 🏗️ Model Architecture

The hybrid model combines:
- **CNN layers**: Extract local patterns and features
- **Bidirectional LSTM**: Capture long-term dependencies in both directions
- **GRU layers**: Efficient sequence modeling
- **Dropout layers**: Prevent overfitting

## 📊 Technical Indicators

- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands**
- **Simple Moving Averages** (SMA)
- **Exponential Moving Averages** (EMA)

## 🎯 Model Performance

The app provides comprehensive evaluation metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Model comparison charts
- Prediction vs Actual price plots

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── stock_forecasting.py   # Standalone forecasting script
├── models/               # Saved model files
├── .streamlit/           # Streamlit configuration
├── pyproject.toml        # Project dependencies
└── README.md            # Project documentation
```

## 🔧 Configuration

Customize the model parameters in the Streamlit sidebar:
- Stock ticker symbol
- Date range
- Number of timesteps
- Dropout rate
- Training epochs
- Model architecture

## 📈 Supported Stocks

The app supports any stock ticker available on Yahoo Finance (e.g., AAPL, GOOGL, MSFT, TSLA, etc.)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock price predictions should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.

## 🙏 Acknowledgments

- Yahoo Finance for providing stock data
- Facebook Prophet for time series forecasting
- TensorFlow team for the deep learning framework
- Streamlit for the amazing web app framework
