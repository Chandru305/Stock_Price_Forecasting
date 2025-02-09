# Stock Market Predictor 📈

A powerful web application built with Streamlit that predicts stock prices using machine learning. The app provides real-time stock data analysis, price predictions, technical indicators, and news sentiment analysis.

## 🌟 Features

- **Real-time Stock Data**: Fetch and display current stock information
- **Price Prediction**: ML-based price predictions with customizable timeframes
- **Technical Analysis**: Moving averages and trading volume analysis
- **News Sentiment**: Real-time news sentiment analysis for stocks
- **Interactive Charts**: Visual representation of stock data and predictions
- **Company Information**: Comprehensive company overview and metrics

## 🛠️ Technologies Used

- Python 3.x
- Streamlit
- TensorFlow/Keras
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/stock-market-predictor.git
cd stock-market-predictor
```

Install required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
numpy
pandas
yfinance
keras
streamlit
matplotlib
scikit-learn
requests
```

## 🚀 Usage

1. Launch the application using:
   ```bash
   streamlit run app.py
   ```
2. Enter a stock symbol (e.g., `GOOG` for Google)
3. Select the time period for analysis
4. Adjust prediction days using the slider
5. View various analyses including:
   - Company overview
   - Price charts
   - Moving averages
   - Trading volume
   - Price predictions
   - News sentiment

## 📊 Features in Detail

### Company Overview
- Company name and basic information
- Current stock price and market cap
- Daily returns and trading volume

### Technical Analysis
- Historical price charts
- 50, 100, and 200-day moving averages
- Trading volume analysis

### Price Predictions
- Machine learning-based price predictions
- Future trend forecasting
- Model performance metrics (MAE, MSE, RMSE)

### News Sentiment
- Real-time news updates
- Sentiment analysis of news articles
- Overall sentiment score

## 🔑 API Keys

The application uses the Alpha Vantage API for news sentiment analysis. You'll need to:

1. Get an API key from [Alpha Vantage](https://www.alphavantage.co/)
2. Replace the API key in the code:
   ```python
   apikey = 'YOUR_API_KEY'
   ```

## 📈 Model Details

The application uses a pre-trained Keras model for price predictions. The model should be named `Stock Predictions Model.keras` and placed in the root directory.
