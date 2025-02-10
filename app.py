import numpy as np
import pandas as pd
import yfinance as yf
import requests
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import os
import tensorflow as tf

# Configure GPU memory growth to avoid memory issues
def configure_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU configuration successful")
        else:
            print("No GPU devices found. Using CPU")
    except Exception as e:
        print(f"GPU configuration failed: {str(e)}. Using CPU")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure GPU at startup
configure_gpu()

# Page configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .main {
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model loading with error handling
@st.cache_resource
def load_model_cached():
    try:
        model_path = 'Stock Predictions Model.keras'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Improved data fetching with retry mechanism
def fetch_stock_data(stock_symbol, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Create a Ticker object
            ticker = yf.Ticker(stock_symbol)
            
            # Verify the ticker is valid
            if not ticker.info:
                raise ValueError(f"Invalid stock symbol: {stock_symbol}")
            
            # Download the data
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for stock symbol: {stock_symbol}")
            
            return data, ticker
            
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(f"Failed to fetch stock data after {max_retries} attempts: {str(e)}")
            continue

# Safe get company info
def get_safe_company_info(ticker):
    try:
        info = ticker.info
        return {
            'Company Name': info.get('longName', info.get('shortName', 'N/A')),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 0),
            'Description': info.get('longBusinessSummary', 'N/A')
        }
    except Exception as e:
        return {
            'Company Name': 'N/A',
            'Sector': 'N/A',
            'Industry': 'N/A',
            'Market Cap': 0,
            'Description': 'N/A'
        }

# Prediction function with error handling
def make_predictions(model, data, prediction_days, scaler):
    try:
        if len(data) < 100:
            raise ValueError("Insufficient data for predictions (need at least 100 data points)")
            
        data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])
        
        past_100_days = data_train.tail(100)
        data_test = pd.concat([past_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)
        
        x = []
        y = []
        for i in range(100, data_test_scale.shape[0]):
            x.append(data_test_scale[i-100:i])
            y.append(data_test_scale[i, 0])
        
        x, y = np.array(x), np.array(y)
        
        # Verify data shapes
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Failed to prepare data for predictions")
            
        predict = model.predict(x, verbose=0)
        scale = 1/scaler.scale_
        predict = predict * scale
        y = y * scale
        
        # Future predictions
        future_x = data_test_scale[-100:]
        future_x = np.expand_dims(future_x, axis=0)
        future_predict = []
        
        for _ in range(prediction_days):
            pred = model.predict(future_x, verbose=0)
            future_predict.append(pred[0, 0])
            future_x = np.append(future_x[:, 1:, :], np.expand_dims(pred, axis=2), axis=1)
        
        future_predict = np.array(future_predict) * scale
        
        return y, predict, future_predict
    except Exception as e:
        raise Exception(f"Error making predictions: {str(e)}")

# Main content
st.title('Stock Market Predictor')

# Sidebar
with st.sidebar:
    st.title("📈 Configuration")
    stock = st.text_input('Enter Stock Symbol', 'AAPL').upper()  # Convert to uppercase
    
    st.subheader("Date Range")
    date_options = {
        '1 Year': 365,
        '5 Years': 365*5,
        '10 Years': 365*10
    }
    selected_range = st.selectbox('Select Time Period', list(date_options.keys()))
    end_date = datetime.today()
    start_date = end_date - timedelta(days=date_options[selected_range])
    
    st.subheader("Prediction Settings")
    prediction_days = st.slider('Future Prediction Days', 30, 200, 100)

try:
    # Load model
    model = load_model_cached()
    if model is None:
        st.stop()
    
    # Fetch data with improved error handling
    with st.spinner('Fetching stock data...'):
        data, ticker = fetch_stock_data(stock, start_date, end_date)
        
        if data is None or ticker is None:
            st.error("Failed to fetch stock data. Please check the stock symbol.")
            st.stop()

    # Company Overview with improved error handling
    st.header('Company Overview')
    col1, col2 = st.columns([2, 1])
    
    with col1:
        company_info = get_safe_company_info(ticker)
        
        # Display company metrics
        metrics = {
            'Company Name': company_info['Company Name'],
            'Sector': company_info['Sector'],
            'Industry': company_info['Industry'],
            'Market Cap': f"${company_info['Market Cap']:,}" if company_info['Market Cap'] else 'N/A',
            'Current Price': f"${data.Close[-1]:.2f}" if not data.empty else 'N/A',
            'Volume': f"{data.Volume[-1]:,}" if not data.empty else 'N/A'
        }
        
        for metric, value in metrics.items():
            st.metric(label=metric, value=value)
    
    with col2:
        if not data.empty:
            daily_return = data.Close.pct_change()[-1] * 100
            st.metric(
                "Daily Return",
                f"{daily_return:.2f}%",
                delta=f"{daily_return:.2f}%",
                delta_color="normal"
            )

    # Stock Data Section
    if not data.empty:
        st.header('Stock Price History')
        tab1, tab2, tab3 = st.tabs(["Price Chart", "Moving Averages", "Trading Volume"])

        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, data.Close)
            ax.set_title(f'{stock} Stock Price')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            st.pyplot(fig)
            plt.close()

        with tab2:
            ma_50 = data.Close.rolling(50).mean()
            ma_100 = data.Close.rolling(100).mean()
            ma_200 = data.Close.rolling(200).mean()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, data.Close, label='Price')
            ax.plot(data.index, ma_50, label='MA50')
            ax.plot(data.index, ma_100, label='MA100')
            ax.plot(data.index, ma_200, label='MA200')
            ax.set_title('Moving Averages')
            ax.legend()
            st.pyplot(fig)
            plt.close()

        with tab3:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(data.index, data.Volume)
            ax.set_title('Trading Volume')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volume')
            st.pyplot(fig)
            plt.close()

        # Price Prediction Section
        st.header('Price Predictions')
        
        with st.spinner('Generating predictions...'):
            scaler = MinMaxScaler(feature_range=(0, 1))
            y, predict, future_predict = make_predictions(model, data, prediction_days, scaler)

            # Plot predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y, label='Actual Price', color='green')
            ax.plot(predict, label='Predicted Price', color='red')
            ax.plot(range(len(y), len(y) + len(future_predict)), 
                    future_predict, 
                    label='Future Prediction', 
                    color='blue', 
                    linestyle='--')
            ax.set_title('Price Predictions')
            ax.legend()
            st.pyplot(fig)
            plt.close()

            # Model Performance Metrics
            st.subheader('Model Performance')
            col1, col2, col3 = st.columns(3)
            
            mae = mean_absolute_error(y, predict)
            mse = mean_squared_error(y, predict)
            rmse = np.sqrt(mse)
            
            col1.metric("Mean Absolute Error", f"${mae:.2f}")
            col2.metric("Mean Squared Error", f"${mse:.2f}")
            col3.metric("Root Mean Squared Error", f"${rmse:.2f}")

        # News Sentiment Analysis
        st.header('News Sentiment Analysis')
        
        try:
            with st.spinner('Fetching news...'):
                url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock}&limit=10&sort=RELEVANCE&apikey=7VJKITL4V5PV4DTV'
                response = requests.get(url, timeout=10)
                news_data = response.json()

                if 'feed' in news_data and news_data['feed']:
                    articles = []
                    total_sentiment = 0
                    
                    for item in news_data['feed']:
                        if 'ticker_sentiment' in item and item['ticker_sentiment']:
                            ticker_sentiment = item['ticker_sentiment'][0]
                            sentiment_score = float(ticker_sentiment['ticker_sentiment_score'])
                            total_sentiment += sentiment_score
                            
                            articles.append({
                                'Title': item['title'],
                                'Time': datetime.strptime(item['time_published'], '%Y%m%dT%H%M%S').strftime('%Y-%m-%d %H:%M'),
                                'Sentiment': ticker_sentiment['ticker_sentiment_label'],
                                'Score': f"{sentiment_score:.2f}"
                            })
                    
                    if articles:
                        df = pd.DataFrame(articles)
                        st.dataframe(df, hide_index=True)
                        
                        avg_sentiment = total_sentiment / len(articles)
                        st.metric(
                            "Overall Sentiment",
                            f"{avg_sentiment:.2f}",
                            delta=f"{avg_sentiment:.2f}",
                            delta_color="normal"
                        )
                    else:
                        st.warning("No sentiment data available for this stock.")
                else:
                    st.warning("No news articles found for the specified stock symbol.")
        except Exception as e:
            st.warning(f"Error fetching news data: {str(e)}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please try again with a different stock symbol or time range.")
