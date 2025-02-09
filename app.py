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

# Sidebar
with st.sidebar:
    st.title("📈 Configuration")
    
    # Stock input
    stock = st.text_input('Enter Stock Symbol', 'GOOG')
    
    # Date range selector
    st.subheader("Date Range")
    date_options = {
        '1 Year': 365,
        '5 Years': 365*5,
        '10 Years': 365*10
    }
    selected_range = st.selectbox('Select Time Period', list(date_options.keys()))
    end_date = datetime.today()
    start_date = end_date - timedelta(days=date_options[selected_range])
    
    # Prediction configuration
    st.subheader("Prediction Settings")
    prediction_days = st.slider('Future Prediction Days', 30, 200, 100)

# Main content
st.title('Stock Market Predictor')

try:
    # Load the pre-trained model
    @st.cache_resource
    def load_model_cached():
        return load_model('Stock Predictions Model.keras')
    
    model = load_model_cached()

    # Fetch stock data
    with st.spinner('Fetching stock data...'):
        data = yf.download(stock, start_date, end_date)
        if data.empty:
            st.error("No data found for the specified stock symbol.")
            st.stop()

    # Company Overview
    st.header('Company Overview')
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_info = yf.Ticker(stock).info
        company_metrics = {
            'Company Name': stock_info.get('longName', 'N/A'),
            'Sector': stock_info.get('sector', 'N/A'),
            'Industry': stock_info.get('industry', 'N/A'),
            'Market Cap': f"${stock_info.get('marketCap', 0):,}",
            'Current Price': f"${data.Close[-1]:.2f}",
            'Volume': f"{data.Volume[-1]:,}"
        }
        
        for metric, value in company_metrics.items():
            st.metric(label=metric, value=value)
    
    with col2:
        daily_return = data.Close.pct_change()[-1] * 100
        st.metric(
            "Daily Return",
            f"{daily_return:.2f}%",
            delta=f"{daily_return:.2f}%",
            delta_color="normal"
        )

    # Stock Data Section
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
    
    # Data preparation
    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)
    
    # Prepare sequences
    x = []
    y = []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])
    
    x, y = np.array(x), np.array(y)
    
    # Make predictions
    with st.spinner('Generating predictions...'):
        predict = model.predict(x)
        scale = 1/scaler.scale_
        predict = predict * scale
        y = y * scale
        
        # Future predictions
        future_x = data_test_scale[-100:]
        future_x = np.expand_dims(future_x, axis=0)
        future_predict = []
        
        for _ in range(prediction_days):
            pred = model.predict(future_x)
            future_predict.append(pred[0, 0])
            future_x = np.append(future_x[:, 1:, :], np.expand_dims(pred, axis=2), axis=1)
        
        future_predict = np.array(future_predict) * scale

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
    
    with st.spinner('Fetching news...'):
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock}&limit=10&sort=RELEVANCE&apikey=7VJKITL4V5PV4DTV'
        response = requests.get(url)
        news_data = response.json()

        if 'feed' in news_data:
            articles = []
            total_sentiment = 0
            
            for item in news_data['feed']:
                ticker_sentiment = item['ticker_sentiment'][0]
                sentiment_score = float(ticker_sentiment['ticker_sentiment_score'])
                total_sentiment += sentiment_score
                
                articles.append({
                    'Title': item['title'],
                    'Time': datetime.strptime(item['time_published'], '%Y%m%dT%H%M%S').strftime('%Y-%m-%d %H:%M'),
                    'Sentiment': ticker_sentiment['ticker_sentiment_label'],
                    'Score': f"{sentiment_score:.2f}"
                })
            
            # Display news
            df = pd.DataFrame(articles)
            st.dataframe(df, hide_index=True)
            
            # Average sentiment
            avg_sentiment = total_sentiment / len(articles)
            sentiment_color = 'green' if avg_sentiment > 0 else 'red'
            st.metric(
                "Overall Sentiment",
                f"{avg_sentiment:.2f}",
                delta=f"{avg_sentiment:.2f}",
                delta_color="normal"
            )
        else:
            st.warning("No news articles found for the specified stock symbol.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check the stock symbol and try again.")