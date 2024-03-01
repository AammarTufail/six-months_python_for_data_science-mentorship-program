# Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# setting the side bar to collapsed taa k footer jo ha wo sahi dikhay
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


# Title
app_name = 'Stock Market Forecasting App'
st.title(app_name)
st.subheader('This app is created to forecast the stock market price of the selected company.')
# Add an image from an online resource
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Take input from the user of the app about the start and end date

# Sidebar
st.sidebar.header('Select the parameters from below')

start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
# Add ticker symbol list
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select the company', ticker_list)

# Fetch data from user inputs using yfinance library
data = yf.download(ticker, start=start_date, end=end_date)
# Add Date as a column to the dataframe
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# Plot the data
st.header('Data Visualization')
st.subheader('Plot of the data')
st.write("**Note:** Select your specific date range on the sidebar, or zoom in on the plot and select your specific column")
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)

# Add a select box to choose the column for forecasting
column = st.selectbox('Select the column to be used for forecasting', data.columns[1:])

# Subsetting the data
data = data[['Date', column]]
st.write("Selected Data")
st.write(data)

# ADF test to check stationarity
st.header('Is data Stationary?')
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the data
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model='additive', period=12)
st.write(decomposition.plot())
# Make same plot in Plotly
st.write("## Plotting the decomposition in Plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend, title='Trend', width=1000, height=400, labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal, title='Seasonality', width=1000, height=400,
labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid, title='Residuals', width=1000, height=400,
labels={'x': 'Date', 'y': 'Price'}).update_traces(line_color='Red', line_dash='dot'))

# Model selection
models = ['SARIMA', 'Random Forest', 'LSTM', 'Prophet']
selected_model = st.sidebar.selectbox('Select the model for forecasting', models)

if selected_model == 'SARIMA':
    # SARIMA Model
    # User input for SARIMA parameters
    p = st.slider('Select the value of p', 0, 5, 2)
    d = st.slider('Select the value of d', 0, 5, 1)
    q = st.slider('Select the value of q', 0, 5, 2)
    seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

    model = sm.tsa.statespace.SARIMAX(data[column], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
    model = model.fit()

    # Print model summary
    st.header('Model Summary')
    st.write(model.summary())
    st.write("---")

    # Forecasting using SARIMA
    st.write("<p style='color:green; font-size: 50px; font-weight: bold;'>Forecasting the data with SARIMA</p>",
             unsafe_allow_html=True)

    forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
    # Predict the future values
    predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period)
    predictions = predictions.predicted_mean
    # Add index to the predictions
    predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
    predictions = pd.DataFrame(predictions)
    predictions.insert(0, "Date", predictions.index, True)
    predictions.reset_index(drop=True, inplace=True)
    st.write("Predictions", predictions)
    st.write("Actual Data", data)
    st.write("---")

    # Plot the data
    fig = go.Figure()
    # Add actual data to the plot
    fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    # Add predicted data to the plot
    fig.add_trace(
        go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode='lines', name='Predicted',
                   line=dict(color='red')))
    # Set the title and axis labels
    fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    # Display the plot
    st.plotly_chart(fig)


elif selected_model == 'Random Forest':
    # Random Forest Model
    st.header('Random Forest Regression')

    # Splitting data into training and testing sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Feature engineering
    train_X, train_y = train_data['Date'], train_data[column]
    test_X, test_y = test_data['Date'], test_data[column]

    # Initialize and fit the Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    rf_model.fit(train_X.values.reshape(-1, 1), train_y.values)

    # Predict the future values
    predictions = rf_model.predict(test_X.values.reshape(-1, 1))

    # Calculate mean squared error
    mse = mean_squared_error(test_y, predictions)
    rmse = np.sqrt(mse)

    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Combine training and testing data for plotting
    combined_data = pd.concat([train_data, test_data])

    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_data["Date"], y=combined_data[column], mode='lines', name='Actual',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_data["Date"], y=predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (Random Forest)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
    st.plotly_chart(fig)

elif selected_model == 'LSTM':
    # LSTM Model
    st.header('Long Short-Term Memory (LSTM)')

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

    # Split the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    # Create sequences for LSTM model
    def create_sequences(dataset, seq_length):
        X, y = [], []
        for i in range(len(dataset) - seq_length):
            X.append(dataset[i:i + seq_length, 0])
            y.append(dataset[i + seq_length, 0])
        return np.array(X), np.array(y)

    seq_length = st.slider('Select the sequence length', 1, 30, 10)

    train_X, train_y = create_sequences(train_data, seq_length)
    test_X, test_y = create_sequences(test_data, seq_length)

    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    # Build the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(train_X.shape[1], 1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))

    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(train_X, train_y, epochs=10, batch_size=16)

    # Predict the future values
    train_predictions = lstm_model.predict(train_X)
    test_predictions = lstm_model.predict(test_X)
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)

    # Calculate mean squared error
    train_mse = mean_squared_error(train_data[seq_length:], train_predictions)
    train_rmse = np.sqrt(train_mse)
    test_mse = mean_squared_error(test_data[seq_length:], test_predictions)
    test_rmse = np.sqrt(test_mse)

    st.write(f"Train RMSE: {train_rmse}")
    st.write(f"Test RMSE: {test_rmse}")

    # Combine training and testing data for plotting
    train_dates = data['Date'][:train_size + seq_length]
    test_dates = data['Date'][train_size + seq_length:]
    combined_dates = pd.concat([train_dates, test_dates])
    combined_predictions = np.concatenate([train_predictions, test_predictions])

    # Plot the data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=combined_dates, y=data[column], mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=test_dates, y=combined_predictions, mode='lines', name='Predicted',
                             line=dict(color='red')))
    fig.update_layout(title='Actual vs Predicted (LSTM)', xaxis_title='Date', yaxis_title='Price',
                      width=1000, height=400)
    st.plotly_chart(fig)

elif selected_model == 'Prophet':
    # Prophet Model
    st.header('Facebook Prophet')

    # Prepare the data for Prophet
    prophet_data = data[['Date', column]]
    prophet_data = prophet_data.rename(columns={'Date': 'ds', column: 'y'})

    # Create and fit the Prophet model
    prophet_model = Prophet()
    prophet_model.fit(prophet_data)

    # Forecast the future values
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)

    # Plot the forecast
    fig = prophet_model.plot(forecast)
    plt.title('Forecast with Facebook Prophet')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig)

st.write("Model selected:", selected_model)

# urls of the images
github_url = "https://img.icons8.com/fluent/48/000000/github.png"
twitter_url = "https://img.icons8.com/color/48/000000/twitter.png"
medium_url = "https://img.icons8.com/?size=48&id=BzFWSIqh6bCr&format=png"

# redirect urls
github_redirect_url = "https://github.com/Muhammad-Ali-Butt"
twitter_redirect_url = "https://twitter.com/Data_Maestro"
medium_redirect_url = "https://medium.com/@Data_Maestro"

# adding a footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0; 
    width: 100%;
    background-color: #f5f5f5;
    color: #000000;
    text-align: center;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="footer">Made with ❤️ by Muhammad Ali Butt<a href="{github_redirect_url}"><img src="{github_url}" width="30" height="30"></a>'
             f'<a href="{twitter_redirect_url}"><img src="{twitter_url}" width="30" height="30"></a>'
             f'<a href="{medium_redirect_url}"><img src="{medium_url}" width="30" height="30"></a> | Credits: Dr.Ammaar Tufail</div>', unsafe_allow_html=True)