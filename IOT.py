import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet

# Load IoT data set into a Pandas DataFrame
df = pd.read_csv('iot_data.csv', index_col='timestamp', parse_dates=True)

# Resample the data to hourly frequency and fill missing values with linear interpolation
df = df.resample('H').interpolate()

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(df, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposition components
plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

# Check the stationarity of the time series
def stationarity_check(timeseries):
    # Calculate rolling statistics
    rolling_mean = timeseries.rolling(window=24).mean()
    rolling_std = timeseries.rolling(window=24).std()

    # Plot rolling statistics
    plt.figure(figsize=(12,8))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

stationarity_check(df)

# Plot the autocorrelation and partial autocorrelation plots
plot_acf(df, lags=50)
plot_pacf(df, lags=50)
plt.show()

# Create ARIMA model and fit it to the data
model = ARIMA(df, order=(2,1,2))
results_ARIMA = model.fit()

# Predict values with ARIMA model
predictions_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_cumsum = predictions_ARIMA.cumsum()
predictions_ARIMA = predictions_ARIMA.add(df['iot_value'].iloc[0], fill_value=0)

# Calculate mean squared error
mse = mean_squared_error(df['iot_value'], predictions_ARIMA)
print('Mean Squared Error:', mse)

# Plot ARIMA predictions against the original data
plt.figure(figsize=(12,8))
plt.plot(df.index, df['iot_value'], label='Original')
plt.plot(predictions_ARIMA.index, predictions_ARIMA, label='ARIMA')
plt.title('ARIMA Predictions vs. Original Data')
plt.legend(loc='best')
plt.show()

Create Exponential Smoothing model and fit it to the data
model = ExponentialSmoothing(df, seasonal_periods=24, trend='add', seasonal='add').fit()

Predict values with Exponential Smoothing model
predictions_ES = model.predict(start=df.index[0], end=df.index[-1])

Calculate mean squared error
mse = mean_squared_error(df['iot_value'], predictions_ES)
print('Mean Squared Error:', mse)

Plot Exponential Smoothing predictions against the original data
plt.figure(figsize=(12,8))
plt.plot(df.index, df['iot_value'], label='Original')
plt.plot(predictions_ES.index, predictions_ES, label='Exponential Smoothing')
plt.title('Exponential Smoothing Predictions vs. Original Data')
plt.legend(loc='best')
plt.show()

Create Prophet model and fit it to the data
df_prophet = df.reset_index()
df_prophet.columns = ['ds', 'y']
model = Prophet(seasonality_mode='additive', yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
model.fit(df_prophet)

Predict values with Prophet model
future = model.make_future_dataframe(periods=24, freq='H')
predictions_Prophet = model.predict(future)

Calculate mean squared error
mse = mean_squared_error(df['iot_value'], predictions_Prophet['yhat'][:-24])
print('Mean Squared Error:', mse)

Plot Prophet predictions against the original data
plt.figure(figsize=(12,8))
plt.plot(df.index, df['iot_value'], label='Original')
plt.plot(predictions_Prophet['ds'][:-24], predictions_Prophet['yhat'][:-24], label='Prophet')
plt.title('Prophet Predictions vs. Original Data')
plt.legend(loc='best')
plt.show()
