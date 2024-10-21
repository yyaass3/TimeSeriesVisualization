import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

data = pd.read_csv(r"C:\Users\padidar\Downloads\stock_data.csv", parse_dates=True, index_col='Date')
sb.set(style='whitegrid')
plt.figure(figsize=(12, 6))

# Salary
sb.lineplot(data=data, x='Date', y='High', label='High Price', color='blue')
plt.xlabel('Date')
plt.ylabel('High')
plt.title('Share Highest Price Over Time')
plt.show()

# Monthly
data_resampled = data.resample('M').mean()
sb.lineplot(data=data_resampled, x=data_resampled.index, y='High', label='Month Wise Average High Price', color='blue')
plt.xlabel('Date(monthly)')
plt.ylabel('High')
plt.title('Monthly Resampling Highest Price Over Time')
plt.show()

# Auto Correlation
plot_acf(data['Volume'], lags=50)
plt.xlabel('lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()

# Detecting Stationarity
result = adfuller(data['High'])
print('ADF statistics: ', result[0])
print('p-value: ', result[1])
print('critical values: ', result[4])

# Differencing
data['High_Diff'] = data['High'].diff()
plt.figure(figsize=(12, 6))
plt.plot(data['High'], label='Original High', color='blue')
plt.plot(data['High_Diff'], label='Differenced High', color='green', linestyle='--')
plt.legend()
plt.title('Original vs Differenced High')
plt.show()

# Moving Average
window_size = 120
data['high_smooth'] = data['High'].rolling(window=window_size).mean()
plt.figure(figsize=(12, 6))

plt.plot(data['High'], label='Original High', color='blue')
plt.plot(data['high_smooth'], label=f'Moving Average (Window={window_size})', color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('High')
plt.title('Original vs Moving Average')
plt.legend()
plt.show()

# removing the Null values
data.dropna(subset=['High_Diff'], inplace=True)
data_combined = pd.concat([data['High'], data['High_Diff']], axis=1)
print(data['High_Diff'].head(15))
