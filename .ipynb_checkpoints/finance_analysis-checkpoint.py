import pandas as pd             # for data manipulation and analysis
import yfinance as yf           # finance data from Yahoo Finance
import matplotlib.pyplot as plt # for plotting
import seaborn as sns           # for plotting, improved visualisation

ticker = "^GSPC"    # S&P 500 index
start_date = "2020-01-01"
end_date = "2024-01-01"
sp500 = yf.download(ticker, start=start_date, end=end_date)

print(sp500.head())

plt.figure(figsize=(12, 6))
sns.lineplot(x=sp500.index, y=sp500['Close'].squeeze(), label='S&P 500 Close Price')
plt.title('S&P 500 Closing Price Over Time')
plt.ylabel('Closing Price')
plt.xlabel('Date')
plt.legend()
plt.show()