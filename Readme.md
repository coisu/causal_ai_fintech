S&P 500 
- Market index data from Standard & Poor's
- A kind of financial grades of State
- based on 500 American companies' stocks (Apple, Microsoft, Amazon, Google, Tesla, JP Morgan, Coca-cola etc)


## S&P 500's Role in the Economy and Investments

Indicator of the U.S. Economy
    → When the S&P 500 rises, it signifies economic growth.
    → Conversely, a sharp decline in the S&P 500 may indicate an impending economic crisis (e.g., the 2008 financial crisis, the 2020 COVID-19 crash).

A Key Metric for Investors
    → Many investors use the S&P 500 as a reference for making investment decisions.
    → It helps answer questions like, "Should I buy or sell stocks now?"

Impact on Retirement Funds and ETFs
    → Financial products like ETFs (Exchange-Traded Funds) often track the S&P 500.
    → For example, if the S&P 500 increases by 10%, ETFs tracking it will also rise by 10%.


# Feature Engineering
    - for Causal Analysis: Preparing Data for "Impact of Interest Rate Changes on S&P 500"

    To better utilize the data for causal analysis, we need to preprocess and engineer features that help us analyze the relationship between interest rate changes and the S&P 500 index.

✅ Features
Before analyzing S&P 500 data, let's generate additional variables (features) to enhance the study.

📌 New Features to Add:

Daily Returns
    → (Current Closing Price - Previous Closing Price) / Previous Closing Price
Moving Average
    → The average closing price over a specific period

    #### Moving Average (MA) Interpretation
    📌 How to Interpret Moving Averages

    Short-term Moving Average (20-day MA) → Shows the average price trend over the past month.
    Medium-term Moving Average (50-day MA) → Reflects the price movement over the past few months.
    Long-term Moving Average (200-day MA) → Identifies long-term market trends.
    📌 How to Read the Chart:

    If the price is above the moving average → Uptrend 📈
    If the price drops below the moving average → Downtrend 📉
    📌 Key Trading Signals:

    When the short-term MA (20-day) crosses above the long-term MA (200-day)
    → Golden Cross: Strong bullish signal (uptrend expected).
    When the short-term MA crosses below the long-term MA
    → Death Cross: Bearish signal (potential downtrend).


Volatility (Standard Deviation)
    → A metric to analyze market fluctuations
Volume Change Rate
    → (Current Trading Volume - Previous Trading Volume) / Previous Trading Volume

# Causal Inference

## Difference-in-Differences (DiD)
    Control Group and Treated Group

## Granger Causality Test
    FED interest changement
