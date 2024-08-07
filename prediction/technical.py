import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt

# Last ned data
ticker = 'BTC-USD'
data = yf.download(ticker, period='max', interval='1wk')

# Beregn tekniske indikatorer
data['SMA'] = ta.sma(data['Close'], timeperiod=200)
data['EMA'] = ta.ema(data['Close'], timeperiod=200)
data['RSI'] = ta.rsi(data['Close'], timeperiod=12)
macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
data = data.join(macd)

# Plot pris og tekniske indikatorer
plt.figure(figsize=(20,12))

# Prisdata og SMA
plt.subplot(2, 2, 1)
plt.plot(data['Close'], label='Close Price')
plt.plot(data['SMA'], label='200-day SMA', linestyle='--')
plt.plot(data['EMA'], label='200-day EMA')
plt.title('BTC-USD Close Price and SMA and EMA Crossover')
plt.legend()

# RSI
plt.subplot(2, 1, 2)
plt.plot(data['RSI'], label='RSI', color='orange')
plt.axhline(70, linestyle='--', color='red')
plt.axhline(30, linestyle='--', color='green')
plt.title('BTC-USD RSI')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(data['MACD_12_26_9'], label='MACD', color='purple')
plt.axhline(0, linestyle='--', color='black')
plt.plot(data['MACDs_12_26_9'], label='MACD Signal Line', color='green')
plt.plot(data['MACDh_12_26_9'], label='MACD Histogram', color='red')
plt.title('BTC-USD MACD')
plt.legend()

plt.tight_layout()
plt.show()