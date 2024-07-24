import yfinance as yf
import json
import os
import datetime as dt

# Bygg stien til config.json relativt til skriptets plassering
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')

# Laste inn konfigurasjonsfilen
with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config['symbol']

def fetch_historical_data(symbol=symbol, period='1mo',):
    data = yf.download(symbol, period=period, interval="5m")
    return data

if __name__ == "__main__":
    data = fetch_historical_data(symbol)
    data.to_csv(f'{symbol}_historical.csv')
    print(f"Historical data for {symbol} saved to csv file")