import os
import logging
import json
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_historical_data(symbol):
    url = f"https://api-testnet.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=15"
    # Example API call to fetch historical data for the symbol with 15-minute intervals

    response = requests.get(url)
    logging.info(f"Fetching historical data for {symbol}...")
    logging.info(f"Response from API: {response.status_code}")

    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}")

    data = response.json()['result']
    df = pd.DataFrame(data)
    
    # Assuming the API returns a column 'start_at' and it represents a timestamp
    df['start_at'] = pd.to_datetime(df['start_at'], unit='s')
    df.set_index('start_at', inplace=True)
    
    return df

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as file:
        config = json.load(file)

    symbol = config['symbol']
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    historical_data = fetch_historical_data(symbol)
    historical_data.to_csv(os.path.join(data_dir, f'historical_{symbol}_data.csv'))
    logging.info(f"Historical data for {symbol} saved to csv file")
