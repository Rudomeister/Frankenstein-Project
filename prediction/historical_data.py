import pandas as pd
from pybit.unified_trading import HTTP
import dateparser
import time
import os
import logging
import json

session = HTTP(testnet=False, api_key=os.environ.get('BYBIT_API_KEY'), api_secret=os.environ.get('BYBIT_SECRET_KEY'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def GetHistoricalData(currency, start_date, end_date, interval):
    start_time = dateparser.parse(start_date)
    end_time = dateparser.parse(end_date)
    start_ts = int(start_time.timestamp() * 1000)  # ms
    end_ts = int(end_time.timestamp() * 1000)  # ms

    df = pd.DataFrame(columns=['startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume'])

    while True:
        logging.info(f"Fetching data from {pd.to_datetime(start_ts, unit='ms')} to {pd.to_datetime(end_ts, unit='ms')}")
        bars = session.get_kline(symbol=currency, interval=str(interval), start=start_ts, category="linear", limit=720)

        if 'result' not in bars or bars['result'] is None:
            logging.error("API call failed or returned no data.")
            break
        
        rows = []
        idx = len(bars['result']['list']) - 1

        while idx >= 0:
            bar = bars['result']['list'][idx]
            rows.append({
                'startTime': pd.to_datetime(int(bar[0]), unit='ms'),
                'openPrice': bar[1],
                'highPrice': bar[2],
                'lowPrice': bar[3],
                'closePrice': bar[4],
                'volume': bar[5]
            })
            idx -= 1
        
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        
        # Oppdater start_ts til tiden for den nyeste dataraden pluss ett intervall
        if len(rows) > 0:
            last_timestamp = int(bars['result']['list'][0][0])  # Tiden til den siste dataraden
            start_ts = last_timestamp + (interval * 60 * 1000)  # Legg til intervallet i millisekunder
        
        logging.info(f"Fetched {len(rows)} rows. Next start timestamp: {pd.to_datetime(start_ts, unit='ms')}")

        # Avslutt hvis vi har hentet all data
        if start_ts >= end_ts:
            break

        time.sleep(0.02)

    # Sørg for at dataene er sortert kronologisk
    df = df.sort_values(by='startTime')

    # Endre kolonnenavnene for å matche det som forventes i resten av prosjektet
    df.rename(columns={
        'startTime': 'Date',
        'openPrice': 'Open',
        'highPrice': 'High',
        'lowPrice': 'Low',
        'closePrice': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    return df


if __name__ == "__main__":
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r') as file:
         config = json.load(file)
         
    symbol = config['symbol']
    interval = int(config['interval'])

    start_date = "January 01, 2023 00:00 UTC"
    end_date = "December 31, 2023 23:59 UTC"

    data = GetHistoricalData(symbol, start_date, end_date, interval)

    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Inkluder intervallet i filnavnet
    data.to_csv(os.path.join(data_dir, "historical_{}_{}_min_data.csv".format(symbol, interval)), index=False)
    logging.info("Historical data saved to csv file")
