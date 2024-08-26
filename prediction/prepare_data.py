import pandas as pd
import json
import logging
import os
import pandas_ta as ta

# Sett opp logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config["symbol"]
interval = config["interval"]

def calculate_technical_indicators(df):
    # Beregn RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Beregn MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_diff'] = macd['MACDh_12_26_9']
    return df

def prepare_data(symbol, interval):
    try:
        # Les inn historiske data
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, '..', 'data')
        historical_data_path = os.path.join(data_dir, f'historical_{symbol}_{interval}_min_data.csv')
        historical_data = pd.read_csv(historical_data_path)
        logging.info(f'Read {len(historical_data)} rows of historical data from {historical_data_path}')

        historical_data.rename(columns={
            'startTime': 'Date',
            'openPrice': 'Open',
            'highPrice': 'High',
            'lowPrice': 'Low',
            'closePrice': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        # Beregn tekniske indikatorer på historiske data
        historical_data = calculate_technical_indicators(historical_data)

        # Les inn sentiment data
        sentiment_data_path = os.path.join(data_dir, f'processed_{symbol}_news.json')
        with open(sentiment_data_path, 'r') as file:
            sentiment_data = json.load(file)
        logging.info(f"Read {len(sentiment_data)} rows of sentiment data from {sentiment_data_path}")

        # Kombiner data
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)

        # Agreger sentimentdataene til daglige verdier
        sentiment_df = sentiment_df.groupby(sentiment_df['Date'].dt.date).agg({
            'combined_score': 'mean'
        }).reset_index()
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

        # Sjekk om 'Date' kolonnen eksisterer i historical_data, hvis ikke, bruk den første kolonnen som dato
        if 'Date' not in historical_data.columns:
            historical_data.rename(columns={'Datetime': 'Date'}, inplace=True)

        historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_localize(None)

        # Ekstra logging for å se de første radene av hver data frame
        logging.info(f"First few rows of historical data:\n{historical_data.head()}")
        logging.info(f"First few rows of sentiment data:\n{sentiment_df.head()}")

        # Kombiner data basert på 'Date'
        combined_data = pd.merge(historical_data, sentiment_df, on='Date', how='outer')
        logging.info(f'Combined data contains {len(combined_data)} rows after merging')

        combined_data = combined_data[['Date', 'Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'combined_score']].dropna()
        combined_data = combined_data.set_index('Date')

        # Lagre de kombinerte dataene
        combined_data_path = os.path.join(data_dir, f'combined_{symbol}_data.csv')
        combined_data.to_csv(combined_data_path)
        logging.info(f"Combined data for {symbol} saved to csv file at {combined_data_path}")

    except Exception as e:
        logging.error(f"Error in preparing data: {e}")
        raise

if __name__ == "__main__":
    prepare_data(symbol, interval)
