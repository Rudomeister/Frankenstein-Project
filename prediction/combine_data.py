import pandas as pd
import json
import logging
import os

# Sett opp logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Les konfigurasjonsfilen
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config['symbol']
interval = config['interval']

def combine_data(symbol):
    try:
        # Les inn tekniske data
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, '..', 'data')
        technical_data_path = os.path.join(data_dir, f'technical_{symbol}_{interval}_min_data.csv')
        technical_data = pd.read_csv(technical_data_path)
        logging.info(f"Read {len(technical_data)} rows of technical data from {technical_data_path}")

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

        # Sjekk om 'Date' kolonnen eksisterer i technical_data, hvis ikke, bruk den første kolonnen som dato
        if 'Date' not in technical_data.columns:
            technical_data.rename(columns={'Datetime': 'Date'}, inplace=True)

        technical_data['Date'] = pd.to_datetime(technical_data['Date']).dt.tz_localize(None)

        # Ekstra logging for å se de første radene av hver data frame
        logging.info(f"First few rows of technical data:\n{technical_data.head()}")
        logging.info(f"First few rows of sentiment data:\n{sentiment_df.head()}")

        # Kombiner data basert på 'Date'
        combined_data = pd.merge(technical_data, sentiment_df, on='Date', how='outer')
        logging.info(f'Combined data contains {len(combined_data)} rows after merging')

        combined_data = combined_data[['Date', 'Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'combined_score']].dropna()
        combined_data = combined_data.set_index('Date')

        # Lagre de kombinerte dataene
        combined_data_path = os.path.join(data_dir, f'combined_{symbol}_{interval}_min_data.csv')
        combined_data.to_csv(combined_data_path)
        logging.info(f"Combined data for {symbol} with interval {interval} saved to csv file at {combined_data_path}")

    except Exception as e:
        logging.error(f"Error in combining data: {e}")

if __name__ == "__main__":
    combine_data(symbol)
