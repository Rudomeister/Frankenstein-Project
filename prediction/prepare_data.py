import pandas as pd
import json
import os
import logging

# Sett opp logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def round_time_to_nearest_quarter(datetime_series):
    return datetime_series.dt.floor('15T')

def prepare_data(symbol):
    # Les inn historiske data
    historical_data = pd.read_csv(f'{symbol}_historical.csv')
    logging.info(f'Read {len(historical_data)} rows of historical data')

    # Les inn sentiment data
    with open(f'processed_{symbol}_news.json', 'r') as file:
        sentiment_data = json.load(file)
    logging.info(f'Read {len(sentiment_data)} rows of sentiment data')

    # Kombiner data
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Forsikre oss om at begge datasett bruker kolonnenavnet 'Date' for dato
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)
    
    # Sjekk om 'Date' kolonnen eksisterer i historical_data, hvis ikke, bruk den første kolonnen som dato
    if 'Date' not in historical_data.columns:
        historical_data.rename(columns={historical_data.columns[0]: 'Date'}, inplace=True)
    
    historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.tz_localize(None)
    
    # Runde ned tidspunktene i sentimentdataene til nærmeste kvarter
    sentiment_df['Date'] = round_time_to_nearest_quarter(sentiment_df['Date'])
    historical_data['Date'] = round_time_to_nearest_quarter(historical_data['Date'])
    
    # Fjern dupliserte kolonner hvis de eksisterer
    sentiment_df.drop(columns=['date'], inplace=True)
    historical_data.drop(columns=['date'], inplace=True, errors='ignore')

    # Ekstra logging for å se de første radene av hver data frame
    logging.info(f"First few rows of historical data:\n{historical_data.head()}")
    logging.info(f"First few rows of sentiment data:\n{sentiment_df.head()}")

    # Kombiner data basert på 'Date'
    combined_data = pd.merge(historical_data, sentiment_df, on='Date', how='inner')
    logging.info(f'Combined data contains {len(combined_data)} rows after merging')
    
    combined_data = combined_data[['Date', 'Close', 'combined_score']]
    combined_data = combined_data.set_index('Date')

    return combined_data

if __name__ == "__main__":
    SYMBOL = 'BTC-USD'
    combined_data = prepare_data(SYMBOL)
    combined_data.to_csv(f'combined_{SYMBOL}_data.csv')
    logging.info(f"Combined data for {SYMBOL} saved to csv file")
