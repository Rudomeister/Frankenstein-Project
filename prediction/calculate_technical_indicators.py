import pandas as pd
import pandas_ta as ta
import os
import json
import logging

# Sett opp logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Les konfigurasjonsfilen
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config['symbol']
interval = config['interval']

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

if __name__ == "__main__":
    try:
        # Les inn de historiske dataene
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, '..', 'data')
        historical_data_path = os.path.join(data_dir, f'historical_{symbol}_{interval}_min_data.csv')
        
        df = pd.read_csv(historical_data_path)
        logging.info(f"Read {len(df)} rows from {historical_data_path}")

        # Beregn tekniske indikatorer
        df = calculate_technical_indicators(df)
        
        # Lagre de oppdaterte dataene
        technical_data_path = os.path.join(data_dir, f'technical_{symbol}_{interval}_min_data.csv')
        df.to_csv(technical_data_path, index=False)
        
        logging.info(f"Calculated technical indicators and saved to {technical_data_path}")

    except Exception as e:
        logging.error(f"Error in calculating technical indicators: {e}")
