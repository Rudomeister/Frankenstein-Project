import pandas as pd
import matplotlib.pyplot as plt
import json
import os

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')

with open(config_path, 'r') as file:
    config = json.load(file)
symbol = config['symbol']
project_root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_root, '..', 'data')
    
historical_data_path = os.path.join(data_dir, f'historical_{symbol}_data.csv')
historical_data = pd.read_csv(historical_data_path)
# Last de historiske dataene
btc_data = pd.read_csv(historical_data_path)
btc_data['Date'] = pd.to_datetime(btc_data['Date'])
# Last de fremtidige prediksjonene
future_predictions = pd.read_csv(os.path.join(data_dir, f'future_{symbol}_predictions.csv'))
future_predictions['Date'] = pd.to_datetime(future_predictions['Date'])

# Kombiner de historiske dataene med fremtidige prediksjoner
all_data = pd.concat([btc_data[['Date', 'Close']], future_predictions.rename(columns={'Prediction': 'Close'})])

# Plot de historiske og fremtidige prediksjonene
plt.figure(figsize=(12, 6))
plt.plot(btc_data['Date'], btc_data['Close'], label='True Values')
plt.plot(all_data['Date'], all_data['Close'], label='Predictions', linestyle='--')
plt.axvline(x=btc_data['Date'].max(), color='r', linestyle='--', label='Prediction Start')
plt.legend()
plt.show()
