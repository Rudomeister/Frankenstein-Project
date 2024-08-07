import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os
import json
import logging
import sys

# Sett opp logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

if len(sys.argv) < 2:
    raise ValueError("Symbol argument is missing. Usage: python train_model.py <symbol>")

symbol = sys.argv[1]

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), :])
        Y.append(data[i + look_back, 0])  # Forutsi 'Close' verdien
    return np.array(X), np.array(Y)

def train_model(symbol):
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, '..', 'data')
        combined_data_path = os.path.join(data_dir, f'combined_{symbol}_data.csv')
        
        dataset = pd.read_csv(combined_data_path)
        dataset = dataset[['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'combined_score']].values.astype('float32')

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        look_back = 1
        X, Y = create_dataset(dataset, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

        model = Sequential()
        model.add(LSTM(4, input_shape=(look_back, X.shape[2])))  # Endre input_shape til (look_back, 6)
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, Y, epochs=20, batch_size=1, verbose=2)

        model_path = os.path.join(data_dir, f'model_{symbol}.keras')
        model.save(model_path)
        logging.info(f'Saved model for {symbol} at {model_path}')

    except Exception as e:
        logging.error(f"Error in training model: {e}")

if __name__ == "__main__":
    train_model(symbol)
