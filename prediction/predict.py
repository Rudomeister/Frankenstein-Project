import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import json
import logging

# Sett opp logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Bruker levelname

# Les konfigurasjonsfilen
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config['symbol']

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), :])
        Y.append(data[i + look_back, 0])  # Forutsi 'Close' verdien
    return np.array(X), np.array(Y)

def predict(symbol):
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, '..', 'data')
        combined_data_path = os.path.join(data_dir, f'combined_{symbol}_data.csv')
        model_path = os.path.join(data_dir, f'model_{symbol}.h5')

        dataset = pd.read_csv(combined_data_path)
        dataset = dataset[['Close', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff', 'combined_score']].values.astype('float32')

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        look_back = 1
        X, Y = create_dataset(dataset, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

        model = load_model(model_path)
        predictions = model.predict(X)

        # Invers transform for Close verdien
        predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], dataset.shape[1] - 1)))))[:, 0]

        prediction_data_path = os.path.join(data_dir, f'predictions_{symbol}.csv')
        pd.DataFrame(predictions, columns=['Predicted_Close']).to_csv(prediction_data_path, index=False)
        logging.info(f"Saved predictions for {symbol} at {prediction_data_path}")

    except Exception as e:
        logging.error(f"Error in predicting: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise ValueError("Symbol argument is missing. Usage: python predict.py <symbol>")
    symbol = sys.argv[1]
    predict(symbol)
