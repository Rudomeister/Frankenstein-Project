import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import os
import json
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')


with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config['symbol']

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_model(symbol):
    dataset = pd.read_csv(f'combined_{symbol}_data.csv')
    dataset = dataset[['Close', 'combined_score']].values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    look_back = 1
    X, Y = create_dataset(dataset, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=20, batch_size=1, verbose=2)

    model.save(f'{symbol}_model.keras')
    print(f'Saved model for {symbol}')
    return model, scaler

if __name__ == "__main__":
    train_model(symbol)
