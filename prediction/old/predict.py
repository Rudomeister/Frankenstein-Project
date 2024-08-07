import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

def visualize_predictions(symbol):
    
    model = load_model(f'{symbol}_model.keras')
    dataset = pd.read_csv(f'combined_{symbol}_data.csv')
    dataset = dataset[['Close', 'combined_score']].values.astype('float32')

    # Skaler bare 'Close' verdiene
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_values = dataset[:, 0].reshape(-1, 1)  # Bare 'Close' kolonnen
    scaled_close = scaler.fit_transform(close_values)

    # Bruk de skalerte 'Close' verdiene sammen med 'combined_score'
    scaled_dataset = np.hstack((scaled_close, dataset[:, 1].reshape(-1, 1)))

    look_back = 1
    X, Y = create_dataset(scaled_dataset, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    predictions = model.predict(X)
    
    # Inverter bare 'Close' verdiene
    predictions = scaler.inverse_transform(predictions)
    Y_true = scaler.inverse_transform(Y.reshape(-1, 1))

    plt.plot(Y_true, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    SYMBOL = 'BTC-USD'
    visualize_predictions(SYMBOL)
