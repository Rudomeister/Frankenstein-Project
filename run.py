import fetch_news
import analyze_sentiment
import prediction.historical_data
import prediction.prepare_data
import prediction.predict
import prediction.train_model
import trading_decision
import json
import os

config_path = os.path.join(os.path.dirname(__file__), '.', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

# All this file is supposed to do is to run the files as they are sequentially.

# Fetch news
fetch_news.fetch_news(config)

# Analyze sentiment
analyze_sentiment.analyze_sentiment(config)

# Get historical data
prediction.historical_data.get_historical_data(config)

# Prepare data for prediction
prediction.prepare_data.prepare_data(config)

# Train the prediction model
prediction.train_model.train_model(config)

# Make predictions
prediction.predict.predict(config)

# Make trading decisions
trading_decision.make_decision(config)
