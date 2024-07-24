import os
import json
import logging
import argparse
from fetch_news import fetch_news
from analyze_sentiment import analyze_sentiment
from trading_decision import aggregate_sentiment, make_trading_decision, get_ai_response
from prediction.historical_data import fetch_historical_data
from prediction.prepare_data import prepare_data
from prediction.train_model import train_model
from prediction.predict import predict
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(config):
    symbol = config['symbol']
    logging.info(f'Starting process for symbol: {symbol}')
    
    # Hente nyhetsdata
    news_articles = fetch_news(symbol)
    with open(f'raw_{symbol}_news.json', 'w') as file:
        json.dump(news_articles, file)
    logging.info('Fetched and saved news data')
    
    # Sentimentanalyse
    processed_news = analyze_sentiment(news_articles)
    with open(f'processed_{symbol}_news.json', 'w') as file:
        json.dump(processed_news, file)
    logging.info('Processed news data for sentiment analysis')

    # Hente historiske data
    historical_data = fetch_historical_data(symbol)
    historical_data.to_csv(f'{symbol}_historical.csv')
    logging.info('Fetched historical data')

    # Forbered data
    combined_data = prepare_data(symbol)
    combined_data.to_csv(f'combined_{symbol}_data.csv')
    logging.info('Prepared combined data for model training')

    # Tren modell
    model, scaler = train_model(symbol)
    logging.info('Trained Keras regression model')

    # Lag prediksjoner
    predictions = predict(symbol)
    logging.info('Made predictions using the Keras model')

    # Aggregere sentimentdata
    avg_positive_score, avg_negative_score = aggregate_sentiment(processed_news)
    trading_decision = make_trading_decision(avg_positive_score, avg_negative_score)

    logging.info(f"Average Positive Sentiment Score: {avg_positive_score}")
    logging.info(f"Average Negative Sentiment Score: {avg_negative_score}")
    logging.info(f"Trading Decision: {trading_decision}")

    # Konsultere AI-agent
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    conversation_history = [
        {"role": "system", "content": "Du er en elite trader i verdensklasse. Du har dyp kunnskap om markedsanalyse, maskinlæring og handelsstrategier. Samarbeid med ditt team av eksperter for å ta de beste beslutningene."},
        {"role": "user", "content": f"Sentimentanalyse viser en gjennomsnittlig positiv score på {avg_positive_score} og en gjennomsnittlig negativ score på {avg_negative_score}. Beslutningen er å {trading_decision}. Hva er din vurdering?"}
    ]
    
    ai_response = get_ai_response(client, conversation_history)
    logging.info(f"AI Response: {ai_response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some symbols.')
    parser.add_argument('--config', type=str, help='Path to config file', default='config.json')
    args = parser.parse_args()

    # Bygg stien til config.json relativt til main.py
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as file:
        config = json.load(file)

    main(config)
