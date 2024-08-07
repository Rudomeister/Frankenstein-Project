import json
import numpy as np
from openai import OpenAI
import os
import logging

# Sett opp logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Les konfigurasjonsfilen
config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config['symbol']

def aggregate_sentiment(processed_news):
    positive_scores = [news['combined_score'] for news in processed_news if news['combined_sentiment'] == 'POSITIVE']
    negative_scores = [news['combined_score'] for news in processed_news if news['combined_sentiment'] == 'NEGATIVE']
    
    avg_positive_score = np.mean(positive_scores) if positive_scores else 0
    avg_negative_score = np.mean(negative_scores) if negative_scores else 0
    
    return avg_positive_score, avg_negative_score

def make_trading_decision(avg_positive_score, avg_negative_score):
    if avg_positive_score > avg_negative_score:
        decision = "BUY"
    else:
        decision = "SELL"
    
    return decision

def get_ai_response(client, messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        presence_penalty=0.5,
        top_p=1,
        stream=False
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    try:
        # Les sentimentdata
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, '..', 'data')
        news_data_path = os.path.join(data_dir, f'processed_{symbol}_news.json')

        logging.info(f"Looking for file: {news_data_path}")
        
        with open(news_data_path, 'r') as file:
            processed_news = json.load(file)
        
        # Agreger sentimentdata
        avg_positive_score, avg_negative_score = aggregate_sentiment(processed_news)
        trading_decision = make_trading_decision(avg_positive_score, avg_negative_score)

        logging.info(f"Average Positive Sentiment Score: {avg_positive_score}")
        logging.info(f"Average Negative Sentiment Score: {avg_negative_score}")
        logging.info(f"Trading Decision: {trading_decision}")

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        conversation_history = [
            {"role": "system", "content": "Du er en elite trader i verdensklasse. Du har dyp kunnskap om markedsanalyse, maskinlæring og handelsstrategier. Samarbeid med ditt team av eksperter for å ta de beste beslutningene."},
            {"role": "user", "content": f"Sentimentanalyse viser en gjennomsnittlig positiv score på {avg_positive_score} og en gjennomsnittlig negativ score på {avg_negative_score}. Beslutningen er å {trading_decision}. Hva er din vurdering?"}
        ]
        
        ai_response = get_ai_response(client, conversation_history)
        logging.info(f"AI Response: {ai_response}")

    except Exception as e:
        logging.error(f"Error in trading decision script: {e}")
