import json
import numpy as np
from openai import OpenAI
import os
config_path = os.path.join(os.path.dirname(__file__), '.', 'config.json')
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
    message = response.choices[0].message
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    SYMBOL = symbol
    with open(f'processed_{SYMBOL}_news.json', 'r') as file:
        processed_news = json.load(file)
    
    avg_positive_score, avg_negative_score = aggregate_sentiment(processed_news)
    trading_decision = make_trading_decision(avg_positive_score, avg_negative_score)

    print(f"Average Positive Sentiment Score: {avg_positive_score}")
    print(f"Average Negative Sentiment Score: {avg_negative_score}")
    print(f"Trading Decision: {trading_decision}")

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    conversation_history = [
        {"role": "system", "content": "Du er en elite trader i verdensklasse. Du har dyp kunnskap om markedsanalyse, maskinlæring og handelsstrategier. Samarbeid med ditt team av eksperter for å ta de beste beslutningene."},
        {"role": "user", "content": f"Sentimentanalyse viser en gjennomsnittlig positiv score på {avg_positive_score} og en gjennomsnittlig negativ score på {avg_negative_score}. Beslutningen er å {trading_decision}. Hva er din vurdering?"}
    ]
    
    ai_response = get_ai_response(client, conversation_history)
    print(f"AI Response: {ai_response}")
