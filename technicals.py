import pandas as pd
import json
import numpy as np
from openai import OpenAI
import os

def aggregate_sentiment(processed_news):
    positive_scores = [news['combined_score'] for news in processed_news if news['combined_sentiment'] == 'POSITIVE']
    negative_scores = [news['combined_score'] for news in processed_news if news['combined_sentiment'] == 'NEGATIVE']
    
    avg_positive_score = np.mean(positive_scores) if positive_scores else 0
    avg_negative_score = np.mean(negative_scores) if negative_scores else 0
    
    return avg_positive_score, avg_negative_score

def make_trading_decision(avg_positive_score, avg_negative_score, rsi, sma_50, sma_200):
    if avg_positive_score > avg_negative_score:
        if rsi < 30:
            decision = "BUY (RSI indicates oversold)"
        elif sma_50 > sma_200:
            decision = "BUY (Golden cross detected)"
        else:
            decision = "BUY"
    else:
        if rsi > 70:
            decision = "SELL (RSI indicates overbought)"
        elif sma_50 < sma_200:
            decision = "SELL (Death cross detected)"
        else:
            decision = "SELL"
    
    return decision

def get_ai_response(client, messages):
    response = client.chat.completions.create(
        model="gpt-4",
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
    SYMBOL = 'BTC-USD'
    with open(f'processed_{SYMBOL}_news.json', 'r') as file:
        processed_news = json.load(file)
    
    avg_positive_score, avg_negative_score = aggregate_sentiment(processed_news)

    # Les inn data med tekniske indikatorer
    technical_data = pd.read_csv(f'technical_{SYMBOL}_data.csv', parse_dates=['Date'], index_col='Date')
    latest_data = technical_data.iloc[-1]  # Få den nyeste raden med data
    
    rsi = latest_data['RSI']
    sma_50 = latest_data['SMA_50']
    sma_200 = latest_data['SMA_200']

    trading_decision = make_trading_decision(avg_positive_score, avg_negative_score, rsi, sma_50, sma_200)

    print(f"Average Positive Sentiment Score: {avg_positive_score}")
    print(f"Average Negative Sentiment Score: {avg_negative_score}")
    print(f"RSI: {rsi}")
    print(f"SMA 50: {sma_50}")
    print(f"SMA 200: {sma_200}")
    print(f"Trading Decision: {trading_decision}")

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    conversation_history = [
        {"role": "system", "content": "Du er en elite trader i verdensklasse. Du har dyp kunnskap om markedsanalyse, maskinlæring og handelsstrategier. Samarbeid med ditt team av eksperter for å ta de beste beslutningene."},
        {"role": "user", "content": f"Sentimentanalyse viser en gjennomsnittlig positiv score på {avg_positive_score} og en gjennomsnittlig negativ score på {avg_negative_score}. RSI er {rsi}, SMA 50 er {sma_50}, og SMA 200 er {sma_200}. Beslutningen er å {trading_decision}. Hva er din vurdering?"}
    ]
    
    ai_response = get_ai_response(client, conversation_history)
    print(f"AI Response: {ai_response}")
