import json
import pandas as pd
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
interval = config['interval']

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
        # Les kombinerte data med tekniske indikatorer og sentimentdata
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(project_root, '..', 'data')
        combined_data_path = os.path.join(data_dir, f'combined_{symbol}_{interval}_min_data.csv')
        
        df = pd.read_csv(combined_data_path)
        
        # Ta de siste 30 linjene med data
        historical_data_str = df.tail(60).to_string(index=False)
        
        avg_positive_score = df['combined_score'].mean()

        logging.info(f"Average Sentiment Score: {avg_positive_score}")

        conversation_history = [
            {"role": "system", "content": "Du er en elite trader i verdensklasse. Du har dyp kunnskap om markedsanalyse og handelsstrategier. Samarbeid med ditt team av eksperter for å ta de beste beslutningene."},
            {"role": "user", "content": f"Her er de siste 60 dagene for ticker-symbolet {symbol} med historiske data og tekniske indikatorer:\n{historical_data_str}\nSentimentanalyse viser en gjennomsnittlig sentiment score på {avg_positive_score}. Hva synes du vi bør gjøre basert på disse dataene? Ta en beslutning og begrunn den."}
        ]
        
        ai_response = get_ai_response(client, conversation_history)
        logging.info(historical_data_str)
        logging.info(f"AI Response: {ai_response}")

    except Exception as e:
        logging.error(f"Error in trading decision script: {e}")
