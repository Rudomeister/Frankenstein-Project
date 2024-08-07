import requests
import json
import os

config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)

symbol = config['symbol']
def fetch_news(symbol):
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    news_data = response.json()
    return news_data['articles']

if __name__ == "__main__":
    news_articles = fetch_news(symbol)
    with open(f'data/raw_{symbol}_news.json', 'w') as file:
        json.dump(news_articles, file)
