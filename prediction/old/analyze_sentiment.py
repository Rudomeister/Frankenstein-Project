import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from scipy.special import softmax
import os


config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_path, 'r') as file:
    config = json.load(file)
    
symbol = config['symbol']
def analyze_sentiment_with_vader(description):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(description)
    sentiment = 'POSITIVE' if vs['compound'] >= 0 else 'NEGATIVE'
    score = vs['compound']
    return sentiment, score

def analyze_sentiment_with_bert(description):
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    inputs = tokenizer.encode_plus(description, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)
    scores = outputs[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment = 'POSITIVE' if np.argmax(scores) > 2 else 'NEGATIVE'
    score = float(scores[np.argmax(scores)])  # Konverter til vanlig float
    return sentiment, score

def analyze_sentiment(news_articles):
    processed_news = []
    for article in news_articles:
        description = article['description'] if article['description'] else ''
        date = article['publishedAt'] if 'publishedAt' in article else None

        vader_sentiment, vader_score = analyze_sentiment_with_vader(description)
        bert_sentiment, bert_score = analyze_sentiment_with_bert(description)

        combined_sentiment = 'POSITIVE' if (vader_score + bert_score) / 2 >= 0 else 'NEGATIVE'
        combined_score = float((vader_score + bert_score) / 2)  # Konverter til vanlig float
        
        processed_news.append({
            'description': description,
            'date': date,  # Inkluderer datoen
            'vader_sentiment': vader_sentiment,
            'vader_score': float(vader_score),  # Konverter til vanlig float
            'bert_sentiment': bert_sentiment,
            'bert_score': bert_score,
            'combined_sentiment': combined_sentiment,
            'combined_score': combined_score
        })
    return processed_news

if __name__ == "__main__":
    with open(f'data/raw_{symbol}_news.json', 'r') as file:
        news_articles = json.load(file)
    
    processed_news = analyze_sentiment(news_articles)
    with open(f'data/processed_{symbol}_news.json', 'w') as file:
        json.dump(processed_news, file)
