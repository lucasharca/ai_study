import pandas as pd
import numpy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import transformers
from transformers import pipeline

data = pd.read_csv('./sentiment_analysis/practical_task/book_reviews_sample.csv')

print(data.head())
print(data['reviewText'][0])

# For sentiment analysis we don't to lemmatization, stemming or remove stopwords. 
# Because small words that would be cut can change the meaning/sentiment of the sentence
# We can also keep punctuation, but for this example we'll remove it

data['reviewText_clean'] = data['reviewText'].str.lower()
data['reviewText_clean'] = data.apply(lambda x: re.sub(r"[^\w\s]", "", x['reviewText_clean']), axis=1)


# Vader uses a rule-based sentiment analysis method

vader_sentiment = SentimentIntensityAnalyzer()

# New column to store the sentiment scores
data['vader_sentiment_score'] = data['reviewText_clean'].apply(lambda review: vader_sentiment.polarity_scores(review)['compound'])

# convert into positive, negative or neutral categories. For that we'll create some bins
# Bins: The value ranges that define how the scores are divided

bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']

# pd.cut(column, bins, labels)
#   -> The first argument is the column we want to divide into ranges
#   -> The bins parameters defines the numerical boundaries of those ranges 
#   -> The labels specifies the names we want to assign to each bin

data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)

transformer_pipeline = pipeline("sentiment-analysis")

transformer_labels = []

for review in data['reviewText_clean'].values:
    sentiment_list = transformer_pipeline(review)
    sentiment_label = [sent['label'] for sent in sentiment_list]
    transformer_labels.append(sentiment_label)

data['transformer_sentiment_label'] = transformer_labels


print(data.head())

