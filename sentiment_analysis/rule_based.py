from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


sentence_1 = "i had a great time at the movie it was really funny"
sentence_2 = "i had a great time at the movie but the parking was terrible"
sentence_3 = "i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie"

sentence_test = "I enjoyed playing the new pokémon game, but the tutorial was too long and that I hated" 

def run_text_blob(sentence: str):
    print(sentence)

    sentiment_score = TextBlob(sentence)
    print(sentiment_score.sentiment.polarity)


def run_vader(sentence: str):
    vader_sentiment = SentimentIntensityAnalyzer()
    
    print(vader_sentiment.polarity_scores(sentence))



run_text_blob(sentence_test)
run_vader(sentence_test)
