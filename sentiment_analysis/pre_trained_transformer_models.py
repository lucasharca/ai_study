import transformers
from transformers import pipeline
# pipeline() provides a quick way to use powerful 
# pre-trained models without needing to build them from scratch

sentiment_pipeline = pipeline('sentiment-analysis')

sentence_1 = "i had a great time at the movie it was really funny"
sentence_2 = "i had a great time at the movie but the parking was terrible"
sentence_3 = "i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie"

sentence_test = "I enjoyed playing the new pokémon game, but the tutorial was too long" 
# The sentence anlysis here will return a label (neutral, negative, positive), 
# and a score that indicates how sure of it the model is 
print(sentence_1)
print(sentiment_pipeline(sentence_1))
print(sentiment_pipeline(sentence_2))
print(sentiment_pipeline(sentence_3))
print(sentiment_pipeline(sentence_4))
print(sentiment_pipeline(sentence_test))

specific_model = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
print(sentence_1)
print(specific_model(sentence_1))