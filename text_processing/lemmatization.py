# Lemmatization reduces the word to a meaningful base form while preserving its meaning
# More sophisticated, as it references a predefined dictionary to find the correct base form of a word
# It knows the context of the word and uses that information to reduce it to its proper base form
# We usually end up with real, meaningful words
# Tradeoff -> We may still have more unique words in our dataset compared to stemming


import nltk

nltk.download('wordnet')
# Extensive lexical database of English 
# Built in dictionary that the lemmatizer uses to make sure the base form it produces are real words

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

connect_tokens = ['connecting', 'connected', 'connectivity', 'connect', 'connects']
learn_tokens = ['learned', 'learning', 'learn', 'learns', 'learner', 'learners']
likes_tokens = ['likes', 'better', 'worse']


for t in connect_tokens:
    print(t, ": ", lemmatizer.lemmatize(t))

for t in learn_tokens:
    print(t, ": ", lemmatizer.lemmatize(t))

for t in likes_tokens:
    print(t, ": ", lemmatizer.lemmatize(t))