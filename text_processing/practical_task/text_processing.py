import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd

data = pd.read_csv("./text_processing/practical_task/tripadvisor_hotel_reviews.csv")

print(data.info())
print(data.head())
print(data['Review'][0])

data['review_lowercase'] = data['Review'].str.lower()

print(data.head())

en_stopwords = stopwords.words('english')
en_stopwords.remove('not')

# The apply function lets us take on column and perform a custom operation on every value in it
# Here we basically telling python to go through each review in this collumn and apply a function 

# First we remove the stopwords
data['review_no_stopwords'] = data['review_lowercase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

print(data['Review'][0])
print(data['review_no_stopwords'][0])

# We then remove punctuation and in this case substitute the * for the word star
data['review_no_stopwords_no_punct'] = data.apply(lambda x: re.sub(r"[*]", "star", x['review_no_stopwords']), axis=1)
print("########################")
print(data['review_no_stopwords_no_punct'][2])

data['review_no_stopwords_no_punct'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['review_no_stopwords_no_punct']), axis=1)
print("########################")
print(data['review_no_stopwords_no_punct'][2])


print("########################")

# Now we tokenize it

data['tokenized'] = data.apply(lambda x: word_tokenize(x['review_no_stopwords_no_punct']), axis=1)
print(data['tokenized'][0])

print("########################")
# Stemming
ps = PorterStemmer()

data['stemmed'] = data['tokenized'].apply(lambda tokens: [ps.stem(token) for token in tokens])
print(data['stemmed'][0])


print("########################")
# let's compare it to Lemmatization
lemmatizer = WordNetLemmatizer()
data['lemmatized'] = data['tokenized'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
print(data['lemmatized'][0])

# We now have a clean dataset with no punctuation, lowecased, tokenized, stemmed and lemmatized
# Now we need to prepare our text in the right format
# Right now each row in the 'lemmatized' column contains a separate list of tokens
# Each review is stored as its own list of lemmatized words
# We need to combine these smaller lists into one long list that contains every token from all reviews
print("########################")
tokens_clean = sum(data['lemmatized'], [])

unigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts())

print(unigrams)
