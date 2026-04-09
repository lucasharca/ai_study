# Topic modeling helps us uncover the main themes that run though a collection of documents
# Here we'll use the Latent Dirichlet Allocation Algorithm (LDA)
# To do this we'll:
# - Prepare the text
# - Build the structures the LDA expects
# - Train the model with Gensim (Python Lib designed for text analysis)

import pandas as pd
import re
from  nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
import gensim.corpora as corpora

data = pd.read_csv('./topic_modeling/files/news_articles.csv')
articles = data['content']

# Take just the content of the article, lowercase and remove punctuation
articles = articles.str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))

# Stop word removal
en_stopwords = stopwords.words("english")
articles = articles.apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# Tokenize
articles = articles.apply(lambda x: word_tokenize(x))

# Stemming (done for speed as we have a lot of text)
# We can Lemmatize as well
ps = PorterStemmer()
articles = articles.apply(lambda tokens: [ps.stem(token) for token in tokens])

# Here we create a dictionary, which will map every unique word in our dataset to a unique ID number
# Later this will alloow the LDA model to work with the text in a structured way
dictionary = corpora.Dictionary(articles)

# Now we create a document term matrix: A table that show which words appear in each document, and how often
# This way the article is represented as a bag of words vector, the format LDA model needs
doc_term = [dictionary.doc2bow(text) for text in articles]


# Now, to begin modeling, we first need to decide how many topics we want the LDA model to extract
# In this example, we'll set the number of topics to two.
num_topics = 2

# This function needs three key arguments 
# corpus: our document term matrix; 
# id2word: dictionary which maps word IDs back to actual words;
# num_topics: the number of topics we want 

lda_model = gensim.models.LdaModel(
    corpus=doc_term, 
    id2word=dictionary, 
    num_topics=num_topics
    )

# In this result, since we are analyzing news articles, these topics are not very informative
# this suggests we may need to adjust the number of topics, 
# clean the data set further or explore the data more to find meaningful topics

print(lda_model.print_topics(num_topics=num_topics, num_words=5))