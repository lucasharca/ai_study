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
from gensim.models import LsiModel

from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt



def main():
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


    # Trying the LSA model

    lsa_model = LsiModel(doc_term, num_topics=num_topics, id2word=dictionary)
    print(lsa_model.print_topics(num_topics=num_topics, num_words=5))

    # To figure out the optimal number of topics we can use coherence scores
    # For that we'll use the coherence model class from Gensim
    # We'll create a for-loop where we'll iterate though different number of topics 
    # and put these coherence scores into a list for inspection
    
    # create two empty lists
    # one for our coherence values and one for our models
    # We then specify the minimum and maximum number of topics we want
    coherence_values = []
    model_list =[]
    min_topics = 2
    max_topics = 11

    # for num_topics_i in range(min_topics, max_topics+1):
    #     # We create a model where the number of topics changes for each run, 
    #     # starting with the minimum and going up to the maximum
    #     print("in the loop")
    #     model = LsiModel(doc_term, num_topics=num_topics_i, id2word=dictionary, random_seed=0)
    #     model_list.append(model)
    #     coherence_model = CoherenceModel(model=model, texts=articles, dictionary=dictionary, coherence='c_v')
    #     coherence_values.append(coherence_model.get_coherence())

    # # This plot will show us that the coherence score is the highest at 3 topics
    # # So it suggests that the model iwth three topics gives the most meaningful grouping of words 
    
    # plt.plot(range(min_topics, max_topics+1), coherence_values)
    # plt.xlabel("Number of Topics")
    # plt.ylabel("Coherence score")
    # plt.legend(("coherence_values"), loc="best")
    # plt.show()

    # With this information at hand, we can create our final model using three topics by
    # setting thhe final num_topics to three 

    final_n_topics = 3
    lsa_model_f = LsiModel(doc_term, num_topics=final_n_topics, id2word=dictionary)

    print("## FINAL ##")
    print(lsa_model_f.print_topics(num_topics=final_n_topics, num_words=5))




if __name__ == "__main__":
    main()