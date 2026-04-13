import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from spacy import displacy
# from spacy.tokenizer import tokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LsiModel, TfidfModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # set plot options
    plt.rcParams['figure.figsize'] = (12,8)
    default_plot_color = "#00bfbf"

    data = pd.read_csv("./fake_news/fake_news_data.csv")


    # data['fake_or_factual'].value_counts().plot(kind='bar', color=default_plot_color)
    # plt.title("hello")

    # Here we show that is well divided. Half being factual and half being fake
    # plt.show()

    # exploring our data through POS tags

    nlp = spacy.load('en_core_web_sm')

    # We split the dataset into fake and factual news 
    # so we can later compare the POS tags that occur between each
    fake_news = data[data['fake_or_factual'] == 'Fake News']
    fact_news = data[data['fake_or_factual'] == 'Factual News']


    # Since we're using dataframes, we want to use nlp.pipe over our news 'text' column
    fake_spaceydocs = list(nlp.pipe(fake_news['text']))
    fact_spaceydoc = list(nlp.pipe(fact_news['text']))

    # Create a function to extract the tags for each of the documents in our data
    def extract_token_tags(doc:spacy.tokens.doc.Doc):
        return [(i.text, i.ent_type_, i.pos_) for i in doc]

    fake_tagsdf = []
    fact_tagsdf = []
    columns = ["token", "ner_tag", "pos_tag"]

    for ix, doc in enumerate(fake_spaceydocs):
        tags = extract_token_tags(doc)
        tags = pd.DataFrame(tags)
        tags.columns = columns
        fake_tagsdf.append(tags)

    for ix, doc in enumerate(fact_spaceydoc):
        tags = extract_token_tags(doc)
        tags = pd.DataFrame(tags)
        tags.columns = columns
        fact_tagsdf.append(tags)


    fake_tagsdf = pd.concat(fake_tagsdf)
    fact_tagsdf = pd.concat(fact_tagsdf)


    pos_count_fake = fake_tagsdf.groupby(['token', 'pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    pos_count_fact = fact_tagsdf.groupby(['token', 'pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    # print(pos_count_fake.head(10))
    # print(pos_count_fact.head(10))

    # At this point we can see that there are many punctuation and stopwords

    # print(pos_count_fake.groupby('pos_tag')['token'].count().sort_values(ascending=False).head(10))
    # print(pos_count_fact.groupby('pos_tag')['token'].count().sort_values(ascending=False).head(10))

    # Here we can see the more common used nouns
    # print(pos_count_fake[pos_count_fake.pos_tag == 'NOUN'][:15])
    # print(pos_count_fact[pos_count_fake.pos_tag == 'NOUN'][:15])

    # We'll now extract named entities
    # This is better to do before we do any further cleaning of our data
    # As we saw before, punctuation and capitalized words can provide important context in this step

    top_entities_fake = fake_tagsdf[fake_tagsdf['ner_tag'] != ""].groupby(['token', 'ner_tag']).size().reset_index(name="counts").sort_values(by='counts', ascending=False)
    top_entities_fact = fact_tagsdf[fact_tagsdf['ner_tag'] != ""].groupby(['token', 'ner_tag']).size().reset_index(name="counts").sort_values(by='counts', ascending=False)

    # print(top_entities_fact)
    # print(top_entities_fake)

    # ner_palette = {
    #     'ORG':  sns.color_palette("Set2").as_hex()[0],
    #     'GPE': sns.color_palette("Set2").as_hex()[1],
    #     'NORP': sns.color_palette("Set2").as_hex()[2],
    #     'PERSON': sns.color_palette("Set2").as_hex()[3],
    #     'DATE':sns.color_palette("Set2").as_hex()[4],
    #     'CARDINAL': sns.color_palette("Set2").as_hex()[5],
    #     'PERCENT': sns.color_palette("Set2").as_hex()[6]
    # }

    # sns.barplot(
    #     x = 'counts', 
    #     y = 'token',
    #     hue = 'ner_tag',
    #     palette= ner_palette,
    #     data = top_entities_fact[:10],
    #     orient='h',
    #     dodge=False
    # ).set(title="Most common named entities in fake news")

    # Preprocess text data (cleaning text)
    # Removing all content that comes before the first "-"
    data["text_clean"] = data.apply(lambda x: re.sub(r"^[^-]*-\s", "", x['text']), axis=1)
    # Lowercasing
    data["text_clean"] = data['text_clean'].str.lower()
    # Remove Punctuation
    data["text_clean"] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x["text_clean"]), axis=1)
    # Remove Stopwords
    en_stopwords = stopwords.words("english")
    data["text_clean"] = data['text_clean'].apply(lambda x: " ".join(word for word in x.split() if word not in en_stopwords))
    #Tokenize the text
    data['text_clean'] = data.apply(lambda x: word_tokenize(x["text_clean"]), axis=1)
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    data['text_clean'] = data['text_clean'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
    # Most common ngrams
    tokens_clean = sum(data['text_clean'], [])
    unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()).reset_index()[:10]
    bigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()).reset_index()[:10]

    unigrams['token'] = unigrams['index'].apply(lambda x: x[0])

    # sns.barplot(
    #     x="count",
    #     y="token",
    #     data=unigrams,
    #     orient="h",
    #     palette=[default_plot_color],
    #     hue="token", legend=False
    #     ).set(title="Most Common Unigrams after preprocessing")


    # Sentiment analysis (Using Vader Sentiment)
    # print(unigrams)

    vader_sentiment = SentimentIntensityAnalyzer()
    data['vader_sentiment_score'] = data['text'].apply(lambda x: vader_sentiment.polarity_scores(x)['compound'])


    bins = [-1, -0.1, 0.1, 1]
    names = ["negative", "neutral", "positive"]

    data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)

    # print(data.head())

    # Create a bar chart of the different labels in each of the data

    # data['vader_sentiment_label'].value_counts().plot.bar(color=default_plot_color)

    # sns.countplot(
    #     x = 'fake_or_factual',
    #     hue = 'vader_sentiment_label',
    #     palette= sns.color_palette("hls"),
    #     data=data
    # ).set(title="Sentiment by news type")

    # Topic modeling - Latent Dirichlet Allocation 
    # Vectorize the text

    fake_news_text = data[data['fake_or_factual'] == 'Fake News']['text_clean'].reset_index(drop=True)
    dictionary_fake = corpora.Dictionary(fake_news_text)

    #bag of words
    doc_term_fake = [dictionary_fake.doc2bow(text) for text in fake_news_text]

    # Generating coherence score to determine the optimal number of topics
    # coherence_values = []
    # model_list = []

    min_topics = 2
    max_topics = 11

    # for num_topics_i in range(min_topics, max_topics+1):
    #     print("current: ", num_topics_i)
    #     model = gensim.models.LdaModel(doc_term_fake, num_topics=num_topics_i, id2word=dictionary_fake)
    #     model_list.append(model)
    #     coherence_model = CoherenceModel(model=model, texts=fake_news_text, dictionary=dictionary_fake, coherence='c_v')
    #     coherence_values.append(coherence_model.get_coherence())

    # # plot this coherence values to see what they look like for each of the iterations
    # plt.plot(range(min_topics, max_topics+1), coherence_values)
    # plt.xlabel('Number of topics')
    # plt.ylabel("Coherence Scores")
    # plt.legend(("Coherence values"), loc="best")
    # print(doc_term_fake)

    # The above was commented because it takes some time to process the coherence scores
    
    # In my example, the best score was at 5 topics
    # In the video lesson, the coherence score was 7

    # I'll use my result here
    num_topics_lda = 7
    lda_model = gensim.models.LdaModel(corpus=doc_term_fake, id2word=dictionary_fake, num_topics=num_topics_lda)
    # print(lda_model.print_topics(num_topics=num_topics_lda, num_words=10))

    # plt.show()

    # TF-IDF - Defining functions
    def tfidf_corpus(doc_term_matrix):
        tfidf = TfidfModel(corpus= doc_term_matrix, normalize= True)
        corpus_tfidf = tfidf[doc_term_matrix]
        return corpus_tfidf
    
    def get_coherence_scores(corpus, dictionary, text, min_topics, max_topics):
        coherence_values = []
        model_list = []
        for num_topics_i in range(min_topics, max_topics+1):
            print("working on: ", num_topics_i)
            model = LsiModel(corpus, num_topics=num_topics_i, id2word=dictionary)
            model_list.append(model)
            coherence_model = CoherenceModel(model = model, texts=text, dictionary=dictionary, coherence="c_v")
            coherence_values.append(coherence_model.get_coherence())
        
        plt.plot(range(min_topics, max_topics+1), coherence_values)
        plt.xlabel("Number of topics")
        plt.ylabel("Coherence Scores")
        plt.legend(("cohrence_values"), loc="best")
        plt.show()

    # Create representation
    corpus_tfidf_fake = tfidf_corpus(doc_term_fake)
    # get_coherence_scores(corpus_tfidf_fake, dictionary_fake, fake_news_text, min_topics=2, max_topics=11)

    # In my example, the best score was at 4 topics
    # In the video lesson, the coherence score was 7
    lsa_model = LsiModel(corpus_tfidf_fake, id2word=dictionary_fake, num_topics=7)
    # This will answer the question: What are the different topics appearing in fake news?
    # print(lsa_model.print_topics())

    #################
    # Can we create a custom classifier to accurately classify fake news versus factual news in our dataset?
    #################

    X = [','.join(map(str, l)) for l in data['text_clean']]
    Y = data['fake_or_factual']

    countvec = CountVectorizer()
    countvec_fit = countvec.fit_transform(X)

    bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns=countvec.get_feature_names_out())
    X_train, X_test, y_train, y_test = train_test_split(bag_of_words, Y, test_size=0.3)

    # Create Classifier with simple logistic regression
    lr = LogisticRegression(random_state=0).fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    print(accuracy_score(y_pred_lr, y_test))

    print(classification_report(y_test, y_pred_lr))

    # This model is fine with a high accuracy score
    # But out of interest, let's see what it would look like if we use SVM (support vector machine)

    svm = SGDClassifier().fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    print(accuracy_score(y_pred_svm, y_test))
    print(classification_report(y_test, y_pred_svm))


if __name__ == "__main__":
    main()