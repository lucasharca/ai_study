import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re 
import pandas as pd
import matplotlib.pyplot as plt


bbc_data = pd.read_csv("./text_tagging/practical_task/bbc_news.csv")
en_stopwords = stopwords.words('english')

# print(bbc_data.head())
# print(bbc_data.info())

titles = pd.DataFrame(bbc_data['title'])

# Clean Data
titles['lowercase'] = titles['title'].str.lower()
titles['no_stopwords'] = titles['lowercase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))
titles['no_stopwords_no_punct'] = titles.apply(lambda x: re.sub(r"[^\w\s]", "", x['no_stopwords']), axis=1)

#tokenization
titles['tokens_raw'] = titles.apply(lambda x: word_tokenize(x['title']), axis=1)
titles['tokens_clean'] = titles.apply(lambda x: word_tokenize(x['no_stopwords_no_punct']), axis=1)

# Lemmatizing

lemmatizer = WordNetLemmatizer()
titles["tokens_clean_lemmatized"] = titles['tokens_clean'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

tokens_raw_list = sum(titles['tokens_raw'], [])
tokens_clean_list = sum(titles['tokens_clean_lemmatized'], [])

# print(titles.head())
# print(tokens_raw_list)
# print(tokens_clean_list)

# POS Tagging

nlp = spacy.load('en_core_web_sm')
spacy_doc = nlp(' '.join(tokens_raw_list))

# Dataframe to store the result of our POS tagging
# Easly see which word belongs to which part of speech

pos_df = pd.DataFrame(columns=['token', 'pos_tag'])

for token in spacy_doc:
    pos_df = pd.concat([pos_df,
                        pd.DataFrame.from_records([{'token': token.text, 'pos_tag': token.pos_}])], ignore_index=True)


# Create a token frequency count
# How many times each word appears in our document

pos_df_counts = pos_df.groupby(['token', 'pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

# We can filter ir to see the most common nouns, verbs, adjectives, etc. in our dataframe
nouns = pos_df_counts[pos_df_counts.pos_tag == "NOUN"][0:10]

print(nouns)

verbs = pos_df_counts[pos_df_counts.pos_tag == "VERB"][0:10]

print(verbs)

adj = pos_df_counts[pos_df_counts.pos_tag == "ADJ"][0:10]

print(adj)


# Named Entities Recognition (NER)

ner_df = pd.DataFrame(columns=['token', 'ner_tag'])
for token in spacy_doc.ents:
    # Checks if a value is missing. If it exists then continue (If a token has a valid label, process it) 
    if pd.isna(token.label_) is False:
        ner_df = pd.concat([ner_df, pd.DataFrame.from_records(
            [{'token': token.text, 'ner_tag': token.label_}]
        )], ignore_index=True)


# Check most commom names in our dataset
ner_df_counts = ner_df.groupby(['token', 'ner_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
print(ner_df_counts.head(10))