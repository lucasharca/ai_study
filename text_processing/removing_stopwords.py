# Stopwords are gluewords like "and", "of", "a" and "to" that helps language flow
# But in natural language processing and machine learning they don't mean much 

# We'll use NLTK -> Natural Language Toolkit

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

en_stopwords = stopwords.words('english')

# print(en_stopwords)

sentence = "it was too far to go to the shop and he did not want her to walk"

sentence_no_stopwords = ' '.join([word for word in sentence.split() if word not in en_stopwords])


print(sentence_no_stopwords)

# We can customize it by adding words we want to keep out of the sentence 
# or removing words that we want to keep

en_stopwords.remove('did')
en_stopwords.remove('not')

en_stopwords.append('go')

sentence_no_stopwords_custom = ' '.join([word for word in sentence.split() if word not in en_stopwords])

print(sentence_no_stopwords_custom)