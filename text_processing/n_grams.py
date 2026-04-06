import nltk
import pandas as pd
# Pandas is one of the most important python libraries for data analysis
# designed to make working with structured data fast and intuitive
# Core feature -> Data frame -> Works like a spreadsheet in python
# Perfect for preparing text data for analysis, keeping it organized and connecting with other steps in machine learning
import matplotlib.pyplot as plt
# Lib for creating chart and visuzlization
# Can take the results of our text analysis and turn them into graphs


tokens = ['the', 'rise', 'of', 'artificial', 'intelligence', 'has', 'led', 'to', 'significant', 'advancements', 'in', 'natural', 'language', 'processing', 'computer', 'vision', 'and', 'other', 'fields', 'machine', 'learning', 'algorithms', 'are', 'becoming', 'more', 'sophisticated', 'enabling', 'computers', 'to', 'perform', 'complex', 'tasks', 'that', 'were', 'once', 'thought', 'to', 'be', 'the', 'exclusive', 'domain', 'of', 'humans', 'with', 'the', 'advent', 'of', 'deep', 'learning', 'neural', 'networks', 'have', 'become', 'even', 'more', 'powerful', 'capable', 'of', 'processing', 'vast', 'amounts', 'of', 'data', 'and', 'learning', 'from', 'it', 'in', 'ways', 'that', 'were', 'not', 'possible', 'before', 'as', 'a', 'result', 'ai', 'is', 'increasingly', 'being', 'used', 'in', 'a', 'wide', 'range', 'of', 'industries', 'from', 'healthcare', 'to', 'finance', 'to', 'transportation', 'and', 'its', 'impact', 'is', 'only', 'set', 'to', 'grow', 'in', 'the', 'years', 'to', 'come']
print(tokens)

# unigrams = (pd.Series(nltk.ngrams(tokens, 1))).value_counts()
# bigrams = (pd.Series(nltk.ngrams(tokens, 2))).value_counts()
trigrams = (pd.Series(nltk.ngrams(tokens, 3))).value_counts()

print(trigrams)

# unigrams[:10].sort_values().plot.barh(color="lightsalmon", width=.9, figsize=(12,8))
# bigrams[:10].sort_values().plot.barh(color="lightsalmon", width=.9, figsize=(12,8))
trigrams[:10].sort_values().plot.barh(color="lightsalmon", width=.9, figsize=(12,8))

plt.title("10 most frequently occuring unigrams")

plt.show()