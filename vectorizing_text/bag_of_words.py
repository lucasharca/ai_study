# Bag of words model example
import pandas as pd
# sklearn is a powerful machine learning lib in Python, CountVectorizer is one of its tools
# Trasnforms a collection of text documents into a matrix of tokens counts
# breaks the text into words (tokens) and counts how often each word appears
# this gives us the numerical representation we need to apply machine learning techniques to text data

from sklearn.feature_extraction.text import CountVectorizer

data = [' Most shark attacks occur about 10 feet from the beach since that is where the people are',
        'the efficiency with which he paired the socks in the drawer was quite admirable',
        'carol drank the blood as if she were a vampire',
        'giving directions that the mountains are to the west only works when you can see them',
        'the sign said there was road work ahead so he decided to speed up',
        'the gruff old man sat in the back of the bait shop grumbling to himself as he scooped out a handful of worms']

# 1 - initialize the countvectorizer class
countvec = CountVectorizer()

# 2 - apply countvec to our text data
#   - Fit means that countvectorizer looks through the text and learns which unique words appear in the data
#   - Transform then converts the text into numbers by creating a matrix that counts how often each words occurs
#   - By using fit_transform() we combine these two steps in one line of code

countvec_fit = countvec.fit_transform(data)

bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns=countvec.get_feature_names_out())
print(bag_of_words)

