import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# For this exercise we'll use a sample dataset of sentences paired with their sentiment labels
# Our goal: to build an algorithhm that can learn from this data and then classify new sentences by their sentiment

data = pd.DataFrame(
    [("i love spending time with my friends and family", "positive"),
    ("that was the best meal i've ever had in my life", "positive"),
    ("i feel so grateful for everything i have in my life", "positive"),
    ("i received a promotion at work and i couldn't be happier", "positive"),
    ("watching a beautiful sunset always fills me with joy", "positive"),
    ("my partner surprised me with a thoughtful gift and it made my day", "positive"),
    ("i am so proud of my daughter for graduating with honors", "positive"),
    ("listening to my favorite music always puts me in a good mood", "positive"),
    ("i love the feeling of accomplishment after completing a challenging task", "positive"),
    ("i am excited to go on vacation next week", "positive"),
    ("i feel so overwhelmed with work and responsibilities", "negative"),
    ("the traffic during my commute is always so frustrating", "negative"),
    ("i received a parking ticket and it ruined my day", "negative"),
    ("i got into an argument with my partner and we're not speaking", "negative"),
    ("i have a headache and i feel terrible", "negative"),
    ("i received a rejection letter for the job i really wanted", "negative"),
    ("my car broke down and it's going to be expensive to fix", "negative"),
    ("i'm feeling sad because i miss my friends who live far away", "negative"),
    ("i'm frustrated because i can't seem to make progress on my project", "negative"),
    ("i'm disappointed because my team lost the game", "negative")],
    columns=['text', 'sentiment'])

# Now we should shuffle the Dataset 
# to make sure we have a good mix of positive and negative sentences instead of having them grouped

data = data.sample(frac=1).reset_index(drop=True)

# Prepare the inputs for our algorithm
X = data['text']
y = data['sentiment']

# Text vectorization: Converting our text into numbers so that the machine learning model can work with them
# Here we'll use the bag of words approach with sklearn CountVectorizer()
#   Each Word = column; 
#   Each Sentence = row; 
#   Value = how many times each word appears in a sentence

countvec = CountVectorizer()

countvec_fit = countvec.fit_transform(X)
bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns=countvec.get_feature_names_out())

# Next: Split the data into training sets and testing sets
# training set: Data using for learning
# testing set: Data used for evaluation. 
#   It helps us check how well the model deals with data it hasn't seen before
# When we split the data we separate into 
#   features: The input data (X) -> (in our case) The bag of words sentences
#   Labels: The output data (y) -> The sentiment for each sentences (positive / negative)
# 
# Here we'll create for variables
#   X_train: -> training features, or sentences in bag of words
#   X_test: -> testing features
#   y_train: -> training labels, or sentiment values
#   y_test: -> testing labels
# We set test size equals 0.3 -> 30% of the data is used for testing and 70% for training
# random_state() makes sure we have the same train and test split each time we run the code
#
# With this setup, X_train and y_train will be used to train the logistic regression model
# while X_test and y_test will help us evaluate its accuracy 

X_train, X_test, y_train, y_test = train_test_split(bag_of_words, y, test_size=0.3, random_state=7)

# Now we train our logistic regression model
# The fir method takes in our training data and teaches the model to find patterns
lr = LogisticRegression(random_state=1).fit(X_train, y_train)

# Next, we want to see how well our model does in data it hasn't seen before
# This is where the testing comes in
# We use the predict() method on Xtest to generate predictions

y_pred_lr = lr.predict(X_test)


# The first metric we'll measure is accuracy: percentage of sentences the model predicted correctly
# To calculate this, we compare the true labels with predicted labels

result_acc_score = accuracy_score(y_pred_lr, y_test)
print("accuracy: ", result_acc_score)

# Now we'll print classification_report(): More detailed information on the performance
# The report includes some additional metrics:
#   - Precision: Out of all sentences, what proportion were actually correct
#   - Recall: Out of all sentences, what proportion did the model correctly find
#   -  f1-Score: A single number from 0 - 1, that combines precision and recall
#       - f1 score of 0.29 means the balance between precision and recall is poor
#       
class_report = classification_report(y_test, y_pred_lr, zero_division=0)
print(class_report)

# print(bag_of_words)