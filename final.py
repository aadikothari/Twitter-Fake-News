"""
Machine Learning ECE 4424/CS 4824 Python Script
Twitter Fake News Detection Algorithm
Created by: Aadi Kothari, Pradyuman Mehta, Devangini Talwar, Campbell Dalen
"""

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np

import pandas as pd
import string

import nltk
from nltk import tokenize
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time


# Start runtime counter
start_time = time.time()

"""
DATA PREPROCESSING: Cleaning the data for feeding to machine
Aadi Kothari
"""

# Read fake news
fake = pd.read_csv("archive/Fake.csv")
fake['target'] = 'fake'

# Read real news
real = pd.read_csv("archive/True.csv")
real['target'] = 'real'

# real + fake TOTAL data combined
# Alternative to deprecating df.append method
data = pd.concat([real, fake]).reset_index(drop=True)

# removing unwanted params
data.drop(["date"], axis=1)
data.drop(["subject"], axis=1)
data.drop(["title"], axis=1)

# Convert to LOWERCASE
data['text'] = data['text'].apply(lambda x: x.lower())

def punctuation_removal(text):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")
    return text


data['text'] = data['text'].apply(punctuation_removal)

stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in (stop)]))

token_space = tokenize.WhitespaceTokenizer()

# counter(data[data["target"] == "true"], "text", 20)

# Function to plot the confusion matrix (code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.show()


plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data.target, test_size=0.3, random_state=1)


def tokens(x):
    return x.split(',')


# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
                                         solver='lbfgs'))])

# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("Accuracy [Neural Network]: {}%".format(round(accuracy_score(y_test, prediction)*100, 2)))
print("F1 Score [Neural Network]: {}%".format(round(f1_score(y_test, prediction, pos_label='fake')*100, 2)))

# Decision Tree MODEL
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion='entropy',
                                                  max_depth=20,
                                                  splitter='best',
                                                  random_state=42))])

# cm = metrics.confusion_matrix(y_test, prediction)
# plot_confusion_matrix(cm, classes=['Fake', 'Real'])

# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("Accuracy [Decision Tree]: {}%".format(round(accuracy_score(y_test, prediction)*100, 2)))
print("F1 Score [Decision Tree]: {}%".format(round(f1_score(y_test, prediction, pos_label='fake')*100, 2)))

print("--- %s seconds ---" % (time.time() - start_time))
