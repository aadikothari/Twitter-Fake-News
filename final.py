"""
Machine Learning ECE 4424/CS 4824 Python Script
Twitter Fake News Detection Algorithm
Created by: Aadi Kothari, Pradyuman Mehta, Devangini Talwar, Campbell Dalen
"""

# Imports for dependencies and libraries
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

import pandas as pd
import string

import nltk
# Need it only ONE-TIME. Remove after first use.
nltk.download('stopwords')
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

# removing unwanted params from dataset
data.drop(["date"], axis=1)
data.drop(["subject"], axis=1)
data.drop(["title"], axis=1)

# Convert to LOWERCASE
data['text'] = data['text'].apply(lambda x: x.lower())

# Remove unwanted puncutations
def punctuation_removal(text):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for ele in text:
        if ele in punc:
            text = text.replace(ele, "")
    return text


# Applying that to the current data
data['text'] = data['text'].apply(punctuation_removal)

"""
DATA EXPLORATION/PROCESSING
"""

# Simple NLP stopwords functionality to remove filler words278
# Increases accuracy and reduces time
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
X_train, X_test, y_train, y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=1)

"""
MODELS and EVALUATION METRICS
"""

# NEURAL NETWORK MODEL
base = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', MLPClassifier(alpha=1e-09, hidden_layer_sizes=(5, 5), random_state=1, solver='adam'))])
model = base.fit(X_train, y_train)
prediction = model.predict(X_test)

# ACCURACY
print("Accuracy [Neural Network]: {}%".format(
    round(accuracy_score(y_test, prediction)*100, 2)))
# F1 SCORE
print("F1 Score [Neural Network]: {}%".format(
    round(f1_score(y_test, prediction, pos_label='fake')*100, 2)))

# (PLACE CONFUSION MATRIX HERE TO CHECK FOR NEURAL NETWORK MODEL)

# CONFUSION MATRIX (uncomment if needed)
mat = confusion_matrix(y_test, prediction)
plt.figure(figsize=(3, 3))
sns.heatmap(mat, annot=True, fmt='d', cmap="gray", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Decision Tree MODEL
base = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', DecisionTreeClassifier(criterion='gini', max_depth=15, splitter='random', random_state=80))])
model = base.fit(X_train, y_train)
prediction = model.predict(X_test)

# ACCURACY
print("Accuracy [Decision Tree]: {}%".format(
    round(accuracy_score(y_test, prediction)*100, 2)))
#F1 SCORE
print("F1 Score [Decision Tree]: {}%".format(
    round(f1_score(y_test, prediction, pos_label='fake')*100, 2)))

# RUNTIME Calculation
print("--- RUNTIME: %s seconds ---" % (time.time() - start_time))


