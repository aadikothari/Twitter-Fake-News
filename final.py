from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords
import pandas as pd
from sklearn.utils import shuffle
import string
import nltk
from nltk import tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

fake = pd.read_csv("archive/Fake.csv")
true = pd.read_csv("archive/True.csv")

fake['target'] = 'fake'
true['target'] = 'true'

data = pd.concat([fake, true]).reset_index(drop=True)

data = data.reset_index(drop=True)

data.drop(["date"], axis=1, inplace=True)
data.drop(["title"], axis=1, inplace=True)

data['text'] = data['text'].apply(lambda x: x.lower())


def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


data['text'] = data['text'].apply(punctuation_removal)

nltk.download('stopwords')
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in (stop)]))

# print(data.groupby(['subject'])['text'].count())
# data.groupby(['subject'])['text'].count().plot(kind="bar")
# plt.show()

# print(data.groupby(['target'])['text'].count())
# data.groupby(['target'])['text'].count().plot(kind="bar")
# plt.show()

token_space = tokenize.WhitespaceTokenizer()


def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                 "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color='blue')
    ax.set(ylabel="Count")
    plt.xticks(rotation='vertical')
    plt.show()

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


plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data.target, test_size=0.3, random_state=1)


def tokens(x):
    return x.split(',')

# # Vectorizing and applying TF-IDF
# pipe = Pipeline([('vect', CountVectorizer()),
#                  ('tfidf', TfidfTransformer()),
#                  ('model', MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
#                                          solver='lbfgs'))])

# # Fitting the model
# model = pipe.fit(X_train, y_train)
# # Accuracy
# prediction = model.predict(X_test)
# print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100, 2)))


# Decision Tree MODEL
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion='entropy',
                                                  max_depth=20,
                                                  splitter='best',
                                                  random_state=42))])

# MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1, solver='lbfgs')

# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100, 2)))
