import pandas as pd
import spacy
import re
import string
import numpy as np
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm

# https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# loading the data from folder
data = "archive/Tweets.csv"
data_read = pd.read_csv(data)


tweets = data_read.iloc[:, 10].values
sentiments = data_read.iloc[:, 1].values
tweets = tweets[0:3000]
sentiments = sentiments[0:3000]

stop_words = stopwords.words("english")

def preprocess(tweet):
    # lowercase
    tweet.lower()


    tweet = re.sub(r"[0-9]", '', tweet)
    tweet = re.sub(r'#\S+', '', tweet)
    tweet = re.sub(r'\S@\S+', '', tweet)
    tweet = re.sub(r'\S+com', '', tweet)
    table = tweet.maketrans("", "", string.punctuation)
    tweet2 = tweet.translate(table)
    tokens = word_tokenize(tweet2)
    filtered_tokens = []
    for item in tokens:
        if item not in stop_words:
            filtered_tokens.append(item)
    stemmer = PorterStemmer()
    stemmed_words = []
    for item in filtered_tokens:
        stemmed_words.append(stemmer.stem(item))
    return " ".join(filtered_tokens)

def vectorizer(tweet):
    tweet = tweet
    vectorized_data = TfidfVectorizer()
    vectorized_data = vectorized_data.fit_transform(tweet).toarray()

    return vectorized_data


pre_processed_data = []

for item in tweets:
    tweet = preprocess(item)
    pre_processed_data.append(tweet)

X = vectorizer(pre_processed_data)

y = []
for item in sentiments:
    if item == "positive":
        y.append(1)
    elif item == "negative":
        y.append(-1)
    else:
        y.append(0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = svm.SVC()
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))