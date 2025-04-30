import json
import random
import pickle
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load data
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

corpus = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = pattern.lower().split()  # replaced word_tokenize to avoid punkt_tab error
        lemmas = [lemmatizer.lemmatize(w) for w in tokens]
        corpus.append(' '.join(lemmas))
        labels.append(intent['tag'])

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Train model
clf = MultinomialNB()
clf.fit(X, labels)

# Save model and vectorizer
pickle.dump(clf, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

