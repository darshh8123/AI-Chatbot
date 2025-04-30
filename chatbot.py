import json
import random
import pickle
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

def clean_text(text):
    tokens = text.lower().split()
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(lemmas)

def get_response(user_input):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    tag = model.predict(vect)[0]

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, I don't understand."

print("Chatbot: Hello! Type 'quit' to exit.")
while True:
    msg = input("You: ")
    if msg.lower() == "quit":
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", get_response(msg))
