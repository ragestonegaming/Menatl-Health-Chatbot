from flask import Flask, render_template, request, jsonify
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

app = Flask(__name__)

# Load intents.json
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

lemmer = WordNetLemmatizer()

# Tokenize and lemmatize user input for pattern matching
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Normalize input text
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower()))

# Find intent matching the user's input
def get_response(user_input):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if user_input in pattern:
                return random.choice(intent['responses'])
    return "I am sorry, I don't understand that."

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_response = request.args.get('msg').lower()
    return get_response(user_response)

if __name__ == "__main__":
    app.run(debug=True)
