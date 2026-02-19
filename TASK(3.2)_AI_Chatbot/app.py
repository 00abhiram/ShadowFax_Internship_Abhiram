import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
import nltk
# --- ADD THESE LINES FOR DEPLOYMENT ---
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
# --------------------------------------
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# --- Part 1: Load Resources (Cached for Speed) ---
@st.cache_resource
def load_resources():
    # Load the trained model and pickle files
    try:
        model = load_model('chatbot_model.h5')
        intents = json.loads(open('intents.json').read())
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
        return model, intents, words, classes
    except FileNotFoundError:
        return None, None, None, None

model, intents, words, classes = load_resources()

if model is None:
    st.error("Error: Could not find model files. Please run 'train_bot.py' first!")
    st.stop()

# --- Part 2: Helper Functions for Prediction ---

def clean_up_sentence(sentence):
    # Tokenize and lemmatize user input
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    # Convert sentence into a bag of words array (0s and 1s)
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # Predict the intent of the sentence
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    # Pick a random response from the predicted intent
    if not intents_list:
        return "I'm sorry, I don't understand that."
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# --- Part 3: The Streamlit UI ---

st.title("🤖 ShadowFox AI Chatbot")
st.markdown("Ask me anything about your internship or AI!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # 1. Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get Bot Response
    ints = predict_class(prompt)
    response = get_response(ints, intents)

    # 3. Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})