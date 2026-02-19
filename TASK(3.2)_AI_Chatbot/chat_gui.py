"""
AI Chatbot GUI Interface - Week 3: ShadowFox AIML Internship
Advanced Level Project

This script:
1. Loads the trained model and preprocessing data
2. Provides a Tkinter GUI for chatting
3. Predicts intent and generates responses

Author: ShadowFox AIML Intern
Date: February 2026
"""

import os
import json
import pickle
import random
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    from colorama import init, Fore, Style
    init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, 'intents.json')
MODEL_PATH = os.path.join(BASE_DIR, 'chatbot_model.h5')
WORDS_PATH = os.path.join(BASE_DIR, 'words.pkl')
CLASSES_PATH = os.path.join(BASE_DIR, 'classes.pkl')

ERROR_THRESHOLD = 0.25

lemmatizer = WordNetLemmatizer()


def load_chatbot_data():
    """Load the trained model and preprocessing data."""
    print("[INFO] Loading chatbot model and data...")
    
    with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
        intents = json.load(f)
    print(f"[INFO] Loaded intents from: {INTENTS_PATH}")
    
    with open(WORDS_PATH, 'rb') as f:
        words = pickle.load(f)
    print(f"[INFO] Loaded words vocabulary: {len(words)} words")
    
    with open(CLASSES_PATH, 'rb') as f:
        classes = pickle.load(f)
    print(f"[INFO] Loaded classes: {classes}")
    
    model = load_model(MODEL_PATH)
    print(f"[INFO] Loaded model from: {MODEL_PATH}")
    
    return intents, words, classes, model


def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = word_tokenize(sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    """Convert sentence to bag of words array."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for s_word in sentence_words:
        for i, word in enumerate(words):
            if word == s_word:
                bag[i] = 1
    
    return np.array(bag)


def predict_class(sentence, model, words, classes):
    """Predict the intent class for a given sentence."""
    bow = bag_of_words(sentence, words)
    
    res = model.predict(np.array([bow]), verbose=0)[0]
    
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({
            'intent': classes[r[0]],
            'probability': str(r[1])
        })
    
    return return_list


def get_response(intents_list, intents_json):
    """Get a random response for the predicted intent."""
    if not intents_list:
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    
    tag = intents_list[0]['intent']
    probability = float(intents_list[0]['probability'])
    
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
    
    return "I'm not sure how to respond to that. Can you try asking something else?"


def chatbot_response(message, model, words, classes, intents):
    """Generate a chatbot response for the user message."""
    intents_list = predict_class(message, model, words, classes)
    response = get_response(intents_list, intents)
    
    intent_info = intents_list[0] if intents_list else {'intent': 'unknown', 'probability': '0'}
    
    return response, intent_info


class ChatbotGUI:
    """Tkinter GUI for the AI Chatbot."""
    
    def __init__(self, root, intents, words, classes, model):
        self.root = root
        self.intents = intents
        self.words = words
        self.classes = classes
        self.model = model
        
        self.setup_gui()
        self.display_welcome_message()
    
    def setup_gui(self):
        """Setup the GUI components."""
        self.root.title("ShadowFox AI Chatbot - Week 3")
        self.root.geometry("600x700")
        self.root.resizable(True, True)
        self.root.configure(bg='#2C3E50')
        
        header_frame = tk.Frame(self.root, bg='#1ABC9C', height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text="🤖 ShadowFox AI Assistant",
            font=('Helvetica', 18, 'bold'),
            bg='#1ABC9C',
            fg='white'
        )
        header_label.pack(pady=15)
        
        chat_frame = tk.Frame(self.root, bg='#2C3E50')
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_log = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Helvetica', 11),
            bg='#ECF0F1',
            fg='#2C3E50',
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.chat_log.pack(fill=tk.BOTH, expand=True)
        
        self.chat_log.tag_configure('user', foreground='#2980B9', font=('Helvetica', 11, 'bold'))
        self.chat_log.tag_configure('bot', foreground='#27AE60', font=('Helvetica', 11, 'bold'))
        self.chat_log.tag_configure('user_msg', foreground='#2C3E50')
        self.chat_log.tag_configure('bot_msg', foreground='#2C3E50')
        self.chat_log.tag_configure('info', foreground='#7F8C8D', font=('Helvetica', 9, 'italic'))
        
        input_frame = tk.Frame(self.root, bg='#2C3E50')
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.message_entry = tk.Entry(
            input_frame,
            font=('Helvetica', 12),
            bg='#ECF0F1',
            fg='#2C3E50',
            relief=tk.FLAT,
            insertbackground='#2C3E50'
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 10))
        self.message_entry.bind('<Return>', self.send_message)
        self.message_entry.focus_set()
        
        self.send_button = tk.Button(
            input_frame,
            text="Send 📤",
            font=('Helvetica', 12, 'bold'),
            bg='#1ABC9C',
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            command=self.send_message,
            padx=20,
            pady=8
        )
        self.send_button.pack(side=tk.RIGHT)
        
        footer_frame = tk.Frame(self.root, bg='#34495E', height=30)
        footer_frame.pack(fill=tk.X)
        footer_frame.pack_propagate(False)
        
        footer_label = tk.Label(
            footer_frame,
            text="ShadowFox AIML Internship - Week 3 Project",
            font=('Helvetica', 9),
            bg='#34495E',
            fg='#BDC3C7'
        )
        footer_label.pack(pady=5)
    
    def display_welcome_message(self):
        """Display welcome message when GUI starts."""
        welcome_msg = (
            "Hello! I'm the ShadowFox AI Assistant.\n"
            "I can help you with questions about:\n"
            "• ShadowFox organization\n"
            "• Internship duration and structure\n"
            "• Skills you'll learn\n"
            "• Projects you'll build\n\n"
            "Type your message below and press Enter or click Send!\n"
        )
        self.insert_message("Bot", welcome_msg, is_bot=True)
    
    def insert_message(self, sender, message, is_bot=False, intent_info=None):
        """Insert a message into the chat log."""
        self.chat_log.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M")
        
        if is_bot:
            self.chat_log.insert(tk.END, f"🤖 Bot ", 'bot')
            self.chat_log.insert(tk.END, f"[{timestamp}]\n", 'info')
            self.chat_log.insert(tk.END, f"{message}\n\n", 'bot_msg')
        else:
            self.chat_log.insert(tk.END, f"👤 You ", 'user')
            self.chat_log.insert(tk.END, f"[{timestamp}]\n", 'info')
            self.chat_log.insert(tk.END, f"{message}\n", 'user_msg')
            if intent_info:
                self.chat_log.insert(tk.END, f"[Intent: {intent_info['intent']} | Confidence: {float(intent_info['probability'])*100:.1f}%]\n\n", 'info')
        
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.see(tk.END)
    
    def send_message(self, event=None):
        """Handle sending a message."""
        message = self.message_entry.get().strip()
        
        if not message:
            return
        
        self.message_entry.delete(0, tk.END)
        
        response, intent_info = chatbot_response(
            message, self.model, self.words, self.classes, self.intents
        )
        
        self.insert_message("You", message, is_bot=False, intent_info=intent_info)
        
        self.insert_message("Bot", response, is_bot=True)
        
        if COLORAMA_AVAILABLE:
            print(f"{Fore.CYAN}You: {Style.RESET_ALL}{message}")
            print(f"{Fore.GREEN}Bot: {Style.RESET_ALL}{response}")
            print(f"{Fore.YELLOW}[Intent: {intent_info['intent']} | Confidence: {float(intent_info['probability'])*100:.1f}%]{Style.RESET_ALL}\n")
        else:
            print(f"You: {message}")
            print(f"Bot: {response}")
            print(f"[Intent: {intent_info['intent']} | Confidence: {float(intent_info['probability'])*100:.1f}%]\n")


def main():
    """Main function to run the chatbot GUI."""
    print("\n" + "=" * 70)
    print("AI CHATBOT GUI - WEEK 3 SHADOWFOX INTERNSHIP")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    required_files = [INTENTS_PATH, MODEL_PATH, WORDS_PATH, CLASSES_PATH]
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"\n[ERROR] Required file not found: {filepath}")
            print("[INFO] Please run 'python train_bot.py' first to train the model!")
            messagebox.showerror(
                "Model Not Found",
                f"Required file not found:\n{filepath}\n\nPlease run 'python train_bot.py' first!"
            )
            return
    
    intents, words, classes, model = load_chatbot_data()
    
    print("\n[INFO] Starting GUI...")
    print("[INFO] Chat messages will also appear in this terminal.\n")
    print("=" * 70)
    
    root = tk.Tk()
    app = ChatbotGUI(root, intents, words, classes, model)
    root.mainloop()
    
    print("\n[INFO] Chatbot closed.")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()