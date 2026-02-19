"""
AI Chatbot Training Script - Week 3: ShadowFox AIML Internship
Advanced Level Project

This script:
1. Loads intents.json knowledge base
2. Performs tokenization and lemmatization using NLTK
3. Creates Bag of Words representation
4. Builds and trains a Neural Network
5. Saves model and preprocessing data

Author: ShadowFox AIML Intern
Date: February 2026
"""

import os
import json
import pickle
import random
import numpy as np
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

print("[INFO] Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, 'intents.json')
MODEL_PATH = os.path.join(BASE_DIR, 'chatbot_model.h5')
WORDS_PATH = os.path.join(BASE_DIR, 'words.pkl')
CLASSES_PATH = os.path.join(BASE_DIR, 'classes.pkl')

EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NESTEROV = True

lemmatizer = WordNetLemmatizer()


def load_intents(filepath):
    """Load intents from JSON file."""
    print("\n" + "=" * 70)
    print("STEP 1: LOADING INTENTS DATA")
    print("=" * 70)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        intents = json.load(f)
    
    print(f"\n[INFO] Loaded intents from: {filepath}")
    print(f"[INFO] Number of intents: {len(intents['intents'])}")
    
    for intent in intents['intents']:
        print(f"       - {intent['tag']}: {len(intent['patterns'])} patterns, {len(intent['responses'])} responses")
    
    return intents


def preprocess_data(intents):
    """
    Preprocess intents data:
    - Tokenize patterns
    - Lemmatize words
    - Create vocabulary and classes
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)
    
    words = []
    classes = []
    documents = []
    ignore_chars = ['?', '!', '.', ',', "'", '"']
    
    print("\n[INFO] Tokenizing and lemmatizing patterns...")
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = word_tokenize(pattern.lower())
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
    words = sorted(set(words))
    
    classes = sorted(set(classes))
    
    print(f"\n[INFO] Preprocessing Results:")
    print(f"       - Total unique words (vocabulary): {len(words)}")
    print(f"       - Total classes (intents): {len(classes)}")
    print(f"       - Total documents (patterns): {len(documents)}")
    
    print(f"\n[INFO] Classes: {classes}")
    print(f"\n[INFO] Sample vocabulary (first 20 words): {words[:20]}")
    
    return words, classes, documents


def create_training_data(words, classes, documents):
    """
    Create Bag of Words (X) and One-Hot encoded labels (Y).
    """
    print("\n" + "=" * 70)
    print("STEP 3: CREATING TRAINING DATA")
    print("=" * 70)
    
    training = []
    output_empty = [0] * len(classes)
    
    print("\n[INFO] Creating Bag of Words representation...")
    
    for document in documents:
        bag = []
        word_patterns = document[0]
        
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        
        training.append([bag, output_row])
    
    random.shuffle(training)
    training = np.array(training, dtype=object)
    
    X_train = np.array(list(training[:, 0]))
    Y_train = np.array(list(training[:, 1]))
    
    print(f"\n[INFO] Training Data Shape:")
    print(f"       - X_train (Bag of Words): {X_train.shape}")
    print(f"       - Y_train (One-Hot Labels): {Y_train.shape}")
    
    return X_train, Y_train


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(input_size, output_size):
    """
    Build Sequential Neural Network.
    Architecture: Input -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(64, ReLU) -> Dropout(0.5) -> Output(Softmax)
    """
    print("\n" + "=" * 70)
    print("STEP 4: BUILDING NEURAL NETWORK")
    print("=" * 70)
    
    model = Sequential([
        Dense(128, input_shape=(input_size,), activation='relu', name='hidden_layer_1'),
        Dropout(0.5, name='dropout_1'),
        Dense(64, activation='relu', name='hidden_layer_2'),
        Dropout(0.5, name='dropout_2'),
        Dense(output_size, activation='softmax', name='output_layer')
    ])
    
    sgd = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=NESTEROV)
    
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n[INFO] Model Architecture:")
    print(f"       - Input Layer: {input_size} neurons (vocabulary size)")
    print(f"       - Hidden Layer 1: 128 neurons (ReLU) + Dropout(0.5)")
    print(f"       - Hidden Layer 2: 64 neurons (ReLU) + Dropout(0.5)")
    print(f"       - Output Layer: {output_size} neurons (Softmax)")
    
    print(f"\n[INFO] Optimizer: SGD (lr={LEARNING_RATE}, momentum={MOMENTUM}, nesterov={NESTEROV})")
    print(f"[INFO] Loss: categorical_crossentropy")
    
    print("\n[INFO] Model Summary:")
    model.summary()
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, Y_train):
    """Train the neural network."""
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING THE MODEL")
    print("=" * 70)
    
    print(f"\n[INFO] Training Configuration:")
    print(f"       - Epochs: {EPOCHS}")
    print(f"       - Batch Size: {BATCH_SIZE}")
    print(f"       - Training Samples: {len(X_train)}")
    
    print("\n[INFO] Starting training...")
    print("-" * 70)
    
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    print("-" * 70)
    print("\n[INFO] Training completed!")
    
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    print(f"\n[RESULTS] Final Training Loss: {final_loss:.4f}")
    print(f"[RESULTS] Final Training Accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
    
    return history


# ============================================================================
# SAVING
# ============================================================================

def save_model_and_data(model, words, classes):
    """Save model and preprocessing data."""
    print("\n" + "=" * 70)
    print("STEP 6: SAVING MODEL AND DATA")
    print("=" * 70)
    
    model.save(MODEL_PATH)
    print(f"\n[INFO] Model saved to: {MODEL_PATH}")
    
    with open(WORDS_PATH, 'wb') as f:
        pickle.dump(words, f)
    print(f"[INFO] Words saved to: {WORDS_PATH}")
    
    with open(CLASSES_PATH, 'wb') as f:
        pickle.dump(classes, f)
    print(f"[INFO] Classes saved to: {CLASSES_PATH}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to train the chatbot."""
    print("\n"+ "=" * 70)
    print("AI CHATBOT TRAINING - WEEK 3 SHADOWFOX INTERNSHIP")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    intents = load_intents(INTENTS_PATH)
    
    words, classes, documents = preprocess_data(intents)
    
    X_train, Y_train = create_training_data(words, classes, documents)
    
    model = build_model(len(X_train[0]), len(Y_train[0]))
    
    history = train_model(model, X_train, Y_train)
    
    save_model_and_data(model, words, classes)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\n[OUTPUT FILES]")
    print(f"   - Model: {MODEL_PATH}")
    print(f"   - Words: {WORDS_PATH}")
    print(f"   - Classes: {CLASSES_PATH}")
    print(f"\n[NEXT STEP] Run 'python chat_gui.py' to start the chatbot!")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
