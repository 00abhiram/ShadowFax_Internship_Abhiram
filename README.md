# 🦊 ShadowFox AI/ML Internship Portfolio

**Author:** Abhi Ram Reddy K.  
**Institution:** Pallavi Engineering College  
**Duration:** February 2026

Welcome to my ShadowFox internship repository! This repo contains the complete, end-to-end Machine Learning projects I built during my 4-week AI/ML virtual internship. 

Going into this, my goal was to move past basic tutorials and experience the real-world grit of software engineering. This journey taught me that training models is only half the job; the real engineering happens when you are debugging environment conflicts, wrestling with deployment servers, and turning raw code into interactive, user-facing applications.

Here is a breakdown of the projects I built, the challenges I faced, and what I learned along the way.

---

## 🖼️ Week 1: Image Tagging (Deep Learning & Computer Vision)
**Objective:** Build a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.

* **What I Built:** A Deep Learning model utilizing TensorFlow/Keras. I implemented Data Augmentation (rotation, zooming, flipping) to hit the "Expert Level" criteria and prevent model overfitting.
* **The Real Challenge:** Before writing a single line of training code, my environment completely broke. My global Python version (3.14) was incompatible with TensorFlow, and VS Code kept enforcing broken virtual environments. I had to learn how to downgrade to Python 3.12.9, manage system PATH variables, and force IDE interpreter switches.
* **Outcome:** Seeing the CNN successfully train for 15 epochs and generate clean accuracy/loss plots was the ultimate relief after hours of environment debugging.

## 🚗 Week 2: Car Price Predictor (Regression & Feature Engineering)
**Objective:** Develop a regression model to estimate the selling price of used vehicles.

* **What I Built:** A Random Forest Regressor trained on historical car data. I performed extensive feature engineering—calculating the actual age of the vehicles and writing manual one-hot encoding logic for categorical data like fuel and transmission types.
* **The Win:** The model achieved an astonishing **R-squared score of 0.96**. Plotting the "Actual vs. Predicted" scatter graph showed a beautifully tight diagonal line. I then wrapped this model into a local Streamlit web application so users could input car details and get live predictions.

## 🤖 Week 3: Context-Aware AI Chatbot (NLP & Cloud Deployment)
**Objective:** Build an intelligent chatbot that understands user intents, going beyond simple string-matching.

* **What I Built:** A Deep Learning chatbot using a Sequential Neural Network, NLTK for tokenization and lemmatization, and a custom JSON knowledge base. 
* **The Real Challenge:** I wanted to deploy this app live on Streamlit Community Cloud. While it ran perfectly on my Asus laptop, the cloud server instantly crashed because it lacked my local NLTK dictionary corpuses (like 'punkt' and 'wordnet'). I had to rewrite my `app.py` script to force the server to download these dependencies on startup and rigorously configure my `requirements.txt` file.
* **Outcome:** Successfully deploying the chatbot live to the web. Sharing the URL and watching people interact with an AI I built from scratch was the most rewarding moment of the internship.

---

## 🛠️ Tech Stack & Tools Learned
* **Languages:** Python
* **Machine Learning:** TensorFlow, Keras, Scikit-Learn
* **NLP:** NLTK
* **Data Processing:** Pandas, NumPy
* **Deployment & UI:** Streamlit, Streamlit Community Cloud
* **Version Control:** Git, GitHub

## 📈 Final Thoughts
This internship bridged the massive gap between college theory and industry reality. I am walking away with a deep appreciation for the full-stack ML lifecycle—from data preprocessing to cloud deployment—and the resilience required to debug complex dependency issues.

Thank you to the ShadowFox team for pushing me to build these end-to-end projects!

---
*Feel free to explore the code in the folders above!*
