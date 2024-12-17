import streamlit as st
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import re

# Preprocessing function (applies to both models)
def preprocess_text(text):
    """
    Basic text preprocessing to clean the input text for sentiment analysis.
    This includes:
    - Lowercasing the text
    - Removing special characters and digits
    - Removing extra whitespace
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove anything that is not a letter
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text

# BERT-based sentiment analysis pipeline setup
@st.cache_resource
def load_bert_pipeline():
    """Load and cache the BERT pipeline for sentiment analysis."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to make sentiment predictions using BERT
def predict_sentiment_bert(text):
    bert_pipeline = load_bert_pipeline()
    result = bert_pipeline(text)
    sentiment = result[0]['label']
    return sentiment

# Naive Bayes model setup
@st.cache_resource
def load_naive_bayes_model():
    """Load and cache the Naive Bayes model."""
    with open("naive_bayes_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer
# Function to make sentiment predictions using Naive Bayes
def predict_sentiment_naive_bayes(text):
    model, vectorizer = load_naive_bayes_model()
    # Preprocess and vectorize the input text
    text = preprocess_text(text)
    vectorized_text = vectorizer.transform([text])
    sentiment = model.predict(vectorized_text)
    return sentiment[0]

# Streamlit app
st.title("Sentiment Analysis with BERT and Naive Bayes")

# Option to select model
model_choice = st.radio("Select Model:", ("BERT (DistilBERT)", "Naive Bayes"))

# User input for sentiment analysis
user_input = st.text_area("Enter text for sentiment analysis:")

# Show appropriate result based on model choice
if user_input:
    if model_choice == "BERT (DistilBERT)":
        sentiment = predict_sentiment_bert(user_input)
        st.write(f"Sentiment (BERT): {sentiment}")
    elif model_choice == "Naive Bayes":
        sentiment = predict_sentiment_nb(user_input)
        st.write(f"Sentiment (Naive Bayes): {sentiment}")
