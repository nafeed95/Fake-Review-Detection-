import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def load_model_and_tools():
    try:
        model = tf.keras.models.load_model("app/model/best_cnn_lstm_model.h5")
        with open("app/model/tokenizer.pkl", "rb") as handle:
            tokenizer = pickle.load(handle)
        print("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print("Failed to load model/tokenizer:", str(e))
        return None, None

def predict_review(review_text, category, rating, model, tokenizer):
    combined = f"{category} {rating} {review_text}"
    cleaned = clean_text(combined)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=256, padding='post')

    prediction = model.predict(padded)[0][0]

    label = "fake" if prediction > 0.5 else "genuine"
    return label, float(prediction)
