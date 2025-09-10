from pathlib import Path
import pickle
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent  # This is MovieGenreClassifier/

MODEL_PATH = BASE_DIR / "models" / "movie_genre_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# Load trained model and vectorizer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# NLTK setup
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("🎬 Movie Genre Classifier")
st.write("Enter a movie description and get the predicted genre!")

description = st.text_area("Movie Description:")

if st.button("Predict Genre"):
    if description.strip() == "":
        st.warning("Please enter a movie description!")
    else:
        processed = preprocess_text(description)
        if len(processed.split()) < 3:
            st.error("Invalid or insufficient description. Please enter a valid movie description.")
        else:
            vect = vectorizer.transform([processed])
            # Predict probabilities
            probs = model.predict_proba(vect)[0]
            classes = model.classes_
            # Top 3 predictions
            top_idx = probs.argsort()[::-1][:3]
            top_genres = [(classes[i], probs[i]) for i in top_idx]

            st.success("Predicted Genres (Top 3):")
            for genre, prob in top_genres:
                st.write(f"**{genre}**: {prob*100:.2f}%")
