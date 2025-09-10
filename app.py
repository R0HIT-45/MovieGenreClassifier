import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained model and vectorizer
with open("movie_genre_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# NLTK setup
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and lemmatize
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
        # Check if input is valid
        if len(processed) < 3:
            st.error("Invalid or insufficient description. Please enter a valid movie description.")
        else:
            vect = vectorizer.transform([processed])
            prediction = model.predict(vect)[0]
            st.success(f"Predicted Genre: **{prediction}**")
