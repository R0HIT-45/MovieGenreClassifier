import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load trained model and vectorizer
with open("movie_genre_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("🎬 Movie Genre Predictor")
print("Type 'exit' to quit.\n")

def is_valid_text(text):
    """Check if input is a valid movie description."""
    text = text.strip()
    # Must contain at least 3 alphabetic words
    words = re.findall(r"[a-zA-Z]+", text)
    return len(words) >= 3

while True:
    user_input = input("Enter movie description: ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not is_valid_text(user_input):
        print("❌ Invalid input. Please provide a proper movie description.\n")
        continue

    # Transform input
    X_input = vectorizer.transform([user_input])

    # Predict probabilities
    probs = model.predict_proba(X_input)[0]
    classes = model.classes_

    # Get top 3 predictions
    top_idx = np.argsort(probs)[::-1][:3]
    top_genres = [(classes[i], probs[i]) for i in top_idx]

    print("\nPredicted Genres (Top 3):")
    for genre, prob in top_genres:
        print(f" - {genre} ({prob*100:.2f}%)")
    print()
