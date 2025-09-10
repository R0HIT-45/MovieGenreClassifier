import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings

# Suppress warnings for ill-defined metrics
warnings.filterwarnings("ignore")

# Load preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")

# Features and labels
X = df["Clean_Description"]
y = df["Genre"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_vect = vectorizer.fit_transform(X)

# Train-test split with stratification to handle imbalanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Logistic Regression with balanced class weight
model = LogisticRegression(max_iter=1000, class_weight="balanced")

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer safely
with open("movie_genre_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved successfully!")
