import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from pathlib import Path

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Path to CSV
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)
print("Dataset loaded from:", CSV_PATH)
print("Dataset loaded.")
print("Shape:", df.shape)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean text: lowercase, remove punctuation, numbers, stopwords, and lemmatize."""
    
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return " ".join(cleaned_tokens)

# Apply preprocessing to dataset
df["Clean_Description"] = df["Description"].apply(preprocess_text)

print("Sample original text:\n", df["Description"].iloc[0])
print("\nSample cleaned text:\n", df["Clean_Description"].iloc[0])

# Save cleaned dataset
df.to_csv("preprocessed_data.csv", index=False)
print("\n✅ Preprocessed data saved as 'preprocessed_data.csv'.")
