import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("preprocessed_data.csv")  # If raw dataset, replace with your CSV path
# Example: df = pd.read_csv("Genre Classification Dataset/train_data.txt", sep="\t", names=["ID", "Genre", "Description"])

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
