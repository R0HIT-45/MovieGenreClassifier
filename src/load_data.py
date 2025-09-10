import pandas as pd
import os

# Path to downloaded dataset folder
dataset_folder = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "hijest", "genre-classification-dataset-imdb", "versions", "1", "Genre Classification Dataset")

# List files
print("Files in dataset folder:", os.listdir(dataset_folder))

# Load train_data.txt
train_file = os.path.join(dataset_folder, "train_data.txt")
df = pd.read_csv(train_file, sep="\t")  # assuming tab-separated
print("Shape of dataset:", df.shape)
print("Columns:", df.columns)
print("\nFirst 5 rows:\n", df.head())

# Save for preprocessing
df.to_csv("raw_data.csv", index=False)
