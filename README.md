# Movie Genre Classifier

A Machine Learning project that predicts the genre of a movie based on its description. The project leverages Natural Language Processing (NLP) techniques, including text preprocessing, TF-IDF vectorization, and machine learning classification models.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Model Performance](#model-performance)  
- [Future Improvements](#future-improvements)  
- [License](#license)  

---

## Project Overview

This project allows users to input a movie description, and the model predicts the most likely genre. The workflow includes:

1. **Data Loading:** Load training and testing datasets.  
2. **Data Preprocessing:** Clean text, remove stopwords, and lemmatize words using NLTK.  
3. **Feature Extraction:** Convert movie descriptions into TF-IDF vectors.  
4. **Model Training:** Train a machine learning classifier (Random Forest) on the processed dataset.  
5. **Prediction:** Predict genres of new movie descriptions.  

---

## Features

- Predict movie genres from a textual description.  
- Supports multiple genres like Action, Drama, Comedy, Horror, Sci-Fi, Animation, Documentary, etc.  
- Handles invalid or meaningless input gracefully.  
- Fully offline capable after model training.  
- CLI-based interaction; ready for extension to a web UI.

---

## Technologies Used

- Python 3.10+  
- Pandas  
- Scikit-learn  
- NLTK (Natural Language Toolkit)  
- Pickle (for saving/loading models and vectorizers)  

---

## Installation

1. **Clone the repository**  
```bash
git clone https://github.com/R0HIT-45/MovieGenreClassifier.git
cd MovieGenreClassifier
