import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import string
from nltk.corpus import stopwords

# Load the dataset
data = pd.read_csv("/mnt/data/netflixData.csv")
print(data.head())

# Select relevant columns
data = data[["Title", "Description", "Content Type", "Genres"]]
print(data.head())

# Drop rows with missing values
data = data.dropna()
print("Data after dropping NaNs:")
print(data.head())

# Download NLTK stopwords
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# Function to clean text
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Clean the Title and Genres columns
data["Title"] = data["Title"].apply(clean)
data["Genres"] = data["Genres"].apply(clean)
print("Data after cleaning:")
print(data.head())

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english")

# Fit and transform the vectorizer on the Genres column
tfidf_matrix = tfidf.fit_transform(data["Genres"].tolist())
print("TF-IDF Matrix shape:", tfidf_matrix.shape)

# Compute the cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix)
print("Cosine similarity matrix shape:", similarity.shape)

# Create a Series to map titles to their indices
indices = pd.Series(data.index, index=data['Title']).drop_duplicates()
print("Indices Series:")
print(indices.head())

# Function to get Netflix recommendations
def netFlix_recommendation(title, similarity=similarity):
    if title not in indices:
        return f"Title '{title}' not found in the dataset."
    index = indices[title]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[:10]
    movie_indices = [i[0] for i in similarity_scores]
    return data['Title'].iloc[movie_indices]

# Example: Get recommendations for a specific title
recommendations = netFlix_recommendation("girlfriend")
print("Recommendations for 'girlfriend':")
print(recommendations)
