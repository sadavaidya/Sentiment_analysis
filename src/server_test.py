import requests
import pandas as pd
from gensim.models import Word2Vec
import numpy as np

# Load Word2Vec model (adjust path if needed)
word2vec_model = Word2Vec.load("models\word2vec.model")

# Function to convert text into Word2Vec vectors
def text_to_vector(text):
    words = text.split()
    vector = np.mean(
        [word2vec_model.wv[word] for word in words if word in word2vec_model.wv] 
        or [np.zeros(300)], axis=0
    )
    return vector

# Sample test data
texts = ["This product is bad"]

# Convert text to vectors and ensure correct shape
vectorized_data = np.array([text_to_vector(text) for text in texts])

# Reshape data into proper 2D array
vectorized_data = vectorized_data.reshape(-1, 100)  # Assuming 300D Word2Vec vectors

# Prepare payload in MLflow's new format
payload = {"instances": vectorized_data.tolist()}  # Ensures correct JSON format

# Send request
url = "http://localhost:1234/invocations"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, json=payload)
print("Response:", response.json())
