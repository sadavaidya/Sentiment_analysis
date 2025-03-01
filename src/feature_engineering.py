import numpy as np
import pandas as pd
import gensim

# Load Word2Vec model
try:
    word2vec_model = gensim.models.Word2Vec.load("models/word2vec.model")
    print("Word2Vec model loaded successfully!")
except FileNotFoundError:
    print("Error: word2vec.model not found. Check the file path.")
    exit()

# Load tokenized tweets dataset
try:
    df = pd.read_csv("data/tokenized_tweets.csv")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: tokenized_tweets.csv not found. Check the file path.")
    exit()

# Ensure 'tokenized_text' and 'target' columns exist
if "tokenized_text" not in df.columns or "target" not in df.columns:
    print("Error: Required columns not found in dataset.")
    exit()

# Convert tokenized text from string back to list (if needed)
df["tokenized_text"] = df["tokenized_text"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Function to get sentence embedding (average of word embeddings)
def get_sentence_embedding(words, model):
    valid_words = [word for word in words if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)  # Return zero vector if no words in vocab
    return np.mean([model.wv[word] for word in valid_words], axis=0)

# Convert each tweet into a numerical vector
print("Generating sentence embeddings...")
df["embedding"] = df["tokenized_text"].apply(lambda x: get_sentence_embedding(x, word2vec_model))

# Convert embeddings to a structured feature matrix
X = np.vstack(df["embedding"].values)
y = df["target"].values  # Labels for classification

# Save the processed feature set
np.save("data/X_features.npy", X)
np.save("data/y_labels.npy", y)
print("Feature extraction completed and saved successfully!")
