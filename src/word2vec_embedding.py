import pandas as pd
import gensim
from gensim.models import Word2Vec

# Load tokenized tweets
try:
    df = pd.read_csv("data/tokenized_tweets.csv")  # Ensure the correct path
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: tokenized_tweets.csv not found. Check the file path.")
    exit()

# Ensure 'tokenized_text' column exists
if "tokenized_text" not in df.columns:
    print("Error: 'tokenized_text' column not found in dataset.")
    exit()

# Convert tokenized text from string back to list (if necessary)
df["tokenized_text"] = df["tokenized_text"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Train Word2Vec model
print("Training Word2Vec model...")
word2vec_model = Word2Vec(sentences=df["tokenized_text"], vector_size=100, window=5, min_count=2, workers=4)

# Save the Word2Vec model
word2vec_model.save("models/word2vec.model")
print("Word2Vec model saved successfully!")

# Save word vectors separately (optional)
word2vec_model.wv.save_word2vec_format("models/word_vectors.bin", binary=True)
print("Word vectors saved successfully!")
