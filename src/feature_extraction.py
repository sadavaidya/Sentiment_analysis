import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK resources are downloaded
nltk.download("punkt")

# Load the preprocessed data
try:
    df = pd.read_csv("data\processed_tweets.csv")  # Ensure the correct path
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: preprocessed_tweets.csv not found. Check the file path.")
    exit()

# Ensure 'cleaned_text' column exists
if "cleaned_text" not in df.columns:
    print("Error: 'cleaned_text' column not found in dataset.")
    exit()

# Convert to string and handle NaN values
df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

# Tokenize tweets
df["tokenized_text"] = df["cleaned_text"].apply(word_tokenize)

# Save tokenized tweets
df.to_csv("data/tokenized_tweets.csv", index=False)

print("Feature extraction (tokenization) completed successfully!")
