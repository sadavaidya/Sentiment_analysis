import pandas as pd
import re
import spacy
import time
from multiprocessing import Pool
from nltk.corpus import stopwords

# Load SpaCy model (faster than NLTK)
nlp = spacy.load("en_core_web_sm")

# Load stopwords once
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    """Preprocesses a single tweet: lowercasing, removing URLs, special characters, and stopwords."""
    if not isinstance(text, str):  # Handle missing values
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers

    # Use SpaCy tokenizer (much faster than NLTK)
    doc = nlp(text)
    tokens = [token.text for token in doc if token.text not in stop_words]

    return " ".join(tokens)  # Return cleaned text as a string

def parallel_preprocessing(texts):
    """Uses multiprocessing to process texts in parallel."""
    with Pool(processes=4) as pool:  # Use 4 CPU cores
        cleaned_texts = pool.map(preprocess_text, texts)
    return cleaned_texts

if __name__ == "__main__":  # <== FIX: Needed for multiprocessing on Windows
    # Load dataset with column names (if missing headers)
    column_names = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv("data/tweets.csv", encoding="ISO-8859-1", names=column_names, header=None)

    # Sample 10,000 rows for testing (modify based on system performance)
    df_sample = df.sample(10000)

    # Start timing
    start_time = time.time()

    # Apply preprocessing with multiprocessing
    df_sample["cleaned_text"] = parallel_preprocessing(df_sample["text"])

    # Print execution time
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

    # Save processed data
    df_sample.to_csv("data/processed_tweets.csv", index=False)

    print("Processed data saved successfully!")
