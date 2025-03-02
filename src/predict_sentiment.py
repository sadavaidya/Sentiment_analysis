import joblib
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import re
import string

# Load the trained model
model_path = "models/sentiment_model.pkl"
word2vec_path = "models/word2vec.model"

try:
    clf = joblib.load(model_path)
    word2vec = Word2Vec.load(word2vec_path)
    print("Model and Word2Vec loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Ensure training is completed.")
    exit()

# Text preprocessing function (same as used earlier)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"\@\w+|\#", "", text)  # Remove mentions & hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)
    return tokens

# Convert text to Word2Vec feature vector
def text_to_vector(text, word2vec_model):
    tokens = preprocess_text(text)
    vector_size = word2vec_model.vector_size
    vector = np.zeros(vector_size)
    count = 0
    
    for word in tokens:
        if word in word2vec_model.wv:
            vector += word2vec_model.wv[word]
            count += 1
    
    return vector / count if count > 0 else vector

# Get user input
while True:
    user_input = input("Enter a tweet (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    
    # Convert input text to features
    feature_vector = text_to_vector(user_input, word2vec).reshape(1, -1)
    
    # Predict sentiment
    prediction = clf.predict(feature_vector)
    
    sentiment_map = {0: "Negative 😡", 2: "Neutral 😐", 4: "Positive 😊"}
    print(f"Predicted Sentiment: {sentiment_map.get(prediction[0], 'Unknown')}")

print("Prediction session ended.")
