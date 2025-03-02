import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load extracted features
try:
    X = np.load("data/X_features.npy")
    y = np.load("data/y_labels.npy")
    print("Features and labels loaded successfully!")
except FileNotFoundError:
    print("Error: Feature files not found. Ensure previous steps are completed.")
    exit()

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")

# Initialize and train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training the model...")
clf.fit(X_train, y_train)
print("Model training completed!")

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(clf, "models/sentiment_model.pkl")
print("Model saved successfully as sentiment_model.pkl!")
