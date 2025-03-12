# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Load extracted features
# try:
#     X = np.load("data/X_features.npy")
#     y = np.load("data/y_labels.npy")
#     print("Features and labels loaded successfully!")
# except FileNotFoundError:
#     print("Error: Feature files not found. Ensure previous steps are completed.")
#     exit()

# # Split dataset into training and test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")

# # Initialize and train a RandomForest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# print("Training the model...")
# clf.fit(X_train, y_train)
# print("Model training completed!")

# # Evaluate the model
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Test Accuracy: {accuracy:.4f}")
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # Save the trained model
# joblib.dump(clf, "models/sentiment_model.pkl")
# print("Model saved successfully as sentiment_model.pkl!")

import mlflow
import mlflow.sklearn  # If using an sklearn model, otherwise use the relevant module
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Load data
X_train = np.load("data/X_train.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test = np.load("data/y_test.npy")

# Start MLflow run
with mlflow.start_run():
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_dict(report, "classification_report.json")

    # Save model
    model_path = "models/sentiment_model"
    mlflow.sklearn.save_model(model, model_path)

    print(f"Model logged with accuracy: {accuracy}")


