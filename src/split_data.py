import numpy as np
import joblib
from sklearn.model_selection import train_test_split

# Load the feature vectors and labels
X = np.load("data/X_features.npy")  # Ensure this file exists
y = np.load("data/y_labels.npy")  # Ensure this file exists

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save split data
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("Data successfully split and saved!")
