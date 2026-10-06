import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib
import os


# --------------------------------------------------
# Paths
# --------------------------------------------------

EMBEDDINGS_PATH = "data/embeddings/train_embeddings.npy"
LABELS_PATH = "data/embeddings/train_labels.npy"

MODEL_DIR = "models"

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR,
    "face_classifier.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "label_encoder.pkl"
)


# --------------------------------------------------
# Create model directory
# --------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


print("Loading training embeddings...")
print("--------------------------------")


# --------------------------------------------------
# Load data
# --------------------------------------------------

X = np.load(
    EMBEDDINGS_PATH
)

y = np.load(
    LABELS_PATH
)


print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------------------------
# Encode identity names as numbers
# --------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)


print("\nIdentities:")
print("--------------------------------")

for number, identity in enumerate(
    label_encoder.classes_
):
    print(
        f"{number}: {identity}"
    )


# --------------------------------------------------
# Create classifier
# --------------------------------------------------

print("\nCreating Logistic Regression...")
print("--------------------------------")


classifier = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("Training classifier...")


classifier.fit(
    X,
    y_encoded
)


print("Training completed!")


# --------------------------------------------------
# Training predictions
# --------------------------------------------------

train_predictions = classifier.predict(
    X
)


train_accuracy = accuracy_score(
    y_encoded,
    train_predictions
)


print("\nTraining accuracy:")
print(
    f"{train_accuracy * 100:.2f}%"
)


# --------------------------------------------------
# Save classifier
# --------------------------------------------------

joblib.dump(
    classifier,
    CLASSIFIER_PATH
)

joblib.dump(
    label_encoder,
    LABEL_ENCODER_PATH
)


print("\n================================")
print("Classifier saved successfully!")
print("================================")

print(
    "Classifier:",
    CLASSIFIER_PATH
)

print(
    "Label encoder:",
    LABEL_ENCODER_PATH
)