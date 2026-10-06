import joblib
import os


CLASSIFIER_PATH = "models/face_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


print("Checking classifier files...")
print("--------------------------------")

print(
    "Classifier exists:",
    os.path.exists(CLASSIFIER_PATH)
)

print(
    "Encoder exists:",
    os.path.exists(ENCODER_PATH)
)


# Try loading with Joblib

classifier = joblib.load(CLASSIFIER_PATH)
label_encoder = joblib.load(ENCODER_PATH)


print("\nClassifier loaded successfully!")
print("Classifier type:", type(classifier))

print("\nLabel encoder loaded successfully!")
print("Encoder type:", type(label_encoder))

print("\nClasses:")
print(label_encoder.classes_)