import os
import cv2
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

DETECTOR_PATH = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = "models/face_recognition_sface_2021dec.onnx"

CLASSIFIER_PATH = "models/face_classifier.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

TEST_DIR = "data/test"


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading models...")
print("--------------------------------")


detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_PATH,
    ""
)


classifier = joblib.load(
    CLASSIFIER_PATH
)


label_encoder = joblib.load(
    LABEL_ENCODER_PATH
)


print("Models loaded successfully!")


# --------------------------------------------------
# Storage
# --------------------------------------------------

y_true = []
y_pred = []

successful = 0
failed = 0


print("\nEvaluating test images...")
print("--------------------------------")


# --------------------------------------------------
# Get identities
# --------------------------------------------------

people = sorted(
    os.listdir(TEST_DIR)
)


# --------------------------------------------------
# Process test images
# --------------------------------------------------

for person in people:

    person_dir = os.path.join(
        TEST_DIR,
        person
    )

    if not os.path.isdir(person_dir):
        continue


    images = sorted([
        file
        for file in os.listdir(person_dir)
        if file.lower().endswith(".jpg")
    ])


    person_success = 0
    person_failed = 0


    for image_name in images:

        image_path = os.path.join(
            person_dir,
            image_name
        )


        # Load image
        image = cv2.imread(
            image_path
        )


        if image is None:
            failed += 1
            person_failed += 1
            continue


        # Image dimensions
        height, width = image.shape[:2]


        # Set YuNet input size
        detector.setInputSize(
            (width, height)
        )


        # Detect face
        _, faces = detector.detect(
            image
        )


        if faces is None or len(faces) == 0:
            failed += 1
            person_failed += 1
            continue


        # Use first detected face
        face = faces[0]


        # Align face
        aligned_face = recognizer.alignCrop(
            image,
            face
        )


        # Extract SFace feature
        feature = recognizer.feature(
            aligned_face
        )


        # Convert to 1D
        feature = feature.flatten()


        # Predict class
        prediction = classifier.predict(
            [feature]
        )


        # Convert number back to identity
        predicted_identity = (
            label_encoder.inverse_transform(
                prediction
            )[0]
        )


        # Store actual and predicted
        y_true.append(
            person
        )

        y_pred.append(
            predicted_identity
        )


        successful += 1
        person_success += 1


    print(
        f"{person}: "
        f"{person_success} successful, "
        f"{person_failed} failed"
    )


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\n================================")
print("TEST RESULTS")
print("================================")

print(
    "Successful test images:",
    successful
)

print(
    "Failed test images:",
    failed
)

print(
    f"Test accuracy: {accuracy * 100:.2f}%"
)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification Report:")
print("--------------------------------")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=label_encoder.classes_
)


print("\nConfusion Matrix:")
print("--------------------------------")

print(matrix)
