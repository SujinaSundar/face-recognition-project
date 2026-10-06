import cv2
import os
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# ========================================
# 1. Paths
# ========================================

DETECTOR_PATH = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = "models/face_recognition_sface_2021dec.onnx"
CLASSIFIER_PATH = "models/face_nn_classifier.pth"

TRAIN_LABELS_PATH = "data/embeddings/train_labels.npy"
TEST_PATH = "data/test"


# ========================================
# 2. Neural network
# ========================================

class FaceClassifier(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ========================================
# 3. Load label encoder
# ========================================

train_labels = np.load(
    TRAIN_LABELS_PATH
)

label_encoder = LabelEncoder()

label_encoder.fit(train_labels)

print("Number of identities:", len(label_encoder.classes_))


# ========================================
# 4. Load YuNet
# ========================================

detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


# ========================================
# 5. Load SFace
# ========================================

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_PATH,
    ""
)


# ========================================
# 6. Load neural network
# ========================================

model = FaceClassifier(
    input_size=128,
    num_classes=len(label_encoder.classes_)
)

model.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location="cpu"
    )
)

model.eval()

print("Neural network loaded successfully!")


# ========================================
# 7. Evaluate test images
# ========================================

y_true = []
y_pred = []

successful = 0
failed = 0


for person_name in sorted(os.listdir(TEST_PATH)):

    person_folder = os.path.join(
        TEST_PATH,
        person_name
    )

    if not os.path.isdir(person_folder):
        continue

    image_files = sorted(
        os.listdir(person_folder)
    )

    for image_name in image_files:

        image_path = os.path.join(
            person_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            continue

        # --------------------------------
        # Detect face
        # --------------------------------

        height, width = image.shape[:2]

        detector.setInputSize(
            (width, height)
        )

        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            failed += 1
            continue

        # Use first detected face
        face = faces[0]

        # --------------------------------
        # Align face using SFace
        # --------------------------------

        aligned_face = recognizer.alignCrop(
            image,
            face
        )

        # --------------------------------
        # Extract SFace embedding
        # --------------------------------

        feature = recognizer.feature(
            aligned_face
        )

        # --------------------------------
        # Convert to PyTorch tensor
        # --------------------------------

        feature_tensor = torch.tensor(
            feature,
            dtype=torch.float32
        )

        # --------------------------------
        # Predict identity
        # --------------------------------

        with torch.no_grad():

            output = model(
                feature_tensor
            )

            predicted_class = torch.argmax(
                output,
                dim=1
            ).item()

        predicted_name = (
            label_encoder
            .inverse_transform(
                [predicted_class]
            )[0]
        )

        # --------------------------------
        # Store result
        # --------------------------------

        y_true.append(person_name)
        y_pred.append(predicted_name)

        successful += 1


# ========================================
# 8. Calculate accuracy
# ========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


# ========================================
# 9. Print results
# ========================================

print("\n========================================")
print("NEURAL NETWORK TEST RESULTS")
print("========================================")

print("Total test images:", successful + failed)
print("Successfully processed:", successful)
print("Failed:", failed)

print(
    f"\nTest accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=label_encoder.classes_,
        zero_division=0
    )
)