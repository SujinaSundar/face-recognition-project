import cv2
import os


MODEL_PATH = "models/face_recognition_sface_2021dec.onnx"
IMAGE_PATH = "data/train/George_W_Bush/001.jpg"


print("Loading SFace...")
print("--------------------------------")


# Check files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# Load SFace
recognizer = cv2.FaceRecognizerSF.create(
    MODEL_PATH,
    ""
)


# Load image
image = cv2.imread(
    IMAGE_PATH
)


print("Image loaded!")
print("Original image shape:", image.shape)


# Resize face image
face = cv2.resize(
    image,
    (112, 112)
)


print("Resized face shape:", face.shape)


# Extract face feature
feature = recognizer.feature(
    face
)


print("Feature extracted successfully!")
print("Feature shape:", feature.shape)
print("Feature data type:", feature.dtype)


print("\nFirst 10 feature values:")
print(feature[0][:10])