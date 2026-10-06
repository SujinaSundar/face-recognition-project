import cv2
import os


DETECTOR_PATH = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = "models/face_recognition_sface_2021dec.onnx"

IMAGE_PATH = "data/train/George_W_Bush/001.jpg"


print("Loading models...")
print("--------------------------------")


# Load YuNet
detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


# Load SFace
recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_PATH,
    ""
)


# Load image
image = cv2.imread(
    IMAGE_PATH
)


if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


print("Image loaded!")
print("Image shape:", image.shape)


# Tell YuNet the actual image size
height, width = image.shape[:2]

detector.setInputSize(
    (width, height)
)


# Detect face
_, faces = detector.detect(
    image
)


if faces is None:
    raise RuntimeError(
        "No face detected!"
    )


print("Faces detected:", len(faces))


# Take the first detected face
face = faces[0]


print("\nDetection result:")
print("--------------------------------")

print("Bounding box:")
print(face[:4])

print("\nLandmarks:")
print(face[4:14])

print("\nConfidence:")
print(face[-1])


# Align face for SFace
aligned_face = recognizer.alignCrop(
    image,
    face
)


print("\nAligned face created!")
print("Aligned face shape:", aligned_face.shape)


# Extract embedding
feature = recognizer.feature(
    aligned_face
)


print("\nFeature extracted!")
print("Feature shape:", feature.shape)


print("\nFirst 10 feature values:")
print(feature[0][:10])