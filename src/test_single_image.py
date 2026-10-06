import cv2
import pickle,joblib
import os


# ========================================
# 1. Paths
# ========================================

DETECTOR_PATH = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = "models/face_recognition_sface_2021dec.onnx"
CLASSIFIER_PATH = "models/face_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

IMAGE_PATH = "data/test/George_W_Bush/021.jpg"


# ========================================
# 2. Check files
# ========================================

for path in [
    DETECTOR_PATH,
    RECOGNIZER_PATH,
    CLASSIFIER_PATH,
    ENCODER_PATH,
    IMAGE_PATH
]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )


# ========================================
# 3. Load YuNet detector
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
# 4. Load SFace recognizer
# ========================================

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_PATH,
    ""
)


# ========================================
# 5. Load Logistic Regression classifier
# ========================================

classifier = joblib.load(
    CLASSIFIER_PATH
)


# ========================================
# 6. Load label encoder
# ========================================

label_encoder = joblib.load(
    ENCODER_PATH
)


print("All models loaded successfully!")
# ========================================
# 7. Load image
# ========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        "Could not load image!"
    )

print("Image loaded!")
print("Image shape:", image.shape)


# ========================================
# 8. Detect face
# ========================================

height, width = image.shape[:2]

detector.setInputSize(
    (width, height)
)

_, faces = detector.detect(image)


if faces is None or len(faces) == 0:

    print("No face detected!")

    raise SystemExit


print("Faces detected:", len(faces))


# ========================================
# 9. Process first detected face
# ========================================

face = faces[0]

x, y, w, h = face[:4]

print("\nBounding box:")
print(
    f"x={int(x)}, "
    f"y={int(y)}, "
    f"width={int(w)}, "
    f"height={int(h)}"
)

print(
    f"Detection confidence: {face[-1]:.2f}"
)


# ========================================
# 10. Align face
# ========================================

aligned_face = recognizer.alignCrop(
    image,
    face
)

print(
    "Aligned face shape:",
    aligned_face.shape
)


# ========================================
# 11. Extract SFace embedding
# ========================================

feature = recognizer.feature(
    aligned_face
)

print(
    "Embedding shape:",
    feature.shape
)


# ========================================
# 12. Predict identity
# ========================================

predicted_class = classifier.predict(
    feature
)[0]

predicted_name = label_encoder.inverse_transform(
    [predicted_class]
)[0]


# ========================================
# 13. Prediction probability
# ========================================

probabilities = classifier.predict_proba(
    feature
)[0]

confidence = probabilities[predicted_class]


# ========================================
# 14. Display result
# ========================================

print("\n========================================")
print("RECOGNITION RESULT")
print("========================================")

print("Predicted identity:", predicted_name)
print(
    f"Recognition confidence: "
    f"{confidence * 100:.2f}%"
)


# ========================================
# 15. Draw result
# ========================================

x1 = int(x)
y1 = int(y)
x2 = int(x + w)
y2 = int(y + h)

cv2.rectangle(
    image,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    f"{predicted_name} ({confidence * 100:.1f}%)",
    (x1, max(y1 - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)


# ========================================
# 16. Save result
# ========================================

output_path = "results/single_image_result.jpg"

cv2.imwrite(
    output_path,
    image
)

print("\nResult saved to:")
print(output_path)