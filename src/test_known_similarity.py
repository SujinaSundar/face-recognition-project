import cv2
import numpy as np


# ========================================
# 1. Paths
# ========================================

DETECTOR_PATH = (
    "models/face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_PATH = (
    "models/face_recognition_sface_2021dec.onnx"
)

EMBEDDINGS_PATH = (
    "data/embeddings/train_embeddings.npy"
)

LABELS_PATH = (
    "data/embeddings/train_labels.npy"
)

IMAGE_PATH = (
    "data/test/George_W_Bush/021.jpg"
)


# ========================================
# 2. Load known embeddings
# ========================================

known_embeddings = np.load(
    EMBEDDINGS_PATH
)

known_labels = np.load(
    LABELS_PATH
)

print(
    "Known embeddings:",
    known_embeddings.shape
)


# ========================================
# 3. Load models
# ========================================

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


# ========================================
# 4. Load image
# ========================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ========================================
# 5. Detect face
# ========================================

height, width = image.shape[:2]

detector.setInputSize(
    (width, height)
)

_, faces = detector.detect(
    image
)


if faces is None or len(faces) == 0:
    raise RuntimeError(
        "No face detected!"
    )


print(
    "Faces detected:",
    len(faces)
)


# ========================================
# 6. Align face
# ========================================

face = faces[0]

aligned_face = recognizer.alignCrop(
    image,
    face
)


# ========================================
# 7. Extract embedding
# ========================================

feature = recognizer.feature(
    aligned_face
)


print(
    "Embedding shape:",
    feature.shape
)


# ========================================
# 8. Compare with known embeddings
# ========================================

similarities = []

for known_embedding in known_embeddings:

    score = recognizer.match(
        feature,
        known_embedding.reshape(1, -1),
        cv2.FaceRecognizerSF_FR_COSINE
    )

    similarities.append(
        float(score)
    )


# ========================================
# 9. Find best matches
# ========================================

similarities = np.array(
    similarities
)

top_indices = np.argsort(
    similarities
)[::-1][:10]


print("\nTop 10 matches:")
print("--------------------------------")

for index in top_indices:

    print(
        f"{known_labels[index]:35s} "
        f"Similarity: {similarities[index]:.4f}"
    )


# ========================================
# 10. Best match
# ========================================

best_index = top_indices[0]

print("\n========================================")
print("BEST MATCH")
print("========================================")

print(
    "Identity:",
    known_labels[best_index]
)

print(
    f"Similarity: "
    f"{similarities[best_index]:.4f}"
)