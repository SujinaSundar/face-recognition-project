import cv2
import numpy as np
import os


# --------------------------------------------------
# Paths
# --------------------------------------------------

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"

PROTOTYPES_PATH = "data/embeddings/prototypes.npy"
PROTOTYPE_LABELS_PATH = "data/embeddings/prototype_labels.npy"

TEST_DIR = "data/test"


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading models...")

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNITION_MODEL,
    ""
)

prototypes = np.load(PROTOTYPES_PATH)
prototype_labels = np.load(PROTOTYPE_LABELS_PATH)

print("Models loaded successfully!")
print("Prototypes:", prototypes.shape)
print("Identities:", len(prototype_labels))


# --------------------------------------------------
# Extract embedding
# --------------------------------------------------

def get_embedding(image):

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # Use first detected face
    face = faces[0]

    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    embedding = recognizer.feature(
        aligned_face
    )

    return embedding


# --------------------------------------------------
# Calculate similarity scores
# --------------------------------------------------

def calculate_scores(embedding):

    scores = []

    for prototype in prototypes:

        prototype = prototype.reshape(1, -1)

        similarity = recognizer.match(
            embedding,
            prototype,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        scores.append(float(similarity))

    return np.array(scores)


# --------------------------------------------------
# Variables
# --------------------------------------------------

genuine_scores = []
impostor_scores = []

successful = 0
failed = 0


# --------------------------------------------------
# Process test dataset
# --------------------------------------------------

print("\n========================================")
print("VALIDATING SIMILARITY SCORES")
print("========================================")

for identity in sorted(os.listdir(TEST_DIR)):

    identity_dir = os.path.join(
        TEST_DIR,
        identity
    )

    if not os.path.isdir(identity_dir):
        continue

    for filename in sorted(os.listdir(identity_dir)):

        image_path = os.path.join(
            identity_dir,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            continue

        embedding = get_embedding(image)

        if embedding is None:
            failed += 1
            continue

        scores = calculate_scores(embedding)

        # ------------------------------------------
        # Genuine score
        # ------------------------------------------

        correct_index = np.where(
            prototype_labels == identity
        )[0][0]

        genuine_score = scores[correct_index]

        # ------------------------------------------
        # Impostor score
        # ------------------------------------------

        incorrect_scores = np.delete(
            scores,
            correct_index
        )

        impostor_score = np.max(
            incorrect_scores
        )

        genuine_scores.append(
            genuine_score
        )

        impostor_scores.append(
            impostor_score
        )

        successful += 1


# --------------------------------------------------
# Convert to NumPy arrays
# --------------------------------------------------

genuine_scores = np.array(
    genuine_scores
)

impostor_scores = np.array(
    impostor_scores
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\n========================================")
print("VALIDATION COMPLETE")
print("========================================")

print("Successful:", successful)
print("Failed:", failed)

print("\nGENUINE SCORES")
print("----------------------------------------")
print(
    "Minimum:",
    f"{genuine_scores.min():.4f}"
)
print(
    "Maximum:",
    f"{genuine_scores.max():.4f}"
)
print(
    "Mean:",
    f"{genuine_scores.mean():.4f}"
)

print("\nIMPOSTOR SCORES")
print("----------------------------------------")
print(
    "Minimum:",
    f"{impostor_scores.min():.4f}"
)
print(
    "Maximum:",
    f"{impostor_scores.max():.4f}"
)
print(
    "Mean:",
    f"{impostor_scores.mean():.4f}"
)


# --------------------------------------------------
# Percentiles
# --------------------------------------------------

print("\nGENUINE SCORE PERCENTILES")
print("----------------------------------------")

for percentile in [5, 10, 25, 50, 75, 90, 95]:

    value = np.percentile(
        genuine_scores,
        percentile
    )

    print(
        f"{percentile}%: {value:.4f}"
    )


print("\nIMPOSTOR SCORE PERCENTILES")
print("----------------------------------------")

for percentile in [5, 10, 25, 50, 75, 90, 95, 99]:

    value = np.percentile(
        impostor_scores,
        percentile
    )

    print(
        f"{percentile}%: {value:.4f}"
    )