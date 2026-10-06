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

# Known test image
KNOWN_IMAGE = "data/test/George_W_Bush/021.jpg"


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
print("Prototype shape:", prototypes.shape)
print()


# --------------------------------------------------
# Function to extract face embedding
# --------------------------------------------------

def get_embedding(image):

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # Use first detected face
    face = faces[0]

    # Align face using YuNet landmarks
    aligned_face = recognizer.alignCrop(
        image,
        face
    )

    # Extract SFace embedding
    feature = recognizer.feature(
        aligned_face
    )

    return feature


# --------------------------------------------------
# Function to compare with prototypes
# --------------------------------------------------

def compare_with_prototypes(embedding):

    scores = []

    for prototype in prototypes:

        # Convert prototype to OpenCV format
        prototype = prototype.reshape(1, -1)

        similarity = recognizer.match(
            embedding,
            prototype,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        scores.append(float(similarity))

    scores = np.array(scores)

    # Sort from highest similarity to lowest
    sorted_indices = np.argsort(scores)[::-1]

    return scores, sorted_indices


# --------------------------------------------------
# Test known person
# --------------------------------------------------

print("========================================")
print("TEST 1: KNOWN PERSON")
print("========================================")

image = cv2.imread(KNOWN_IMAGE)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {KNOWN_IMAGE}"
    )

embedding = get_embedding(image)

if embedding is None:
    raise RuntimeError("No face detected!")

print("Embedding shape:", embedding.shape)

scores, sorted_indices = compare_with_prototypes(
    embedding
)

print("\nTop 5 matches:")

for i in range(5):

    index = sorted_indices[i]

    print(
        f"{prototype_labels[index]} "
        f"-> {scores[index]:.4f}"
    )

best_index = sorted_indices[0]

print("\nBEST MATCH")
print("Identity:", prototype_labels[best_index])
print("Similarity:", f"{scores[best_index]:.4f}")
# --------------------------------------------------
# Test unknown person using webcam
# --------------------------------------------------

print("\n========================================")
print("TEST 2: UNKNOWN PERSON")
print("========================================")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam started.")
print("Look at the camera.")
print("Press SPACE to capture.")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    cv2.putText(
        frame,
        "Press SPACE to capture | Q to quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Prototype Similarity Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        embedding = get_embedding(frame)

        if embedding is None:

            print("No face detected. Try again.")

        else:

            print("\nFace detected!")

            scores, sorted_indices = compare_with_prototypes(
                embedding
            )

            print("\nTop 5 matches:")

            for i in range(5):

                index = sorted_indices[i]

                print(
                    f"{prototype_labels[index]} "
                    f"-> {scores[index]:.4f}"
                )

            best_index = sorted_indices[0]

            print("\nBEST MATCH")
            print(
                "Identity:",
                prototype_labels[best_index]
            )
            print(
                "Similarity:",
                f"{scores[best_index]:.4f}"
            )

            break

    elif key == ord("q"):

        break

cap.release()
cv2.destroyAllWindows()