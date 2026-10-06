import os
import cv2
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

DETECTOR_PATH = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = "models/face_recognition_sface_2021dec.onnx"

TRAIN_DIR = "data/train_augmented"

OUTPUT_DIR = "data/embeddings"

EMBEDDINGS_PATH = os.path.join(
    OUTPUT_DIR,
    "train_embeddings.npy"
)

LABELS_PATH = os.path.join(
    OUTPUT_DIR,
    "train_labels.npy"
)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("Loading models...")
print("--------------------------------")


# --------------------------------------------------
# Load YuNet
# --------------------------------------------------

detector = cv2.FaceDetectorYN.create(
    DETECTOR_PATH,
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)


# --------------------------------------------------
# Load SFace
# --------------------------------------------------

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNIZER_PATH,
    ""
)


print("Models loaded successfully!")


# --------------------------------------------------
# Storage
# --------------------------------------------------

embeddings = []
labels = []

successful = 0
failed = 0


print("\nStarting embedding extraction...")
print("--------------------------------")


# --------------------------------------------------
# Get identities
# --------------------------------------------------

people = sorted(
    os.listdir(TRAIN_DIR)
)


# --------------------------------------------------
# Process every identity
# --------------------------------------------------

for person in people:

    person_dir = os.path.join(
        TRAIN_DIR,
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


    # --------------------------------------------------
    # Process every image
    # --------------------------------------------------

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


        # Get image dimensions
        height, width = image.shape[:2]


        # Tell YuNet the image size
        detector.setInputSize(
            (width, height)
        )


        # Detect face
        _, faces = detector.detect(
            image
        )


        # No face detected
        if faces is None or len(faces) == 0:

            failed += 1
            person_failed += 1

            continue


        # Use the first detected face
        face = faces[0]


        # Align face
        aligned_face = recognizer.alignCrop(
            image,
            face
        )


        # Extract SFace embedding
        feature = recognizer.feature(
            aligned_face
        )


        # Convert to 1D array
        feature = feature.flatten()


        # Store embedding
        embeddings.append(
            feature
        )


        # Store identity
        labels.append(
            person
        )


        successful += 1
        person_success += 1


    print(
        f"{person}: "
        f"{person_success} successful, "
        f"{person_failed} failed"
    )


# --------------------------------------------------
# Convert to NumPy arrays
# --------------------------------------------------

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

labels = np.array(
    labels
)


# --------------------------------------------------
# Save
# --------------------------------------------------

np.save(
    EMBEDDINGS_PATH,
    embeddings
)

np.save(
    LABELS_PATH,
    labels
)


# --------------------------------------------------
# Final results
# --------------------------------------------------

print("\n================================")
print("Embedding extraction completed!")
print("================================")

print(
    "Successful images:",
    successful
)

print(
    "Failed images:",
    failed
)

print(
    "Embedding shape:",
    embeddings.shape
)

print(
    "Labels shape:",
    labels.shape
)

print(
    "Embeddings saved to:",
    EMBEDDINGS_PATH
)

print(
    "Labels saved to:",
    LABELS_PATH
)