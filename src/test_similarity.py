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


# ========================================
# 2. Load known embeddings
# ========================================

known_embeddings = np.load(
    EMBEDDINGS_PATH
)

known_labels = np.load(
    LABELS_PATH
)

print("Known embeddings:", known_embeddings.shape)
print("Known labels:", known_labels.shape)


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

print("Models loaded successfully!")


# ========================================
# 4. Open webcam
# ========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open webcam!"
    )

print("Webcam started.")
print("Look at the camera.")
print("Press Q to quit.")


# ========================================
# 5. Real-time similarity
# ========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break


    # ------------------------------------
    # Detect face
    # ------------------------------------

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)


    if faces is not None:

        # Use first detected face
        face = faces[0]


        # --------------------------------
        # Align face
        # --------------------------------

        aligned_face = recognizer.alignCrop(
            frame,
            face
        )


        # --------------------------------
        # Extract embedding
        # --------------------------------

        feature = recognizer.feature(
            aligned_face
        )


        # --------------------------------
        # Calculate cosine similarity
        # --------------------------------

        similarities = []

        for known_embedding in known_embeddings:

            score = recognizer.match(
                feature,
                known_embedding.reshape(1, -1),
                cv2.FaceRecognizerSF_FR_COSINE
            )

            similarities.append(score)


        # --------------------------------
        # Find best match
        # --------------------------------

        best_index = int(
            np.argmax(similarities)
        )

        best_score = float(
            similarities[best_index]
        )

        best_name = known_labels[
            best_index
        ]


        # --------------------------------
        # Print result
        # --------------------------------

        print(
            f"Best match: {best_name} | "
            f"Similarity: {best_score:.4f}"
        )


    # ------------------------------------
    # Display
    # ------------------------------------

    cv2.imshow(
        "SFace Similarity Test",
        frame
    )


    # ------------------------------------
    # Quit
    # ------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ========================================
# 6. Cleanup
# ========================================

cap.release()
cv2.destroyAllWindows()

print("Similarity test stopped.")