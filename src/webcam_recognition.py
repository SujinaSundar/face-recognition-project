import cv2
import numpy as np

from detector import FaceDetector


# ========================================
# 1. Model paths
# ========================================

SFACE_MODEL_PATH = (
    "models/face_recognition_sface_2021dec.onnx"
)

PROTOTYPES_PATH = (
    "data/embeddings/prototypes.npy"
)

PROTOTYPE_LABELS_PATH = (
    "data/embeddings/prototype_labels.npy"
)


# ========================================
# 2. Recognition threshold
# ========================================
#
# Based on our validation:
#
# Maximum impostor similarity = 0.4041
# Minimum genuine similarity  = 0.5840
#
# We selected 0.50 as the threshold.
#
# similarity >= 0.50 -> Known
# similarity <  0.50 -> Unknown
#

SIMILARITY_THRESHOLD = 0.50


# ========================================
# 3. Load face detector
# ========================================

detector = FaceDetector(
    confidence_threshold=0.8
)


# ========================================
# 4. Load SFace
# ========================================

recognizer = cv2.FaceRecognizerSF.create(
    SFACE_MODEL_PATH,
    ""
)


# ========================================
# 5. Load identity prototypes
# ========================================

prototypes = np.load(
    PROTOTYPES_PATH
)

prototype_labels = np.load(
    PROTOTYPE_LABELS_PATH
)


print("All models loaded successfully!")

print(
    "Number of identity prototypes:",
    len(prototypes)
)

print(
    "Similarity threshold:",
    SIMILARITY_THRESHOLD
)

print("Face recognition started.")
print("Press Q to quit.")


# ========================================
# 6. Open webcam
# ========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not open webcam.")
    exit()


# ========================================
# 7. Real-time loop
# ========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read frame.")
        break


    # ------------------------------------
    # Detect faces
    # ------------------------------------

    faces = detector.detect(
        frame
    )


    # ------------------------------------
    # Process every detected face
    # ------------------------------------

    for face in faces:

        x1, y1, x2, y2 = face["box"]

        detection_confidence = (
            face["confidence"]
        )


        # --------------------------------
        # Get YuNet landmarks
        # --------------------------------

        landmarks = face["landmarks"]


        # --------------------------------
        # Reconstruct YuNet face result
        # --------------------------------
        #
        # SFace alignCrop() expects:
        #
        # [x, y, w, h,
        #  landmark1,
        #  landmark2,
        #  landmark3,
        #  landmark4,
        #  landmark5,
        #  confidence]
        #

        x = float(x1)
        y = float(y1)

        w = float(
            x2 - x1
        )

        h = float(
            y2 - y1
        )

        sface_face = [
            x,
            y,
            w,
            h,
            *landmarks,
            detection_confidence
        ]


        # --------------------------------
        # Convert to NumPy array
        # --------------------------------

        sface_face = np.array(
            sface_face,
            dtype=np.float32
        )


        # --------------------------------
        # Align face
        # --------------------------------

        aligned_face = (
            recognizer.alignCrop(
                frame,
                sface_face
            )
        )


        # --------------------------------
        # Extract SFace embedding
        # --------------------------------

        feature = (
            recognizer.feature(
                aligned_face
            )
        )


        # --------------------------------
        # Compare with all prototypes
        # --------------------------------

        similarities = []

        for prototype in prototypes:

            prototype = prototype.reshape(
                1,
                -1
            )

            similarity = (
                recognizer.match(
                    feature,
                    prototype,
                    cv2.FaceRecognizerSF_FR_COSINE
                )
            )

            similarities.append(
                float(similarity)
            )


        # --------------------------------
        # Convert scores to NumPy array
        # --------------------------------

        similarities = np.array(
            similarities
        )


        # --------------------------------
        # Find best matching identity
        # --------------------------------

        best_index = np.argmax(
            similarities
        )

        best_similarity = (
            similarities[best_index]
        )

        best_identity = (
            prototype_labels[best_index]
        )


        # --------------------------------
        # Unknown face rejection
        # --------------------------------

        if best_similarity >= SIMILARITY_THRESHOLD:

            predicted_name = str(
                best_identity
            )

        else:

            predicted_name = "Unknown"


        # --------------------------------
        # Print recognition result
        # --------------------------------

        print(
            f"Predicted: {predicted_name} | "
            f"Similarity: {best_similarity:.4f}"
        )


        # --------------------------------
        # Draw bounding box
        # --------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # --------------------------------
        # Draw identity
        # --------------------------------

        label = (
            f"{predicted_name} "
            f"{best_similarity:.2f}"
        )

        cv2.putText(
            frame,
            label,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # --------------------------------
        # Draw detection confidence
        # --------------------------------

        detection_label = (
            f"Detection: "
            f"{detection_confidence:.2f}"
        )

        cv2.putText(
            frame,
            detection_label,
            (
                x1,
                y2 + 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1
        )


    # ------------------------------------
    # Number of detected faces
    # ------------------------------------

    cv2.putText(
        frame,
        f"Faces detected: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # ------------------------------------
    # Display threshold
    # ------------------------------------

    cv2.putText(
        frame,
        f"Threshold: {SIMILARITY_THRESHOLD:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


    # ------------------------------------
    # Display webcam
    # ------------------------------------

    cv2.imshow(
        "Real-Time Face Recognition",
        frame
    )


    # ------------------------------------
    # Quit with Q
    # ------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ========================================
# 8. Cleanup
# ========================================

cap.release()

cv2.destroyAllWindows()

print("Face recognition stopped.")