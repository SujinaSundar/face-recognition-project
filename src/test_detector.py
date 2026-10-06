import cv2

from detector import FaceDetector


# -----------------------------
# Load image
# -----------------------------

image_path = "test.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Could not load image.")
    exit()


# -----------------------------
# Initialize detector
# -----------------------------

detector = FaceDetector()


# -----------------------------
# Detect faces
# -----------------------------

faces = detector.detect(image)


print(f"Faces detected: {len(faces)}")


# -----------------------------
# Draw detections
# -----------------------------

for face in faces:

    x1, y1, x2, y2 = face["box"]
    confidence = face["confidence"]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Face {confidence:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


# -----------------------------
# Display
# -----------------------------

cv2.imshow(
    "Face Detection",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()