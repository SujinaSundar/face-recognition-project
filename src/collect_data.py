import cv2
import os

from detector import FaceDetector


# ==========================================
# Configuration
# ==========================================

person_name = input("Enter person name: ").strip()

if not person_name:
    print("Invalid name.")
    exit()


save_dir = os.path.join(
    "data",
    "raw",
    person_name
)

os.makedirs(save_dir, exist_ok=True)


NUM_IMAGES = 30


# ==========================================
# Initialize face detector
# ==========================================

detector = FaceDetector(
    confidence_threshold=0.8
)


# ==========================================
# Find existing images
# ==========================================

existing_images = [
    file
    for file in os.listdir(save_dir)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]


count = len(existing_images)


# ==========================================
# Start webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()


print("\n======================================")
print("FACE DATA COLLECTION")
print("======================================")
print(f"Person       : {person_name}")
print(f"Target       : {NUM_IMAGES} images")
print(f"Already saved: {count} images")
print("--------------------------------------")
print("Press S → Save face")
print("Press Q → Quit")
print("======================================\n")


# ==========================================
# Webcam loop
# ==========================================

while count < NUM_IMAGES:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    faces = detector.detect(frame)


    # --------------------------------------
    # Draw all detected faces
    # --------------------------------------

    for face in faces:

        x1, y1, x2, y2 = face["box"]
        confidence = face["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Face {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # --------------------------------------
    # Display status
    # --------------------------------------

    cv2.putText(
        frame,
        f"Saved: {count}/{NUM_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "S = Save | Q = Quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Face Data Collection",
        frame
    )


    # --------------------------------------
    # Keyboard input
    # --------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------
    # Save face
    # --------------------------------------

    if key == ord("s"):

        # We only save when exactly one face
        # is detected.

        if len(faces) == 0:

            print("No face detected. Try again.")

            continue


        if len(faces) > 1:

            print(
                "Multiple faces detected. "
                "Please keep only one person in frame."
            )

            continue


        # Get detected face
        x1, y1, x2, y2 = faces[0]["box"]


        # Make sure coordinates stay
        # inside the image

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)


        # Crop face
        face_crop = frame[
            y1:y2,
            x1:x2
        ]


        # Validate crop
        if face_crop.size == 0:

            print("Invalid face crop.")

            continue


        # Save
        count += 1

        image_path = os.path.join(
            save_dir,
            f"{count:03d}.jpg"
        )

        cv2.imwrite(
            image_path,
            face_crop
        )

        print(
            f"Saved face: {image_path}"
        )


    # --------------------------------------
    # Quit
    # --------------------------------------

    elif key == ord("q"):

        print("Stopping data collection...")

        break


# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()


print("\n======================================")
print("DATA COLLECTION COMPLETED")
print(f"Person : {person_name}")
print(f"Images : {count}")
print("======================================")