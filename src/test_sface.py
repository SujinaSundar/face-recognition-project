import cv2
import os


MODEL_PATH = "models/face_recognition_sface_2021dec.onnx"


print("Checking SFace model...")
print("--------------------------------")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"SFace model not found: {MODEL_PATH}"
    )


recognizer = cv2.FaceRecognizerSF.create(
    MODEL_PATH,
    ""
)


print("SFace model loaded successfully!")
print("Model path:", MODEL_PATH)