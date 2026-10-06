import cv2
import os


class FaceDetector:

    def __init__(
        self,
        model_path="models/face_detection_yunet_2023mar.onnx",
        input_size=(320, 320),
        confidence_threshold=0.8
    ):

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face detector model not found: {model_path}"
            )

        self.input_size = input_size
        self.confidence_threshold = confidence_threshold

        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            confidence_threshold,
            0.3,
            5000
        )

    def detect(self, frame):

        height, width = frame.shape[:2]

        # YuNet needs the current image size
        self.detector.setInputSize(
            (width, height)
        )

        _, faces = self.detector.detect(frame)

        results = []

        if faces is None:
            return results

        for face in faces:

            # Bounding box
            x, y, w, h = face[:4]

            # Confidence
            confidence = float(face[-1])

            # Convert bounding box
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            # Store detection information
            results.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence,

                # 5 facial landmarks:
                # left eye, right eye, nose,
                # left mouth corner, right mouth corner
                "landmarks": face[4:14]
            })

        return results