import streamlit as st
import cv2
import numpy as np
import os

from src.detector import FaceDetector


# ============================================================
# CONFIGURATION
# ============================================================

SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"

PROTOTYPES_FILE = "data/embeddings/prototypes.npy"
LABELS_FILE = "data/embeddings/prototype_labels.npy"

SIMILARITY_THRESHOLD = 0.50


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Face Detection & Recognition System")

st.write(
    "Detect and recognize faces using YuNet + SFace."
)


# ============================================================
# INPUT MODE
# ============================================================

input_mode = st.radio(
    "Choose input source:",
    ["Upload Image", "Webcam"],
    horizontal=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # YuNet detector
    detector = FaceDetector()

    # SFace recognizer
    recognizer = cv2.FaceRecognizerSF.create(
        SFACE_MODEL,
        ""
    )

    # Load prototypes
    prototypes = np.load(
        PROTOTYPES_FILE
    )

    # Load identity labels
    prototype_labels = np.load(
        LABELS_FILE
    )

    return (
        detector,
        recognizer,
        prototypes,
        prototype_labels
    )


detector, recognizer, prototypes, prototype_labels = load_models()


# ============================================================
# INPUT
# ============================================================

uploaded_file = None
camera_image = None


if input_mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )


elif input_mode == "Webcam":

    camera_image = st.camera_input(
        "Take a picture"
    )


# ============================================================
# GET IMAGE
# ============================================================

image_bytes = None


if input_mode == "Upload Image" and uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()


elif input_mode == "Webcam" and camera_image is not None:

    image_bytes = camera_image.getvalue()


# ============================================================
# PROCESS IMAGE
# ============================================================

if image_bytes is not None:

    # --------------------------------------------------------
    # Convert bytes to NumPy array
    # --------------------------------------------------------

    file_bytes = np.asarray(
        bytearray(image_bytes),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Decode image using OpenCV
    # --------------------------------------------------------

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )


    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if image is None:

        st.error(
            "Could not read the image."
        )

    else:

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        results = detector.detect(image)

        output_image = image.copy()

        recognition_results = []


        # ----------------------------------------------------
        # Process each detected face
        # ----------------------------------------------------

        for result in results:

            x1, y1, x2, y2 = result["box"]

            detection_confidence = result["confidence"]

            landmarks = result["landmarks"]


            # ------------------------------------------------
            # Create SFace face information
            # ------------------------------------------------

            face = np.zeros(
                (1, 15),
                dtype=np.float32
            )


            face[0, 0:4] = [
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ]


            face[0, 4:14] = landmarks

            face[0, 14] = detection_confidence


            # ------------------------------------------------
            # Align face
            # ------------------------------------------------

            aligned_face = recognizer.alignCrop(
                image,
                face
            )


            # ------------------------------------------------
            # Extract SFace embedding
            # ------------------------------------------------

            feature = recognizer.feature(
                aligned_face
            )


            # ------------------------------------------------
            # Compare with prototypes
            # ------------------------------------------------

            similarities = []


            for prototype in prototypes:

                similarity = recognizer.match(
                    feature,
                    prototype.reshape(1, -1),
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                similarities.append(
                    float(similarity)
                )


            # ------------------------------------------------
            # Find best match
            # ------------------------------------------------

            best_index = int(
                np.argmax(similarities)
            )


            best_similarity = similarities[
                best_index
            ]


            predicted_name = str(
                prototype_labels[best_index]
            )


            # ------------------------------------------------
            # Unknown rejection
            # ------------------------------------------------

            if best_similarity >= SIMILARITY_THRESHOLD:

                display_name = predicted_name

            else:

                display_name = "Unknown"


            # ------------------------------------------------
            # Save recognition result
            # ------------------------------------------------

            recognition_results.append(
                {
                    "name": display_name,
                    "similarity": best_similarity,
                    "detection": detection_confidence
                }
            )


            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                output_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        output_image = cv2.cvtColor(
            output_image,
            cv2.COLOR_BGR2RGB
        )


        st.image(
            output_image,
            caption=f"Detected Faces: {len(results)}",
            width="stretch"
        )


        # ====================================================
        # RECOGNITION RESULTS
        # ====================================================

        st.subheader("Recognition Results")


        if len(recognition_results) == 0:

            st.warning(
                "No faces detected."
            )

        else:

            for i, result in enumerate(
                recognition_results,
                start=1
            ):

                name = result["name"]

                similarity = result["similarity"]

                detection = result["detection"]


                st.markdown(
                    f"### Face {i}: **{name}**"
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Similarity",
                        f"{similarity:.2f}"
                    )


                with col2:

                    st.metric(
                        "Detection Confidence",
                        f"{detection:.2f}"
                    )


                if name == "Unknown":

                    st.warning(
                        "This person is not in the known identities."
                    )

                else:

                    st.success(
                        f"Recognized as {name}"
                    )


        # ====================================================
        # SYSTEM INFORMATION
        # ====================================================

        with st.expander("System Information"):

            st.write(
                f"**Detected faces:** {len(results)}"
            )

            st.write(
                f"**Recognition threshold:** "
                f"{SIMILARITY_THRESHOLD:.2f}"
            )

            st.write(
                "**Recognition model:** SFace"
            )

            st.write(
                "**Face detector:** YuNet"
            )
# ============================================================
# ACCURACY VS EPOCHS
# ============================================================

st.subheader("Training Accuracy vs Epochs")

accuracy_plot = "results/accuracy_vs_epochs.png"

if os.path.exists(accuracy_plot):

    st.image(
        accuracy_plot,
        caption="Training and Validation Accuracy",
        width="stretch"
    )

else:

    st.warning(
        "Accuracy plot not found."
    )