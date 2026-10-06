# Face Detection and Recognition System

An end-to-end face detection and recognition system built with **OpenCV, YuNet, SFace, PyTorch, Scikit-learn, and Streamlit**.

## Overview

This project detects and recognizes multiple faces from uploaded images and webcam captures.

**Key components**

- **YuNet** – face and facial landmark detection
- **SFace** – deep face feature extraction
- **128-dimensional embeddings** – compact face representation
- **Cosine similarity** – identity matching
- **Prototype-based recognition** – known / unknown face classification
- **Logistic Regression** and **PyTorch neural network** – embedding classifiers
- **Streamlit** – interactive user interface

## Dataset

The system uses the **Labeled Faces in the Wild (LFW)** dataset.

| Item | Value |
|---|---|
| Identities | 20 |
| Images per identity | 25 |
| Total images | 500 |
| Training images | 400 |
| Testing images | 100 |

### Data Augmentation

Training images were augmented with:

- Rotation
- Horizontal flipping
- Scaling
- Brightness and contrast adjustment

The augmented training set contains approximately **2,000 images**.

## System Workflow

```text
Dataset
   ↓
Train / Test Split
   ↓
Data Augmentation
   ↓
YuNet Face Detection
   ↓
Face Alignment
   ↓
SFace Feature Extraction
   ↓
128-Dimensional Face Embedding
   ↓
Prototype Generation
   ↓
Cosine Similarity
   ↓
Known / Unknown Recognition
   ↓
Streamlit Interface
```

## Model Architecture

### Face Detection

YuNet detects faces and facial landmarks in the input image.

### Face Recognition

SFace generates a 128-dimensional feature vector for each detected face.

### Identity Matching

1. A **prototype embedding** is created for each identity by averaging that identity's training embeddings.
2. The embedding of an input face is compared with all identity prototypes using **cosine similarity**.
3. If the best similarity is at or above the threshold of **0.50**, the face is labeled with that identity; otherwise it is labeled **Unknown**.

The 0.50 threshold was selected by validating genuine and impostor similarity scores.

## Model Evaluation

### Logistic Regression

| Metric | Result |
|---|---|
| Training accuracy | 100% |
| Test accuracy | 100% (on 99 successfully processed test images) |

One test image failed during face detection and was excluded from evaluation.

### Neural Network (PyTorch)

A neural network classifier was trained on the SFace embeddings.

| Setting | Value |
|---|---|
| Input | 128 features |
| Hidden layer | 64 neurons |
| Output | 20 identities |
| Training epochs | 20 |
| Validation accuracy | 94.81% |
| Test accuracy | 96.97% (on 99 successfully processed test images) |

The training vs. validation accuracy plot is available at
[`results/accuracy_vs_epochs.png`](results/accuracy_vs_epochs.png).

## Streamlit Application

The web app provides:

- Image upload
- Webcam image capture
- Face detection
- Identity recognition
- Similarity score display
- Known / Unknown classification

## Technologies

- Python
- OpenCV
- YuNet
- SFace
- PyTorch
- Scikit-learn
- NumPy
- Pandas
- Albumentations
- Matplotlib
- Streamlit

## Project Structure

```text
face-recognition-project/
│
├── app.py
├── README.md
├── .gitignore
│
├── src/
│   ├── detector.py
│   ├── collect_data.py
│   ├── create_lfw_dataset.py
│   ├── split_data.py
│   ├── augment_data.py
│   ├── extract_embeddings.py
│   ├── create_prototypes.py
│   ├── train_classifier.py
│   ├── train_nn_classifier.py
│   ├── evaluate_model.py
│   ├── evaluate_nn.py
│   └── webcam_recognition.py
│
└── results/
    ├── accuracy_vs_epochs.png
    └── single_image_result.jpg
```

## Installation

Create and activate a Python virtual environment, then install the dependencies:

```bash
pip install opencv-python numpy pandas scikit-learn matplotlib albumentations torch streamlit joblib
```

## Running the Application

From the project root:

```bash
streamlit run app.py
```

The application will open in your browser.

## Important Note

The dataset, pretrained model files (YuNet and SFace), embeddings, and trained model artifacts are excluded from this repository via `.gitignore`. This keeps the repository lightweight and avoids committing large generated files.

To run the project locally, you will need to download the YuNet and SFace model files, prepare the dataset, and regenerate the embeddings, prototypes, and classifiers using the scripts in `src/`.

## Future Improvements

- Real-time video stream recognition
- Face tracking
- Liveness detection
- Age and gender estimation
- Larger and more diverse datasets
- Improved unknown-face rejection
