\# Face Detection and Recognition System



An end-to-end face detection and recognition system built using OpenCV, YuNet, SFace, PyTorch, Scikit-learn, and Streamlit.



\## Project Overview



This project detects and recognizes multiple faces from images and webcam input.



The system uses:



\- \*\*YuNet\*\* for face detection

\- \*\*SFace\*\* for deep face feature extraction

\- \*\*128-dimensional face embeddings\*\* for face representation

\- \*\*Cosine similarity\*\* for identity matching

\- \*\*Prototype-based recognition\*\* for known/unknown face classification

\- \*\*Logistic Regression\*\* and \*\*PyTorch Neural Network\*\* classifiers

\- \*\*Streamlit\*\* for the user interface



\## Dataset



The system uses the \*\*Labeled Faces in the Wild (LFW)\*\* dataset.



\- 20 identities

\- 25 images per identity

\- Total: 500 images

\- Training: 400 images

\- Testing: 100 images



\### Data Augmentation



Training images were augmented using:



\- Rotation

\- Horizontal flipping

\- Scaling

\- Brightness and contrast adjustment



The augmented training set contains approximately 2,000 images.



\## System Workflow



```text

Dataset

&#x20;  ↓

Train / Test Split

&#x20;  ↓

Data Augmentation

&#x20;  ↓

YuNet Face Detection

&#x20;  ↓

Face Alignment

&#x20;  ↓

SFace Feature Extraction

&#x20;  ↓

128-Dimensional Face Embedding

&#x20;  ↓

Prototype Generation

&#x20;  ↓

Cosine Similarity

&#x20;  ↓

Known / Unknown Recognition

&#x20;  ↓

Streamlit Interface

