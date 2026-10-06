import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ----------------------------------------
# 1. Load embeddings
# ----------------------------------------

X = np.load("data/embeddings/train_embeddings.npy")
labels = np.load("data/embeddings/train_labels.npy")

print("Embeddings loaded!")
print("X shape:", X.shape)
print("Labels shape:", labels.shape)


# ----------------------------------------
# 2. Convert identity names to numbers
# ----------------------------------------

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

print("\nNumber of identities:", len(label_encoder.classes_))
print("Classes:")
print(label_encoder.classes_)


# ----------------------------------------
# 3. Train-validation split
# ----------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nData split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ----------------------------------------
# 4. Convert NumPy arrays to PyTorch tensors
# ----------------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

print("\nTensor shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ----------------------------------------
# 5. Use CPU
# ----------------------------------------

device = torch.device("cpu")

print("\nDevice:", device)
# ----------------------------------------
# 6. Define neural network
# ----------------------------------------

class FaceClassifier(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ----------------------------------------
# 7. Create model
# ----------------------------------------

input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)

model = FaceClassifier(
    input_size=input_size,
    num_classes=num_classes
)

model = model.to(device)

print("\nNeural network created!")
print(model)

# ----------------------------------------
# 8. Loss function
# ----------------------------------------

criterion = nn.CrossEntropyLoss()


# ----------------------------------------
# 9. Optimizer
# ----------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


print("\nLoss function:")
print(criterion)

print("\nOptimizer:")
print(optimizer)

# ----------------------------------------
# 10. Training settings
# ----------------------------------------

epochs = 20

train_accuracies = []
val_accuracies = []
train_losses = []


# ----------------------------------------
# 11. Train the model
# ----------------------------------------

for epoch in range(epochs):

    # Training mode
    model.train()

    # Forward pass
    outputs = model(X_train)

    # Calculate loss
    loss = criterion(outputs, y_train)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model weights
    optimizer.step()

    # ------------------------------------
    # Calculate training accuracy
    # ------------------------------------

    predicted = torch.argmax(outputs, dim=1)

    train_accuracy = (
        (predicted == y_train).float().mean().item()
    )

    # ------------------------------------
    # Validation
    # ------------------------------------

    model.eval()

    with torch.no_grad():

        val_outputs = model(X_val)

        val_predicted = torch.argmax(
            val_outputs,
            dim=1
        )

        val_accuracy = (
            (val_predicted == y_val)
            .float()
            .mean()
            .item()
        )

    # Save metrics
    train_losses.append(loss.item())
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Print results
    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {loss.item():.4f} "
        f"Train Accuracy: {train_accuracy * 100:.2f}% "
        f"Validation Accuracy: {val_accuracy * 100:.2f}%"
    )
# ----------------------------------------
# 12. Save trained model
# ----------------------------------------

import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

torch.save(
    model.state_dict(),
    "models/face_nn_classifier.pth"
)

np.save(
    "results/train_accuracies.npy",
    np.array(train_accuracies)
)

np.save(
    "results/val_accuracies.npy",
    np.array(val_accuracies)
)

np.save(
    "results/train_losses.npy",
    np.array(train_losses)
)

print("\nTraining completed!")
print("Model saved: models/face_nn_classifier.pth")
print("Training history saved in results/")