import torch
import torch.nn as nn


# ----------------------------------------
# 1. Define the same network architecture
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
# 2. Create model
# ----------------------------------------

model = FaceClassifier(
    input_size=128,
    num_classes=20
)


# ----------------------------------------
# 3. Load trained weights
# ----------------------------------------

model.load_state_dict(
    torch.load(
        "models/face_nn_classifier.pth",
        map_location="cpu"
    )
)


# ----------------------------------------
# 4. Evaluation mode
# ----------------------------------------

model.eval()


print("Saved neural network loaded successfully!")
print("Input features:", 128)
print("Output classes:", 20)
print("Model is ready for evaluation!")