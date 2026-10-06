import numpy as np
import matplotlib.pyplot as plt
import os


# ----------------------------------------
# 1. Load training history
# ----------------------------------------

train_accuracies = np.load(
    "results/train_accuracies.npy"
)

val_accuracies = np.load(
    "results/val_accuracies.npy"
)


# ----------------------------------------
# 2. Create epoch numbers
# ----------------------------------------

epochs = range(1, len(train_accuracies) + 1)


# ----------------------------------------
# 3. Create plot
# ----------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracies * 100,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracies * 100,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title("Face Recognition Accuracy vs Epochs")

plt.legend()
plt.grid(True)


# ----------------------------------------
# 4. Save plot
# ----------------------------------------

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/accuracy_vs_epochs.png",
    dpi=300,
    bbox_inches="tight"
)

print("Accuracy plot created successfully!")
print("Saved to: results/accuracy_vs_epochs.png")

plt.show()