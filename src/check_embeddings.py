import numpy as np
from collections import Counter


EMBEDDINGS_PATH = "data/embeddings/train_embeddings.npy"
LABELS_PATH = "data/embeddings/train_labels.npy"


print("Loading embeddings...")
print("--------------------------------")


# Load saved data
embeddings = np.load(
    EMBEDDINGS_PATH
)

labels = np.load(
    LABELS_PATH
)


print("Embeddings loaded!")
print("Embedding shape:", embeddings.shape)
print("Labels shape:", labels.shape)


# Check NaN values
nan_count = np.isnan(
    embeddings
).sum()


# Check infinite values
inf_count = np.isinf(
    embeddings
).sum()


print("\nData validation:")
print("--------------------------------")

print(
    "NaN values:",
    nan_count
)

print(
    "Infinite values:",
    inf_count
)


# Number of identities
unique_labels = np.unique(
    labels
)


print(
    "Number of identities:",
    len(unique_labels)
)


# Class distribution
counts = Counter(
    labels
)


print("\nImages per identity:")
print("--------------------------------")


for identity, count in sorted(
    counts.items()
):
    print(
        f"{identity}: {count}"
    )


# Embedding statistics
print("\nEmbedding statistics:")
print("--------------------------------")

print(
    "Minimum value:",
    embeddings.min()
)

print(
    "Maximum value:",
    embeddings.max()
)

print(
    "Mean value:",
    embeddings.mean()
)

print(
    "Standard deviation:",
    embeddings.std()
)


print("\n================================")
print("Embedding validation completed!")
print("================================")