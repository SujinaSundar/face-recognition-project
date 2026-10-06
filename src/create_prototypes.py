import numpy as np
import os


# ========================================
# 1. Load embeddings and labels
# ========================================

embeddings = np.load(
    "data/embeddings/train_embeddings.npy"
)

labels = np.load(
    "data/embeddings/train_labels.npy"
)


print("Embeddings:", embeddings.shape)
print("Labels:", labels.shape)


# ========================================
# 2. Find identities
# ========================================

identities = np.unique(labels)

print("Identities:", len(identities))


# ========================================
# 3. Create one prototype per identity
# ========================================

prototypes = []
prototype_labels = []


for identity in identities:

    # Select embeddings belonging
    # to this identity
    identity_embeddings = embeddings[
        labels == identity
    ]

    # Calculate mean embedding
    prototype = np.mean(
        identity_embeddings,
        axis=0
    )

    # Normalize prototype
    norm = np.linalg.norm(prototype)

    if norm > 0:
        prototype = prototype / norm

    prototypes.append(prototype)
    prototype_labels.append(identity)

    print(
        f"{identity}: "
        f"{len(identity_embeddings)} embeddings"
    )


# ========================================
# 4. Convert to NumPy arrays
# ========================================

prototypes = np.array(
    prototypes,
    dtype=np.float32
)

prototype_labels = np.array(
    prototype_labels
)


# ========================================
# 5. Save prototypes
# ========================================

os.makedirs(
    "data/embeddings",
    exist_ok=True
)

np.save(
    "data/embeddings/prototypes.npy",
    prototypes
)

np.save(
    "data/embeddings/prototype_labels.npy",
    prototype_labels
)


# ========================================
# 6. Print final result
# ========================================

print("\n========================================")
print("PROTOTYPES CREATED")
print("========================================")

print(
    "Prototype shape:",
    prototypes.shape
)

print(
    "Labels shape:",
    prototype_labels.shape
)

print(
    "\nSaved:"
)

print(
    "data/embeddings/prototypes.npy"
)

print(
    "data/embeddings/prototype_labels.npy"
)