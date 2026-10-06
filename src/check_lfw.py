from sklearn.datasets import fetch_lfw_people
from collections import Counter


# Download / load LFW dataset
lfw = fetch_lfw_people(
    min_faces_per_person=20,
    resize=0.5
)


# Basic information
print("Dataset loaded successfully!")
print("--------------------------------")

print("Number of images:", len(lfw.images))

print("Number of identities:", len(lfw.target_names))

print("Image shape:", lfw.images.shape)

print("\nFirst 10 identities:")
print(lfw.target_names[:10])


# -----------------------------------------
# Count images per identity
# -----------------------------------------

counts = Counter(lfw.target)


print("\nImages per identity:")
print("--------------------------------")

for identity_id, count in counts.items():

    name = lfw.target_names[identity_id]

    print(
        f"{name}: {count} images"
    )
    