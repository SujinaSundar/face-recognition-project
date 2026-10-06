from sklearn.datasets import fetch_lfw_people
from collections import Counter
import os
import cv2


# =====================================================
# 1. Load LFW dataset
# =====================================================

print("Loading LFW dataset...")

lfw = fetch_lfw_people(
    min_faces_per_person=20,
    resize=0.5
)

print("LFW loaded!")
print("Total images:", len(lfw.images))
print("Total identities:", len(lfw.target_names))


# =====================================================
# 2. Count images for each identity
# =====================================================

counts = Counter(lfw.target)


# =====================================================
# 3. Select identities with >= 25 images
# =====================================================

eligible_identities = [
    identity_id
    for identity_id, count in counts.items()
    if count >= 25
]


print("\nEligible identities:", len(eligible_identities))


# =====================================================
# 4. Sort identities by number of images
# =====================================================

eligible_identities.sort(
    key=lambda identity_id: counts[identity_id],
    reverse=True
)


# =====================================================
# 5. Select top 20 identities
# =====================================================

selected_identities = eligible_identities[:20]


print("\nSelected identities:")
print("--------------------------------")

for identity_id in selected_identities:

    name = lfw.target_names[identity_id]

    print(
        f"{name}: {counts[identity_id]} images"
    )


# =====================================================
# 6. Create output directory
# =====================================================

output_dir = os.path.join(
    "data",
    "raw",
    "lfw"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# =====================================================
# 7. Save 25 images per identity
# =====================================================

IMAGES_PER_PERSON = 25


print("\nCreating dataset...")
print("--------------------------------")


for identity_id in selected_identities:

    person_name = lfw.target_names[identity_id]

    # Make folder name safe for Windows
    safe_name = (
        person_name
        .replace(" ", "_")
        .replace(".", "")
    )

    person_dir = os.path.join(
        output_dir,
        safe_name
    )

    os.makedirs(
        person_dir,
        exist_ok=True
    )


    # Find images belonging to this person
    image_indices = [
        i
        for i, target in enumerate(lfw.target)
        if target == identity_id
    ]


    # Take first 25
    image_indices = image_indices[
        :IMAGES_PER_PERSON
    ]


    # Save images
    for count, index in enumerate(
        image_indices,
        start=1
    ):

        image = lfw.images[index]

        # Convert float image [0,1]
        # to uint8 [0,255]
        image = (
            image * 255
        ).astype("uint8")


        image_path = os.path.join(
            person_dir,
            f"{count:03d}.jpg"
        )


        cv2.imwrite(
            image_path,
            image
        )


    print(
        f"{person_name}: "
        f"{len(image_indices)} images saved"
    )


print("\n================================")
print("Dataset creation completed!")
print("================================")
print("Total identities:", len(selected_identities))
print(
    "Total images:",
    len(selected_identities) * IMAGES_PER_PERSON
)