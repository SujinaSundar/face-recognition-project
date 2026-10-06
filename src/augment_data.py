import os
import cv2
import albumentations as A


TRAIN_DIR = "data/train"
AUGMENTED_DIR = "data/train_augmented"

AUGMENTATIONS_PER_IMAGE = 4


# Create output directory
os.makedirs(
    AUGMENTED_DIR,
    exist_ok=True
)


# Augmentation pipeline
transform = A.Compose([
    A.Rotate(
        limit=15,
        p=0.7
    ),

    A.HorizontalFlip(
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),

    A.Affine(
        scale=(0.9, 1.1),
        p=0.5
    )
])


print("Starting data augmentation...")
print("--------------------------------")


total_images = 0


# Get all people
people = sorted(
    os.listdir(TRAIN_DIR)
)


for person in people:

    person_dir = os.path.join(
        TRAIN_DIR,
        person
    )

    if not os.path.isdir(person_dir):
        continue


    # Create person's augmented directory
    output_person_dir = os.path.join(
        AUGMENTED_DIR,
        person
    )

    os.makedirs(
        output_person_dir,
        exist_ok=True
    )


    # Get training images
    images = sorted([
        file
        for file in os.listdir(person_dir)
        if file.lower().endswith(".jpg")
    ])


    for image_name in images:

        image_path = os.path.join(
            person_dir,
            image_name
        )

        image = cv2.imread(
            image_path
        )


        if image is None:
            continue


        # Save original image
        original_output = os.path.join(
            output_person_dir,
            image_name
        )

        cv2.imwrite(
            original_output,
            image
        )


        # Create augmented images
        base_name = os.path.splitext(
            image_name
        )[0]


        for aug_number in range(
            1,
            AUGMENTATIONS_PER_IMAGE + 1
        ):

            augmented = transform(
                image=image
            )

            augmented_image = augmented[
                "image"
            ]


            augmented_name = (
                f"{base_name}_aug_{aug_number:02d}.jpg"
            )


            augmented_path = os.path.join(
                output_person_dir,
                augmented_name
            )


            cv2.imwrite(
                augmented_path,
                augmented_image
            )


            total_images += 1


    print(
        f"{person}: "
        f"{len(images)} original images "
        f"→ {len(images) * 5} total images"
    )


print("\n================================")
print("Data augmentation completed!")
print("================================")

print(
    "Augmented training images:",
    total_images
)

print(
    "Original + augmented images:",
    total_images + 400
)