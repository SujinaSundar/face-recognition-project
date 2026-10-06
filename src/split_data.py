import os
import shutil


SOURCE_DIR = "data/raw/lfw"

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

TRAIN_IMAGES_PER_PERSON = 20
TEST_IMAGES_PER_PERSON = 5


# Create train and test directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


print("Starting train/test split...")
print("--------------------------------")


# Get all identity folders
people = sorted(os.listdir(SOURCE_DIR))


total_train = 0
total_test = 0


for person in people:

    source_person_dir = os.path.join(
        SOURCE_DIR,
        person
    )

    # Make sure it is a directory
    if not os.path.isdir(source_person_dir):
        continue


    # Get all image files
    images = sorted([
        file
        for file in os.listdir(source_person_dir)
        if file.lower().endswith(".jpg")
    ])


    # First 20 → training
    train_images = images[:TRAIN_IMAGES_PER_PERSON]

    # Last 5 → testing
    test_images = images[
        TRAIN_IMAGES_PER_PERSON:
        TRAIN_IMAGES_PER_PERSON + TEST_IMAGES_PER_PERSON
    ]


    # Create person's train directory
    train_person_dir = os.path.join(
        TRAIN_DIR,
        person
    )

    os.makedirs(
        train_person_dir,
        exist_ok=True
    )


    # Create person's test directory
    test_person_dir = os.path.join(
        TEST_DIR,
        person
    )

    os.makedirs(
        test_person_dir,
        exist_ok=True
    )


    # Copy training images
    for image in train_images:

        source_path = os.path.join(
            source_person_dir,
            image
        )

        destination_path = os.path.join(
            train_person_dir,
            image
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        total_train += 1


    # Copy testing images
    for image in test_images:

        source_path = os.path.join(
            source_person_dir,
            image
        )

        destination_path = os.path.join(
            test_person_dir,
            image
        )

        shutil.copy2(
            source_path,
            destination_path
        )

        total_test += 1


    print(
        f"{person}: "
        f"{len(train_images)} train, "
        f"{len(test_images)} test"
    )


print("\n================================")
print("Train/test split completed!")
print("================================")

print("Total training images:", total_train)
print("Total testing images:", total_test)
print("Total images:", total_train + total_test)