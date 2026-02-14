import os
import random
import shutil

# Paths
base_dir = r"D:\summer project\crop_disease_detector\dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Ratio (keep 10% in test, move 90% to train)
test_keep_ratio = 0.1

# Loop through each class in test/
for class_name in os.listdir(test_dir):
    test_class_dir = os.path.join(test_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)

    # Ensure it's a directory (ignore files like .DS_Store)
    if not os.path.isdir(test_class_dir):
        continue

    os.makedirs(train_class_dir, exist_ok=True)

    # List all images in test class
    images = os.listdir(test_class_dir)
    random.shuffle(images)

    # How many to keep in test
    keep_count = int(len(images) * test_keep_ratio)

    # Move the rest to train
    for img in images[keep_count:]:
        src = os.path.join(test_class_dir, img)
        dst = os.path.join(train_class_dir, img)
        shutil.move(src, dst)

    print(f"✅ Class '{class_name}': kept {keep_count}, moved {len(images) - keep_count}")

print("🎯 Dataset rebalanced: Test set reduced to ~10%, rest moved to train.")
