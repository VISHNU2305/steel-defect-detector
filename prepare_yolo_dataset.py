import os
import shutil
import random

# Paths
train_path = r"C:\Users\pendy\OneDrive\Desktop\project1\defect_dataset\train"
output_path = r"C:\Users\pendy\OneDrive\Desktop\project1\yolo_dataset"

defect_types = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Create YOLO folder structure
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', split), exist_ok=True)

print("Folders created!")

# Copy images and create label files
for class_id, defect in enumerate(defect_types):
    folder = os.path.join(train_path, defect)
    images = os.listdir(folder)
    random.shuffle(images)

    # 80% train, 20% val
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for split, img_list in [('train', train_images), ('val', val_images)]:
        for img_file in img_list:
            # Copy image
            src = os.path.join(folder, img_file)
            dst = os.path.join(output_path, 'images', split, img_file)
            shutil.copy(src, dst)

            # Create label file (YOLO format: class x_center y_center width height)
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.bmp', '.txt')
            label_path = os.path.join(output_path, 'labels', split, label_file)
            with open(label_path, 'w') as f:
                # Full image bounding box (normalized)
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("Images and labels created!")

# Create dataset.yaml config file
yaml_content = f"""path: {output_path}
train: images/train
val: images/val

nc: 6
names: ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
"""

yaml_path = os.path.join(output_path, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print("dataset.yaml created!")
print(f"\nYOLO dataset ready at: {output_path}")