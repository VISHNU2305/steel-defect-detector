import os
import cv2
import matplotlib.pyplot as plt

train_path = r"C:\Users\pendy\OneDrive\Desktop\project1\defect_dataset\train"
test_path = r"C:\Users\pendy\OneDrive\Desktop\project1\defect_dataset\test"
valid_path = r"C:\Users\pendy\OneDrive\Desktop\project1\defect_dataset\valid"

defect_types = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Count images in each category
print("=" * 40)
print("IMAGE COUNT PER DEFECT TYPE")
print("=" * 40)

train_counts = []
for defect in defect_types:
    folder = os.path.join(train_path, defect)
    count = len(os.listdir(folder))
    train_counts.append(count)
    print(f"{defect}: {count} images")

print("=" * 40)
print(f"Total train images: {sum(train_counts)}")

# Check image size of first image
sample_folder = os.path.join(train_path, 'Crazing')
sample_img_path = os.path.join(sample_folder, os.listdir(sample_folder)[0])
sample_img = cv2.imread(sample_img_path)
print(f"\nImage size (H x W x Channels): {sample_img.shape}")

# Plot bar chart of image counts
plt.figure(figsize=(10, 5))
bars = plt.bar(defect_types, train_counts, color=['red','blue','green','orange','purple','cyan'])
plt.title("Number of Training Images per Defect Type", fontsize=14)
plt.xlabel("Defect Type")
plt.ylabel("Number of Images")

# Add count on top of each bar
for bar, count in zip(bars, train_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(count), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig("data_distribution.jpg")
print("\nSaved data_distribution.jpg — check your project1 folder!")