import cv2
import matplotlib.pyplot as plt
import os

# Path to train folder
dataset_path = r"C:\Users\pendy\OneDrive\Desktop\project1\defect_dataset\train"

defect_types = os.listdir(dataset_path)
print("Defect types found:", defect_types)

# Show one sample image from each defect type
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, defect in enumerate(defect_types[:6]):
    folder = os.path.join(dataset_path, defect)
    sample_image_path = os.path.join(folder, os.listdir(folder)[0])

    img = cv2.imread(sample_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(defect, fontsize=14)
    axes[i].axis("off")

plt.suptitle("NEU Metal Surface Defects - All 6 Types", fontsize=16)
plt.tight_layout()
plt.savefig("defect_samples.jpg")
print("Saved! Check project1 folder for defect_samples.jpg")