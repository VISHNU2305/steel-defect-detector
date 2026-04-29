import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

train_path = r"C:\Users\pendy\OneDrive\Desktop\project1\defect_dataset\train"

# Pick one sample image
sample_img_path = os.path.join(train_path, 'Scratches', 
                  os.listdir(os.path.join(train_path, 'Scratches'))[0])

# Read original
original = cv2.imread(sample_img_path)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Step 1 - Grayscale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Step 2 - Histogram Equalization (improves contrast)
equalized = cv2.equalizeHist(gray)

# Step 3 - Gaussian Blur (removes noise)
blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

# Step 4 - Edge Detection (highlights defect boundaries)
edges = cv2.Canny(blurred, 50, 150)

# Show all steps
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(original_rgb)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(gray, cmap='gray')
axes[1].set_title("Grayscale")
axes[1].axis("off")

axes[2].imshow(equalized, cmap='gray')
axes[2].set_title("Contrast Enhanced")
axes[2].axis("off")

axes[3].imshow(edges, cmap='gray')
axes[3].set_title("Edge Detection")
axes[3].axis("off")

plt.suptitle("Preprocessing Steps on Scratches Defect", fontsize=14)
plt.tight_layout()
plt.savefig("preprocessing_steps.jpg")
print("Saved preprocessing_steps.jpg!")