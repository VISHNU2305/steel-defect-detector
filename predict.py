import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load the trained model
model = tf.keras.models.load_model("defect_classifier.h5")
defect_types = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Pick 6 random test images (one from each defect type)
test_path = r"C:\Users\pendy\OneDrive\Desktop\project1\defect_dataset\test"

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, defect in enumerate(defect_types):
    folder = os.path.join(test_path, defect)
    img_file = os.listdir(folder)[0]
    img_path = os.path.join(folder, img_file)

    # Preprocess same way as training
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64))
    img_equalized = cv2.equalizeHist(img_resized)
    img_normalized = img_equalized / 255.0
    img_input = img_normalized.reshape(1, 64, 64, 1)

    # Predict
    prediction = model.predict(img_input, verbose=0)
    predicted_class = defect_types[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    color = 'green' if predicted_class == defect else 'red'
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(
        f"Actual: {defect}\nPredicted: {predicted_class}\nConfidence: {confidence:.1f}%",
        color=color, fontsize=10)
    axes[i].axis("off")

plt.suptitle("Model Predictions on Test Images", fontsize=14)
plt.tight_layout()
plt.savefig("predictions.jpg")
print("Saved predictions.jpg!")