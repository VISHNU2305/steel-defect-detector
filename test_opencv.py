import cv2
import numpy as np

# Create a blank black image
image = np.zeros((400, 400, 3), dtype=np.uint8)

# Draw a rectangle (simulating a defect bounding box)
cv2.rectangle(image, (100, 100), (300, 300), (0, 255, 0), 3)

# Add text
cv2.putText(image, "Defect Detected", (90, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Save the image instead of showing it
cv2.imwrite("test_output.jpg", image)
print("Image saved! Check your project1 folder for test_output.jpg")