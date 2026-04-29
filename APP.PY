import os
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import base64

app = Flask(__name__)

# Load CNN model
cnn_model = tf.keras.models.load_model("defect_classifier.h5")
defect_types = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Load YOLO model
yolo_model = YOLO(r"C:\Users\pendy\OneDrive\Desktop\project1\yolo_runs\defect_detector-5\weights\best.pt")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # CNN Prediction
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_equalized = cv2.equalizeHist(img_resized)
    img_normalized = img_equalized / 255.0
    img_input = img_normalized.reshape(1, 64, 64, 1)
    prediction = cnn_model.predict(img_input, verbose=0)
    predicted_class = defect_types[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # YOLO Detection — draw bounding boxes
    results = yolo_model(img, verbose=False)
    annotated = results[0].plot()

    # Convert annotated image to base64 to send to frontend
    _, buffer = cv2.imencode('.jpg', annotated)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'defect_type': predicted_class,
        'confidence': round(confidence, 2),
        'annotated_image': annotated_b64
    })

if __name__ == '__main__':
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(debug=False)