# 🔍 Steel Surface Defect Detection System

An AI-powered quality inspection system that automatically detects and classifies 
surface defects in steel manufacturing using Computer Vision and Deep Learning.

---

## 🎯 Problem Statement

In steel manufacturing, surface defects directly impact product quality and safety. 
Manual inspection is slow, inconsistent, and expensive. This system automates 
defect detection using AI — reducing inspection time and improving accuracy.

---

## 🏭 Real World Application

This type of system is used in industries like:
- Automotive manufacturing (Toyota, Denso, Honda)
- Steel production lines
- Quality control in electronics manufacturing (Sony, Hitachi)

---

## 🔬 Defect Types Detected

| Defect | Severity | Description |
|--------|----------|-------------|
| Crazing | Medium | Network of fine surface cracks |
| Inclusion | High | Foreign material embedded in steel |
| Patches | Medium | Irregular surface oxidation areas |
| Pitted | Low | Small holes from corrosion |
| Rolled-in Scale | High | Oxide scale pressed into surface |
| Scratches | Low | Linear marks from mechanical contact |

---

## 🧠 Model Architecture

### CNN Classifier
- 3 Convolutional layers with MaxPooling
- Dense layers with Dropout for regularization
- Trained on 1,656 images across 6 defect classes
- **Training Accuracy: 96.45%**
- **Validation Accuracy: 90.96%**

### YOLOv8 Object Detector
- Fine-tuned YOLOv8s pretrained model
- Detects and localizes defects with bounding boxes
- Real-time inference capability

---

## 📊 Dataset

- **Source:** NEU Metal Surface Defects Dataset
- **Total Images:** 1,656 training images
- **Classes:** 6 defect types
- **Image Size:** 200x200 pixels
- **Distribution:** Perfectly balanced (276 images per class)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow / Keras |
| Object Detection | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Analysis | NumPy, Matplotlib |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/VISHNU2305/steel-defect-detector
cd steel-defect-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the web application
```bash
python app.py
```

### 4. Open in browser

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 96.45% |
| Validation Accuracy | 90.96% |
| Real-world Test Confidence | 96.64% |
| Inference Time | < 1 second |

---

## 🇯🇵 Inspiration

This project is inspired by the Japanese manufacturing philosophy of 
**Monozukuri** (ものづくり) — the art of making things with precision, 
quality, and dedication. Zero-defect manufacturing is a core principle 
in Japanese industry, and this system aims to support that goal through AI.

---

## 👨‍💻 Authors

**Vishnu** — 3rd Year Computer Science Student, Jain University Bangalore  
**Meghana** — 3rd Year Computer Science Student, Jain University Bangalore
GitHub: github.com/Meghanamahanandareddy
