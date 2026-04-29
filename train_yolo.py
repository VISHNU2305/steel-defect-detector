from ultralytics import YOLO

# Load a pretrained YOLOv8 small model
model = YOLO('yolov8s.pt')

# Train on our defect dataset
results = model.train(
    data=r"C:\Users\pendy\OneDrive\Desktop\project1\yolo_dataset\dataset.yaml",
    epochs=50,
    imgsz=200,
    batch=8,
    name='defect_detector',
    project=r"C:\Users\pendy\OneDrive\Desktop\project1\yolo_runs",
    patience=50,
    verbose=True
)

print("\nTraining complete!")
print(f"Best model saved at: {results.save_dir}")