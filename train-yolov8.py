# train_yolov8.py

from ultralytics import YOLO
import torch, gc
import os
from pathlib import Path

def check_dataset(image_dir, label_dir):
    print(f"🔍 Checking dataset at: {image_dir}")
    missing_labels = []
    for img_file in Path(image_dir).glob("*.*"):
        label_file = Path(label_dir) / (img_file.stem + ".txt")
        if not label_file.exists():
            missing_labels.append(img_file.name)
    if missing_labels:
        print(f"{len(missing_labels)} images are missing label files.")
        print("Examples:", missing_labels[:5])
    else:
        print("All images have corresponding label files.")

def main():
    # Step 1: Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Check dataset integrity
    check_dataset("dataset_yolo/train/images", "dataset_yolo/train/labels")
    check_dataset("dataset_yolo/valid/images", "dataset_yolo/valid/labels")
    check_dataset("dataset_yolo/test/images", "dataset_yolo/test/labels")

    # Step 3: Load and train YOLOv8
    model = YOLO("yolov8m.pt")

    model.train(
        data="dataset_yolo/data.yaml",
        epochs=100,
        imgsz=416,
        batch=1,
        device="cuda",
        workers=4,
        name="custom_yolov8_script_train"
    )

    print("Training complete!")
    print(f"Final GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

# Required on Windows to prevent multiprocessing crash
if __name__ == "__main__":
    main()
