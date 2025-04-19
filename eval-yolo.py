import os
import time
import torch
import io
import contextlib
import re
from ultralytics import YOLO


def evaluate_map(model_path, data_yaml, split="test", imgsz=640, batch=1, device="cuda"):
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split=split, imgsz=imgsz, batch=batch, device=device)
    print("Available result keys:", results.results_dict.keys())
    map_50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
    map_50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0.0)
    return map_50, map_50_95


def evaluate_speed(model_path, image_dir, device="cuda"):
    model = YOLO(model_path)
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])
    start = time.time()
    for path in image_paths:
        _ = model(path, device=device)
    elapsed = time.time() - start
    avg_ms = (elapsed / len(image_paths)) * 1000
    return avg_ms


def evaluate_model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


def main():
    model_path = "runs/detect/custom_yolov8_script_train/weights/best.pt"
    data_yaml = "dataset_yolo/data.yaml"
    test_image_dir = "dataset_yolo/test/images"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nEvaluating YOLOv8 on test set...")

    # Accuracy
    map_50, map_50_95 = evaluate_map(model_path, data_yaml, split="test", device=device)

    # Speed
    avg_time = evaluate_speed(model_path, test_image_dir, device=device)

    # Size
    model_size_mb = evaluate_model_size(model_path)

    # Results
    print("\nYOLOv8 Evaluation Results:")
    print(f"mAP@0.5:         {map_50:.4f}")
    print(f"mAP@0.5:0.95:    {map_50_95:.4f}")
    print(f"Inference Speed: {avg_time:.2f} ms/image")
    print(f"Model File Size: {model_size_mb:.2f} MB")


if __name__ == "__main__":
    main()