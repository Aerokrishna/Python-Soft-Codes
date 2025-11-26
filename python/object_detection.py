#!/usr/bin/env python3
"""
yolov8_webcam.py
Live webcam object detection using YOLOv8 (ultralytics) + OpenCV.
Press 'q' to quit.
"""

import time
import argparse
import cv2
from ultralytics import YOLO
import numpy as np


def draw_boxes(frame, boxes, scores, classes, names):
    """Draw bounding boxes, labels and confidence on the frame."""
    for (xyxy, conf, cls) in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{names.get(int(cls), str(int(cls)))} {conf:.2f}"
        # box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def main(args):
    # Load YOLOv8 model
    model = YOLO(args.model)
    if args.device is not None:
        try:
            model.to(args.device)
        except Exception:
            pass

    # Open webcam
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera index.")
        return

    # FPS calculation
    prev_time = time.time()
    fps = 0.0
    smoothing = 0.9  # exponential moving average

    # Get class names
    names = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else {}
    if not names and hasattr(model, "names"):
        names = model.names

    print("Starting webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: empty frame received. Exiting.")
            break

        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO inference
        results = model(img_rgb, imgsz=args.imgsz, conf=args.conf, verbose=False)
        res = results[0]

        boxes, scores, classes = [], [], []
        if hasattr(res, "boxes") and res.boxes is not None:
            for box in res.boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                conf = float(box.conf.cpu().numpy().item())
                cls = int(box.cls.cpu().numpy().item())
                boxes.append(xyxy)
                scores.append(conf)
                classes.append(cls)

        # Draw detections on frame
        draw_boxes(frame, boxes, scores, classes, names)

        # Print detected object info to terminal
        if boxes:
            print("\nDetected objects:")
            for (xyxy, cls) in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, xyxy)
                obj_name = names.get(int(cls), f"Class {cls}")
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                print(f" {obj_name} POSE : {cx} {cy}")

        # FPS calculation
        now = time.time()
        instant_fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        fps = (smoothing * fps) + ((1 - smoothing) * instant_fps)

        # Show FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        # Display frame
        cv2.imshow("YOLOv8 Webcam", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExiting on user request.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 webcam object detection (OpenCV).")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="path to YOLOv8 model or model name (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default: 0.25)")
    parser.add_argument("--device", type=str, default=None,
                        help="device to run on, e.g. 'cpu' or 'cuda:0' (default: None -> auto)")
    args = parser.parse_args()
    main(args)
