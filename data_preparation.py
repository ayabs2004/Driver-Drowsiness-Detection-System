"""
data_preparation.py – Dataset Preprocessing for Driver Drowsiness Detection
=============================================================================
Converts a YOLOv8-format dataset into Keras flow_from_directory structure.

Fixes applied:
  - face class is skipped (not used by the CNN)
  - no grayscale/CLAHE conversion — crops saved as real color (BGR)
  - valid_crops is a local variable (no cross-split state leak)
  - all 4 target classes are balanced to equal counts in train split
  - valid/test splits are never oversampled
"""

import os
import shutil
import cv2

SOURCE_DIR = r"c:\Users\aya\Documents\driver ai project\Driver Drowsiness.v4-640x640-no-augmentation.yolov8"
OUTPUT_DIR = r"c:\Users\aya\Documents\driver ai project\processed_data"

# YOLO class order from the dataset's classes.txt / data.yaml
# Index 2 ('face') is skipped during saving — we only train on eye/mouth classes
YOLO_CLASSES  = ['closed_eye', 'closed_mouth', 'face', 'open_eye', 'open_mouth']
TRAIN_CLASSES = ['closed_eye', 'closed_mouth', 'open_eye', 'open_mouth']  # 4 classes only


def process_data(split):
    image_dir = os.path.join(SOURCE_DIR, split, "images")
    label_dir = os.path.join(SOURCE_DIR, split, "labels")

    for cls in TRAIN_CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    if not os.path.exists(label_dir):
        print(f"Label directory {label_dir} not found. Skipping {split}.")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"\nProcessing {len(label_files)} files in '{split}'...")

    # FIX: local list — no cross-split state leak
    valid_crops = []

    for label_file in label_files:
        img_name = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            img_name = label_file.replace('.txt', '.png')
            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id   = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w        = float(parts[3])
            h        = float(parts[4])

            cls_name = YOLO_CLASSES[cls_id]

            # FIX: skip face class entirely — not a CNN target
            if cls_name == 'face':
                continue

            x1 = max(0, int((x_center - w / 2) * img_w))
            y1 = max(0, int((y_center - h / 2) * img_h))
            x2 = min(img_w, int((x_center + w / 2) * img_w))
            y2 = min(img_h, int((y_center + h / 2) * img_h))

            cw, ch = x2 - x1, y2 - y1
            if cw < 20 or ch < 20:
                continue

            aspect = cw / float(ch)
            if aspect < 0.3 or aspect > 3.5:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Blur filter on grayscale — but the saved crop stays in real color (BGR)
            gray_for_blur = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray_for_blur, cv2.CV_64F).var() < 80:
                continue

            # FIX: crop is real color, never converted to grayscale for saving
            valid_crops.append((cls_name, label_file.replace('.txt', ''), i, crop))

    # ── Oversampling: train split only ────────────────────────────────────
    class_crops = {cls: [] for cls in TRAIN_CLASSES}
    for cls_name, base_name, i, crop in valid_crops:
        if cls_name in class_crops:
            class_crops[cls_name].append((base_name, i, crop))

    counts = {cls: len(class_crops[cls]) for cls in TRAIN_CLASSES}
    print(f"  Raw counts: {counts}")

    if split == 'train':
        max_count = max(counts.values()) if counts else 0
        print(f"  Balancing all classes to {max_count} samples each.")
    else:
        max_count = None  # no oversampling for valid/test

    for cls_name, crops in class_crops.items():
        if not crops:
            continue

        if split == 'train' and max_count:
            # FIX: distribute copies evenly so total == max_count
            n = len(crops)
            base  = max_count // n
            extra = max_count % n
            copies_list = [base + (1 if idx < extra else 0) for idx in range(n)]
        else:
            copies_list = [1] * len(crops)

        for idx, (base_name, i, crop) in enumerate(crops):
            for copy_idx in range(copies_list[idx]):
                save_name = f"{base_name}_{i}_copy{copy_idx}.jpg"
                save_path = os.path.join(OUTPUT_DIR, split, cls_name, save_name)
                if copy_idx == 0:
                    cv2.imwrite(save_path, crop)          # original color
                else:
                    cv2.imwrite(save_path, cv2.flip(crop, 1))  # horizontally flipped

    final = {cls: len(os.listdir(os.path.join(OUTPUT_DIR, split, cls)))
             for cls in TRAIN_CLASSES
             if os.path.exists(os.path.join(OUTPUT_DIR, split, cls))}
    print(f"  Saved counts: {final}")


if __name__ == "__main__":
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing old data at: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    process_data("train")
    process_data("valid")
    process_data("test")
    print("\nData preparation complete.")