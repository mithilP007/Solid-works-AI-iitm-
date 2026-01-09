
import os
import random
import cv2
from ultralytics import YOLO

# Paths
VAL_IMG_DIR = r"d:\Solid works datasets datas\datasets\val\images"
VAL_LBL_DIR = r"d:\Solid works datasets datas\datasets\val\labels"
MODEL_PATH = r"d:\Solid works datasets datas\runs\detect\train_yolo11_advanced\weights\best.pt"

# Load Model
model = YOLO(MODEL_PATH)
native_names = model.names
print(f"Native Model Names: {native_names}")

# Get list of images
image_files = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith('.png')]
random.shuffle(image_files)
selected_files = image_files[:5]

print("\n--- ANALYSIS START ---\n")

for img_file in selected_files:
    base_name = os.path.splitext(img_file)[0]
    lbl_file = os.path.join(VAL_LBL_DIR, base_name + ".txt")
    img_path = os.path.join(VAL_IMG_DIR, img_file)
    
    print(f"Checking Image: {img_file}")
    
    # Check Ground Truth
    ground_truth_classes = []
    if os.path.exists(lbl_file):
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    ground_truth_classes.append(cls_id)
        
        gt_names = [native_names[cid] for cid in ground_truth_classes]
        print(f"  Ground Truth IDs: {ground_truth_classes}")
        print(f"  Ground Truth Names (using native mapping): {gt_names}")
    else:
        print("  No Label File Found")

    # Run Prediction
    results = model.predict(img_path, verbose=False)
    pred_classes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            pred_classes.append(cls_id)
            
    pred_names = [native_names[cid] for cid in pred_classes]
    print(f"  Model Predicted IDs: {pred_classes}")
    print(f"  Model Predicted Names (native): {pred_names}")

    # Check Swap Logic Effect
    swapped_names = []
    for cid in pred_classes:
        name = native_names[cid]
        # Simulate App Logic
        if cid == 0: name = 'nut'   # App swaps 0 -> nut
        if cid == 2: name = 'bolt'  # App swaps 2 -> bolt
        swapped_names.append(name)
    
    print(f"  App Logic Output would be: {swapped_names}")
    
    if ground_truth_classes == pred_classes:
        print("  Result: MATCH (Model aligns with Training Labels)")
        if swapped_names != pred_names:
            print("  CRITICAL: App Logic CHANGED the correct prediction! Swap might be wrong.")
    else:
        print("  Result: MISMATCH")

    print("-" * 30)

