from ultralytics import YOLO
import json
import os

model_path = r"d:\Solid works datasets datas\runs\detect\train_yolo11_advanced\weights\best.pt"
print(f"Checking model: {model_path}")

if os.path.exists(model_path):
    try:
        model = YOLO(model_path)
        print("\n--- Model Class Names ---")
        print(json.dumps(model.names, indent=4))
        
        # Explicit check for known indices
        print("\n--- Diagnostic Check ---")
        expected = {0: 'bolt', 1: 'locatingpin', 2: 'nut', 3: 'washer'}
        
        matches = True
        for id, name in expected.items():
            if model.names.get(id) != name:
                print(f"MISMATCH: ID {id} should be '{name}', but is '{model.names.get(id)}'")
                matches = False
            else:
                print(f"MATCH: ID {id} -> {name}")
                
        if matches:
            print("\n✅ Model structure matches standard mapping.")
        else:
            print("\n❌ Model structure DIFFERS from standard mapping.")

        with open('model_names.json', 'w') as f:
            json.dump(model.names, f)
            print("\nSaved to model_names.json")
            
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model not found at specified path.")
