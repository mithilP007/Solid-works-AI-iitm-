"""
Advanced YOLO11 Pipeline with TTA and Pseudo-Labeling
For SOLIDWORKS parts detection (bolt, locatingpin, nut, washer)
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import os
from tqdm import tqdm

# Configuration
BASE_DIR = Path(r"d:\Solid works datasets datas")
DATA_YAML = BASE_DIR / "data.yaml"
DATASET_DIR = BASE_DIR / "datasets"
TEST_IMG_DIR = Path(r"d:\Solid works datasets datas\test\test")
RUNS_DIR = BASE_DIR / "runs"

# Class mapping
CLASS_NAMES = ['bolt', 'locatingpin', 'nut', 'washer']

class CustomPipeline:
    def __init__(self, model_name="yolo11n.pt"):
        """
        Initialize YOLO11 pipeline
        Args:
            model_name: Model size or weights path
        """
        print(f"Initializing CustomPipeline with {model_name}")
        self.model = YOLO(model_name)
        self.model_name = model_name
        
    def train_advanced(self, data_yaml, epochs=30, imgsz=1024, batch=-1, device=0):
        """
        Advanced training with optimized hyperparameters for competition.
        Recommended per user: imgsz=1024, AdamW, mosaic=1.0, close_mosaic=10, box=7.5
        """
        print("\n" + "="*60)
        print("STARTING ADVANCED TRAINING")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Epochs: {epochs}")
        print(f"Image Size: {imgsz}")
        print(f"Batch Size: {batch}")
        print(f"Device: {device}")
        print("="*60 + "\n")
        
        # Train with optimized settings matching successful sibling runs
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            augment=True,
            mosaic=1.0,
            close_mosaic=10,         # Final 10 epochs without mosaic for edge sharpening
            mixup=0.1,              # Added mixup for better generalization
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            val=True,
            save=True,
            plots=True,
            project=str(RUNS_DIR / 'detect'),
            name='train_yolo11_advanced',
            exist_ok=True
        )
        
        print("\n✅ Training complete!")
        print(f"Best model saved at: {self.model.trainer.best}")
        return results
    
    def generate_pseudo_labels(self, unlabeled_path, conf_threshold=0.90, save_labels=True):
        """
        Generate pseudo-labels for unlabeled data
        Args:
            unlabeled_path: Path to unlabeled images
            conf_threshold: Confidence threshold for pseudo-labels
            save_labels: Whether to save label files
        """
        print("\n" + "="*60)
        print("GENERATING PSEUDO-LABELS")
        print("="*60)
        print(f"Unlabeled path: {unlabeled_path}")
        print(f"Confidence threshold: {conf_threshold}")
        print("="*60 + "\n")
        
        unlabeled_path = Path(unlabeled_path)
        images = list(unlabeled_path.glob("*.png"))
        
        print(f"Found {len(images)} images")
        
        pseudo_label_dir = unlabeled_path.parent / "labels"
        pseudo_label_dir.mkdir(exist_ok=True)
        
        labeled_count = 0
        total_objects = 0
        
        for img_path in tqdm(images, desc="Generating labels"):
            # Predict with high confidence threshold
            results = self.model.predict(
                str(img_path),
                conf=conf_threshold,
                verbose=False
            )
            
            if save_labels and len(results) > 0:
                result = results[0]
                
                # Check if any detections
                if result.boxes is not None and len(result.boxes) > 0:
                    # Save YOLO format labels
                    label_path = pseudo_label_dir / f"{img_path.stem}.txt"
                    
                    with open(label_path, 'w') as f:
                        for box in result.boxes:
                            # Get YOLO format coordinates
                            cls = int(box.cls[0])
                            x_center, y_center, width, height = box.xywhn[0].tolist()
                            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            total_objects += 1
                    
                    labeled_count += 1
        
        print(f"\n✅ Pseudo-labeling complete!")
        print(f"   Images with labels: {labeled_count}/{len(images)}")
        print(f"   Total objects detected: {total_objects}")
        print(f"   Labels saved to: {pseudo_label_dir}")
        
        return labeled_count, total_objects
    
    def predict_with_tta(self, source, conf=0.25, save=True):
        """
        Predict with Test-Time Augmentation for maximum accuracy
        Args:
            source: Image source (path or directory)
            conf: Confidence threshold
            save: Save visualizations
        """
        print("\n" + "="*60)
        print("PREDICTION WITH TTA")
        print("="*60)
        print(f"Source: {source}")
        print(f"Confidence: {conf}")
        print(f"TTA: Enabled")
        print("="*60 + "\n")
        
        # Predict with augmentation
        results = self.model.predict(
            source,
            augment=True,  # Enable TTA
            conf=conf,
            save=save,
            verbose=True
        )
        
        print(f"\n✅ Prediction complete! Processed {len(results)} images")
        return results
    
    def generate_submission(self, test_img_dir, output_csv="submission.csv", conf=0.25, use_tta=True):
        """
        Generate submission CSV from test images
        Args:
            test_img_dir: Directory with test images
            output_csv: Output CSV filename
            conf: Confidence threshold
            use_tta: Use Test-Time Augmentation
        """
        print("\n" + "="*60)
        print("GENERATING SUBMISSION")
        print("="*60)
        print(f"Test images: {test_img_dir}")
        print(f"Output CSV: {output_csv}")
        print(f"Confidence: {conf}")
        print(f"TTA: {use_tta}")
        print("="*60 + "\n")
        
        test_img_dir = Path(test_img_dir)
        images = sorted(list(test_img_dir.glob("*.png")))
        
        print(f"Found {len(images)} test images")
        
        # Prepare results dictionary
        results_dict = {
            'image_name': [],
            'bolt': [],
            'locatingpin': [],
            'nut': [],
            'washer': []
        }
        
        for img_path in tqdm(images, desc="Processing test images"):
            # Predict
            results = self.model.predict(
                str(img_path),
                augment=use_tta,
                conf=conf,
                verbose=False
            )
            
            # Count detections per class
            class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # bolt, locatingpin, nut, washer
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    class_counts[cls] += 1
            
            # Add to results
            results_dict['image_name'].append(img_path.name)
            results_dict['bolt'].append(class_counts[0])
            results_dict['locatingpin'].append(class_counts[1])
            results_dict['nut'].append(class_counts[2])
            results_dict['washer'].append(class_counts[3])
        
        # Create DataFrame and save
        df = pd.DataFrame(results_dict)
        output_path = BASE_DIR / output_csv
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Submission generated!")
        print(f"   Total images: {len(df)}")
        print(f"   Saved to: {output_path}")
        print(f"\nPreview:")
        print(df.head(10))
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Advanced Pipeline")
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'pseudo-label', 'predict', 'submit'],
                       help='Pipeline mode')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Model size (yolo11n/s/m/l/x.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                       help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--tta', action='store_true',
                       help='Enable Test-Time Augmentation')
    parser.add_argument('--weights', type=str, default=None,
                       help='Path to trained weights (for prediction/submission)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    if args.weights:
        pipeline = CustomPipeline(args.weights)
    else:
        pipeline = CustomPipeline(args.model)
    
    # Execute based on mode
    if args.mode == 'train':
        pipeline.train_advanced(
            DATA_YAML,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
        
    elif args.mode == 'pseudo-label':
        if not args.weights:
            print("⚠ Warning: Using base model for pseudo-labeling. Specify --weights for better results.")
        pipeline.generate_pseudo_labels(
            TEST_IMG_DIR,
            conf_threshold=0.90
        )
        
    elif args.mode == 'predict':
        pipeline.predict_with_tta(
            TEST_IMG_DIR,
            conf=args.conf,
            save=True
        )
        
    elif args.mode == 'submit':
        pipeline.generate_submission(
            TEST_IMG_DIR,
            output_csv="submission.csv",
            conf=args.conf,
            use_tta=args.tta
        )

if __name__ == "__main__":
    main()
