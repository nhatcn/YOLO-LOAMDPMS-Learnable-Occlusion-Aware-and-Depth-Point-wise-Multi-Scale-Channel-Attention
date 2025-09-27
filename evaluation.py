import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from collections import defaultdict
import json
from sklearn.metrics import average_precision_score
from scipy import stats
import pandas as pd
import albumentations as A
import random

# ========== Custom YOLO Wrapper ==========
class CustomYOLOWithModules:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.names = self.model.names

    def predict(self, frame, imgsz=640, conf=0.25, verbose=False):
        return self.model.predict(frame, imgsz=imgsz, conf=conf, verbose=verbose)

# ========== Data Augmentation ==========
class DataAugmentation:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            ], p=0.7),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.MotionBlur(blur_limit=6, p=0.3),
            ], p=0.5),
            
            A.OneOf([
                A.RandomRotate90(p=0.2),
                A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.Affine(scale=0.9, translate_percent=0.1, rotate=10, 
                        mode=cv2.BORDER_CONSTANT, p=0.5),
            ], p=0.6),
            
            A.OneOf([
                A.HorizontalFlip(p=0.4),
                A.VerticalFlip(p=0.2),
            ], p=0.4),
            
            A.BBoxSafeRandomCrop(erosion_rate=0.0, p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['class_labels']))
    
    def augment_image_and_labels(self, image, bboxes, class_labels):
        """Apply augmentation to image and corresponding bounding boxes"""
        try:
            augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return augmented['image'], augmented['bboxes'], augmented['class_labels']
        except:
            # If augmentation fails, return original
            return image, bboxes, class_labels

# ========== Enhanced Evaluation with Statistical Tests ==========
class StatisticalModelEvaluator:
    def __init__(self, model_path, ground_truth_folder, output_folder="detected_images"):
        self.model = CustomYOLOWithModules(model_path)
        self.class_names = self.model.names
        self.ground_truth_folder = ground_truth_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Store metrics for each run
        self.reset_metrics()

    def reset_metrics(self):
        """Reset metrics for new evaluation run"""
        self.metrics = {
            'total_tp': 0, 'total_fp': 0, 'total_fn': 0,
            'total_predictions': 0, 'total_ground_truth': 0,
            'gt_confidences': [], 'gt_labels': [],
            'inference_times': [],
            'image_ious': {}
        }
        self.best_image = {'name': None, 'avg_iou': 0.0}

    def load_ground_truth(self, image_name):
        base_name = os.path.splitext(image_name)[0]
        gt_file = os.path.join(self.ground_truth_folder, base_name + ".txt")
        ground_truth = []
        if not os.path.exists(gt_file):
            return []

        with open(gt_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # Class mapping
                    cx, cy, w, h = map(float, parts[1:5])
                    ground_truth.append({'class_id': class_id, 'center_x': cx, 'center_y': cy, 'width': w, 'height': h})
        return ground_truth

    def yolo_to_xyxy(self, cx, cy, w, h, iw, ih):
        x1 = int((cx - w / 2) * iw)
        y1 = int((cy - h / 2) * ih)
        x2 = int((cx + w / 2) * iw)
        y2 = int((cy + h / 2) * ih)
        return x1, y1, x2, y2

    def calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def draw_boxes(self, frame, predictions):
        img = frame.copy()
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            class_name = self.class_names[pred['class_id']]
            conf = pred['confidence']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return img

    def evaluate_on_augmented_images(self, augmented_images_data, save_image=False, run_id=0):
        """Evaluate model on pre-augmented images"""
        self.reset_metrics()
        
        for image_name, (augmented_frame, original_gt) in augmented_images_data.items():
            h, w = augmented_frame.shape[:2]
            
            # Convert original ground truth to boxes
            gt_boxes = [{'bbox': self.yolo_to_xyxy(g['center_x'], g['center_y'], g['width'], g['height'], w, h),
                         'class_id': g['class_id']} for g in original_gt]

            # Measure inference time
            start_time = time.time()
            results = self.model.predict(augmented_frame)[0]
            inference_time = time.time() - start_time
            self.metrics['inference_times'].append(inference_time)

            predictions = [{'bbox': tuple(results.boxes.xyxy[i].int().tolist()),
                            'confidence': float(results.boxes.conf[i]),
                            'class_id': int(results.boxes.cls[i])}
                           for i in range(len(results.boxes))] if results.boxes is not None else []

            # Save detected image (only for first run to avoid clutter)
            if save_image and run_id == 0:
                detected_img = self.draw_boxes(augmented_frame, predictions)
                output_path = os.path.join(self.output_folder, f"detected_{image_name}")
                cv2.imwrite(output_path, detected_img)

            # Calculate metrics for this frame
            self._calculate_frame_metrics(predictions, gt_boxes, image_name)
        
        return self.calculate_final_metrics()

    def _calculate_frame_metrics(self, predictions, gt_boxes, image_name, iou_thres=0.5):
        """Calculate TP, FP, FN for a single frame"""
        frame_tp = 0
        frame_fp = 0
        frame_ious = []
        matched_gt = [False] * len(gt_boxes)

        for pred in predictions:
            best_iou = 0
            best_gt_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if matched_gt[idx] or pred['class_id'] != gt['class_id']:
                    continue
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou >= iou_thres:
                frame_tp += 1
                matched_gt[best_gt_idx] = True
                frame_ious.append(best_iou)
                self.metrics['gt_confidences'].append(pred['confidence'])
                self.metrics['gt_labels'].append(1)
            else:
                frame_fp += 1
                self.metrics['gt_confidences'].append(pred['confidence'])
                self.metrics['gt_labels'].append(0)

        frame_fn = len(gt_boxes) - sum(matched_gt)

        # Update metrics
        self.metrics['total_tp'] += frame_tp
        self.metrics['total_fp'] += frame_fp
        self.metrics['total_fn'] += frame_fn
        self.metrics['total_predictions'] += len(predictions)
        self.metrics['total_ground_truth'] += len(gt_boxes)

        # Track best image
        avg_iou = sum(frame_ious) / len(frame_ious) if frame_ious else 0.0
        self.metrics['image_ious'][image_name] = avg_iou
        if avg_iou > self.best_image['avg_iou']:
            self.best_image = {'name': image_name, 'avg_iou': avg_iou}

    def calculate_final_metrics(self):
        tp = self.metrics['total_tp']
        fp = self.metrics['total_fp']
        fn = self.metrics['total_fn']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        try:
            map_score = average_precision_score(self.metrics['gt_labels'], self.metrics['gt_confidences'])
        except:
            map_score = 0.0

        # Calculate speed metrics
        inference_times = self.metrics['inference_times']
        avg_inference_time = sum(inference_times) / len(inference_times) * 1000 if inference_times else 0
        fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0

        return {
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'map_score': map_score,
            'tp': tp, 'fp': fp, 'fn': fn,
            'total_predictions': self.metrics['total_predictions'],
            'total_ground_truth': self.metrics['total_ground_truth'],
            'avg_inference_time_ms': avg_inference_time,
            'fps': fps,
            'best_detected_image': self.best_image
        }

# ========== Augmented Dataset Generator ==========
def generate_augmented_datasets(image_folder, ground_truth_folder, n_runs=20):
    """
    Pre-generate augmented datasets for fair comparison
    Each run will have the same augmented images for all models
    """
    print(f"🔄 Pre-generating {n_runs} augmented datasets...")
    
    # Get list of images
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])
    
    augmented_datasets = []
    
    for run_id in range(n_runs):
        print(f"  📁 Generating dataset {run_id + 1}/{n_runs}")
        
        # Initialize augmentation with specific seed for this run
        augmenter = DataAugmentation(seed=run_id * 42)
        
        # Store augmented data for this run
        run_data = {}
        
        for image_name in image_files:
            img_path = os.path.join(image_folder, image_name)
            frame = cv2.imread(img_path)
            
            if frame is not None:
                # Load original ground truth
                original_gt = load_ground_truth_for_image(ground_truth_folder, image_name)
                
                if len(original_gt) > 0:
                    # Convert ground truth to albumentations format
                    bboxes = [(g['center_x'], g['center_y'], g['width'], g['height']) for g in original_gt]
                    class_labels = [g['class_id'] for g in original_gt]
                    
                    # Apply augmentation
                    augmented_frame, augmented_bboxes, augmented_labels = augmenter.augment_image_and_labels(
                        frame, bboxes, class_labels
                    )
                    
                    # Convert back to our format
                    augmented_gt = []
                    for i, (bbox, cls) in enumerate(zip(augmented_bboxes, augmented_labels)):
                        augmented_gt.append({
                            'class_id': cls,
                            'center_x': bbox[0],
                            'center_y': bbox[1], 
                            'width': bbox[2],
                            'height': bbox[3]
                        })
                else:
                    # No ground truth, just augment image
                    augmented_frame, _, _ = augmenter.augment_image_and_labels(frame, [], [])
                    augmented_gt = original_gt
                
                # Store augmented data
                run_data[image_name] = (augmented_frame, augmented_gt)
        
        augmented_datasets.append(run_data)
    
    print(f"✅ Generated {len(augmented_datasets)} augmented datasets")
    return augmented_datasets

def load_ground_truth_for_image(ground_truth_folder, image_name):
    """Helper function to load ground truth for a single image"""
    base_name = os.path.splitext(image_name)[0]
    gt_file = os.path.join(ground_truth_folder, base_name + ".txt")
    ground_truth = []
    
    if not os.path.exists(gt_file):
        return []

    with open(gt_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                # Class mapping
                if class_id == 3:
                    class_id = 1
                elif class_id == 8:
                    class_id = 3
                elif class_id == 6:
                    class_id = 0
                cx, cy, w, h = map(float, parts[1:5])
                ground_truth.append({'class_id': class_id, 'center_x': cx, 'center_y': cy, 'width': w, 'height': h})
    
    return ground_truth

# ========== Statistical Testing Functions ==========
def perform_statistical_tests(baseline_runs, comparison_runs, method_name="Comparison"):
    """Perform statistical significance tests between baseline and comparison methods"""
    results = {}
    
    print(f"\n====== STATISTICAL SIGNIFICANCE TESTS: {method_name} vs Baseline ======")
    # print(f"Number of paired runs compared: {len(baseline_runs)}")
    print("Weather Condition: Fog")
    # Perform paired t-tests for each metric
    metrics_to_test = ['precision', 'recall', 'f1_score', 'map_score']
    
    for metric in metrics_to_test:
        baseline_values = np.array([run[metric] for run in baseline_runs])
        comparison_values = np.array([run[metric] for run in comparison_runs])
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(comparison_values, baseline_values)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(comparison_values, baseline_values, 
                                                       alternative='two-sided', zero_method='wilcox')
        except:
            wilcoxon_stat, wilcoxon_p = 0, 1.0
        
        # Effect size (Cohen's d for paired samples)
        diff = comparison_values - baseline_values
        cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        # Mean improvement
        baseline_mean = np.mean(baseline_values)
        comparison_mean = np.mean(comparison_values)
        improvement = ((comparison_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
        
        # Confidence interval for the difference
        diff_mean = np.mean(diff)
        diff_std = np.std(diff, ddof=1)
        n = len(diff)
        t_crit = stats.t.ppf(0.975, n-1)  # 95% CI
        ci_lower = diff_mean - t_crit * (diff_std / np.sqrt(n))
        ci_upper = diff_mean + t_crit * (diff_std / np.sqrt(n))
        
        results[metric] = {
            'baseline_mean': baseline_mean,
            'baseline_std': np.std(baseline_values),
            'comparison_mean': comparison_mean,
            'comparison_std': np.std(comparison_values),
            'improvement_percent': improvement,
            't_statistic': t_stat,
            'p_value': p_value,
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_p_value': wilcoxon_p,
            'cohen_d': cohen_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01
        }
        
        # Print results
        print(f"\n{metric.upper()}:")
        print(f"  Baseline: {baseline_mean:.4f} ± {np.std(baseline_values):.4f}")
        print(f"  {method_name}: {comparison_mean:.4f} ± {np.std(comparison_values):.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")
        print(f"  Effect size (Cohen's d): {cohen_d:.4f}")
        print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")
    
    return results

def create_significance_table(statistical_comparisons):
    """Create a simple text table showing metrics, improvements, p-values, and significance."""
    lines = []
    lines.append("Statistical Significance of Model Improvements (Paired Evaluation)")
    lines.append("=" * 80)
    lines.append(f"{'Method':<20} {'Metric':<12} {'Baseline':<10} {'Comparison':<10} {'Improvement(%)':<15} {'p-value':<12} {'Significant':<12}")
    lines.append("-" * 80)
    
    for method_name, metrics in statistical_comparisons.items():
        for metric_name, stats in metrics.items():
            signif = "*" if stats['significant_005'] else ""
            if stats['significant_001']:
                signif = "**"
            if not signif:
                signif = "No"
            lines.append(
                f"{method_name:<20} {metric_name:<12} "
                f"{stats['baseline_mean']:<10.4f} {stats['comparison_mean']:<10.4f} "
                f"{stats['improvement_percent']:+<15.2f} {stats['p_value']:<12.6f} {signif:<12}"
            )
        lines.append("-" * 80)
    
    lines.append("\nNote: * p < 0.05, ** p < 0.01")
    lines.append("Each model evaluated on IDENTICAL augmented datasets for fair comparison")
    
    return "\n".join(lines)

def save_statistical_summary(statistical_comparisons, output_file="statistical_summary_fair.txt"):
    """Save a summary of statistical test results"""
    
    with open(output_file, 'w') as f:
        f.write("FAIR PAIRED STATISTICAL SIGNIFICANCE SUMMARY\n")
        f.write("="*50 + "\n")
        f.write("All models evaluated on IDENTICAL augmented datasets\n")
        f.write("This ensures fair paired comparison\n\n")
        
        for method_name, results in statistical_comparisons.items():
            f.write(f"{method_name} vs Baseline:\n")
            f.write("-" * 30 + "\n")
            
            for metric, stats in results.items():
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Baseline: {stats['baseline_mean']:.4f} ± {stats['baseline_std']:.4f}\n")
                f.write(f"  Comparison: {stats['comparison_mean']:.4f} ± {stats['comparison_std']:.4f}\n")
                f.write(f"  Improvement: {stats['improvement_percent']:+.2f}%\n")
                f.write(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")
                f.write(f"  p-value: {stats['p_value']:.6f}\n")
                f.write(f"  Effect size (Cohen's d): {stats['cohen_d']:.4f}\n")
                f.write(f"  Significant (p<0.05): {'Yes' if stats['significant_005'] else 'No'}\n")
                f.write(f"  Significant (p<0.01): {'Yes' if stats['significant_001'] else 'No'}\n\n")
            f.write("\n")
    
    print(f"Statistical summary saved to: {output_file}")

# ========== Main Fair Evaluation Function ==========
def run_fair_statistical_evaluation_with_augmentation(n_runs=20):
    """
    Run FAIR evaluation with data augmentation and statistical testing
    All models are evaluated on the SAME augmented datasets for each run
    """
    
    # Define model configurations
    models = {
        "YOLOv8(Baseline)": "bestv8m.pt",
        "YOLOv8+DPMS": "bestDPMS.pt", 
        "YOLOv8+LOAM": "bestloama0.7.pt",
        "YOLOv8+LOAM+DPMS": "besta0.7.pt"
    }
    
    image_folder = "train/images"
    ground_truth_folder = "train/labels"
    
    # Pre-generate augmented datasets for fair comparison
    augmented_datasets = generate_augmented_datasets(image_folder, ground_truth_folder, n_runs)
    
    # Store all results for each model and each run
    all_results = {}
    
    # Evaluate each model on the SAME augmented datasets
    for method_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skipping {method_name}")
            continue
            
        print(f"\n🚀 Evaluating {method_name} on {n_runs} pre-generated augmented datasets...")
        
        output_folder = f"detected_images_{method_name.replace('+', '_').replace('(', '_').replace(')', '_')}"
        evaluator = StatisticalModelEvaluator(model_path, ground_truth_folder, output_folder=output_folder)
        
        # Store results for each run
        method_runs = []
        
        # Evaluate on each pre-generated augmented dataset
        for run_id, augmented_data in enumerate(augmented_datasets):
            print(f"  📊 Run {run_id + 1}/{n_runs}")
            
            # Evaluate on this specific augmented dataset
            run_metrics = evaluator.evaluate_on_augmented_images(
                augmented_data, 
                save_image=(run_id == 0), 
                run_id=run_id
            )
            
            method_runs.append(run_metrics)
            print(f"    P: {run_metrics['precision']:.4f}, R: {run_metrics['recall']:.4f}, F1: {run_metrics['f1_score']:.4f}, mAP: {run_metrics['map_score']:.4f}")
        
        # Store all runs for this method
        all_results[method_name] = method_runs
        
        # Calculate summary statistics
        precisions = [run['precision'] for run in method_runs]
        recalls = [run['recall'] for run in method_runs]
        f1_scores = [run['f1_score'] for run in method_runs]
        map_scores = [run['map_score'] for run in method_runs]
        
        print(f"✅ {method_name} Summary ({n_runs} runs):")
        print(f"   Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"   Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"   F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"   mAP: {np.mean(map_scores):.4f} ± {np.std(map_scores):.4f}")
    
    # Perform statistical tests
    if "YOLOv8(Baseline)" in all_results:
        baseline_runs = all_results["YOLOv8(Baseline)"]
        statistical_comparisons = {}
        
        for method_name, method_runs in all_results.items():
            if method_name != "YOLOv8(Baseline)":
                statistical_comparisons[method_name] = perform_statistical_tests(
                    baseline_runs, method_runs, method_name
                )
        
        # Generate text table with significance
        significance_table = create_significance_table(statistical_comparisons)
        
        # Save statistical results
        with open("statistical_significance_results_fair_paired.json", 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Convert all numpy types
            json_compatible = {}
            for method, stats in statistical_comparisons.items():
                json_compatible[method] = {}
                for metric, values in stats.items():
                    json_compatible[method][metric] = {k: convert_numpy(v) for k, v in values.items()}
            
            json.dump(json_compatible, f, indent=2)
        
        # Save summary
        save_statistical_summary(statistical_comparisons, "statistical_summary_fair_paired.txt")
        
        # Save text table
        with open("significance_table_fair_paired.txt", 'w') as f:
            f.write(significance_table)
        
        # Save detailed results
        results_df = []
        for method_name, method_runs in all_results.items():
            for run_id, run_metrics in enumerate(method_runs):
                row = {
                    'Method': method_name,
                    'Run': run_id + 1,
                    'Precision': run_metrics['precision'],
                    'Recall': run_metrics['recall'],
                    'F1_Score': run_metrics['f1_score'],
                    'mAP': run_metrics['map_score'],
                    'FPS': run_metrics['fps']
                }
                results_df.append(row)
        
        df = pd.DataFrame(results_df)
        df.to_csv("detailed_fair_paired_results.csv", index=False)
        
        print(f"\n📊 Fair paired statistical significance results saved to:")
        print("  - statistical_significance_results_fair_paired.json")
        print("  - statistical_summary_fair_paired.txt")
        print("  - significance_table_fair_paired.txt")
        print("  - detailed_fair_paired_results.csv")
        print("\n📋 Fair Paired Results Table:")
        print(significance_table)
        
        print(f"\n🔬 FAIR EVALUATION COMPLETED!")
        print(f"✅ All models evaluated on IDENTICAL {n_runs} augmented datasets")
        print("✅ Paired t-tests now statistically valid and meaningful")
    
    return all_results

if __name__ == "__main__":
    print("🔬 Running FAIR Statistical Evaluation with Data Augmentation...")
    print("🎯 Key improvement: All models evaluated on IDENTICAL augmented datasets")
    print("📊 This ensures valid paired statistical comparisons")
    print("Weather Condition: Sand")
    print("Number of augmented runs: 20")
    print("\n" + "="*60)
    print("FAIR EVALUATION PROCESS:")
    print("1. Pre-generate N augmented datasets with different seeds")
    print("2. Run ALL models on augmented dataset #1")
    print("3. Run ALL models on augmented dataset #2") 
    print("4. ... repeat for all N datasets")
    print("5. Now paired t-test compares like-with-like!")
    print("="*60 + "\n")
    
    results = run_fair_statistical_evaluation_with_augmentation(n_runs=20)