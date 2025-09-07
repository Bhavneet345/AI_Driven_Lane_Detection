"""
Evaluation metrics for lane detection models.
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path

class LaneDetectionMetrics:
    """Comprehensive metrics for lane detection evaluation."""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_predictions = 0
        self.total_ground_truth = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.iou_scores = []
        self.processing_times = []
    
    def add_prediction(
        self,
        predicted_lanes: List[np.ndarray],
        ground_truth_lanes: List[np.ndarray],
        processing_time: Optional[float] = None
    ):
        """
        Add a prediction for evaluation.
        
        Args:
            predicted_lanes: List of predicted lane points
            ground_truth_lanes: List of ground truth lane points
            processing_time: Time taken for inference
        """
        if processing_time is not None:
            self.processing_times.append(processing_time)
        
        # Calculate IoU for each predicted lane
        for pred_lane in predicted_lanes:
            best_iou = 0
            matched = False
            
            for gt_lane in ground_truth_lanes:
                iou = self._calculate_lane_iou(pred_lane, gt_lane)
                best_iou = max(best_iou, iou)
                
                if iou >= self.iou_threshold:
                    matched = True
                    break
            
            self.iou_scores.append(best_iou)
            
            if matched:
                self.true_positives += 1
            else:
                self.false_positives += 1
        
        # Count false negatives (ground truth lanes not matched)
        for gt_lane in ground_truth_lanes:
            best_iou = 0
            for pred_lane in predicted_lanes:
                iou = self._calculate_lane_iou(pred_lane, gt_lane)
                best_iou = max(best_iou, iou)
            
            if best_iou < self.iou_threshold:
                self.false_negatives += 1
        
        self.total_predictions += len(predicted_lanes)
        self.total_ground_truth += len(ground_truth_lanes)
    
    def _calculate_lane_iou(self, lane1: np.ndarray, lane2: np.ndarray) -> float:
        """
        Calculate IoU between two lane lines.
        
        Args:
            lane1: First lane points
            lane2: Second lane points
            
        Returns:
            IoU score between 0 and 1
        """
        # Create binary masks for both lanes
        mask1 = self._lane_to_mask(lane1)
        mask2 = self._lane_to_mask(lane2)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _lane_to_mask(self, lane_points: np.ndarray, image_size: Tuple[int, int] = (720, 1280)) -> np.ndarray:
        """
        Convert lane points to binary mask.
        
        Args:
            lane_points: Lane line points
            image_size: Image dimensions (height, width)
            
        Returns:
            Binary mask
        """
        mask = np.zeros(image_size, dtype=np.uint8)
        
        if len(lane_points) < 2:
            return mask
        
        # Draw lane line on mask
        for i in range(len(lane_points) - 1):
            pt1 = tuple(lane_points[i].astype(int))
            pt2 = tuple(lane_points[i + 1].astype(int))
            cv2.line(mask, pt1, pt2, 255, 3)
        
        return mask
    
    def get_precision(self) -> float:
        """Calculate precision."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def get_recall(self) -> float:
        """Calculate recall."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    def get_f1_score(self) -> float:
        """Calculate F1 score."""
        precision = self.get_precision()
        recall = self.get_recall()
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def get_mean_iou(self) -> float:
        """Calculate mean IoU."""
        if not self.iou_scores:
            return 0.0
        return np.mean(self.iou_scores)
    
    def get_accuracy(self) -> float:
        """Calculate accuracy."""
        if self.total_ground_truth == 0:
            return 0.0
        return self.true_positives / self.total_ground_truth
    
    def get_avg_processing_time(self) -> float:
        """Calculate average processing time."""
        if not self.processing_times:
            return 0.0
        return np.mean(self.processing_times)
    
    def get_fps(self) -> float:
        """Calculate average FPS."""
        avg_time = self.get_avg_processing_time()
        if avg_time == 0:
            return 0.0
        return 1.0 / avg_time
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get comprehensive metrics summary."""
        return {
            "precision": self.get_precision(),
            "recall": self.get_recall(),
            "f1_score": self.get_f1_score(),
            "accuracy": self.get_accuracy(),
            "mean_iou": self.get_mean_iou(),
            "avg_processing_time": self.get_avg_processing_time(),
            "fps": self.get_fps(),
            "total_predictions": self.total_predictions,
            "total_ground_truth": self.total_ground_truth,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives
        }
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot metrics visualization."""
        metrics = self.get_metrics_summary()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Lane Detection Metrics', fontsize=16)
        
        # Precision, Recall, F1 Score
        scores = ['Precision', 'Recall', 'F1 Score']
        values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
        
        axes[0, 0].bar(scores, values, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Classification Metrics')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_ylabel('Score')
        
        # IoU distribution
        if self.iou_scores:
            axes[0, 1].hist(self.iou_scores, bins=20, alpha=0.7, color='purple')
            axes[0, 1].set_title('IoU Score Distribution')
            axes[0, 1].set_xlabel('IoU Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(metrics['mean_iou'], color='red', linestyle='--', 
                             label=f'Mean: {metrics["mean_iou"]:.3f}')
            axes[0, 1].legend()
        
        # Processing time distribution
        if self.processing_times:
            axes[1, 0].hist(self.processing_times, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_title('Processing Time Distribution')
            axes[1, 0].set_xlabel('Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(metrics['avg_processing_time'], color='red', linestyle='--',
                             label=f'Mean: {metrics["avg_processing_time"]:.4f}s')
            axes[1, 0].legend()
        
        # Performance summary
        perf_metrics = ['Accuracy', 'Mean IoU', 'FPS']
        perf_values = [metrics['accuracy'], metrics['mean_iou'], metrics['fps']]
        
        bars = axes[1, 1].bar(perf_metrics, perf_values, color=['cyan', 'magenta', 'yellow'])
        axes[1, 1].set_title('Performance Summary')
        axes[1, 1].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, perf_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved: {save_path}")
        
        plt.show()

def evaluate_model_on_dataset(
    model,
    dataset_path: str,
    ground_truth_path: str,
    output_dir: str = "evaluation_results"
) -> LaneDetectionMetrics:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Lane detection model
        dataset_path: Path to test images
        ground_truth_path: Path to ground truth annotations
        output_dir: Directory to save results
        
    Returns:
        Evaluation metrics
    """
    metrics = LaneDetectionMetrics()
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth annotations
    # This would be implemented based on your annotation format
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Process each image
    for image_path in Path(dataset_path).glob("*.jpg"):
        # Load image
        image = cv2.imread(str(image_path))
        
        # Run inference
        start_time = time.time()
        result = model.predict(image)
        processing_time = time.time() - start_time
        
        # Get ground truth for this image
        image_name = image_path.stem
        gt_lanes = ground_truth.get(image_name, [])
        
        # Add to metrics
        metrics.add_prediction(result.lanes, gt_lanes, processing_time)
    
    # Generate report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write("Lane Detection Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        
        summary = metrics.get_metrics_summary()
        for key, value in summary.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Plot metrics
    plot_path = os.path.join(output_dir, "metrics_plot.png")
    metrics.plot_metrics(plot_path)
    
    print(f"Evaluation complete. Results saved to: {output_dir}")
    return metrics

def load_ground_truth(gt_path: str) -> Dict[str, List[np.ndarray]]:
    """
    Load ground truth annotations.
    This is a placeholder - implement based on your annotation format.
    """
    # Placeholder implementation
    # In practice, this would load from JSON, XML, or other format
    return {}

def compare_models(
    models: Dict[str, Any],
    test_data: str,
    ground_truth: str,
    output_dir: str = "model_comparison"
) -> Dict[str, LaneDetectionMetrics]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of model_name -> model
        test_data: Path to test images
        ground_truth: Path to ground truth
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary of model_name -> metrics
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model_on_dataset(model, test_data, ground_truth, 
                                          os.path.join(output_dir, model_name))
        results[model_name] = metrics
    
    # Create comparison plot
    create_comparison_plot(results, output_dir)
    
    return results

def create_comparison_plot(
    results: Dict[str, LaneDetectionMetrics],
    output_dir: str
):
    """Create comparison plot for multiple models."""
    model_names = list(results.keys())
    metrics_names = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'FPS']
    
    # Extract metrics for each model
    data = {}
    for metric in metrics_names:
        data[metric] = [results[name].get_metrics_summary()[metric.lower().replace(' ', '_')] 
                       for name in model_names]
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, (metric, values) in enumerate(data.items()):
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()
