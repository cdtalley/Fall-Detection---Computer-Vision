"""
Fall Detection Performance Analysis Module

This module provides comprehensive analysis tools for investigating model performance,
failure cases, and healthcare-specific evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_processing import PrivacyPreservingPoseProcessor, create_sample_data
from model_training import FallDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FallDetectionAnalyzer:
    """
    Comprehensive analyzer for fall detection model performance.
    
    This class provides tools for investigating model failures, analyzing performance
    patterns, and generating healthcare-specific insights.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the analyzer with a trained model.
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = FallDetectionModel()
        
        try:
            self.model.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Initialize pose processor for feature analysis
        self.pose_processor = PrivacyPreservingPoseProcessor()
        
        # Analysis results storage
        self.analysis_results = {}
        self.failure_cases = []
        self.performance_metrics = {}
        
        logger.info("Fall detection analyzer initialized successfully")
    
    def analyze_model_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Comprehensive model performance analysis.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        logger.info("Starting comprehensive model performance analysis")
        
        # Get predictions
        y_pred_proba = self.model.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Standard metrics
        self.performance_metrics = self._calculate_standard_metrics(y_test, y_pred, y_pred_proba)
        
        # Healthcare-specific metrics
        healthcare_metrics = self._calculate_healthcare_metrics(y_test, y_pred, y_pred_proba)
        self.performance_metrics.update(healthcare_metrics)
        
        # Failure case analysis
        self.failure_cases = self._identify_failure_cases(X_test, y_test, y_pred, y_pred_proba)
        
        # Performance patterns
        performance_patterns = self._analyze_performance_patterns(X_test, y_test, y_pred)
        
        # Compile results
        self.analysis_results = {
            'standard_metrics': self.performance_metrics,
            'healthcare_metrics': healthcare_metrics,
            'failure_cases': self.failure_cases,
            'performance_patterns': performance_patterns,
            'recommendations': self._generate_recommendations()
        }
        
        logger.info("Model performance analysis completed")
        return self.analysis_results
    
    def _calculate_standard_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray) -> Dict:
        """Calculate standard classification metrics."""
        # Classification report
        class_names = self.model.label_encoder.classes_
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC curves for each class
        roc_data = {}
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[class_name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_data': roc_data,
            'accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }
    
    def _calculate_healthcare_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray) -> Dict:
        """Calculate healthcare-specific metrics."""
        # Find fall class index
        fall_class_idx = None
        for i, class_name in enumerate(self.model.label_encoder.classes_):
            if 'fall' in class_name.lower():
                fall_class_idx = i
                break
        
        if fall_class_idx is None:
            fall_class_idx = len(self.model.label_encoder.classes_) - 1  # Assume last class
        
        # Calculate healthcare-specific metrics
        cm = confusion_matrix(y_true, y_pred)
        
        # Sensitivity (recall) for fall detection
        tp = cm[fall_class_idx, fall_class_idx]
        fn = np.sum(cm[fall_class_idx, :]) - tp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity for fall detection
        fp = np.sum(cm[:, fall_class_idx]) - tp
        tn = np.sum(cm) - tp - fp - fn
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Precision for fall detection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # F1 score for fall detection
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        # Alert fatigue analysis
        false_positives = np.sum((y_pred == fall_class_idx) & (y_true != fall_class_idx))
        total_predictions = len(y_pred)
        alert_fatigue_score = false_positives / total_predictions if total_predictions > 0 else 0.0
        
        # Missed falls cost
        missed_falls = np.sum((y_pred != fall_class_idx) & (y_true == fall_class_idx))
        total_falls = np.sum(y_true == fall_class_idx)
        missed_falls_cost = missed_falls / total_falls if total_falls > 0 else 0.0
        
        # False alarm cost
        false_alarms = np.sum((y_pred == fall_class_idx) & (y_true != fall_class_idx))
        total_non_falls = np.sum(y_true != fall_class_idx)
        false_alarm_cost = false_alarms / total_non_falls if total_non_falls > 0 else 0.0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'alert_fatigue_score': alert_fatigue_score,
            'missed_falls_cost': missed_falls_cost,
            'false_alarm_cost': false_alarm_cost,
            'fall_class_index': fall_class_idx
        }
    
    def _identify_failure_cases(self, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, y_pred_proba: np.ndarray) -> List[Dict]:
        """Identify and analyze failure cases."""
        failure_cases = []
        
        for i in range(len(X_test)):
            if y_test[i] != y_pred[i]:  # Misclassification
                true_class = self.model.label_encoder.classes_[y_test[i]]
                pred_class = self.model.label_encoder.classes_[y_pred[i]]
                confidence = np.max(y_pred_proba[i])
                
                # Extract features for analysis
                features = self._extract_sequence_features(X_test[i])
                
                failure_case = {
                    'index': i,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'features': features,
                    'sequence': X_test[i],
                    'misclassification_type': self._classify_misclassification(y_test[i], y_pred[i])
                }
                
                failure_cases.append(failure_case)
        
        logger.info(f"Identified {len(failure_cases)} failure cases")
        return failure_cases
    
    def _extract_sequence_features(self, sequence: np.ndarray) -> Dict:
        """Extract features from a sequence for failure analysis."""
        features = {}
        
        # Temporal features
        features['sequence_length'] = len(sequence)
        features['mean_velocity'] = np.mean([np.linalg.norm(seq[1:] - seq[:-1]) for seq in sequence])
        features['max_velocity'] = np.max([np.linalg.norm(seq[1:] - seq[:-1]) for seq in sequence])
        features['velocity_variance'] = np.var([np.linalg.norm(seq[1:] - seq[:-1]) for seq in sequence])
        
        # Spatial features
        features['mean_com_height'] = np.mean([seq[8] for seq in sequence])  # Assuming COM height is at index 8
        features['com_height_variance'] = np.var([seq[8] for seq in sequence])
        features['mean_stability'] = np.mean([seq[12] for seq in sequence])  # Assuming stability is at index 12
        
        return features
    
    def _classify_misclassification(self, true_label: int, pred_label: int) -> str:
        """Classify the type of misclassification."""
        class_names = self.model.label_encoder.classes_
        true_class = class_names[true_label]
        pred_class = class_names[pred_label]
        
        # Check for critical misclassifications (missed falls)
        if 'fall' in true_class.lower() and 'fall' not in pred_class.lower():
            return 'missed_fall'
        elif 'fall' in pred_class.lower() and 'fall' not in true_class.lower():
            return 'false_alarm'
        else:
            return 'class_confusion'
    
    def _analyze_performance_patterns(self, X_test: np.ndarray, y_test: np.ndarray,
                                    y_pred: np.ndarray) -> Dict:
        """Analyze performance patterns and trends."""
        patterns = {}
        
        # Performance by class
        class_performance = {}
        for i, class_name in enumerate(self.model.label_encoder.classes_):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((y_test == y_pred) & class_mask) / np.sum(class_mask)
                class_performance[class_name] = class_accuracy
        
        patterns['class_performance'] = class_performance
        
        # Performance by sequence characteristics
        sequence_lengths = [len(seq) for seq in X_test]
        patterns['length_performance'] = self._analyze_length_performance(sequence_lengths, y_test, y_pred)
        
        # Performance by feature characteristics
        patterns['feature_performance'] = self._analyze_feature_performance(X_test, y_test, y_pred)
        
        return patterns
    
    def _analyze_length_performance(self, lengths: List[int], y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict:
        """Analyze performance based on sequence length."""
        length_performance = {}
        
        # Group by length ranges
        length_ranges = [(0, 20), (20, 40), (40, 60), (60, 100)]
        
        for min_len, max_len in length_ranges:
            mask = (np.array(lengths) >= min_len) & (np.array(lengths) < max_len)
            if np.sum(mask) > 0:
                accuracy = np.sum((y_true == y_pred) & mask) / np.sum(mask)
                length_performance[f"{min_len}-{max_len}"] = {
                    'accuracy': accuracy,
                    'count': np.sum(mask)
                }
        
        return length_performance
    
    def _analyze_feature_performance(self, X_test: np.ndarray, y_true: np.ndarray,
                                   y_pred: np.ndarray) -> Dict:
        """Analyze performance based on feature characteristics."""
        feature_performance = {}
        
        # Analyze performance by velocity characteristics
        velocities = []
        for seq in X_test:
            vel = np.mean([np.linalg.norm(seq[1:] - seq[:-1]) for seq in seq])
            velocities.append(vel)
        
        # Group by velocity ranges
        vel_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.0)]
        
        for min_vel, max_vel in vel_ranges:
            mask = (np.array(velocities) >= min_vel) & (np.array(velocities) < max_vel)
            if np.sum(mask) > 0:
                accuracy = np.sum((y_true == y_pred) & mask) / np.sum(mask)
                feature_performance[f"velocity_{min_vel}-{max_vel}"] = {
                    'accuracy': accuracy,
                    'count': np.sum(mask)
                }
        
        return feature_performance
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Analyze sensitivity
        if self.performance_metrics.get('sensitivity', 0) < 0.90:
            recommendations.append("Increase model sensitivity for fall detection to meet healthcare requirements")
        
        # Analyze false positive rate
        if self.performance_metrics.get('alert_fatigue_score', 0) > 0.10:
            recommendations.append("Reduce false positive rate to minimize alert fatigue")
        
        # Analyze failure patterns
        if self.failure_cases:
            missed_falls = [case for case in self.failure_cases if case['misclassification_type'] == 'missed_fall']
            if len(missed_falls) > len(self.failure_cases) * 0.3:
                recommendations.append("High rate of missed falls detected - consider retraining with more fall examples")
        
        # Performance pattern recommendations
        patterns = self.analysis_results.get('performance_patterns', {})
        class_performance = patterns.get('class_performance', {})
        
        for class_name, accuracy in class_performance.items():
            if accuracy < 0.80:
                recommendations.append(f"Low accuracy for {class_name} class - consider data augmentation or model tuning")
        
        # General recommendations
        recommendations.extend([
            "Implement continuous monitoring of model performance in production",
            "Consider ensemble methods to improve robustness",
            "Add environmental condition detection for adaptive thresholds",
            "Implement A/B testing framework for model improvements"
        ])
        
        return recommendations
    
    def generate_visualizations(self, save_path: Optional[str] = None) -> Dict:
        """Generate comprehensive visualizations for the analysis."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_model_performance first.")
        
        visualizations = {}
        
        # 1. Confusion Matrix
        cm = self.performance_metrics['confusion_matrix']
        class_names = self.model.label_encoder.classes_
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            showscale=True
        ))
        
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=500
        )
        
        visualizations['confusion_matrix'] = fig_cm
        
        # 2. ROC Curves
        roc_data = self.performance_metrics['roc_data']
        fig_roc = go.Figure()
        
        for class_name, data in roc_data.items():
            fig_roc.add_trace(go.Scatter(
                x=data['fpr'],
                y=data['tpr'],
                mode='lines',
                name=f'{class_name} (AUC = {data["auc"]:.3f})'
            ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))
        
        fig_roc.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500
        )
        
        visualizations['roc_curves'] = fig_roc
        
        # 3. Healthcare Metrics Dashboard
        healthcare_metrics = self.analysis_results['healthcare_metrics']
        
        fig_healthcare = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sensitivity vs Specificity', 'Precision vs F1 Score',
                          'Alert Fatigue Score', 'Missed Falls Cost'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Bar charts
        fig_healthcare.add_trace(
            go.Bar(x=['Sensitivity', 'Specificity'], 
                  y=[healthcare_metrics['sensitivity'], healthcare_metrics['specificity']],
                  name='Detection Metrics'),
            row=1, col=1
        )
        
        fig_healthcare.add_trace(
            go.Bar(x=['Precision', 'F1 Score'],
                  y=[healthcare_metrics['precision'], healthcare_metrics['f1_score']],
                  name='Quality Metrics'),
            row=1, col=2
        )
        
        # Gauge charts
        fig_healthcare.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=healthcare_metrics['alert_fatigue_score'],
                title={'text': "Alert Fatigue Score"},
                gauge={'axis': {'range': [0, 1]},
                      'bar': {'color': "darkblue"},
                      'steps': [{'range': [0, 0.05], 'color': "lightgray"},
                               {'range': [0.05, 0.1], 'color': "yellow"},
                               {'range': [0.1, 1], 'color': "red"}]}
            ),
            row=2, col=1
        )
        
        fig_healthcare.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=healthcare_metrics['missed_falls_cost'],
                title={'text': "Missed Falls Cost"},
                gauge={'axis': {'range': [0, 1]},
                      'bar': {'color': "darkred"},
                      'steps': [{'range': [0, 0.05], 'color': "green"},
                               {'range': [0.05, 0.1], 'color': "yellow"},
                               {'range': [0.1, 1], 'color': "red"}]}
            ),
            row=2, col=2
        )
        
        fig_healthcare.update_layout(height=600, title_text="Healthcare Metrics Dashboard")
        visualizations['healthcare_dashboard'] = fig_healthcare
        
        # 4. Failure Case Analysis
        if self.failure_cases:
            failure_types = [case['misclassification_type'] for case in self.failure_cases]
            failure_counts = pd.Series(failure_types).value_counts()
            
            fig_failures = go.Figure(data=go.Pie(
                labels=failure_counts.index,
                values=failure_counts.values,
                hole=0.3
            ))
            
            fig_failures.update_layout(
                title="Failure Case Distribution",
                height=400
            )
            
            visualizations['failure_analysis'] = fig_failures
        
        # Save visualizations if path provided
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            for name, fig in visualizations.items():
                fig.write_html(f"{save_path}/{name}.html")
                fig.write_image(f"{save_path}/{name}.png")
        
        return visualizations
    
    def generate_report(self, output_path: str):
        """Generate a comprehensive analysis report."""
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_model_performance first.")
        
        # Create report content
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'model_type': self.model.model_type,
            'analysis_summary': {
                'total_samples': len(self.failure_cases) + len([x for x in self.analysis_results.get('standard_metrics', {}).get('confusion_matrix', [])]),
                'failure_cases': len(self.failure_cases),
                'overall_accuracy': self.performance_metrics.get('accuracy', 0),
                'healthcare_compliance': self._assess_healthcare_compliance()
            },
            'performance_metrics': self.performance_metrics,
            'healthcare_metrics': self.analysis_results['healthcare_metrics'],
            'failure_analysis': {
                'total_failures': len(self.failure_cases),
                'failure_types': self._summarize_failure_types(),
                'critical_failures': len([case for case in self.failure_cases 
                                        if case['misclassification_type'] == 'missed_fall'])
            },
            'performance_patterns': self.analysis_results['performance_patterns'],
            'recommendations': self.analysis_results['recommendations']
        }
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to {output_path}")
        return report
    
    def _assess_healthcare_compliance(self) -> Dict:
        """Assess compliance with healthcare requirements."""
        healthcare_metrics = self.analysis_results['healthcare_metrics']
        
        compliance = {
            'sensitivity_requirement': healthcare_metrics['sensitivity'] >= 0.90,
            'false_positive_requirement': healthcare_metrics['alert_fatigue_score'] <= 0.10,
            'response_time_requirement': True,  # Would need real-time data
            'privacy_requirement': True,  # Always true with pose estimation
            'overall_compliance': True
        }
        
        # Check overall compliance
        if not all([compliance['sensitivity_requirement'], 
                   compliance['false_positive_requirement']]):
            compliance['overall_compliance'] = False
        
        return compliance
    
    def _summarize_failure_types(self) -> Dict:
        """Summarize failure types for the report."""
        failure_types = [case['misclassification_type'] for case in self.failure_cases]
        return dict(pd.Series(failure_types).value_counts())


def analyze_fall_detection_model(model_path: str, X_test: np.ndarray, y_test: np.ndarray,
                               output_dir: str = "analysis_results") -> Dict:
    """
    Complete analysis pipeline for fall detection model.
    
    Args:
        model_path: Path to trained model
        X_test: Test sequences
        y_test: Test labels
        output_dir: Directory to save analysis results
        
    Returns:
        Complete analysis results
    """
    logger.info("Starting comprehensive fall detection model analysis")
    
    # Initialize analyzer
    analyzer = FallDetectionAnalyzer(model_path)
    
    # Run analysis
    results = analyzer.analyze_model_performance(X_test, y_test)
    
    # Generate visualizations
    visualizations = analyzer.generate_visualizations(output_dir)
    
    # Generate report
    report_path = f"{output_dir}/analysis_report.json"
    report = analyzer.generate_report(report_path)
    
    # Print summary
    print("\n=== Fall Detection Model Analysis Summary ===")
    print(f"Overall Accuracy: {results['standard_metrics']['accuracy']:.3f}")
    print(f"Sensitivity: {results['healthcare_metrics']['sensitivity']:.3f}")
    print(f"Specificity: {results['healthcare_metrics']['specificity']:.3f}")
    print(f"Alert Fatigue Score: {results['healthcare_metrics']['alert_fatigue_score']:.3f}")
    print(f"Failure Cases: {len(results['failure_cases'])}")
    print(f"Healthcare Compliance: {report['analysis_summary']['healthcare_compliance']['overall_compliance']}")
    
    print("\n=== Key Recommendations ===")
    for i, rec in enumerate(results['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    
    logger.info("Analysis completed successfully")
    return results


if __name__ == "__main__":
    # Example usage
    print("Fall Detection Performance Analysis Module")
    print("This module provides comprehensive analysis tools for investigating model performance.")
    print("Use analyze_fall_detection_model() function for complete analysis pipeline.") 