"""
Real-time Fall Detection Inference Module

This module handles real-time video processing, fall detection predictions, and alert generation
with healthcare-specific considerations for privacy and response time.
"""

import cv2
import numpy as np
import pandas as pd
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path

# Import our modules
from data_processing import PrivacyPreservingPoseProcessor, VideoProcessor, PoseKeypoints
from model_training import FallDetectionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlertEvent:
    """Data class for fall detection alerts."""
    timestamp: datetime
    alert_type: str  # 'fall', 'high_risk', 'normal'
    confidence: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    location: str
    description: str
    metadata: Dict

@dataclass
class PerformanceMetrics:
    """Data class for real-time performance metrics."""
    fps: float
    processing_time_ms: float
    detection_confidence: float
    alert_count: int
    false_positive_rate: float
    sensitivity: float

class RealTimeFallDetector:
    """
    Real-time fall detection system with healthcare-specific optimizations.
    
    This class provides real-time video processing, fall detection, and alert generation
    while maintaining privacy and meeting healthcare response time requirements.
    """
    
    def __init__(self, 
                 model_path: str,
                 camera_source: int = 0,
                 alert_callback: Optional[Callable] = None,
                 confidence_threshold: float = 0.7,
                 alert_cooldown_seconds: int = 30):
        """
        Initialize the real-time fall detector.
        
        Args:
            model_path: Path to trained model
            camera_source: Camera source (0 for webcam, or video file path)
            alert_callback: Optional callback function for alerts
            confidence_threshold: Minimum confidence for alerts
            alert_cooldown_seconds: Cooldown period between alerts
        """
        self.model_path = model_path
        self.camera_source = camera_source
        self.alert_callback = alert_callback
        self.confidence_threshold = confidence_threshold
        self.alert_cooldown_seconds = alert_cooldown_seconds
        
        # Initialize components
        self.pose_processor = PrivacyPreservingPoseProcessor()
        self.video_processor = VideoProcessor(self.pose_processor)
        self.model = FallDetectionModel()
        
        # Load trained model
        try:
            self.model.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Real-time processing state
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.alert_history = []
        self.last_alert_time = None
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(
            fps=0.0,
            processing_time_ms=0.0,
            detection_confidence=0.0,
            alert_count=0,
            false_positive_rate=0.0,
            sensitivity=0.0
        )
        
        # Healthcare-specific settings
        self.healthcare_config = {
            'max_response_time_ms': 90000,  # 90 seconds max response time
            'min_sensitivity': 0.90,  # Minimum 90% sensitivity required
            'max_false_positive_rate': 0.10,  # Maximum 10% false positive rate
            'alert_priority_levels': {
                'fall': 'critical',
                'high_risk': 'high',
                'normal': 'low'
            }
        }
        
        logger.info("Real-time fall detector initialized successfully")
    
    def start_detection(self):
        """Start real-time fall detection."""
        if self.is_running:
            logger.warning("Detection already running")
            return
        
        self.is_running = True
        
        # Start processing threads
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.processing_thread = threading.Thread(target=self._process_frames)
        
        self.capture_thread.start()
        self.processing_thread.start()
        
        logger.info("Real-time fall detection started")
    
    def stop_detection(self):
        """Stop real-time fall detection."""
        self.is_running = False
        
        # Wait for threads to finish
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        
        logger.info("Real-time fall detection stopped")
    
    def _capture_frames(self):
        """Capture frames from camera in a separate thread."""
        cap = cv2.VideoCapture(self.camera_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera source: {self.camera_source}")
            return
        
        frame_count = 0
        start_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                break
            
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put_nowait((frame_count, frame, time.time()))
            except queue.Full:
                # Skip frame if queue is full
                pass
            
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                self.performance_metrics.fps = frame_count / elapsed_time
        
        cap.release()
        logger.info("Frame capture thread stopped")
    
    def _process_frames(self):
        """Process frames for fall detection in a separate thread."""
        sequence_buffer = []
        max_sequence_length = self.model.sequence_length
        
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame_count, frame, timestamp = self.frame_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # Process frame
                result = self.video_processor.process_frame(frame)
                
                if result['keypoints_detected']:
                    # Extract features
                    features = result['features']
                    
                    # Add to sequence buffer
                    sequence_buffer.append(list(features.values()))
                    
                    # Keep only the last N frames
                    if len(sequence_buffer) > max_sequence_length:
                        sequence_buffer.pop(0)
                    
                    # Make prediction if we have enough frames
                    if len(sequence_buffer) == max_sequence_length:
                        sequence = np.array(sequence_buffer)
                        predicted_class, confidence = self.model.predict_sequence(sequence)
                        
                        # Update performance metrics
                        processing_time = (time.time() - start_time) * 1000
                        self.performance_metrics.processing_time_ms = processing_time
                        self.performance_metrics.detection_confidence = confidence
                        
                        # Generate alert if confidence is high enough
                        if confidence >= self.confidence_threshold:
                            self._generate_alert(predicted_class, confidence, timestamp)
                        
                        # Add result to queue for display
                        try:
                            self.result_queue.put_nowait({
                                'frame': frame,
                                'keypoints': result['keypoints'],
                                'prediction': predicted_class,
                                'confidence': confidence,
                                'processing_time': processing_time
                            })
                        except queue.Full:
                            pass
                
                # Check response time compliance
                processing_time = (time.time() - start_time) * 1000
                if processing_time > self.healthcare_config['max_response_time_ms']:
                    logger.warning(f"Processing time ({processing_time:.2f}ms) exceeds healthcare requirement")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
        
        logger.info("Frame processing thread stopped")
    
    def _generate_alert(self, predicted_class: int, confidence: float, timestamp: float):
        """Generate healthcare-appropriate alert."""
        # Check cooldown period
        current_time = datetime.now()
        if (self.last_alert_time and 
            (current_time - self.last_alert_time).seconds < self.alert_cooldown_seconds):
            return
        
        # Map prediction to alert type
        class_names = self.model.label_encoder.classes_
        alert_type = class_names[predicted_class] if predicted_class < len(class_names) else 'unknown'
        
        # Determine severity based on healthcare configuration
        severity = self.healthcare_config['alert_priority_levels'].get(alert_type, 'medium')
        
        # Create alert event
        alert = AlertEvent(
            timestamp=current_time,
            alert_type=alert_type,
            confidence=confidence,
            severity=severity,
            location='Camera 1',  # Could be configurable
            description=f"{alert_type.replace('_', ' ').title()} detected with {confidence:.2f} confidence",
            metadata={
                'predicted_class': predicted_class,
                'processing_timestamp': timestamp,
                'model_type': self.model.model_type
            }
        )
        
        # Add to alert history
        self.alert_history.append(alert)
        self.last_alert_time = current_time
        self.performance_metrics.alert_count += 1
        
        # Log alert
        logger.warning(f"ALERT: {alert.description} (Severity: {severity})")
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get the latest processing result for display."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics
    
    def get_alert_history(self, hours: int = 24) -> List[AlertEvent]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]
    
    def export_alert_log(self, filepath: str):
        """Export alert history to JSON file."""
        alert_data = []
        for alert in self.alert_history:
            alert_data.append({
                'timestamp': alert.timestamp.isoformat(),
                'alert_type': alert.alert_type,
                'confidence': alert.confidence,
                'severity': alert.severity,
                'location': alert.location,
                'description': alert.description,
                'metadata': alert.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        logger.info(f"Alert log exported to {filepath}")
    
    def run_demo(self, display_skeleton: bool = True, save_video: bool = False):
        """
        Run a demonstration of the fall detection system.
        
        Args:
            display_skeleton: Whether to display skeleton overlay
            save_video: Whether to save processed video
        """
        logger.info("Starting fall detection demo")
        
        # Initialize video writer if saving
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter('fall_detection_demo.mp4', fourcc, 30.0, (640, 480))
        
        try:
            self.start_detection()
            
            while True:
                # Get latest result
                result = self.get_latest_result()
                
                if result:
                    frame = result['frame']
                    keypoints = result['keypoints']
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    # Draw skeleton if requested
                    if display_skeleton and keypoints:
                        frame = self.pose_processor.draw_skeleton(frame, keypoints)
                    
                    # Add prediction overlay
                    class_names = self.model.label_encoder.classes_
                    prediction_text = f"{class_names[prediction]}: {confidence:.2f}"
                    
                    # Color based on prediction
                    if prediction == 2:  # Fall
                        color = (0, 0, 255)  # Red
                    elif prediction == 1:  # High risk
                        color = (0, 165, 255)  # Orange
                    else:  # Normal
                        color = (0, 255, 0)  # Green
                    
                    cv2.putText(frame, prediction_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Add performance metrics
                    metrics = self.get_performance_metrics()
                    cv2.putText(frame, f"FPS: {metrics.fps:.1f}", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Processing: {metrics.processing_time_ms:.1f}ms", (10, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if save_video and video_writer:
                        video_writer.write(frame)
                
                # Display frame
                cv2.imshow('Privacy-Preserving Fall Detection', frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        
        finally:
            self.stop_detection()
            cv2.destroyAllWindows()
            
            if video_writer:
                video_writer.release()
            
            logger.info("Fall detection demo completed")


class HealthcareAlertManager:
    """
    Healthcare-specific alert management system.
    
    This class handles alert prioritization, escalation, and integration
    with healthcare workflows.
    """
    
    def __init__(self):
        """Initialize the healthcare alert manager."""
        self.alert_queue = queue.PriorityQueue()
        self.escalation_rules = {
            'critical': {'escalation_time_minutes': 1, 'max_alerts': 3},
            'high': {'escalation_time_minutes': 5, 'max_alerts': 5},
            'medium': {'escalation_time_minutes': 15, 'max_alerts': 10},
            'low': {'escalation_time_minutes': 60, 'max_alerts': 20}
        }
        
        self.alert_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        self.last_escalation = {severity: datetime.now() for severity in self.escalation_rules.keys()}
    
    def process_alert(self, alert: AlertEvent) -> Dict:
        """
        Process and prioritize healthcare alert.
        
        Args:
            alert: Alert event to process
            
        Returns:
            Processing result with action recommendations
        """
        # Calculate priority score (higher = more urgent)
        priority_score = self._calculate_priority_score(alert)
        
        # Add to priority queue
        self.alert_queue.put((-priority_score, alert))  # Negative for max-heap behavior
        
        # Update alert counts
        self.alert_counts[alert.severity] += 1
        
        # Check for escalation
        escalation_needed = self._check_escalation(alert)
        
        # Generate response
        response = {
            'alert_id': f"{alert.timestamp.strftime('%Y%m%d_%H%M%S')}_{alert.alert_type}",
            'priority_score': priority_score,
            'escalation_needed': escalation_needed,
            'recommended_action': self._get_recommended_action(alert, escalation_needed),
            'response_time_requirement': self._get_response_time_requirement(alert.severity)
        }
        
        return response
    
    def _calculate_priority_score(self, alert: AlertEvent) -> float:
        """Calculate priority score for alert."""
        # Base score from severity
        severity_scores = {'critical': 100, 'high': 75, 'medium': 50, 'low': 25}
        base_score = severity_scores.get(alert.severity, 25)
        
        # Confidence bonus
        confidence_bonus = alert.confidence * 20
        
        # Time decay (newer alerts get higher priority)
        time_decay = max(0, 10 - (datetime.now() - alert.timestamp).seconds / 60)
        
        return base_score + confidence_bonus + time_decay
    
    def _check_escalation(self, alert: AlertEvent) -> bool:
        """Check if alert requires escalation."""
        rules = self.escalation_rules[alert.severity]
        current_time = datetime.now()
        
        # Check if escalation time has passed
        if (current_time - self.last_escalation[alert.severity]).seconds > rules['escalation_time_minutes'] * 60:
            # Check if alert count exceeds threshold
            if self.alert_counts[alert.severity] >= rules['max_alerts']:
                self.last_escalation[alert.severity] = current_time
                return True
        
        return False
    
    def _get_recommended_action(self, alert: AlertEvent, escalation_needed: bool) -> str:
        """Get recommended action for alert."""
        if alert.severity == 'critical':
            return "IMMEDIATE RESPONSE REQUIRED - Contact emergency services"
        elif alert.severity == 'high':
            return "URGENT - Notify on-call staff immediately"
        elif escalation_needed:
            return "ESCALATION - Multiple alerts detected, increase monitoring"
        else:
            return "Monitor and document"
    
    def _get_response_time_requirement(self, severity: str) -> str:
        """Get response time requirement for severity level."""
        requirements = {
            'critical': 'Immediate (within 1 minute)',
            'high': 'Urgent (within 5 minutes)',
            'medium': 'Standard (within 15 minutes)',
            'low': 'Routine (within 1 hour)'
        }
        return requirements.get(severity, 'Standard')


def create_healthcare_alert_callback():
    """Create a healthcare-appropriate alert callback function."""
    alert_manager = HealthcareAlertManager()
    
    def alert_callback(alert: AlertEvent):
        """Process alert with healthcare-specific logic."""
        # Process alert through healthcare manager
        response = alert_manager.process_alert(alert)
        
        # Log healthcare response
        logger.info(f"Healthcare Alert Response: {response['recommended_action']}")
        
        # Here you would integrate with actual healthcare systems
        # For example: send SMS, email, or integrate with nurse call system
        
        # Simulate healthcare system integration
        if response['escalation_needed']:
            logger.warning("ESCALATION TRIGGERED - Multiple alerts detected")
        
        if alert.severity in ['critical', 'high']:
            logger.warning(f"URGENT ALERT: {response['recommended_action']}")
    
    return alert_callback


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize detector with healthcare alert callback
        alert_callback = create_healthcare_alert_callback()
        
        detector = RealTimeFallDetector(
            model_path="models/fall_detection_lstm",
            camera_source=0,  # Webcam
            alert_callback=alert_callback,
            confidence_threshold=0.7
        )
        
        # Run demo
        detector.run_demo(display_skeleton=True, save_video=False)
        
    except Exception as e:
        logger.error(f"Error running fall detection: {str(e)}")
        print("Please ensure you have a trained model available at 'models/fall_detection_lstm'")
        print("You can train a model using the model_training.py script first.") 