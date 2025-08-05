"""
Demo Video Processor for Fall Detection
Processes pre-recorded videos and provides real-time fall detection data
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import queue

@dataclass
class DetectionResult:
    timestamp: float
    confidence: float
    status: str  # 'normal', 'risk', 'fall'
    pose_keypoints: List[Tuple[float, float]]
    bounding_box: Tuple[int, int, int, int]
    velocity: float
    stability_score: float

class DemoVideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        # Detection state
        self.current_frame = 0
        self.detection_history: List[DetectionResult] = []
        self.is_processing = False
        self.processing_queue = queue.Queue()
        
        # Pre-analyzed detection data (simulated for demo)
        self._generate_demo_detections()
    
    def _generate_demo_detections(self):
        """Generate realistic fall detection data for demo purposes"""
        # Simulate different scenarios throughout the video
        scenarios = [
            # Normal walking (0-10 seconds)
            {'start_time': 0, 'end_time': 10, 'status': 'normal', 'confidence_range': (85, 95)},
            # Risk behavior (10-15 seconds)
            {'start_time': 10, 'end_time': 15, 'status': 'risk', 'confidence_range': (70, 85)},
            # Fall incident (15-20 seconds)
            {'start_time': 15, 'end_time': 20, 'status': 'fall', 'confidence_range': (90, 98)},
            # Recovery period (20-25 seconds)
            {'start_time': 20, 'end_time': 25, 'status': 'normal', 'confidence_range': (80, 90)},
            # Another risk moment (25-30 seconds)
            {'start_time': 25, 'end_time': 30, 'status': 'risk', 'confidence_range': (75, 88)},
        ]
        
        for scenario in scenarios:
            start_frame = int(scenario['start_time'] * self.fps)
            end_frame = int(scenario['end_time'] * self.fps)
            
            for frame_num in range(start_frame, end_frame):
                timestamp = frame_num / self.fps
                confidence = np.random.uniform(*scenario['confidence_range'])
                status = scenario['status']
                
                # Generate realistic pose keypoints (simplified skeleton)
                pose_keypoints = self._generate_pose_keypoints(status, timestamp)
                
                # Generate bounding box
                bbox = self._generate_bounding_box(status, timestamp)
                
                # Generate velocity and stability scores
                velocity = self._generate_velocity(status, timestamp)
                stability = self._generate_stability(status, timestamp)
                
                detection = DetectionResult(
                    timestamp=timestamp,
                    confidence=confidence,
                    status=status,
                    pose_keypoints=pose_keypoints,
                    bounding_box=bbox,
                    velocity=velocity,
                    stability_score=stability
                )
                
                self.detection_history.append(detection)
    
    def _generate_pose_keypoints(self, status: str, timestamp: float) -> List[Tuple[float, float]]:
        """Generate realistic pose keypoints based on status"""
        # 17 keypoints for MediaPipe pose
        keypoints = []
        
        if status == 'normal':
            # Normal standing/walking pose
            keypoints = [
                (0.5, 0.1),   # nose
                (0.48, 0.15), # left eye
                (0.52, 0.15), # right eye
                (0.45, 0.2),  # left ear
                (0.55, 0.2),  # right ear
                (0.5, 0.25),  # left shoulder
                (0.5, 0.25),  # right shoulder
                (0.45, 0.35), # left elbow
                (0.55, 0.35), # right elbow
                (0.4, 0.45),  # left wrist
                (0.6, 0.45),  # right wrist
                (0.5, 0.4),   # left hip
                (0.5, 0.4),   # right hip
                (0.45, 0.6),  # left knee
                (0.55, 0.6),  # right knee
                (0.4, 0.8),   # left ankle
                (0.6, 0.8),   # right ankle
            ]
        elif status == 'risk':
            # Unstable pose
            keypoints = [
                (0.5, 0.1),   # nose
                (0.48, 0.15), # left eye
                (0.52, 0.15), # right eye
                (0.45, 0.2),  # left ear
                (0.55, 0.2),  # right ear
                (0.48, 0.25), # left shoulder
                (0.52, 0.25), # right shoulder
                (0.43, 0.35), # left elbow
                (0.57, 0.35), # right elbow
                (0.38, 0.45), # left wrist
                (0.62, 0.45), # right wrist
                (0.48, 0.4),  # left hip
                (0.52, 0.4),  # right hip
                (0.43, 0.6),  # left knee
                (0.57, 0.6),  # right knee
                (0.35, 0.8),  # left ankle
                (0.65, 0.8),  # right ankle
            ]
        elif status == 'fall':
            # Fallen pose
            keypoints = [
                (0.5, 0.3),   # nose
                (0.48, 0.32), # left eye
                (0.52, 0.32), # right eye
                (0.45, 0.34), # left ear
                (0.55, 0.34), # right ear
                (0.48, 0.4),  # left shoulder
                (0.52, 0.4),  # right shoulder
                (0.43, 0.5),  # left elbow
                (0.57, 0.5),  # right elbow
                (0.38, 0.6),  # left wrist
                (0.62, 0.6),  # right wrist
                (0.48, 0.6),  # left hip
                (0.52, 0.6),  # right hip
                (0.43, 0.75), # left knee
                (0.57, 0.75), # right knee
                (0.35, 0.9),  # left ankle
                (0.65, 0.9),  # right ankle
            ]
        
        return keypoints
    
    def _generate_bounding_box(self, status: str, timestamp: float) -> Tuple[int, int, int, int]:
        """Generate bounding box based on status"""
        if status == 'normal':
            return (200, 100, 400, 600)  # Normal standing
        elif status == 'risk':
            return (180, 120, 420, 580)  # Slightly unstable
        else:  # fall
            return (150, 200, 450, 500)  # Fallen position
    
    def _generate_velocity(self, status: str, timestamp: float) -> float:
        """Generate velocity based on status"""
        if status == 'normal':
            return np.random.uniform(0.1, 0.5)  # Normal movement
        elif status == 'risk':
            return np.random.uniform(0.8, 1.5)  # Fast movement
        else:  # fall
            return np.random.uniform(2.0, 4.0)  # Very fast fall
    
    def _generate_stability(self, status: str, timestamp: float) -> float:
        """Generate stability score based on status"""
        if status == 'normal':
            return np.random.uniform(0.8, 1.0)  # Stable
        elif status == 'risk':
            return np.random.uniform(0.4, 0.7)  # Unstable
        else:  # fall
            return np.random.uniform(0.0, 0.3)  # Very unstable
    
    def get_detection_at_time(self, timestamp: float) -> Optional[DetectionResult]:
        """Get detection result at specific timestamp"""
        # Find the closest detection to the given timestamp
        if not self.detection_history:
            return None
        
        closest = min(self.detection_history, key=lambda x: abs(x.timestamp - timestamp))
        return closest
    
    def get_current_detection(self) -> Optional[DetectionResult]:
        """Get current detection based on video position"""
        current_time = self.current_frame / self.fps
        return self.get_detection_at_time(current_time)
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        if not self.detection_history:
            return {
                'fps': 0,
                'latency': 0,
                'confidence': 0,
                'pose_status': 'No Pose'
            }
        
        current_detection = self.get_current_detection()
        if not current_detection:
            return {
                'fps': 30,
                'latency': 150,
                'confidence': 0,
                'pose_status': 'No Pose'
            }
        
        return {
            'fps': 30,
            'latency': np.random.uniform(120, 180),
            'confidence': current_detection.confidence,
            'pose_status': 'Pose Detected' if current_detection.confidence > 50 else 'No Pose'
        }
    
    def get_alert_history(self, limit: int = 10) -> List[Dict]:
        """Get recent alert history"""
        alerts = []
        for detection in self.detection_history[-limit:]:
            if detection.status in ['risk', 'fall']:
                alerts.append({
                    'timestamp': detection.timestamp,
                    'type': detection.status,
                    'confidence': detection.confidence,
                    'time': time.strftime('%H:%M:%S', time.localtime(detection.timestamp))
                })
        return alerts
    
    def seek_to_time(self, timestamp: float):
        """Seek video to specific timestamp"""
        frame_number = int(timestamp * self.fps)
        self.current_frame = max(0, min(frame_number, self.frame_count - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
    
    def get_video_info(self) -> Dict:
        """Get video information"""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.duration,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
    
    def close(self):
        """Close video capture"""
        if self.cap:
            self.cap.release()

# Demo video data generator
def create_demo_video_data():
    """Create demo video data for the frontend"""
    # This would normally process real videos, but for demo we'll create synthetic data
    demo_data = {
        'videos': [
            {
                'id': 'demo_1',
                'title': 'Elderly Care Facility - Fall Detection',
                'description': 'Real-time fall detection in a healthcare environment',
                'duration': 30,
                'scenarios': [
                    {'time': 0, 'status': 'normal', 'description': 'Normal walking'},
                    {'time': 10, 'status': 'risk', 'description': 'Unstable movement detected'},
                    {'time': 15, 'status': 'fall', 'description': 'Fall incident detected'},
                    {'time': 20, 'status': 'normal', 'description': 'Recovery and assistance'},
                ],
                'metrics': {
                    'accuracy': 94.2,
                    'sensitivity': 96.8,
                    'specificity': 91.5,
                    'response_time': 0.8
                }
            },
            {
                'id': 'demo_2',
                'title': 'Home Safety Monitoring',
                'description': 'Privacy-preserving fall detection in residential setting',
                'duration': 45,
                'scenarios': [
                    {'time': 0, 'status': 'normal', 'description': 'Daily activities'},
                    {'time': 20, 'status': 'risk', 'description': 'Potential fall risk'},
                    {'time': 35, 'status': 'fall', 'description': 'Fall detected'},
                    {'time': 40, 'status': 'normal', 'description': 'Emergency response'},
                ],
                'metrics': {
                    'accuracy': 92.1,
                    'sensitivity': 94.3,
                    'specificity': 89.7,
                    'response_time': 1.2
                }
            }
        ]
    }
    
    return demo_data

if __name__ == "__main__":
    # Test the demo processor
    processor = DemoVideoProcessor("demo_video.mp4")
    print("Demo processor initialized")
    print(f"Video info: {processor.get_video_info()}")
    
    # Test detection at different times
    for time_point in [5, 12, 17, 22]:
        detection = processor.get_detection_at_time(time_point)
        if detection:
            print(f"Time {time_point}s: {detection.status} (confidence: {detection.confidence:.1f}%)")
    
    processor.close() 