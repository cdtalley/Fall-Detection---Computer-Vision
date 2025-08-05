"""
Privacy-Preserving Fall Detection Data Processing Module

This module handles video input processing, MediaPipe pose estimation, and feature extraction
while maintaining privacy by only working with skeleton keypoints.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PoseKeypoints:
    """Data class to store pose keypoints with privacy-safe structure"""
    nose: Tuple[float, float, float]
    left_eye: Tuple[float, float, float]
    right_eye: Tuple[float, float, float]
    left_ear: Tuple[float, float, float]
    right_ear: Tuple[float, float, float]
    left_shoulder: Tuple[float, float, float]
    right_shoulder: Tuple[float, float, float]
    left_elbow: Tuple[float, float, float]
    right_elbow: Tuple[float, float, float]
    left_wrist: Tuple[float, float, float]
    right_wrist: Tuple[float, float, float]
    left_hip: Tuple[float, float, float]
    right_hip: Tuple[float, float, float]
    left_knee: Tuple[float, float, float]
    right_knee: Tuple[float, float, float]
    left_ankle: Tuple[float, float, float]
    right_ankle: Tuple[float, float, float]

class PrivacyPreservingPoseProcessor:
    """
    Privacy-preserving pose estimation processor using MediaPipe.
    
    This class ensures that only skeleton keypoints are extracted and processed,
    maintaining privacy by avoiding any facial recognition or identifying features.
    """
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the pose processor with privacy-focused configuration.
        
        Args:
            static_image_mode: Whether to process static images or video
            model_complexity: Model complexity (0, 1, or 2)
            smooth_landmarks: Whether to smooth landmarks across frames
            enable_segmentation: Whether to enable segmentation (disabled for privacy)
            smooth_segmentation: Whether to smooth segmentation
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose estimation with privacy-focused settings
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,  # Disabled for privacy
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define keypoint indices for privacy-safe processing
        self.keypoint_indices = {
            'nose': 0,
            'left_eye': 2,
            'right_eye': 5,
            'left_ear': 7,
            'right_ear': 8,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        logger.info("Privacy-preserving pose processor initialized successfully")
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[PoseKeypoints]:
        """
        Extract pose keypoints from a video frame while preserving privacy.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            PoseKeypoints object with normalized coordinates, or None if no pose detected
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Extract keypoints with privacy-safe structure
            landmarks = results.pose_landmarks.landmark
            
            keypoints = PoseKeypoints(
                nose=self._extract_landmark(landmarks, self.keypoint_indices['nose']),
                left_eye=self._extract_landmark(landmarks, self.keypoint_indices['left_eye']),
                right_eye=self._extract_landmark(landmarks, self.keypoint_indices['right_eye']),
                left_ear=self._extract_landmark(landmarks, self.keypoint_indices['left_ear']),
                right_ear=self._extract_landmark(landmarks, self.keypoint_indices['right_ear']),
                left_shoulder=self._extract_landmark(landmarks, self.keypoint_indices['left_shoulder']),
                right_shoulder=self._extract_landmark(landmarks, self.keypoint_indices['right_shoulder']),
                left_elbow=self._extract_landmark(landmarks, self.keypoint_indices['left_elbow']),
                right_elbow=self._extract_landmark(landmarks, self.keypoint_indices['right_elbow']),
                left_wrist=self._extract_landmark(landmarks, self.keypoint_indices['left_wrist']),
                right_wrist=self._extract_landmark(landmarks, self.keypoint_indices['right_wrist']),
                left_hip=self._extract_landmark(landmarks, self.keypoint_indices['left_hip']),
                right_hip=self._extract_landmark(landmarks, self.keypoint_indices['right_hip']),
                left_knee=self._extract_landmark(landmarks, self.keypoint_indices['left_knee']),
                right_knee=self._extract_landmark(landmarks, self.keypoint_indices['right_knee']),
                left_ankle=self._extract_landmark(landmarks, self.keypoint_indices['left_ankle']),
                right_ankle=self._extract_landmark(landmarks, self.keypoint_indices['right_ankle'])
            )
            
            return keypoints
            
        except Exception as e:
            logger.error(f"Error extracting keypoints: {str(e)}")
            return None
    
    def _extract_landmark(self, landmarks: List, index: int) -> Tuple[float, float, float]:
        """Extract a single landmark with normalized coordinates."""
        if index < len(landmarks):
            landmark = landmarks[index]
            return (landmark.x, landmark.y, landmark.z)
        return (0.0, 0.0, 0.0)
    
    def extract_features(self, keypoints: PoseKeypoints) -> Dict[str, float]:
        """
        Extract fall detection features from pose keypoints.
        
        Args:
            keypoints: PoseKeypoints object
            
        Returns:
            Dictionary of computed features for fall detection
        """
        features = {}
        
        # Calculate body proportions and positions
        features.update(self._calculate_body_proportions(keypoints))
        features.update(self._calculate_center_of_mass(keypoints))
        features.update(self._calculate_velocity_features(keypoints))
        features.update(self._calculate_pose_stability(keypoints))
        
        return features
    
    def _calculate_body_proportions(self, keypoints: PoseKeypoints) -> Dict[str, float]:
        """Calculate body proportions for fall detection."""
        features = {}
        
        # Torso length (shoulder to hip)
        left_torso = np.linalg.norm(np.array(keypoints.left_shoulder[:2]) - np.array(keypoints.left_hip[:2]))
        right_torso = np.linalg.norm(np.array(keypoints.right_shoulder[:2]) - np.array(keypoints.right_hip[:2]))
        features['torso_length'] = (left_torso + right_torso) / 2
        
        # Leg length (hip to ankle)
        left_leg = np.linalg.norm(np.array(keypoints.left_hip[:2]) - np.array(keypoints.left_ankle[:2]))
        right_leg = np.linalg.norm(np.array(keypoints.right_hip[:2]) - np.array(keypoints.right_ankle[:2]))
        features['leg_length'] = (left_leg + right_leg) / 2
        
        # Arm length (shoulder to wrist)
        left_arm = np.linalg.norm(np.array(keypoints.left_shoulder[:2]) - np.array(keypoints.left_wrist[:2]))
        right_arm = np.linalg.norm(np.array(keypoints.right_shoulder[:2]) - np.array(keypoints.right_wrist[:2]))
        features['arm_length'] = (left_arm + right_arm) / 2
        
        return features
    
    def _calculate_center_of_mass(self, keypoints: PoseKeypoints) -> Dict[str, float]:
        """Calculate center of mass position and movement."""
        features = {}
        
        # Calculate center of mass (simplified as midpoint between shoulders and hips)
        shoulder_center = np.array([
            (keypoints.left_shoulder[0] + keypoints.right_shoulder[0]) / 2,
            (keypoints.left_shoulder[1] + keypoints.right_shoulder[1]) / 2
        ])
        
        hip_center = np.array([
            (keypoints.left_hip[0] + keypoints.right_hip[0]) / 2,
            (keypoints.left_hip[1] + keypoints.right_hip[1]) / 2
        ])
        
        com = (shoulder_center + hip_center) / 2
        
        features['com_x'] = com[0]
        features['com_y'] = com[1]
        features['com_height'] = 1.0 - com[1]  # Normalized height (1.0 = top of frame)
        
        return features
    
    def _calculate_velocity_features(self, keypoints: PoseKeypoints) -> Dict[str, float]:
        """Calculate velocity-based features (placeholder for temporal analysis)."""
        # This would be implemented with frame history in real-time processing
        features = {}
        
        # Placeholder features - in real implementation, these would be calculated
        # from the difference between current and previous frame keypoints
        features['velocity_magnitude'] = 0.0
        features['acceleration_magnitude'] = 0.0
        
        return features
    
    def _calculate_pose_stability(self, keypoints: PoseKeypoints) -> Dict[str, float]:
        """Calculate pose stability indicators."""
        features = {}
        
        # Calculate shoulder tilt (indicator of balance)
        shoulder_tilt = abs(keypoints.left_shoulder[1] - keypoints.right_shoulder[1])
        features['shoulder_tilt'] = shoulder_tilt
        
        # Calculate hip tilt
        hip_tilt = abs(keypoints.left_hip[1] - keypoints.right_hip[1])
        features['hip_tilt'] = hip_tilt
        
        # Calculate overall stability score
        features['stability_score'] = 1.0 - (shoulder_tilt + hip_tilt) / 2
        
        return features
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: PoseKeypoints) -> np.ndarray:
        """
        Draw skeleton overlay on frame for visualization (privacy-safe).
        
        Args:
            frame: Input video frame
            keypoints: PoseKeypoints object
            
        Returns:
            Frame with skeleton overlay
        """
        # Convert keypoints back to MediaPipe format for drawing
        landmarks = []
        for i in range(33):  # MediaPipe pose has 33 landmarks
            if i in self.keypoint_indices.values():
                # Find the keypoint name for this index
                keypoint_name = [k for k, v in self.keypoint_indices.items() if v == i][0]
                keypoint_value = getattr(keypoints, keypoint_name)
                landmarks.append(keypoint_value)
            else:
                landmarks.append((0.0, 0.0, 0.0))
        
        # Create a mock results object for drawing
        class MockResults:
            def __init__(self, landmarks):
                self.pose_landmarks = MockLandmarks(landmarks)
        
        class MockLandmarks:
            def __init__(self, landmarks):
                self.landmark = [MockLandmark(lm) for lm in landmarks]
        
        class MockLandmark:
            def __init__(self, coords):
                self.x, self.y, self.z = coords
        
        mock_results = MockResults(landmarks)
        
        # Draw the skeleton
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            mock_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return annotated_frame
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> List[Dict[str, float]]:
        """
        Process a video file and extract features from all frames.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save processed video
            
        Returns:
            List of feature dictionaries for each frame
        """
        cap = cv2.VideoCapture(video_path)
        features_list = []
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keypoints
            keypoints = self.extract_keypoints(frame)
            
            if keypoints:
                # Extract features
                features = self.extract_features(keypoints)
                features['frame_number'] = frame_count
                features_list.append(features)
                
                # Draw skeleton if output video requested
                if output_path:
                    annotated_frame = self.draw_skeleton(frame, keypoints)
                    out.write(annotated_frame)
            else:
                # No pose detected, add empty features
                features = {'frame_number': frame_count}
                features_list.append(features)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
        
        cap.release()
        if output_path:
            out.release()
        
        logger.info(f"Video processing complete. Processed {frame_count} frames.")
        return features_list
    
    def __del__(self):
        """Clean up MediaPipe pose processor."""
        if hasattr(self, 'pose'):
            self.pose.close()


class VideoProcessor:
    """High-level video processing class for fall detection."""
    
    def __init__(self, pose_processor: PrivacyPreservingPoseProcessor):
        """
        Initialize video processor.
        
        Args:
            pose_processor: Initialized pose processor
        """
        self.pose_processor = pose_processor
        self.frame_history = []
        self.max_history = 30  # Keep last 30 frames for temporal analysis
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Process a single frame for fall detection.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary containing keypoints, features, and processing metadata
        """
        # Extract keypoints
        keypoints = self.pose_processor.extract_keypoints(frame)
        
        result = {
            'timestamp': len(self.frame_history),
            'keypoints_detected': keypoints is not None,
            'features': {},
            'keypoints': keypoints
        }
        
        if keypoints:
            # Extract features
            features = self.pose_processor.extract_features(keypoints)
            
            # Add temporal features if we have history
            if self.frame_history:
                temporal_features = self._calculate_temporal_features(keypoints)
                features.update(temporal_features)
            
            result['features'] = features
            
            # Update frame history
            self.frame_history.append(keypoints)
            if len(self.frame_history) > self.max_history:
                self.frame_history.pop(0)
        
        return result
    
    def _calculate_temporal_features(self, current_keypoints: PoseKeypoints) -> Dict[str, float]:
        """Calculate temporal features using frame history."""
        features = {}
        
        if len(self.frame_history) < 2:
            return features
        
        # Get previous keypoints
        prev_keypoints = self.frame_history[-1]
        
        # Calculate velocity (movement between frames)
        current_com = np.array([
            (current_keypoints.left_shoulder[0] + current_keypoints.right_shoulder[0]) / 2,
            (current_keypoints.left_shoulder[1] + current_keypoints.right_shoulder[1]) / 2
        ])
        
        prev_com = np.array([
            (prev_keypoints.left_shoulder[0] + prev_keypoints.right_shoulder[0]) / 2,
            (prev_keypoints.left_shoulder[1] + prev_keypoints.right_shoulder[1]) / 2
        ])
        
        velocity = current_com - prev_com
        features['velocity_x'] = velocity[0]
        features['velocity_y'] = velocity[1]
        features['velocity_magnitude'] = np.linalg.norm(velocity)
        
        # Calculate acceleration if we have more history
        if len(self.frame_history) >= 3:
            prev_prev_keypoints = self.frame_history[-2]
            prev_prev_com = np.array([
                (prev_prev_keypoints.left_shoulder[0] + prev_prev_keypoints.right_shoulder[0]) / 2,
                (prev_prev_keypoints.left_shoulder[1] + prev_prev_keypoints.right_shoulder[1]) / 2
            ])
            
            prev_velocity = prev_com - prev_prev_com
            acceleration = velocity - prev_velocity
            features['acceleration_magnitude'] = np.linalg.norm(acceleration)
        
        return features


def create_sample_data():
    """Create sample pose data for testing and demonstration."""
    sample_keypoints = PoseKeypoints(
        nose=(0.5, 0.2, 0.0),
        left_eye=(0.48, 0.18, 0.0),
        right_eye=(0.52, 0.18, 0.0),
        left_ear=(0.45, 0.18, 0.0),
        right_ear=(0.55, 0.18, 0.0),
        left_shoulder=(0.4, 0.35, 0.0),
        right_shoulder=(0.6, 0.35, 0.0),
        left_elbow=(0.3, 0.5, 0.0),
        right_elbow=(0.7, 0.5, 0.0),
        left_wrist=(0.25, 0.65, 0.0),
        right_wrist=(0.75, 0.65, 0.0),
        left_hip=(0.4, 0.65, 0.0),
        right_hip=(0.6, 0.65, 0.0),
        left_knee=(0.4, 0.85, 0.0),
        right_knee=(0.6, 0.85, 0.0),
        left_ankle=(0.4, 0.95, 0.0),
        right_ankle=(0.6, 0.95, 0.0)
    )
    
    return sample_keypoints


if __name__ == "__main__":
    # Example usage
    processor = PrivacyPreservingPoseProcessor()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        keypoints = processor.extract_keypoints(frame)
        
        if keypoints:
            # Extract features
            features = processor.extract_features(keypoints)
            print(f"Features: {features}")
            
            # Draw skeleton
            annotated_frame = processor.draw_skeleton(frame, keypoints)
            cv2.imshow('Privacy-Preserving Fall Detection', annotated_frame)
        else:
            cv2.imshow('Privacy-Preserving Fall Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows() 