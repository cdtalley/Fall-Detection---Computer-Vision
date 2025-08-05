"""
Create Demo Video with Skeleton Detection
Processes a video and generates frames with MediaPipe pose detection overlays
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
import json

class PoseVideoProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_video(self, input_video_path, output_video_path, output_frames_dir):
        """Process video and generate skeleton overlay frames"""
        cap = cv2.VideoCapture(input_video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory
        os.makedirs(output_frames_dir, exist_ok=True)
        
        # Video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_data = []
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose detection
            results = self.pose.process(rgb_frame)
            
            # Create output frame
            output_frame = frame.copy()
            
            # Draw skeleton if pose detected
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # Extract pose data
                pose_data = self.extract_pose_data(results.pose_landmarks, width, height)
                detection_data.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'pose_detected': True,
                    'landmarks': pose_data,
                    'status': self.analyze_pose_status(pose_data)
                })
            else:
                detection_data.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'pose_detected': False,
                    'landmarks': [],
                    'status': 'no_pose'
                })
            
            # Write frame to video
            out.write(output_frame)
            
            # Save individual frame
            frame_filename = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_frames_dir, frame_filename)
            cv2.imwrite(frame_path, output_frame)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        self.pose.close()
        
        # Save detection data
        detection_file = os.path.join(output_frames_dir, 'detection_data.json')
        with open(detection_file, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        print(f"Processing complete! Output saved to {output_video_path}")
        print(f"Detection data saved to {detection_file}")
        
        return detection_data

    def extract_pose_data(self, landmarks, width, height):
        """Extract pose landmark coordinates"""
        pose_data = []
        for landmark in landmarks.landmark:
            pose_data.append({
                'x': landmark.x * width,
                'y': landmark.y * height,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        return pose_data

    def analyze_pose_status(self, pose_data):
        """Analyze pose for fall detection"""
        if not pose_data:
            return 'no_pose'
        
        # Simple fall detection logic
        # Check if person is lying down (y-coordinate of key points)
        nose_y = pose_data[0]['y']  # Nose
        left_ankle_y = pose_data[27]['y']  # Left ankle
        right_ankle_y = pose_data[28]['y']  # Right ankle
        
        # If ankles are at similar height to nose, likely lying down
        if abs(nose_y - left_ankle_y) < 50 and abs(nose_y - right_ankle_y) < 50:
            return 'fall'
        
        # Check for unstable pose (large movement in key points)
        # This is a simplified check
        return 'normal'

def create_sample_video():
    """Create a simple sample video for demo purposes"""
    # Create a simple video with a person walking
    width, height = 640, 480
    fps = 30
    duration = 10  # 10 seconds
    
    output_path = "data/sample_videos/demo_walking.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_num in range(fps * duration):
        # Create a simple frame with a moving rectangle (representing a person)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background
        frame[:] = (50, 50, 50)
        
        # Draw a simple "person" (rectangle that moves)
        x = int(width/2 + 100 * np.sin(frame_num * 0.1))
        y = int(height/2 + 20 * np.sin(frame_num * 0.2))
        
        # Head
        cv2.circle(frame, (x, y-40), 15, (255, 255, 255), -1)
        # Body
        cv2.rectangle(frame, (x-20, y-20), (x+20, y+40), (255, 255, 255), -1)
        # Arms
        cv2.line(frame, (x-20, y-10), (x-40, y+10), (255, 255, 255), 3)
        cv2.line(frame, (x+20, y-10), (x+40, y+10), (255, 255, 255), 3)
        # Legs
        cv2.line(frame, (x-10, y+40), (x-20, y+80), (255, 255, 255), 3)
        cv2.line(frame, (x+10, y+40), (x+20, y+80), (255, 255, 255), 3)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {output_path}")
    return output_path

def main():
    """Main function to create demo video"""
    print("Creating fall detection demo video...")
    
    # Create sample video if it doesn't exist
    sample_video_path = "data/sample_videos/demo_walking.mp4"
    if not os.path.exists(sample_video_path):
        print("Creating sample video...")
        sample_video_path = create_sample_video()
    
    # Process the video
    processor = PoseVideoProcessor()
    
    output_video_path = "data/sample_videos/demo_with_skeleton.mp4"
    output_frames_dir = "data/sample_videos/frames"
    
    detection_data = processor.process_video(
        sample_video_path,
        output_video_path,
        output_frames_dir
    )
    
    print("Demo video creation complete!")
    print(f"Output video: {output_video_path}")
    print(f"Frames directory: {output_frames_dir}")

if __name__ == "__main__":
    main() 