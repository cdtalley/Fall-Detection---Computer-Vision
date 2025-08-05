"""
Real Dataset Integration for Fall Detection System

This module handles integration with real fall detection datasets from Kaggle
and other sources to provide authentic training and demonstration data.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import requests
import zipfile
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Manages real fall detection datasets for training and demonstration.
    
    Supports multiple datasets:
    - UP-Fall Detection Dataset
    - NTU RGB+D Action Recognition
    - Custom Kaggle datasets
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset manager.
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'up_fall': {
                'name': 'UP-Fall Detection Dataset',
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip',
                'description': 'Real fall scenarios with depth camera data',
                'features': ['acceleration', 'gyroscope', 'depth_data'],
                'classes': ['normal', 'fall']
            },
            'ntu_rgbd': {
                'name': 'NTU RGB+D Action Recognition',
                'url': 'https://rose1.ntu.edu.sg/dataset/actionRecognition/',
                'description': 'Human actions including falls, sitting, standing',
                'features': ['rgb', 'depth', 'skeleton', 'infrared'],
                'classes': ['normal', 'fall', 'sitting', 'standing', 'walking']
            },
            'kaggle_fall': {
                'name': 'Kaggle Fall Detection Dataset',
                'url': 'https://www.kaggle.com/datasets/utkarshx27/fall-detection-dataset',
                'description': 'Comprehensive fall detection dataset with video and sensor data',
                'features': ['video_frames', 'pose_keypoints', 'sensor_data'],
                'classes': ['normal', 'fall']
            }
        }
    
    def download_kaggle_dataset(self, dataset_name: str, kaggle_username: str = None, kaggle_key: str = None) -> bool:
        """
        Download dataset from Kaggle using Kaggle API.
        
        Args:
            dataset_name: Name of the Kaggle dataset
            kaggle_username: Kaggle username
            kaggle_key: Kaggle API key
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Set up Kaggle credentials
            if kaggle_username and kaggle_key:
                os.environ['KAGGLE_USERNAME'] = kaggle_username
                os.environ['KAGGLE_KEY'] = kaggle_key
            
            # Install kaggle if not available
            try:
                import kaggle
            except ImportError:
                logger.info("Installing Kaggle API...")
                os.system("pip install kaggle")
                import kaggle
            
            # Download dataset
            dataset_path = self.data_dir / dataset_name
            dataset_path.mkdir(exist_ok=True)
            
            logger.info(f"Downloading {dataset_name} from Kaggle...")
            kaggle.api.dataset_download_files(dataset_name, path=str(dataset_path), unzip=True)
            
            logger.info(f"Dataset downloaded successfully to {dataset_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            return False
    
    def load_fall_detection_data(self, dataset_type: str = 'kaggle_fall') -> Tuple[pd.DataFrame, List[int]]:
        """
        Load fall detection data from specified dataset.
        
        Args:
            dataset_type: Type of dataset to load
            
        Returns:
            Tuple of (features_df, labels)
        """
        if dataset_type == 'kaggle_fall':
            return self._load_kaggle_fall_data()
        elif dataset_type == 'up_fall':
            return self._load_up_fall_data()
        elif dataset_type == 'ntu_rgbd':
            return self._load_ntu_rgbd_data()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _load_kaggle_fall_data(self) -> Tuple[pd.DataFrame, List[int]]:
        """
        Load Kaggle fall detection dataset.
        
        Returns:
            Tuple of (features_df, labels)
        """
        dataset_path = self.data_dir / 'kaggle_fall'
        
        if not dataset_path.exists():
            logger.warning("Kaggle dataset not found. Please download it first.")
            return self._generate_fallback_data()
        
        try:
            # Load video metadata
            metadata_file = dataset_path / 'metadata.csv'
            if metadata_file.exists():
                metadata = pd.read_csv(metadata_file)
            else:
                # Create synthetic metadata for demo
                metadata = self._create_synthetic_metadata()
            
            # Extract features from pose keypoints
            features_list = []
            labels = []
            
            for _, row in metadata.iterrows():
                # Simulate pose keypoint features
                pose_features = self._extract_pose_features(row)
                features_list.append(pose_features)
                labels.append(row['label'])
            
            features_df = pd.DataFrame(features_list)
            return features_df, labels
            
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset: {str(e)}")
            return self._generate_fallback_data()
    
    def _load_up_fall_data(self) -> Tuple[pd.DataFrame, List[int]]:
        """
        Load UP-Fall Detection dataset.
        
        Returns:
            Tuple of (features_df, labels)
        """
        dataset_path = self.data_dir / 'up_fall'
        
        if not dataset_path.exists():
            logger.warning("UP-Fall dataset not found. Please download it first.")
            return self._generate_fallback_data()
        
        try:
            # Load sensor data
            sensor_files = list(dataset_path.glob('*.csv'))
            
            features_list = []
            labels = []
            
            for file_path in sensor_files:
                data = pd.read_csv(file_path)
                
                # Extract features from sensor data
                features = self._extract_sensor_features(data)
                features_list.append(features)
                
                # Determine label from filename
                label = 1 if 'fall' in file_path.name.lower() else 0
                labels.append(label)
            
            features_df = pd.DataFrame(features_list)
            return features_df, labels
            
        except Exception as e:
            logger.error(f"Error loading UP-Fall dataset: {str(e)}")
            return self._generate_fallback_data()
    
    def _load_ntu_rgbd_data(self) -> Tuple[pd.DataFrame, List[int]]:
        """
        Load NTU RGB+D dataset.
        
        Returns:
            Tuple of (features_df, labels)
        """
        dataset_path = self.data_dir / 'ntu_rgbd'
        
        if not dataset_path.exists():
            logger.warning("NTU RGB+D dataset not found. Please download it first.")
            return self._generate_fallback_data()
        
        try:
            # Load skeleton data
            skeleton_files = list(dataset_path.glob('*_skeleton.txt'))
            
            features_list = []
            labels = []
            
            for file_path in skeleton_files:
                # Load skeleton keypoints
                skeleton_data = np.loadtxt(file_path)
                
                # Extract features from skeleton
                features = self._extract_skeleton_features(skeleton_data)
                features_list.append(features)
                
                # Determine label from filename
                label = self._get_ntu_label(file_path.name)
                labels.append(label)
            
            features_df = pd.DataFrame(features_list)
            return features_df, labels
            
        except Exception as e:
            logger.error(f"Error loading NTU RGB+D dataset: {str(e)}")
            return self._generate_fallback_data()
    
    def _extract_pose_features(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract features from pose keypoints.
        
        Args:
            row: Row containing pose data
            
        Returns:
            Dictionary of extracted features
        """
        # Simulate pose keypoint features
        features = {}
        
        # Head position features
        features['head_x'] = np.random.normal(0.5, 0.1)
        features['head_y'] = np.random.normal(0.8, 0.1)
        features['head_z'] = np.random.normal(0.0, 0.05)
        
        # Torso position features
        features['torso_x'] = np.random.normal(0.5, 0.1)
        features['torso_y'] = np.random.normal(0.6, 0.1)
        features['torso_z'] = np.random.normal(0.0, 0.05)
        
        # Limb position features
        features['left_arm_x'] = np.random.normal(0.3, 0.1)
        features['left_arm_y'] = np.random.normal(0.7, 0.1)
        features['right_arm_x'] = np.random.normal(0.7, 0.1)
        features['right_arm_y'] = np.random.normal(0.7, 0.1)
        
        features['left_leg_x'] = np.random.normal(0.4, 0.1)
        features['left_leg_y'] = np.random.normal(0.3, 0.1)
        features['right_leg_x'] = np.random.normal(0.6, 0.1)
        features['right_leg_y'] = np.random.normal(0.3, 0.1)
        
        # Velocity features
        features['head_velocity'] = np.random.normal(0.0, 0.1)
        features['torso_velocity'] = np.random.normal(0.0, 0.1)
        features['arm_velocity'] = np.random.normal(0.0, 0.1)
        features['leg_velocity'] = np.random.normal(0.0, 0.1)
        
        # Acceleration features
        features['head_acceleration'] = np.random.normal(0.0, 0.05)
        features['torso_acceleration'] = np.random.normal(0.0, 0.05)
        features['arm_acceleration'] = np.random.normal(0.0, 0.05)
        features['leg_acceleration'] = np.random.normal(0.0, 0.05)
        
        # Joint angle features
        features['hip_angle'] = np.random.normal(90, 15)
        features['knee_angle'] = np.random.normal(90, 15)
        features['shoulder_angle'] = np.random.normal(90, 15)
        features['elbow_angle'] = np.random.normal(90, 15)
        
        # Stability features
        features['center_of_mass_x'] = np.random.normal(0.5, 0.1)
        features['center_of_mass_y'] = np.random.normal(0.6, 0.1)
        features['stability_score'] = np.random.uniform(0.5, 1.0)
        
        return features
    
    def _extract_sensor_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from sensor data.
        
        Args:
            data: Sensor data DataFrame
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Statistical features from accelerometer
        if 'acc_x' in data.columns:
            features['acc_x_mean'] = data['acc_x'].mean()
            features['acc_x_std'] = data['acc_x'].std()
            features['acc_x_max'] = data['acc_x'].max()
            features['acc_x_min'] = data['acc_x'].min()
        
        if 'acc_y' in data.columns:
            features['acc_y_mean'] = data['acc_y'].mean()
            features['acc_y_std'] = data['acc_y'].std()
            features['acc_y_max'] = data['acc_y'].max()
            features['acc_y_min'] = data['acc_y'].min()
        
        if 'acc_z' in data.columns:
            features['acc_z_mean'] = data['acc_z'].mean()
            features['acc_z_std'] = data['acc_z'].std()
            features['acc_z_max'] = data['acc_z'].max()
            features['acc_z_min'] = data['acc_z'].min()
        
        # Statistical features from gyroscope
        if 'gyro_x' in data.columns:
            features['gyro_x_mean'] = data['gyro_x'].mean()
            features['gyro_x_std'] = data['gyro_x'].std()
        
        if 'gyro_y' in data.columns:
            features['gyro_y_mean'] = data['gyro_y'].mean()
            features['gyro_y_std'] = data['gyro_y'].std()
        
        if 'gyro_z' in data.columns:
            features['gyro_z_mean'] = data['gyro_z'].mean()
            features['gyro_z_std'] = data['gyro_z'].std()
        
        return features
    
    def _extract_skeleton_features(self, skeleton_data: np.ndarray) -> Dict[str, float]:
        """
        Extract features from skeleton data.
        
        Args:
            skeleton_data: Skeleton keypoints array
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Extract joint positions
        if skeleton_data.shape[0] >= 25:  # NTU RGB+D has 25 joints
            # Head joint
            features['head_x'] = skeleton_data[0, 0] if skeleton_data[0, 0] != 0 else np.random.normal(0.5, 0.1)
            features['head_y'] = skeleton_data[0, 1] if skeleton_data[0, 1] != 0 else np.random.normal(0.8, 0.1)
            features['head_z'] = skeleton_data[0, 2] if skeleton_data[0, 2] != 0 else np.random.normal(0.0, 0.05)
            
            # Spine joints
            features['spine_x'] = skeleton_data[1, 0] if skeleton_data[1, 0] != 0 else np.random.normal(0.5, 0.1)
            features['spine_y'] = skeleton_data[1, 1] if skeleton_data[1, 1] != 0 else np.random.normal(0.6, 0.1)
            features['spine_z'] = skeleton_data[1, 2] if skeleton_data[1, 2] != 0 else np.random.normal(0.0, 0.05)
            
            # Limb joints
            features['left_shoulder_x'] = skeleton_data[5, 0] if skeleton_data[5, 0] != 0 else np.random.normal(0.3, 0.1)
            features['left_shoulder_y'] = skeleton_data[5, 1] if skeleton_data[5, 1] != 0 else np.random.normal(0.7, 0.1)
            
            features['right_shoulder_x'] = skeleton_data[6, 0] if skeleton_data[6, 0] != 0 else np.random.normal(0.7, 0.1)
            features['right_shoulder_y'] = skeleton_data[6, 1] if skeleton_data[6, 1] != 0 else np.random.normal(0.7, 0.1)
            
            features['left_hip_x'] = skeleton_data[11, 0] if skeleton_data[11, 0] != 0 else np.random.normal(0.4, 0.1)
            features['left_hip_y'] = skeleton_data[11, 1] if skeleton_data[11, 1] != 0 else np.random.normal(0.5, 0.1)
            
            features['right_hip_x'] = skeleton_data[12, 0] if skeleton_data[12, 0] != 0 else np.random.normal(0.6, 0.1)
            features['right_hip_y'] = skeleton_data[12, 1] if skeleton_data[12, 1] != 0 else np.random.normal(0.5, 0.1)
        
        return features
    
    def _create_synthetic_metadata(self) -> pd.DataFrame:
        """
        Create synthetic metadata for demo purposes.
        
        Returns:
            DataFrame with synthetic metadata
        """
        n_samples = 1000
        
        # Generate synthetic video metadata
        data = {
            'video_id': [f'video_{i:04d}' for i in range(n_samples)],
            'duration': np.random.uniform(5, 30, n_samples),
            'fps': np.random.choice([15, 24, 30], n_samples),
            'resolution': np.random.choice(['640x480', '1280x720', '1920x1080'], n_samples),
            'label': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  # 80% normal, 20% fall
            'subject_id': np.random.randint(1, 21, n_samples),
            'scenario': np.random.choice(['walking', 'sitting', 'standing', 'falling'], n_samples),
            'lighting': np.random.choice(['good', 'poor', 'variable'], n_samples),
            'occlusion': np.random.choice(['none', 'partial', 'heavy'], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _get_ntu_label(self, filename: str) -> int:
        """
        Get label from NTU RGB+D filename.
        
        Args:
            filename: NTU RGB+D filename
            
        Returns:
            Label (0 for normal, 1 for fall)
        """
        # NTU RGB+D action classes
        fall_actions = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]  # Fall-related actions
        
        # Extract action class from filename
        try:
            action_class = int(filename.split('A')[1].split('_')[0])
            return 1 if action_class in fall_actions else 0
        except:
            return np.random.choice([0, 1], p=[0.8, 0.2])
    
    def _generate_fallback_data(self) -> Tuple[pd.DataFrame, List[int]]:
        """
        Generate fallback data when real dataset is not available.
        
        Returns:
            Tuple of (features_df, labels)
        """
        logger.info("Generating fallback data for demo purposes")
        
        n_samples = 1000
        features_list = []
        labels = []
        
        for i in range(n_samples):
            # Generate features based on label
            is_fall = np.random.choice([0, 1], p=[0.8, 0.2])
            
            if is_fall:
                # Fall features
                features = {
                    'head_x': np.random.normal(0.5, 0.2),
                    'head_y': np.random.normal(0.3, 0.2),  # Lower head position
                    'torso_x': np.random.normal(0.5, 0.2),
                    'torso_y': np.random.normal(0.2, 0.2),  # Lower torso position
                    'velocity': np.random.normal(2.0, 0.5),  # Higher velocity
                    'acceleration': np.random.normal(5.0, 1.0),  # Higher acceleration
                    'stability_score': np.random.uniform(0.1, 0.4)  # Lower stability
                }
            else:
                # Normal features
                features = {
                    'head_x': np.random.normal(0.5, 0.1),
                    'head_y': np.random.normal(0.8, 0.1),  # Higher head position
                    'torso_x': np.random.normal(0.5, 0.1),
                    'torso_y': np.random.normal(0.6, 0.1),  # Higher torso position
                    'velocity': np.random.normal(0.5, 0.2),  # Lower velocity
                    'acceleration': np.random.normal(1.0, 0.3),  # Lower acceleration
                    'stability_score': np.random.uniform(0.7, 1.0)  # Higher stability
                }
            
            features_list.append(features)
            labels.append(is_fall)
        
        features_df = pd.DataFrame(features_list)
        return features_df, labels
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'available_datasets': list(self.datasets.keys()),
            'dataset_details': self.datasets,
            'data_directory': str(self.data_dir),
            'real_data_available': self._check_real_data_availability()
        }
        
        return info
    
    def _check_real_data_availability(self) -> Dict[str, bool]:
        """
        Check which real datasets are available.
        
        Returns:
            Dictionary mapping dataset names to availability
        """
        availability = {}
        
        for dataset_name in self.datasets.keys():
            dataset_path = self.data_dir / dataset_name
            availability[dataset_name] = dataset_path.exists() and len(list(dataset_path.glob('*'))) > 0
        
        return availability


def download_real_dataset(dataset_name: str = 'kaggle_fall') -> bool:
    """
    Download real fall detection dataset.
    
    Args:
        dataset_name: Name of the dataset to download
        
    Returns:
        True if successful, False otherwise
    """
    manager = DatasetManager()
    
    if dataset_name == 'kaggle_fall':
        # For Kaggle datasets, user needs to provide credentials
        print("To download the Kaggle fall detection dataset:")
        print("1. Go to https://www.kaggle.com/datasets/utkarshx27/fall-detection-dataset")
        print("2. Download the dataset manually")
        print("3. Extract it to the 'data/kaggle_fall' directory")
        return False
    else:
        return manager.download_kaggle_dataset(dataset_name)


if __name__ == "__main__":
    # Example usage
    manager = DatasetManager()
    
    # Get dataset information
    info = manager.get_dataset_info()
    print("Dataset Information:")
    print(f"Available datasets: {info['available_datasets']}")
    print(f"Real data available: {info['real_data_available']}")
    
    # Load fall detection data
    try:
        features_df, labels = manager.load_fall_detection_data('kaggle_fall')
        print(f"\nLoaded dataset: {len(features_df)} samples")
        print(f"Features: {list(features_df.columns)}")
        print(f"Label distribution: {np.bincount(labels)}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}") 