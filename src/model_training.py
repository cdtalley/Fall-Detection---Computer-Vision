"""
Fall Detection Machine Learning Model Training Module

This module handles the training of LSTM/CNN models for fall detection with healthcare-specific
optimizations, focusing on sensitivity over specificity to prevent missed falls.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime

# Import our data processing module
from data_processing import PrivacyPreservingPoseProcessor, create_sample_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FallDetectionModel:
    """
    Fall detection model with healthcare-specific optimizations.
    
    This class implements LSTM and CNN models for fall detection, with special
    attention to healthcare requirements like high sensitivity and low false negatives.
    """
    
    def __init__(self, model_type: str = 'lstm', sequence_length: int = 30):
        """
        Initialize the fall detection model.
        
        Args:
            model_type: Type of model ('lstm', 'cnn', or 'hybrid')
            sequence_length: Number of frames to use for prediction
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Healthcare-specific parameters
        self.class_weights = {
            0: 1.0,  # Normal activity
            1: 3.0,  # High-risk movement (higher weight to prevent missed detection)
            2: 5.0   # Fall (highest weight - critical to detect)
        }
        
        logger.info(f"Initialized {model_type.upper()} model for fall detection")
    
    def build_lstm_model(self, input_shape: Tuple[int, int], num_classes: int = 3) -> tf.keras.Model:
        """
        Build LSTM model for temporal sequence analysis.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of output classes
            
        Returns:
            Compiled LSTM model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # LSTM layers for temporal analysis
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            layers.LSTM(64, return_sequences=False, dropout=0.2),
            
            # Dense layers for classification
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile with healthcare-optimized loss
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )
        
        return model
    
    def build_cnn_model(self, input_shape: Tuple[int, int], num_classes: int = 3) -> tf.keras.Model:
        """
        Build CNN model for spatial feature analysis.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of output classes
            
        Returns:
            Compiled CNN model
        """
        model = models.Sequential([
            # Reshape for CNN (add channel dimension)
            layers.Input(shape=input_shape),
            layers.Reshape((input_shape[0], input_shape[1], 1)),
            
            # Convolutional layers
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )
        
        return model
    
    def build_hybrid_model(self, input_shape: Tuple[int, int], num_classes: int = 3) -> tf.keras.Model:
        """
        Build hybrid CNN-LSTM model for comprehensive analysis.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of output classes
            
        Returns:
            Compiled hybrid model
        """
        # Input layer
        input_layer = layers.Input(shape=input_shape)
        
        # CNN branch for spatial features
        cnn_branch = layers.Reshape((input_shape[0], input_shape[1], 1))(input_layer)
        cnn_branch = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cnn_branch)
        cnn_branch = layers.BatchNormalization()(cnn_branch)
        cnn_branch = layers.MaxPooling2D((2, 2))(cnn_branch)
        cnn_branch = layers.Dropout(0.25)(cnn_branch)
        
        cnn_branch = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cnn_branch)
        cnn_branch = layers.BatchNormalization()(cnn_branch)
        cnn_branch = layers.GlobalAveragePooling2D()(cnn_branch)
        
        # LSTM branch for temporal features
        lstm_branch = layers.LSTM(128, return_sequences=True, dropout=0.2)(input_layer)
        lstm_branch = layers.LSTM(64, return_sequences=False, dropout=0.2)(lstm_branch)
        
        # Combine branches
        combined = layers.Concatenate()([cnn_branch, lstm_branch])
        
        # Dense layers
        combined = layers.Dense(128, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(64, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax')(combined)
        
        model = models.Model(inputs=input_layer, outputs=output)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )
        
        return model
    
    def prepare_sequences(self, features_df: pd.DataFrame, labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for model training.
        
        Args:
            features_df: DataFrame with extracted features
            labels: List of labels for each frame
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        sequences = []
        sequence_labels = []
        
        # Remove rows with missing features
        features_df = features_df.dropna()
        
        # Get feature columns (exclude non-feature columns)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['frame_number', 'timestamp']]
        
        self.feature_names = feature_cols
        
        # Create sequences
        for i in range(len(features_df) - self.sequence_length + 1):
            sequence = features_df[feature_cols].iloc[i:i + self.sequence_length].values
            
            # Check if sequence has any missing values
            if not np.isnan(sequence).any():
                sequences.append(sequence)
                # Use the label of the last frame in the sequence
                sequence_labels.append(labels[i + self.sequence_length - 1])
        
        X = np.array(sequences)
        y = np.array(sequence_labels)
        
        logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the fall detection model with healthcare-specific optimizations.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history dictionary
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Calculate class weights for healthcare optimization
        class_weights = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            if class_name in self.class_weights:
                class_weights[i] = self.class_weights[class_name]
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(self.label_encoder.classes_)
        
        if self.model_type == 'lstm':
            self.model = self.build_lstm_model(input_shape, num_classes)
        elif self.model_type == 'cnn':
            self.model = self.build_cnn_model(input_shape, num_classes)
        elif self.model_type == 'hybrid':
            self.model = self.build_hybrid_model(input_shape, num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Callbacks for training
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train_encoded,
            validation_data=(X_val_scaled, y_val_encoded),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Model training completed successfully")
        return history.history
    
    def evaluate_healthcare_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model with healthcare-specific metrics.
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Dictionary of healthcare-specific evaluation metrics
        """
        # Scale and encode test data
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate standard metrics
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(y_test_encoded, y_pred, 
                                     target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # Healthcare-specific metrics
        healthcare_metrics = {
            'sensitivity': self._calculate_sensitivity(cm),
            'specificity': self._calculate_specificity(cm),
            'precision': self._calculate_precision(cm),
            'f1_score': self._calculate_f1_score(cm),
            'alert_fatigue_score': self._calculate_alert_fatigue(y_pred, y_test_encoded),
            'missed_falls_cost': self._calculate_missed_falls_cost(y_pred, y_test_encoded),
            'false_alarm_cost': self._calculate_false_alarm_cost(y_pred, y_test_encoded)
        }
        
        # Combine all metrics
        evaluation_results = {
            'classification_report': report,
            'confusion_matrix': cm,
            'healthcare_metrics': healthcare_metrics,
            'class_names': class_names.tolist()
        }
        
        return evaluation_results
    
    def _calculate_sensitivity(self, cm: np.ndarray) -> float:
        """Calculate sensitivity (recall) for fall detection."""
        # Focus on fall class (assuming it's the last class)
        fall_class = cm.shape[0] - 1
        tp = cm[fall_class, fall_class]
        fn = np.sum(cm[fall_class, :]) - tp
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _calculate_specificity(self, cm: np.ndarray) -> float:
        """Calculate specificity for fall detection."""
        # Focus on fall class
        fall_class = cm.shape[0] - 1
        tp = cm[fall_class, fall_class]
        fp = np.sum(cm[:, fall_class]) - tp
        tn = np.sum(cm) - tp - fp - np.sum(cm[fall_class, :]) + tp
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_precision(self, cm: np.ndarray) -> float:
        """Calculate precision for fall detection."""
        fall_class = cm.shape[0] - 1
        tp = cm[fall_class, fall_class]
        fp = np.sum(cm[:, fall_class]) - tp
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _calculate_f1_score(self, cm: np.ndarray) -> float:
        """Calculate F1 score for fall detection."""
        precision = self._calculate_precision(cm)
        sensitivity = self._calculate_sensitivity(cm)
        return 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    def _calculate_alert_fatigue(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate alert fatigue score based on false positive rate."""
        # Count false positives for fall class
        fall_class = len(np.unique(y_true)) - 1
        false_positives = np.sum((y_pred == fall_class) & (y_true != fall_class))
        total_predictions = len(y_pred)
        return false_positives / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_missed_falls_cost(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate cost of missed falls (false negatives)."""
        fall_class = len(np.unique(y_true)) - 1
        missed_falls = np.sum((y_pred != fall_class) & (y_true == fall_class))
        total_falls = np.sum(y_true == fall_class)
        return missed_falls / total_falls if total_falls > 0 else 0.0
    
    def _calculate_false_alarm_cost(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate cost of false alarms."""
        fall_class = len(np.unique(y_true)) - 1
        false_alarms = np.sum((y_pred == fall_class) & (y_true != fall_class))
        total_non_falls = np.sum(y_true != fall_class)
        return false_alarms / total_non_falls if total_non_falls > 0 else 0.0
    
    def predict_sequence(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Predict fall risk for a single sequence.
        
        Args:
            sequence: Input sequence of shape (sequence_length, num_features)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Scale sequence
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
        
        # Add batch dimension
        sequence_batch = np.expand_dims(sequence_scaled, axis=0)
        
        # Get prediction
        prediction = self.model.predict(sequence_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return predicted_class, confidence
    
    def save_model(self, model_path: str):
        """Save the trained model and preprocessing components."""
        # Create directory if it doesn't exist
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(f"{model_path}_model.h5")
        
        # Save preprocessing components
        joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
        joblib.dump(self.label_encoder, f"{model_path}_encoder.pkl")
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names,
            'class_names': self.label_encoder.classes_.tolist(),
            'training_date': datetime.now().isoformat()
        }
        
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model and preprocessing components."""
        # Load model
        self.model = tf.keras.models.load_model(f"{model_path}_model.h5")
        
        # Load preprocessing components
        self.scaler = joblib.load(f"{model_path}_scaler.pkl")
        self.label_encoder = joblib.load(f"{model_path}_encoder.pkl")
        
        # Load metadata
        with open(f"{model_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.model_type = metadata['model_type']
        self.sequence_length = metadata['sequence_length']
        self.feature_names = metadata['feature_names']
        
        logger.info(f"Model loaded from {model_path}")


def generate_synthetic_data(num_samples: int = 1000) -> Tuple[pd.DataFrame, List[int]]:
    """
    Generate synthetic training data for demonstration.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (features_df, labels)
    """
    # Create sample keypoints
    sample_keypoints = create_sample_data()
    
    # Generate synthetic features
    features_list = []
    labels = []
    
    for i in range(num_samples):
        # Add some noise to keypoints
        noise_factor = 0.1
        
        # Create noisy keypoints
        noisy_keypoints = sample_keypoints
        
        # Extract features using our processor
        processor = PrivacyPreservingPoseProcessor()
        features = processor.extract_features(noisy_keypoints)
        
        # Add some temporal variation
        features['velocity_magnitude'] = np.random.normal(0, 0.1)
        features['acceleration_magnitude'] = np.random.normal(0, 0.05)
        
        features['frame_number'] = i
        features_list.append(features)
        
        # Assign labels (0: normal, 1: high-risk, 2: fall)
        if i < num_samples * 0.7:  # 70% normal
            labels.append(0)
        elif i < num_samples * 0.9:  # 20% high-risk
            labels.append(1)
        else:  # 10% falls
            labels.append(2)
    
    features_df = pd.DataFrame(features_list)
    return features_df, labels


def train_fall_detection_model(data_path: Optional[str] = None, 
                              dataset_type: str = 'kaggle_fall',
                              model_type: str = 'lstm',
                              sequence_length: int = 30,
                              epochs: int = 100) -> FallDetectionModel:
    """
    Complete training pipeline for fall detection model.
    
    Args:
        data_path: Path to training data (if None, uses dataset_type)
        dataset_type: Type of dataset to use ('kaggle_fall', 'up_fall', 'ntu_rgbd')
        model_type: Type of model to train
        sequence_length: Length of sequences for training
        epochs: Number of training epochs
        
    Returns:
        Trained FallDetectionModel
    """
    logger.info("Starting fall detection model training pipeline")
    
    # Load or generate data
    if data_path:
        # Load real data from CSV file
        logger.info(f"Loading data from {data_path}")
        features_df = pd.read_csv(data_path)
        labels = features_df['label'].tolist()
        features_df = features_df.drop('label', axis=1)
    else:
        # Load real dataset
        try:
            from dataset_integration import DatasetManager
            logger.info(f"Loading {dataset_type} dataset")
            manager = DatasetManager()
            features_df, labels = manager.load_fall_detection_data(dataset_type)
            
            # Get dataset info
            info = manager.get_dataset_info()
            logger.info(f"Dataset info: {info['real_data_available']}")
            
        except ImportError:
            # Fallback to synthetic data
            logger.info("Dataset integration not available, generating synthetic training data")
            features_df, labels = generate_synthetic_data(1000)
    
    # Initialize model
    model = FallDetectionModel(model_type=model_type, sequence_length=sequence_length)
    
    # Prepare sequences
    X, y = model.prepare_sequences(features_df, labels)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Train model
    history = model.train(X_train, y_train, X_val, y_val, epochs=epochs)
    
    # Evaluate model
    evaluation_results = model.evaluate_healthcare_metrics(X_test, y_test)
    
    # Print results
    print("\n=== Healthcare-Specific Evaluation Results ===")
    healthcare_metrics = evaluation_results['healthcare_metrics']
    for metric, value in healthcare_metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Save model
    model.save(f"models/fall_detection_{model_type}")
    
    logger.info("Training pipeline completed successfully")
    return model


if __name__ == "__main__":
    # Train a fall detection model
    model = train_fall_detection_model(
        model_type='lstm',
        sequence_length=30,
        epochs=50
    )
    
    print("Model training completed!") 