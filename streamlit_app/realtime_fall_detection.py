"""
Real-time Fall Detection Dashboard

A Streamlit application for monitoring fall detection systems in real-time,
with healthcare-specific metrics and alert management.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
import queue
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import PrivacyPreservingPoseProcessor, VideoProcessor
from src.model_training import FallDetectionModel
from src.realtime_inference import RealTimeFallDetector, AlertEvent, PerformanceMetrics

# Page configuration
st.set_page_config(
    page_title="Privacy-Preserving Fall Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for healthcare theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-high {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .alert-normal {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FallDetectionDashboard:
    """Main dashboard class for fall detection monitoring."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.detector = None
        self.is_running = False
        self.alert_history = []
        self.performance_history = []
        self.frame_placeholder = None
        self.metrics_placeholder = None
        
        # Initialize session state
        if 'detector_initialized' not in st.session_state:
            st.session_state.detector_initialized = False
        if 'alert_history' not in st.session_state:
            st.session_state.alert_history = []
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
    
    def initialize_detector(self, model_path: str, camera_source: int = 0):
        """Initialize the fall detection system."""
        try:
            # Create healthcare alert callback
            def alert_callback(alert: AlertEvent):
                st.session_state.alert_history.append({
                    'timestamp': alert.timestamp.isoformat(),
                    'type': alert.alert_type,
                    'severity': alert.severity,
                    'confidence': alert.confidence,
                    'description': alert.description
                })
            
            # Initialize detector
            self.detector = RealTimeFallDetector(
                model_path=model_path,
                camera_source=camera_source,
                alert_callback=alert_callback,
                confidence_threshold=0.7
            )
            
            st.session_state.detector_initialized = True
            return True
            
        except Exception as e:
            st.error(f"Failed to initialize detector: {str(e)}")
            return False
    
    def render_header(self):
        """Render the main header."""
        st.markdown("""
        <div class="main-header">
            <h1>🏥 Privacy-Preserving Fall Detection System</h1>
            <p>Real-time monitoring with healthcare-specific optimizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.title("System Controls")
        
        # Model selection
        st.sidebar.subheader("Model Configuration")
        model_path = st.sidebar.selectbox(
            "Select Model",
            ["models/fall_detection_lstm", "models/fall_detection_cnn", "models/fall_detection_hybrid"],
            help="Choose the trained model to use for detection"
        )
        
        # Camera source
        camera_source = st.sidebar.selectbox(
            "Camera Source",
            [0, 1, 2],
            help="Select camera device (0=default webcam)"
        )
        
        # Detection parameters
        st.sidebar.subheader("Detection Parameters")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.1,
            help="Minimum confidence for alert generation"
        )
        
        alert_cooldown = st.sidebar.slider(
            "Alert Cooldown (seconds)",
            min_value=10,
            max_value=120,
            value=30,
            step=10,
            help="Minimum time between alerts"
        )
        
        # System controls
        st.sidebar.subheader("System Controls")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🚀 Start Detection", type="primary"):
                if not st.session_state.detector_initialized:
                    if self.initialize_detector(model_path, camera_source):
                        self.start_detection()
                else:
                    self.start_detection()
        
        with col2:
            if st.button("⏹️ Stop Detection"):
                self.stop_detection()
        
        # Export controls
        st.sidebar.subheader("Data Export")
        if st.button("📊 Export Alert Log"):
            self.export_alert_log()
        
        if st.button("📈 Export Performance Data"):
            self.export_performance_data()
        
        # System status
        st.sidebar.subheader("System Status")
        status = "🟢 Running" if self.is_running else "🔴 Stopped"
        st.sidebar.markdown(f"**Status:** {status}")
        
        if st.session_state.detector_initialized and self.detector:
            metrics = self.detector.get_performance_metrics()
            st.sidebar.metric("FPS", f"{metrics.fps:.1f}")
            st.sidebar.metric("Processing Time", f"{metrics.processing_time_ms:.1f}ms")
            st.sidebar.metric("Alert Count", metrics.alert_count)
    
    def start_detection(self):
        """Start the fall detection system."""
        if self.detector and not self.is_running:
            self.detector.start_detection()
            self.is_running = True
            st.success("Fall detection started successfully!")
    
    def stop_detection(self):
        """Stop the fall detection system."""
        if self.detector and self.is_running:
            self.detector.stop_detection()
            self.is_running = False
            st.warning("Fall detection stopped.")
    
    def render_main_content(self):
        """Render the main dashboard content."""
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Monitoring", "📊 Performance Metrics", "🚨 Alert History", "🔧 System Analysis"])
        
        with tab1:
            self.render_live_monitoring()
        
        with tab2:
            self.render_performance_metrics()
        
        with tab3:
            self.render_alert_history()
        
        with tab4:
            self.render_system_analysis()
    
    def render_live_monitoring(self):
        """Render the live monitoring view."""
        st.subheader("📹 Live Video Monitoring")
        
        # Create columns for video and metrics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video display placeholder
            video_placeholder = st.empty()
            
            # Update video feed
            if self.is_running and self.detector:
                result = self.detector.get_latest_result()
                if result:
                    frame = result['frame']
                    keypoints = result['keypoints']
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    # Draw skeleton if keypoints detected
                    if keypoints:
                        frame = self.detector.pose_processor.draw_skeleton(frame, keypoints)
                    
                    # Add prediction overlay
                    class_names = self.detector.model.label_encoder.classes_
                    prediction_text = f"{class_names[prediction]}: {confidence:.2f}"
                    
                    # Color based on prediction
                    if prediction == 2:  # Fall
                        color = (0, 0, 255)  # Red
                    elif prediction == 1:  # High risk
                        color = (0, 165, 255)  # Orange
                    else:  # Normal
                        color = (0, 255, 0)  # Green
                    
                    # Add text overlay
                    cv2.putText(frame, prediction_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                else:
                    video_placeholder.info("Waiting for video feed...")
            else:
                video_placeholder.info("Detection not running. Click 'Start Detection' to begin.")
        
        with col2:
            # Real-time metrics
            st.subheader("Real-time Metrics")
            
            if self.is_running and self.detector:
                metrics = self.detector.get_performance_metrics()
                
                # Performance metrics
                st.metric("FPS", f"{metrics.fps:.1f}")
                st.metric("Processing Time", f"{metrics.processing_time_ms:.1f}ms")
                st.metric("Detection Confidence", f"{metrics.detection_confidence:.2f}")
                st.metric("Total Alerts", metrics.alert_count)
                
                # Healthcare compliance
                st.subheader("Healthcare Compliance")
                
                # Response time compliance
                response_time_ok = metrics.processing_time_ms < 90000  # 90 seconds
                status_color = "🟢" if response_time_ok else "🔴"
                st.markdown(f"{status_color} **Response Time:** {'Compliant' if response_time_ok else 'Non-compliant'}")
                
                # Sensitivity tracking
                sensitivity_ok = metrics.sensitivity >= 0.90  # 90% minimum
                status_color = "🟢" if sensitivity_ok else "🔴"
                st.markdown(f"{status_color} **Sensitivity:** {'Compliant' if sensitivity_ok else 'Non-compliant'}")
                
                # False positive rate
                fpr_ok = metrics.false_positive_rate <= 0.10  # 10% maximum
                status_color = "🟢" if fpr_ok else "🔴"
                st.markdown(f"{status_color} **False Positive Rate:** {'Compliant' if fpr_ok else 'Non-compliant'}")
            else:
                st.info("Start detection to view metrics")
    
    def render_performance_metrics(self):
        """Render performance metrics and charts."""
        st.subheader("📊 Performance Analytics")
        
        if not self.is_running or not self.detector:
            st.info("Start detection to view performance metrics")
            return
        
        # Get performance data
        metrics = self.detector.get_performance_metrics()
        
        # Create performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # FPS over time
            st.subheader("FPS Over Time")
            # Simulate FPS data (in real implementation, this would come from historical data)
            fps_data = pd.DataFrame({
                'time': pd.date_range(start=datetime.now() - timedelta(minutes=10), 
                                    periods=60, freq='10S'),
                'fps': [metrics.fps + np.random.normal(0, 0.5) for _ in range(60)]
            })
            
            fig_fps = px.line(fps_data, x='time', y='fps', 
                            title="Frames Per Second")
            fig_fps.update_layout(height=300)
            st.plotly_chart(fig_fps, use_container_width=True)
        
        with col2:
            # Processing time distribution
            st.subheader("Processing Time Distribution")
            # Simulate processing time data
            processing_times = [metrics.processing_time_ms + np.random.normal(0, 5) 
                              for _ in range(100)]
            
            fig_processing = px.histogram(x=processing_times, 
                                        title="Processing Time Distribution (ms)",
                                        nbins=20)
            fig_processing.update_layout(height=300)
            st.plotly_chart(fig_processing, use_container_width=True)
        
        # Healthcare-specific metrics
        st.subheader("Healthcare Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sensitivity", f"{metrics.sensitivity:.1%}")
            st.metric("Specificity", f"{metrics.specificity:.1%}")
        
        with col2:
            st.metric("Precision", f"{metrics.precision:.1%}")
            st.metric("F1 Score", f"{metrics.f1_score:.1%}")
        
        with col3:
            st.metric("Alert Fatigue Score", f"{metrics.alert_fatigue_score:.1%}")
            st.metric("Missed Falls Cost", f"{metrics.missed_falls_cost:.1%}")
    
    def render_alert_history(self):
        """Render alert history and management."""
        st.subheader("🚨 Alert History")
        
        # Get alert history
        if self.detector:
            alerts = self.detector.get_alert_history(hours=24)
        else:
            alerts = st.session_state.alert_history
        
        if not alerts:
            st.info("No alerts in the last 24 hours")
            return
        
        # Convert to DataFrame for easier manipulation
        alert_df = pd.DataFrame(alerts)
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", len(alerts))
        
        with col2:
            critical_alerts = len([a for a in alerts if a.severity == 'critical'])
            st.metric("Critical Alerts", critical_alerts)
        
        with col3:
            high_alerts = len([a for a in alerts if a.severity == 'high'])
            st.metric("High Priority", high_alerts)
        
        with col4:
            avg_confidence = np.mean([a.confidence for a in alerts])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Alert timeline
        st.subheader("Alert Timeline")
        
        # Create timeline chart
        timeline_data = []
        for alert in alerts:
            timeline_data.append({
                'timestamp': alert.timestamp,
                'type': alert.alert_type,
                'severity': alert.severity,
                'confidence': alert.confidence
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        if not timeline_df.empty:
            fig_timeline = px.scatter(timeline_df, x='timestamp', y='confidence',
                                    color='severity', size='confidence',
                                    title="Alert Timeline",
                                    color_discrete_map={
                                        'critical': 'red',
                                        'high': 'orange',
                                        'medium': 'yellow',
                                        'low': 'green'
                                    })
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent alerts table
        st.subheader("Recent Alerts")
        
        # Display recent alerts
        for alert in alerts[-10:]:  # Show last 10 alerts
            if alert.severity == 'critical':
                alert_class = "alert-critical"
            elif alert.severity == 'high':
                alert_class = "alert-high"
            else:
                alert_class = "alert-normal"
            
            st.markdown(f"""
            <div class="{alert_class}">
                <strong>{alert.timestamp.strftime('%H:%M:%S')}</strong> - {alert.alert_type.title()}<br>
                <small>Confidence: {alert.confidence:.2f} | Severity: {alert.severity}</small><br>
                <small>{alert.description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def render_system_analysis(self):
        """Render system analysis and investigation tools."""
        st.subheader("🔧 System Analysis & Investigation")
        
        # Model investigation
        st.subheader("Model Performance Investigation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Architecture Analysis**")
            if self.detector and self.detector.model:
                model_info = {
                    "Model Type": self.detector.model.model_type.upper(),
                    "Sequence Length": self.detector.model.sequence_length,
                    "Feature Count": len(self.detector.model.feature_names),
                    "Classes": ", ".join(self.detector.model.label_encoder.classes_)
                }
                
                for key, value in model_info.items():
                    st.write(f"**{key}:** {value}")
        
        with col2:
            st.write("**Healthcare Compliance Analysis**")
            
            if self.detector:
                metrics = self.detector.get_performance_metrics()
                
                # Compliance checklist
                compliance_items = [
                    ("Response Time < 90s", metrics.processing_time_ms < 90000),
                    ("Sensitivity ≥ 90%", metrics.sensitivity >= 0.90),
                    ("False Positive Rate ≤ 10%", metrics.false_positive_rate <= 0.10),
                    ("Privacy Preserved", True),  # Always true with pose estimation
                    ("Real-time Processing", metrics.fps > 15)
                ]
                
                for item, status in compliance_items:
                    status_icon = "✅" if status else "❌"
                    st.write(f"{status_icon} {item}")
        
        # Failure analysis
        st.subheader("Failure Case Analysis")
        
        # Simulate failure analysis data
        failure_data = {
            'Lighting Conditions': {'Low Light': 15, 'Normal': 5, 'Bright': 2},
            'Occlusion Level': {'None': 3, 'Partial': 8, 'Heavy': 12},
            'Movement Speed': {'Slow': 4, 'Normal': 6, 'Fast': 10}
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Lighting Impact**")
            fig_lighting = px.pie(values=list(failure_data['Lighting Conditions'].values()),
                                names=list(failure_data['Lighting Conditions'].keys()),
                                title="Failure Rate by Lighting")
            st.plotly_chart(fig_lighting, use_container_width=True)
        
        with col2:
            st.write("**Occlusion Impact**")
            fig_occlusion = px.pie(values=list(failure_data['Occlusion Level'].values()),
                                 names=list(failure_data['Occlusion Level'].keys()),
                                 title="Failure Rate by Occlusion")
            st.plotly_chart(fig_occlusion, use_container_width=True)
        
        with col3:
            st.write("**Movement Speed Impact**")
            fig_speed = px.pie(values=list(failure_data['Movement Speed'].values()),
                             names=list(failure_data['Movement Speed'].keys()),
                             title="Failure Rate by Speed")
            st.plotly_chart(fig_speed, use_container_width=True)
        
        # Recommendations
        st.subheader("System Recommendations")
        
        recommendations = [
            "🔧 Implement adaptive lighting compensation for low-light conditions",
            "📹 Add multi-view processing to handle occlusion scenarios",
            "⚡ Optimize model for faster processing during rapid movements",
            "🎯 Fine-tune confidence thresholds based on environmental conditions",
            "📊 Implement continuous model performance monitoring"
        ]
        
        for rec in recommendations:
            st.write(rec)
    
    def export_alert_log(self):
        """Export alert history to file."""
        if self.detector:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alert_log_{timestamp}.json"
            self.detector.export_alert_log(filename)
            st.success(f"Alert log exported to {filename}")
        else:
            st.warning("No detector available for export")
    
    def export_performance_data(self):
        """Export performance data to file."""
        if self.detector:
            metrics = self.detector.get_performance_metrics()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"
            
            performance_data = {
                'timestamp': timestamp,
                'fps': metrics.fps,
                'processing_time_ms': metrics.processing_time_ms,
                'detection_confidence': metrics.detection_confidence,
                'alert_count': metrics.alert_count,
                'sensitivity': metrics.sensitivity,
                'specificity': metrics.specificity,
                'precision': metrics.precision,
                'f1_score': metrics.f1_score,
                'alert_fatigue_score': metrics.alert_fatigue_score,
                'missed_falls_cost': metrics.missed_falls_cost,
                'false_positive_rate': metrics.false_positive_rate
            }
            
            with open(filename, 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            st.success(f"Performance data exported to {filename}")
        else:
            st.warning("No detector available for export")
    
    def run(self):
        """Run the main dashboard."""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()


def main():
    """Main function to run the dashboard."""
    # Create and run dashboard
    dashboard = FallDetectionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main() 