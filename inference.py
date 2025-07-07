#!/usr/bin/env python3
"""
Vishal Patel (me__unpredictable) 2025
During my research on pedestrian intent prediction, I am making this script for educational purposes and includes extensive comments
to help understand the inference pipeline for pedestrian intent Prediction.

JAAD Intent Prediction Inference Script

This script demonstrates how to use a trained intent Prediction model for inference.
It provides three main functionalities:

1. Demo Mode: Evaluates the trained model on test data and shows prediction accuracy
2. List Videos: Discovers and lists all available video IDs in the JAAD dataset
3. Create Preview: Generates annotated video previews showing model predictions in real-time. It only shows the first few sequences
    1. Processes pedestrian sequences from a specific video 
    2. Predicts intent for each pedestrian in the video
It creates separate videos for each pedestrian intent prediction.

Example usage:
    python inference.py --demo                                    # Run accuracy demo
    python inference.py --list_videos                            # List available videos
    python inference.py --create_preview --video_id video_0001   # Create annotated video

Requirements:
    - Trained model file (best_intent_model.pth)
    - Model configuration (best_intent_model_config.pkl)
    - JAAD dataset properly installed and accessible
"""

from turtle import pos
import torch
import torch.utils.data as data
import numpy as np
import os
import argparse
import pickle
import cv2
from jaad_data import JAAD

# import model from train_headless.py
from train_headless import IntentMLP 


# ====================================================================
# DATASET AND DATA LOADING UTILITIES
# ====================================================================

class JAAdDataset(data.Dataset):
    """
    PyTorch Dataset for JAAD intent Prediction
    
    This dataset takes sequence data and creates sliding windows of features
    for training/inference. Each sample contains features from multiple frames
    and the corresponding intent label.
    """
    
    def __init__(self, data, sequence_length=8):
        """
        Args:
            data: Dictionary with 'image', 'pose', 'intent', 'bbox' keys
            sequence_length: Number of frames to use for each prediction
        """
        self.sequence_length = sequence_length
        self.samples = []
        
        # Process each sequence in the data
        for seq_idx in range(len(data['image'])):
            images = data['image'][seq_idx]
            poses = data.get('pose', [None] * len(data['image']))[seq_idx]
            intents = data['intent'][seq_idx]
            bboxes = data.get('bbox', [None] * len(data['image']))[seq_idx]
            
            # Create sliding windows within each sequence
            for i in range(sequence_length - 1, len(images)):
                # Extract features for the sequence window
                features = []
                for j in range(sequence_length):
                    frame_idx = i - sequence_length + 1 + j
                    # Simple normalized features (frame position, sequence position, placeholder bbox)
                    # This creates exactly 6 features per frame to match training: 8 frames × 6 features = 48 total
                    features.extend([
                        frame_idx / len(images),          # Normalized frame position
                        j / sequence_length,              # Normalized sequence position
                        1.0, 1.0, 1.0, 1.0              # Placeholder bbox features (4 values)
                    ])
                
                # Intent label for the current frame
                intent_label = intents[i] if i < len(intents) else 0
                
                self.samples.append((torch.FloatTensor(features), torch.LongTensor([intent_label])))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_jaad_data():
    """
    Load and prepare JAAD dataset for inference

    Returns:
        test_data: Test sequences
    """
    # Initialize JAAD dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    jaad_data_path = os.path.join(current_dir, 'dataset', 'jaad')
    
    jaad = JAAD(data_path=jaad_data_path)
    # Load test data
    test_data = jaad.generate_data_trajectory_sequence(
        image_set='test',
        sample_type='beh',
        seq_type='intention',
        data_split_type='default',
        min_track_size=8,
        fstride=1
    )
    
    
    return test_data


# ====================================================================
# NEURAL NETWORK MODEL
# ====================================================================



class IntentPredictor:
    """
    Wrapper class for loading trained models and making intent predictions
    
    This class handles:
    1. Loading pre-trained model weights and architecture configuration
    2. Making predictions on new feature sequences
    3. Returning both class predictions and confidence scores
    
    If no configuration file is available it uses default parameters.
    """
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model from file 
        """
        config_file = 'best_intent_model_config.pkl'
        model_file = 'best_intent_model.pth'

        # Check if model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file '{model_file}' not found. Please train the model first.")
        
        # Load configuration if available
        if os.path.exists(config_file):
            with open(config_file, 'rb') as f:
                config_data = pickle.load(f)
                config = config_data['model_config']
            
            # Create model with loaded configuration
            self.model = IntentMLP(
                input_size=config['input_size'],
                hidden_sizes=config['hidden_sizes'],
                num_classes=config['num_classes'],
                dropout=config.get('dropout', 0.3)
            )
        else:
            # Use default configuration if config file not found
            self.model = IntentMLP(
                input_size=48,  # 8 frames × 6 features per frame
                hidden_sizes=[512, 256, 128],  # Default hidden sizes
                num_classes=2,  # Binary classification (crossing vs not crossing)
                dropout=0.3  # Default dropout rate
            )
    
        # Load model weights
        state_dict = torch.load(model_file, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, features):
        """
        Make intent prediction on a sequence of features
        
        This method takes feature vectors (typically from multiple frames) and
        returns both the predicted class and confidence score. The features are
        automatically converted to the appropriate tensor format and moved to
        the correct device (CPU/GPU).
        
        Args:
            features: Numpy array or tensor of shape (feature_size,) or (batch_size, feature_size)
            
        Returns:
            prediction: Predicted class (0 = not crossing, 1 = crossing)
            confidence: Confidence score (probability of predicted class)
        """
        # Convert numpy arrays to PyTorch tensors
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Handle single sample vs batch
        if len(features.shape) == 1:
            features = features.unsqueeze(0)  # Add batch dimension
        
        # Move to appropriate device (CPU/GPU)
        features = features.to(self.device)
        
        # Make prediction without computing gradients (inference only)
        with torch.no_grad():
            # Forward pass through the network
            output = self.model(features)
            
            # Apply softmax activation to convert tensor output to probability distribution
            probabilities = torch.softmax(output, dim=1)

            # Get predicted class (highest probability from [not crossing, crossing])
            prediction = torch.argmax(probabilities, dim=1).item()
            # Get confidence (probability of predicted class)
            confidence = probabilities[0][int(prediction)].item()
        
        return prediction, confidence


# ====================================================================
# VIDEO CREATION AND VISUALISATION
# ====================================================================

def create_video_preview(predictor, jaad, video_id='video_0001'):
    """
    Create annotated video preview showing model predictions in real-time
    
    This function demonstrates the complete inference pipeline:
    1. Load pedestrian sequences from a specific video
    2. Extract frames and annotations (bounding boxes, pose (if available), ground truth intents)
    3. Apply the trained model to predict intent for each frame
    4. Create an annotated video with predictions and confidence scores
    
    The output videos show:
    - Bounding boxes around pedestrians (green=crossing intent, red=not crossing intent)
    - Intent prediction labels and confidence scores (green=correct, red=incorrect)
    - Frame and sequence information
    
    Args:
        predictor: Trained IntentPredictor model instance
        jaad: JAAD dataset object for accessing video data
        video_id: ID of the video to process (e.g., 'video_0001')
    """
    # Load trajectory sequences for all videos
    all_video_data = jaad.generate_data_trajectory_sequence(
        image_set='all',      # Use all available data
        sample_type='beh',    # Only pedestrians with behavior annotations
        seq_type='intention', # Intent Prediction sequences
        data_split_type='default',
        min_track_size=8,     # Minimum 8 frames per sequence
        fstride=1            # Use every frame
    )
    
    # Filter sequences that belong to the requested video
    # JAAD contains multiple videos, we need only the specified one
    video_data = {'image': [], 'pose': [], 'intent': [], 'bbox': []}
    for i, image_seq in enumerate(all_video_data['image']):
        # Check if any image path in this sequence belongs to our target video
        found_video = False
        for img_path in image_seq: # Go through all img folders 
            if video_id in img_path: # Check if video_id is found the path (Check conversion process to understand this logic)
                found_video = True
                break
        if found_video:
            video_data['image'].append(image_seq)
            # Safely append corresponding data for other modalities
            if 'pose' in all_video_data and i < len(all_video_data['pose']):
                video_data['pose'].append(all_video_data['pose'][i])
            if 'intent' in all_video_data and i < len(all_video_data['intent']):
                video_data['intent'].append(all_video_data['intent'][i])
            if 'bbox' in all_video_data and i < len(all_video_data['bbox']):
                video_data['bbox'].append(all_video_data['bbox'][i])
    
    if not video_data['image']:
        return
    
    # Process each sequence (limit to 3 for demo purposes)
    videos_created = 0
    max_sequences = 3
    
    for seq_idx in range(min(max_sequences, len(video_data['image']))):
        # Extract data for this sequence
        images = video_data['image'][seq_idx]
        poses = video_data['pose'][seq_idx] if seq_idx < len(video_data.get('pose', [])) else None
        intents = video_data['intent'][seq_idx]
        bboxes = video_data['bbox'][seq_idx] if seq_idx < len(video_data.get('bbox', [])) else None
        
        # Load frame images from disk
        frames = []
        
        for img_path in images:
            path = os.path.join(jaad._images_path, video_id, os.path.basename(img_path))  # Try video subfolder
            frame = cv2.imread(path)
            if frame is not None:
                frames.append(frame)

            # Limit frames to keep demo videos manageable (max 40 frames)
            if len(frames) >= 40:
                break
        
        if len(frames) == 0:
            continue
        
        # Set up video writer for output
        output_file = f'intent_preview_{video_id}_seq_{seq_idx}.mp4'
        
        h, w = frames[0].shape[:2]  # Get frame dimensions
        
        # Initialize video writer with MP4 codec
        fourcc = cv2.VideoWriter.fourcc(*'mp4v') # Setting codec to MP4
        fps = 10  # 10 FPS for clear viewing
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h)) # Initialize video writer

        # Fallback to AVI if MP4 codec fails
        if not video_writer.isOpened():
            fourcc = cv2.VideoWriter.fourcc(*'XVID') 
            output_file = f'intent_preview_{video_id}_seq_{seq_idx}.avi'
            video_writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
            
            if not video_writer.isOpened():
                continue
        
        # Process each frame with model predictions
        sequence_length = 8  # Use 8 frames for each prediction (sliding window)
        
        for i, frame in enumerate(frames):
            frame_copy = frame.copy()  # Work on a copy to preserve original
            
            # Draw initial bounding box (gray color)
            if bboxes is not None and i < len(bboxes) and bboxes[i] is not None:
                bbox = bboxes[i]
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (128, 128, 128), 2)
            # Draw initial pose (if available)
            if poses is not None and i < len(poses) and poses[i] is not None:
                pose = poses[i]
                if len(pose) >= 4:
                    # Draw pose as circles (example: using first 4 points as x,y coordinates)
                    # Assuming pose contains [x1, y1, x2, y2] format
                    for j in range(0, min(4, len(pose)), 2):
                        if j + 1 < len(pose):
                            x, y = int(pose[j]), int(pose[j + 1])
                            cv2.circle(frame_copy, (x, y), 5, (255, 0, 0), -1)

            # Make prediction if we have collected enough frames
            if i >= sequence_length - 1:
                # Create simple feature vector for demonstration
                features = []
                for j in range(sequence_length):
                    frame_idx = max(0, i - sequence_length + 1 + j)
                    # Simple features: normalized frame position, sequence position, placeholder values
                    # This creates exactly 6 features per frame to match training: 8 frames × 6 features = 48 total
                    features.extend([
                        frame_idx / len(frames),  # Frame position in sequence (0 to 1)
                        j / sequence_length,      # Position in sliding window (0 to 1)  
                    ])
                    if poses is not None and frame_idx < len(poses) and poses[frame_idx] is not None:
                        # Add pose features if available (e.g., normalized x, y positions)
                        pose = poses[frame_idx]
                        if len(pose) >= 4:
                            # Use first 4 pose keypoints as features
                            features.extend([pose[0], pose[1], pose[2], pose[3]])
                        else:
                            features.extend([1.0, 1.0, 1.0, 1.0])  # Dummy pose features (4 values)
                    else:
                        # If no pose data, add dummy values
                        features.extend([1.0, 1.0, 1.0, 1.0])  # Dummy pose features (4 values)
                
                # Get model prediction
                prediction, confidence = predictor.predict(np.array(features))
                
                # Update bounding box color based on prediction
                if bboxes is not None and i < len(bboxes) and bboxes[i] is not None:
                    bbox = bboxes[i]
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        # Green for crossing intent, red for not crossing intent (bounding box)
                        color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
                
                # Add prediction text overlay
                # check if prediction is correct
                ground_truth = intents[i] if i < len(intents) else 0
                is_correct = prediction == ground_truth
                
                intent_label = "CROSS" if prediction == 1 else "NOT CROSS"
                if is_correct:
                    text_color = (0, 255, 0)  # Green for correct prediction
                    cv2.putText(frame_copy, f"Intent: {intent_label}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                    cv2.putText(frame_copy, f"Confidence: {confidence:.2f}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                else:
                    text_color = (0, 0, 255)  # Red for incorrect prediction
                    cv2.putText(frame_copy, f"Intent: {intent_label} (Incorrect!!!)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                    cv2.putText(frame_copy, f"Confidence: {confidence:.2f} (Incorrect!!!)", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            else:
                # Show collecting frames message for initial frames
                cv2.putText(frame_copy, "Collecting frames...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Add informational overlay
            cv2.putText(frame_copy, f"Video: {video_id} | Seq: {seq_idx + 1} | Frame: {i+1}/{len(frames)}", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame to video file
            video_writer.write(frame_copy)
        
        # Finalize video file
        video_writer.release()
        
        # Verify video file was created successfully
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"SUCCESS: Video created: {output_file} ({file_size:,} bytes)")
            videos_created += 1

    # Report final results
    if videos_created > 0:
        print(f"SUCCESS: Successfully created {videos_created} annotated video(s)")
        print("Videos show real-time intent predictions with confidence scores")
        print("Bounding boxes: Green=crossing intent, Red=not crossing intent")
        print("Text labels: Green=correct prediction, Red=incorrect prediction")
    else:
        print("No videos were created")


# ====================================================================
# DEMONSTRATION AND EVALUATION FUNCTIONS  
# ====================================================================

def demo_inference():
    """
    TUTORIAL: Demonstrate inference on JAAD test data
    
    This function shows the complete pipeline for evaluating a trained model:
    1. Load JAAD dataset
    2. Extract features using the same method as training
    3. Make predictions and compare with ground truth
    4. Calculate accuracy metrics
    """
    
    # Step 1: Load JAAD dataset using the same method as training
    test_data = load_jaad_data()
    
    # Step 2: Load model configuration to ensure consistency
    config_path = 'best_intent_model_config.pkl'
    sequence_length = 8  # Default: use 8 frames per prediction
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            config_data = pickle.load(f)
        
        # Handle both old and new config formats
        if 'model_config' in config_data:
            # New format from train_headless.py
            config = config_data['model_config']
        else:
            # Old format (backward compatibility)
            config = config_data
            
        sequence_length = config.get('sequence_length', 8)
    
    # Step 3: Create dataset with same parameters as training
    test_dataset = JAAdDataset(test_data, sequence_length=sequence_length)
    
    if len(test_dataset) == 0:
        return
    
    # Step 4: Load the trained model
    predictor = IntentPredictor('best_intent_model.pth')
    
    # Step 5: Evaluate model on test samples
    correct = 0
    total = 0
    
    # Test on first 10 samples for demonstration
    for i in range(min(10, len(test_dataset))):
        features, ground_truth = test_dataset[i]
        features = features.numpy()
        ground_truth = int(ground_truth.item())
        
        prediction, confidence = predictor.predict(features)
        
        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1
        total += 1
    
    if total > 0:
        accuracy = correct / total * 100
        print(f"Accuracy on {total} test samples: {accuracy:.1f}%")


def list_available_videos(jaad):
    """
    Helper function to discover and list available video IDs in the JAAD dataset
    
    This function scans through all trajectory sequences to find unique video IDs,
    helping users discover what videos are available for creating previews.
    
    Args:
        jaad: JAAD dataset object
        
    Returns:
        List of available video ID strings (e.g., ['video_0001', 'video_0002', ...])
    """
    # Load all trajectory data to scan for video IDs
    all_data = jaad.generate_data_trajectory_sequence(
        image_set='all',        # Scan all available data
        sample_type='beh',      # Only pedestrians with behavior annotations
        seq_type='intention',   # Intent Prediction sequences
        data_split_type='default',
        min_track_size=1,       # Use minimal track size to find all videos
        fstride=1
    )
    
    # Extract unique video IDs from image paths
    video_ids = set()
    for image_seq in all_data['image']:
        # Check first image in each sequence to identify video
        for img_path in image_seq[:1]:  # Only check first image per sequence
            path_parts = img_path.split('/')
            # Look for video ID pattern in path
            for part in path_parts:
                if part.startswith('video_'):
                    video_ids.add(part)
                    break
    
    return sorted(list(video_ids))


# ====================================================================
# COMMAND LINE INTERFACE
# ====================================================================


def main():
    """
    Main function implementing the command-line interface for intent Prediction inference
    
    This provides three main modes of operation:
    1. Demo mode: Evaluate model accuracy on test data samples
    2. List videos: Discover available video IDs in the dataset  
    3. Create preview: Generate annotated video with real-time predictions
    """
    parser = argparse.ArgumentParser(
        description='JAAD Intent Prediction Inference Pipeline',
        epilog="""
Examples:
  python inference.py --demo                                    # Test model accuracy
  python inference.py --list_videos                            # Show available videos
  python inference.py --create_preview --video_id video_0001   # Create annotated video
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, default='best_intent_model.pth',
                       help='Path to trained model file (default: best_intent_model.pth)')
    parser.add_argument('--create_preview', action='store_true',
                       help='Create annotated video preview with model predictions')
    parser.add_argument('--video_id', type=str, default='video_0001',
                       help='Video ID for creating preview (default: video_0001)')
    parser.add_argument('--demo', action='store_true',
                       help='Run inference demo on test data to show accuracy')
    parser.add_argument('--list_videos', action='store_true',
                       help='List all available video IDs in the dataset')
    
    args = parser.parse_args()
    
    # Mode 1: Create annotated video preview
    if args.create_preview:
        # Load JAAD dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jaad_data_path = os.path.join(current_dir, 'dataset', 'jaad')
        jaad = JAAD(data_path=jaad_data_path)
        
        # Load trained model
        predictor = IntentPredictor(args.model_path)
        create_video_preview(predictor, jaad, video_id=args.video_id)
    
    # Mode 2: Run accuracy demonstration
    elif args.demo:
        demo_inference()
    
    # Mode 3: List available videos
    elif args.list_videos:
        # Load JAAD dataset and discover videos
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jaad_data_path = os.path.join(current_dir, 'dataset', 'jaad')
        jaad = JAAD(data_path=jaad_data_path)
        available_videos = list_available_videos(jaad)
        
        if available_videos:
            print(f"Found {len(available_videos)} available videos:")
            for i, video_id in enumerate(available_videos, 1):
                print(f"  {i:2d}. {video_id}")
        else:
            print("No videos found in the dataset.")
    
    # No mode specified: Show help
    else:
        print("JAAD Intent Prediction Inference Pipeline")
        print("=" * 50)
        print("This script demonstrates pedestrian intent Prediction using trained models.")
        print("\nAvailable modes:")
        print("  --demo                    : Test model accuracy on sample data")
        print("  --list_videos            : Show all available video IDs")
        print("  --create_preview         : Create annotated video with predictions")
        print("\nFor detailed help:")
        print("  python inference.py --help")
        print("\nQuick start:")
        print("  python inference.py --demo                                    # Test accuracy")
        print("  python inference.py --list_videos                            # Find videos")
        print("  python inference.py --create_preview --video_id video_0001   # Make video")


if __name__ == "__main__":
    main()
