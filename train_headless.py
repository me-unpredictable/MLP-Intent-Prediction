#!/usr/bin/env python3

"""
Vishal Patel (me__unpredictable) 2025
During my research on pedestrian intent prediction, I am making this script for educational purposes and includes extensive comments
to help understand the inference pipeline for pedestrian intent Prediction.

JAAD Intent Prediction Training Script (Headless Version)

This script provides a complete training pipeline for pedestrian intent Prediction
using the JAAD dataset. It includes data loading, model definition, training loop,
evaluation, and model saving functionality.

The "headless" version means it runs without any GUI or interactive components,
making it suitable for:
- Server environments without display
- Automated training pipelines
- Batch processing systems
- Remote training on cloud platforms

Main Sections:
1. Data Loading: Loads JAAD dataset and creates PyTorch datasets
2. Model Definition: Multi-layer perceptron (MLP) for intent Prediction
3. Training Loop: Complete training with validation and early stopping
4. Evaluation: Performance metrics and model assessment
5. Test Evaluation: Final model evaluation on test set
6. Model Saving: Saves best model and configuration for inference

Usage:
    python train_headless.py

The script has following functionalities:
- Load and preprocess the JAAD dataset
- Create train/validation splits
- Train the model with progress monitoring
- Save the best model as 'best_intent_model.pth'
- Save model configuration as 'best_intent_model_config.pkl'
"""

# Import libraries 
import torch                    # PyTorch main library for tensors and neural networks
import torch.nn as nn          # Neural network modules for model creation
import torch.optim as optim    # Optimization algorithms 
import torch.utils.data as data  # Data loading utilities (DataLoader, Dataset)
import os                      # Operating system interface to handle file paths
import pickle                  # Python object serialization to load/save model configuration
import time                    # Time-related functions for tracking training time


# Import scikit learn for evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import our custom JAAD data loading module
from jaad_data import JAAD


# ====================================================================
# PYTORCH DATASET CLASS FOR JAAD DATA
# ====================================================================

class JAAdDataset(data.Dataset):
    """
    PyTorch Dataset class for JAAD intent Prediction
    
    This class converts raw JAAD sequence data into a format suitable for
    PyTorch training. It creates sliding windows of features from pedestrian
    trajectory sequences and associates them with intent labels.
    
    The dataset handles:
    - Converting sequences to fixed-length windows
    - Creating feature vectors from temporal data
    - Associating features with intent labels (0=not crossing, 1=crossing)
    
    Args:
        data: Dictionary containing JAAD sequences with keys:
              - 'image': List of image paths for each sequence
              - 'intent': List of intent labels for each sequence
              - 'bbox': List of bounding boxes for each sequence (optional)
              - 'pose': List of pose data for each sequence (optional)
        sequence_length: Number of frames to use in each training sample
    """
    
    def __init__(self, data, sequence_length=8):
        # Sequence length for creating sliding windows
        self.sequence_length = sequence_length
        
        # Initialise list to store all training samples
        self.samples = []
        
        # Process each sequence in the input data
        # Each sequence represents one pedestrian's trajectory
        for seq_idx in range(len(data['image'])):
            # Get the image paths for this sequence
            images = data['image'][seq_idx]
            
            # Get pose data if available, otherwise use None placeholders
            poses = data.get('pose', [None] * len(data['image']))[seq_idx]
            
            # Get intent labels for this sequence
            intents = data['intent'][seq_idx]
            
            # Get bounding boxes if available, otherwise use None placeholders
            bboxes = data.get('bbox', [None] * len(data['image']))[seq_idx]
            
            # Creating sliding windows within this sequence
            # We start from (sequence_length - 1) because we need enough frames
            # to create a full window
            for i in range(sequence_length - 1, len(images)):
                # Create feature vector for this sliding window
                features = []
                
                # Extract features from each frame in the window
                for j in range(sequence_length):
                    # Calculate the actual frame index in the sequence
                    frame_idx = i - sequence_length + 1 + j
                    
                    # Create simple temporal and positional features
                    # These are normalised to [0, 1] range for better training
                    
                    features.extend([
                        frame_idx / len(images),    # Normalised position in sequence (0 to 1)
                        j / sequence_length,        # Normalised position in window (0 to 1)
                        # 1.0, 1.0, 1.0, 1.0         # Placeholder for bbox/pose features
                    ])

                    if poses and poses[frame_idx] is not None:
                        # Use pose data if available
                        pose = poses[frame_idx]
                        features.extend(pose)
                    elif bboxes and bboxes[frame_idx] is not None:
                        # Use bounding box data if available
                        bbox = bboxes[frame_idx]
                        features.extend(bbox)
                    else:
                        # Use placeholder values if no pose/bbox data
                        features.extend([0.0, 0.0, 0.0, 0.0])

                
                # Get the intent label for the current frame
                # Use 0 as default if index is out of bounds
                intent_label = intents[i] if i < len(intents) else 0
                
                # Convert to PyTorch tensors and add to samples list
                # FloatTensor for features, LongTensor for Prediction labels
                self.samples.append((
                    torch.FloatTensor(features),           # Input features
                    torch.LongTensor([intent_label])       # Target label
                ))
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample by index (required by PyTorch DataLoader)"""
        return self.samples[idx]


# ====================================================================
# NEURAL NETWORK MODEL DEFINITION
# ====================================================================

class IntentMLP(nn.Module):
    """
    Multi-Layer Perceptron for Pedestrian Intent Prediction
    
    This neural network predicts whether a pedestrian intends to cross the street
    based on a sequence of temporal features. The architecture consists of:
    - Multiple fully connected (linear) layers
    - ReLU activation functions for non-linearity
    - Dropout layers for regularisation (prevent overfitting)
    - Final output layer for binary Prediction

    The model takes a flattened feature vector from multiple frames and outputs intent
    class probabilities for "crossing" vs "not crossing".
    
    Args:
        input_size: Size of input feature vector (sequence_length * features_per_frame)
        hidden_sizes: Size of hidden layers (e.g., [512, 256, 128])
        num_classes: Number of output classes (2 for binary Prediction)
        dropout: Dropout probability for regularisation
    """
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=2, dropout=0.3):
        # Initialise the parent class (nn.Module)
        super(IntentMLP, self).__init__()
        
        # Create a list to store all network layers
        layers = []
        
        # Keep track of the previous layer size (starts with input size)
        prev_size = input_size
        
        # Build hidden layers according to the specified architecture
        for hidden_size in hidden_sizes:
            # Add linear (fully connected) layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Add ReLU activation function (introduces non-linearity)
            layers.append(nn.ReLU())
            
            # Add dropout layer for regularisation
            # Randomly sets some neurons to zero during training
            layers.append(nn.Dropout(dropout))
            
            # Update previous size for next layer
            prev_size = hidden_size
        
        # Add final output layer (no activation here, handled by loss function)
        layers.append(nn.Linear(prev_size, num_classes))
        # To add more layer or different type of layers add it here

        # Combine all layers into a sequential model
        # Sequential runs layers in order: input -> layer1 -> layer2 -> ... -> output
        self.network = nn.Sequential(*layers) # pointing to the memory address of the layers list

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Pass input through the entire network
        return self.network(x)


# ====================================================================
# TRAINING CONFIGURATION AND HYPERPARAMETERS
# ====================================================================

class TrainingConfig:
    """
    Configuration class containing all training hyperparameters
    
    This class holds all training settings in one place, making it easy
    to modify hyperparameters and track experimental configurations.
    """
    
    def __init__(self):
        # Data parameters
        self.sequence_length = 8        # Number of frames per training sample
        self.batch_size = 32           # Number of samples processed together
        self.num_workers = 8           # Number of parallel data loading processes (depending on CPU cores, 8 for 16 cores. 
        # Appropriate for my CPU, change it according to your CPU cores)
        
        # Model architecture parameters
        self.hidden_sizes = [512, 256, 128]  # Hidden layer sizes
        self.dropout = 0.3             # Dropout probability for regularisation
        self.num_classes = 2           # Binary Prediction hence 2 classes (cross/not cross)
        
        # Training parameters
        self.num_epochs = 100           # Maximum number of training epochs
        self.learning_rate = 0.001      # Learning rate for Adam optimizer
        self.weight_decay = 1e-4        # L2 regularisation strength (optimizer weight decay optional)
        self.early_stopping_patience = 10  # Stop if no improvement for 10 epochs
        
        # Device configuration
        # Use GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model saving parameters
        self.model_save_path = 'best_intent_model.pth'           # Model weights
        self.config_save_path = 'best_intent_model_config.pkl'   # Model configuration
        
        # Display training progress every 10 batches
        self.print_frequency = 10


# ====================================================================
# TRAINING UTILITIES AND HELPER FUNCTIONS
# ====================================================================

def load_jaad_data():
    """
    Load and prepare JAAD dataset for training
    
    This function Initialises the JAAD dataset, loads trajectory sequences,
    and prepares them for training. It handles:
    - Dataset initialisation and path verification
    - Loading training and validation sequences
    - Data preprocessing and format conversion
    
    Returns:
        train_data: Training sequences dictionary
        val_data: Validation sequences dictionary
    """
    
    # Determine the path to the JAAD dataset
    # Look for dataset in the 'dataset/jaad' subdirectory 
    # make sure to follow the same directory structure as mentioned in the README (# project dir -> dataset -> jaad)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    jaad_data_path = os.path.join(current_dir, 'dataset', 'jaad') 
    
    # Initialise the JAAD dataset object
    # This object provides methods to load and process the dataset
    jaad = JAAD(data_path=jaad_data_path)
    
    # Load training data sequences
    train_data = jaad.generate_data_trajectory_sequence(
        image_set='train',          # Use training split
        sample_type='beh',          # Only pedestrians with behavior annotations
        seq_type='intention',       # Intent Prediction sequences
        data_split_type='default',  # Use default train/val/test split
        min_track_size=8,          # Minimum 8 frames per sequence
        fstride=1                  # Use every frame (no frame skipping)
    )
    
    # Load validation data sequences
    val_data = jaad.generate_data_trajectory_sequence(
        image_set='val',            # Use validation split
        sample_type='beh',          # Only pedestrians with behavior annotations
        seq_type='intention',       # Intent Prediction sequences
        data_split_type='default',  # Use default train/val/test split
        min_track_size=8,          # Minimum 8 frames per sequence
        fstride=1                  # Use every frame (no frame skipping)
    )
    
    # Load test data sequences 
    test_data = jaad.generate_data_trajectory_sequence(
        image_set='test',           # Use test split
        sample_type='beh',          # Only pedestrians with behavior annotations
        seq_type='intention',       # Intent Prediction sequences
        data_split_type='default',  # Use default train/val/test split
        min_track_size=8,          # Minimum 8 frames per sequence
        fstride=1                  # Use every frame (no frame skipping)
    )

    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, config):
    """
    Create PyTorch DataLoaders for training, validation and testing

    DataLoaders handle:
    - Batching: Group samples into mini batches
    - Shuffling: Ransomise order for better training
    - Parallel loading: Use multiple processes for efficiency
    - Memory management: Load data efficiently
    
    Args:
        train_data: Training sequences dictionary
        val_data: Validation sequences dictionary
        test_data: Test sequences dictionary
        config: Training configuration object
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    
    # Create PyTorch Dataset objects from raw data
    train_dataset = JAAdDataset(train_data, sequence_length=config.sequence_length)
    val_dataset = JAAdDataset(val_data, sequence_length=config.sequence_length)
    test_dataset = JAAdDataset(test_data, sequence_length=config.sequence_length)

    # Create DataLoader for training data
    train_loader = data.DataLoader(
        train_dataset,                    # Dataset to load from
        batch_size=config.batch_size,     # Number of samples per batch
        shuffle=True,                     # Ransomise order each epoch
        num_workers=config.num_workers,   # Parallel loading processes
        pin_memory=True                   # Faster GPU transfer if available
        # pin_memory=True allows DataLoader to copy tensors to CUDA pinned memory
        # which can speed up transfer to GPU, especially for large datasets
        # This is useful when using GPUs for training
    )
    
    # Create DataLoader for validation data
    val_loader = data.DataLoader(
        val_dataset,                      # Dataset to load from
        batch_size=config.batch_size,     # Number of samples per batch
        shuffle=False,                    # Don't shuffle validation data
        num_workers=config.num_workers,   # Parallel loading processes
        pin_memory=True                   # Faster GPU transfer if available
    )

    # Creat DataLoader for test data
    test_loader = data.DataLoader(
        test_dataset,                      # Dataset to load from
        batch_size=config.batch_size,     # Number of samples per batch
        shuffle=False,                    # Don't shuffle test data
        num_workers=config.num_workers,   # Parallel loading processes
        pin_memory=True                   # Faster GPU transfer if available
    )

    return train_loader, val_loader, test_loader


def calculate_accuracy(output, labels):
    """
    Calculate Prediction accuracy
    
    Args:
        output: Model predictions of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        
    Returns:
        accuracy: Accuracy as a float between 0 and 1
    """
    # Get predicted class (highest value)
    _, predicted = torch.max(output, 1)
    
    # Compare predictions with ground truth
    correct = (predicted == labels).sum().item()
    
    # Calculate accuracy as percentage
    accuracy = correct / labels.size(0)
    
    return accuracy


def train_one_epoch(model, train_loader, criterion, optimizer, config):
    """
    Train the model for one epoch
    
    An epoch is one complete pass through the entire training dataset.
    This function:
    - Processes all training batches
    - Computes loss and gradients
    - Updates model parameters
    - Tracks training metrics
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimization algorithm (Adam)
        config: Training configuration
        
    Returns:
        avg_loss: Average loss over the epoch
        avg_accuracy: Average accuracy over the epoch
    """
    
    # Set model to training mode
    # This enables dropout and batch normalization training behavior
    model.train()
    
    # Initialise metrics tracking
    total_loss = 0.0        # Sum of all batch losses
    total_accuracy = 0.0    # Sum of all batch accuracies
    num_batches = 0         # Counter for number of batches processed
    
    # Process each batch of training data
    for batch_idx, (features, labels) in enumerate(train_loader):
        # Move data to appropriate device (GPU or CPU)
        features = features.to(config.device)
        labels = labels.to(config.device).squeeze()  # Remove extra dimension
        
        # Clear gradients from previous iteration
        # PyTorch accumulates gradients, so we need to reset them
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        output = model(features)
        
        # Compute loss: how far are predictions from ground truth
        loss = criterion(output, labels)
        
        # Backward pass: compute gradients
        # This calculates how much each parameter should change
        loss.backward()
        
        # Update model parameters using computed gradients
        optimizer.step()
        
        # Calculate accuracy for this batch
        accuracy = calculate_accuracy(output, labels)
        
        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
        
        # Print progress periodically
        if batch_idx % config.print_frequency == 0:
            print(f'Batch {batch_idx:3d}/{len(train_loader):3d} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Accuracy: {accuracy:.4f}')
    
    # Calculate average metrics over the entire epoch
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate_model(model, val_loader, criterion, config):
    """
    Evaluate the model on validation data
    
    Validation is used to:
    - Monitor training progress
    - Detect overfitting
    - Select best model checkpoint
    - Estimate generalisation performance
    
    Args:
        model: Neural network model to evaluate
        val_loader: DataLoader for validation data
        criterion: Loss function (CrossEntropyLoss)
        config: Training configuration
        
    Returns:
        avg_loss: Average validation loss
        avg_accuracy: Average validation accuracy
        detailed_metrics: Dictionary with precision, recall, F1-score
    """
    
    # Set model to evaluation mode
    # This disables dropout and uses batch normalization in inference mode
    model.eval()
    
    # Initialise metrics tracking
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    # Store all predictions and labels for detailed metrics
    all_predictions = []
    all_labels = []
    
    # Disable gradient computation for efficiency
    # We don't need gradients during validation
    with torch.no_grad():
        # Process each validation batch
        for features, labels in val_loader:
            # Move data to appropriate device
            features = features.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            # Forward pass: compute predictions
            output = model(features)
            
            # Compute loss
            loss = criterion(output, labels)
            
            # Calculate accuracy
            accuracy = calculate_accuracy(output, labels)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Store predictions and labels for detailed analysis
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    # Calculate detailed metrics using scikit-learn
    detailed_metrics = {
        'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'f1_score': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }
    
    return avg_loss, avg_accuracy, detailed_metrics

def test_model(model,test_loader, criterion, config):
    """
    Evaluate the model on test data
    
    Testing is used to:
    - Assess final model performance
    - Estimate generalisation to unseen data
    - Provide final metrics for reporting
    
    Args:
        model: Neural network model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function (CrossEntropyLoss)
        config: Training configuration
        
    Returns:
        avg_loss: Average test loss
        avg_accuracy: Average test accuracy
        detailed_metrics: Dictionary with precision, recall, F1-score
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialise metrics tracking
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    # Store all predictions and labels for detailed metrics
    all_predictions = []
    all_labels = []
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Process each test batch
        for features, labels in test_loader:
            # Move data to appropriate device
            features = features.to(config.device)
            labels = labels.to(config.device).squeeze()
            
            # Forward pass: compute predictions
            output = model(features)
            
            # Compute loss
            loss = criterion(output, labels)
            
            # Calculate accuracy
            accuracy = calculate_accuracy(output, labels)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Store predictions and labels for detailed analysis
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    # Calculate detailed metrics using scikit-learn
    detailed_metrics = {
        'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'f1_score': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions)
    }
    
    return avg_loss, avg_accuracy, detailed_metrics


def save_model_and_config(model, config, train_history, val_history):
    """
    Save the trained model and its configuration
    
    This function saves:
    - Model weights and architecture
    - Training configuration
    - Training history for analysis
    
    Args:
        model: Trained neural network model
        config: Training configuration object
        train_history: Training metrics history
        val_history: Validation metrics history
    """
    
    # Save model weights
    torch.save(model.state_dict(), config.model_save_path)
    print(f"Model saved to: {config.model_save_path}")
    
    # Create configuration dictionary for inference
    model_config = {
        'input_size': config.sequence_length * 6,  # 6 features per frame
        'hidden_sizes': config.hidden_sizes,
        'num_classes': config.num_classes,
        'dropout': config.dropout,
        'sequence_length': config.sequence_length
    }
    
    # Save configuration and training history
    save_data = {
        'model_config': model_config,
        'training_config': config.__dict__,
        'train_history': train_history,
        'val_history': val_history
    }
    
    with open(config.config_save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Configuration saved to: {config.config_save_path}")


# ====================================================================
# MAIN TRAINING FUNCTION
# ====================================================================

def train_model():
    """
    Main training function that orchestrates the entire training process
    
    This function:
    1. Loads and prepares the dataset
    2. Creates the model and training setup
    3. Runs the training loop with validation
    4. Implements early stopping
    5. Saves the best model
    6. Provides detailed training statistics
    """
    
    # Record start time for performance tracking
    start_time = time.time()
    
    # Initialise training configuration
    config = TrainingConfig()
    
    # Load dataset
    train_data, val_data, test_data = load_jaad_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data, config)

    # Create model
    input_size = config.sequence_length * 6  # 6 features per frame
    model = IntentMLP(
        input_size=input_size,
        hidden_sizes=config.hidden_sizes,
        num_classes=config.num_classes,
        dropout=config.dropout
    )
    
    # Move model to appropriate device
    model = model.to(config.device)
    
    # Setup training components
    
    # Loss function: CrossEntropyLoss for multi-class Prediction
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam with weight decay for regularisation
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # Reduce when loss stops decreasing
        factor=0.5,      # Multiply LR by 0.5
        patience=5      # Wait 5 epochs before reducing
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0 # Counter for early stopping patience, will use to track how many epochs have passed without improvement
    
    # Training history tracking
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': [], 'detailed_metrics': []}
    
    # Training loop
    print("Starting training loop...")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        # Print epoch header
        print(f"Epoch {epoch+1:3d}/{config.num_epochs}")
        print("-" * 40)
        
        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, config
        )
        
        # Validate the model
        val_loss, val_accuracy, detailed_metrics = validate_model(
            model, val_loader, criterion, config
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Record training history
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_accuracy)
        val_history['loss'].append(val_loss)
        val_history['accuracy'].append(val_accuracy)
        val_history['detailed_metrics'].append(detailed_metrics)
        
        # Print epoch summary
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy:.4f}")
        print(f"  Val F1:     {detailed_metrics['f1_score']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            # New best model found
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save the best model
            print(f"New best model! Validation loss: {val_loss:.4f}")
            save_model_and_config(model, config, train_history, val_history)
            
        else:
            # No improvement
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{config.early_stopping_patience})")
            
            # Check if we should stop early
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
    # Test the final model on test data
    print("Testing the final model on test data...")
    test_loss, test_accuracy, test_metrics = test_model(model, test_loader, criterion, config)
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Total training time: {elapsed_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_history['accuracy'][-1]:.4f}")
    print(f"Final validation F1-score: {val_history['detailed_metrics'][-1]['f1_score']:.4f}")
    print(f"Final validation confusion matrix:")
    print(val_history['detailed_metrics'][-1]['confusion_matrix'])
    print()
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-score: {test_metrics['f1_score']:.4f}")
    print(f"Test confusion matrix:")
    print(test_metrics['confusion_matrix'])
    # Save the final model and configuration
    print("Model files saved:")
    print(f"  - {config.model_save_path}")
    print(f"  - {config.config_save_path}")
    print()
    print("Training complete! You can now use the trained model for inference.")

    # Plot training, validation, and testing loss/accuracy
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.plot(val_history['loss'], label='Val Loss')
    plt.plot(test_metrics['loss'], label='Test Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_history['accuracy'], label='Train Accuracy')
    plt.plot(val_history['accuracy'], label='Val Accuracy')
    plt.plot(test_metrics['accuracy'], label='Test Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # save the plots
    plt.savefig('training_validation_testing_loss_accuracy.png')


# ====================================================================
# SCRIPT ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    """
    Main entry point for the training script
    
    This block only runs when the script is executed directly
    (not when imported as a module)
    """
    
    try:
        # Run the training process
        train_model()
        
        
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C)
        print("\nTraining interrupted by user")
        
    except Exception as e:
        # Handle any other errors
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
