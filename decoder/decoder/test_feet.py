import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import random

class FeetEEGDataset(Dataset):
    def __init__(self, data_dir, run_number, exp_number=1, window_size=256, overlap=0.5):
        """
        Create windowed segments from continuous EEG data
        """
        self.window_size = window_size
        self.overlap = overlap

        # Load data
        tp9_file = f"openclosefeet_run{run_number}_TP9.csv"
        tp10_file = f"openclosefeet_run{run_number}_TP10.csv"
        label_file = f"openclosefeet_run{run_number}_label.csv"

        exp_dir = os.path.join(data_dir, f"exp_{exp_number}")

        tp9_data = pd.read_csv(os.path.join(exp_dir, tp9_file), header=None).values
        tp10_data = pd.read_csv(os.path.join(exp_dir, tp10_file), header=None).values
        labels = pd.read_csv(os.path.join(exp_dir, label_file), header=None).values.flatten()

        # Create windowed segments
        self.segments = []
        self.segment_labels = []

        step = int(window_size * (1 - overlap))

        for i, (tp9_sample, tp10_sample, label) in enumerate(zip(tp9_data, tp10_data, labels)):
            # Create windows from each sample
            for start in range(0, len(tp9_sample) - window_size + 1, step):
                end = start + window_size

                # Stack the two channels
                segment = np.stack([
                    tp9_sample[start:end],
                    tp10_sample[start:end]
                ], axis=0)  # Shape: [2, window_size]

                self.segments.append(segment)
                self.segment_labels.append(label)

        self.segments = np.array(self.segments)
        self.segment_labels = np.array(self.segment_labels)

        print(f"Created {len(self.segments)} segments from run {run_number}")
        print(f"Segment shape: {self.segments.shape}")
        print(f"Label distribution: {np.bincount(self.segment_labels.astype(int))}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        label = torch.tensor(self.segment_labels[idx], dtype=torch.float)
        return segment, label

class TwoChannelLSTMClassifier(nn.Module):
    def __init__(self, input_channels=2, num_classes=1):
        super(TwoChannelLSTMClassifier, self).__init__()

        # Simple CNN architecture
        self.features = nn.Sequential(
            # First conv block
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),

            # Second conv block
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),

            # Third conv block
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # Fixed output size
            nn.Dropout(0.4)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, channels, sequence]
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    print(f"Setting all seeds to {seed}")

    # Python's random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_and_evaluate_model(model_path, data_dir, device, threshold=0.53):
    """Load saved model and evaluate on test data"""
    
    # Create test dataset (same parameters as training)
    test_dataset = FeetEEGDataset(data_dir, exp_number=2, run_number=2, window_size=1024, overlap=0.7)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model with same architecture as training
    model = TwoChannelLSTMClassifier(input_channels=2, num_classes=1).to(device)

    
    # Load saved weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    
    # Evaluate model
    print("Evaluating model...")
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_preds.extend(preds.flatten())
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_results(labels, preds, probs, save_path='test_feet_results.png'):
    """Plot evaluation results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Resting", "Open/Close Feet"],
                yticklabels=["Resting", "Open/Close Feet"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix - Test Feet Results")

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve - Test Feet Results")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    return auc_score

def test_with_different_thresholds(labels, probs):
    """Test model performance with different classification thresholds"""
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\nPerformance at different thresholds:")
    print("-" * 50)
    
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        print(f"Threshold {threshold:.1f}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")

if __name__ == "__main__":
    # Set seeds for reproducibility
    set_all_seeds(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    model_path = 'best_feet_model.pth'  # Path to saved model weights
    #data_dir = "Muse_data_OpenCloseFeet_segmented"  # Path to data directory
    data_dir = "D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\Data\sub-EB-43\Muse_data_OpenCloseFistsFeet_session1_segmented"

    try:
        # Load model and evaluate
        labels, preds, probs = load_and_evaluate_model(model_path, data_dir, device)
        
        # Print classification report
        print("\nClassification Report:")
        print("=" * 50)
        print(classification_report(labels, preds, target_names=["Resting", "Open/Close Feet"]))
        
        # Plot results and get AUC score
        auc_score = plot_results(labels, preds, probs, save_path='test_feet_results.png')
        print(f"\nAUC Score: {auc_score:.3f}")
        
        # Test different thresholds
        test_with_different_thresholds(labels, probs)
        
        # Additional statistics
        print(f"\nTest Dataset Statistics:")
        print(f"Total samples: {len(labels)}")
        print(f"Resting samples: {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")
        print(f"Open/Close Feet samples: {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)")
        
        print(f"\nPrediction Statistics:")
        print(f"Predicted Resting: {np.sum(preds == 0)} ({np.mean(preds == 0)*100:.1f}%)")
        print(f"Predicted Open/Close Feet: {np.sum(preds == 1)} ({np.mean(preds == 1)*100:.1f}%)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Trained the model using the original script")
        print("2. The 'best_feet_model.pth' file exists in the current directory")
        print("3. The correct data directory path")
    except Exception as e:
        print(f"An error occurred: {e}")
