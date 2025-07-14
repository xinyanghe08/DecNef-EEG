
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import random

class TwoChannelEEGDataset(Dataset):
    def __init__(self, data_dir, run_number, exp_number=1, window_size=256, overlap=0.5):
        """
        Create windowed segments from continuous EEG data
        """
        self.window_size = window_size
        self.overlap = overlap

        # Load data
        tp9_file = f"openclosefists_run{run_number}_TP9.csv"
        tp10_file = f"openclosefists_run{run_number}_TP10.csv"
        label_file = f"openclosefists_run{run_number}_label.csv"

        exp_dir = os.path.join(data_dir, f"")

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

def train_simple_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3):
    """Simple training loop"""

    # Calculate class weights
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    pos_weight = torch.tensor(class_weights[1] / class_weights[0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_acc = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            target = target.view(-1, 1).float()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                target_reshaped = target.view(-1, 1).float()
                loss = criterion(output, target_reshaped)
                val_loss += loss.item()

                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target_reshaped).sum().item()
                total += target.size(0)

        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fists_model.pth')

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Load best model
    model.load_state_dict(torch.load('best_fists_model.pth'))
    print(f'Best validation accuracy: {best_val_acc:.4f}')

    return train_losses, val_losses


def evaluate_simple_model(model, test_loader, device, threshold=0.5):
    """Evaluate the model"""
    model.eval()
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



def create_validation_split(dataset, val_ratio=0.2):
    """Create train/validation split"""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size

    return torch.utils.data.random_split(dataset, [train_size, val_size])

if __name__ == "__main__":
    set_all_seeds(42)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets with windowing
    data_dir = r"C:\Users\xhe\Documents\GitHub\DecNef-EEG\decoder\data"

    train_dataset = TwoChannelEEGDataset(data_dir, run_number=2, window_size=1024, overlap=0.7)
    test_dataset = TwoChannelEEGDataset(data_dir, run_number=1, window_size=1024, overlap=0.7)

    # Create train/val split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = TwoChannelLSTMClassifier(input_channels=2, num_classes=1).to(device)

    # Train
    print("Training simple CNN...")
    train_losses, val_loss = train_simple_model(model, train_loader, val_loader, device, epochs=500, lr=0.0001)

    # Evaluate
    labels, preds, probs = evaluate_simple_model(model, test_loader, device, threshold=0.50)

    # Results
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Resting", "Open/Close Fist"]))

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training curves
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_loss, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training Curves Fist')

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Resting", "Open/Close Fist"],
                yticklabels=["Resting", "Open/Close Fist"])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix Fist")

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    axes[2].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve Fist")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('fists_result.png')
    plt.show()

    print(f"\nAUC Score: {auc_score:.3f}")

