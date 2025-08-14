import torch
import torch.nn as nn


# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Attention, self).__init__()
#         self.attn = nn.Linear(hidden_dim, 1)

#     def forward(self, lstm_output):
#         # lstm_output: [batch, seq_len, hidden_dim]
#         scores = self.attn(lstm_output).squeeze(-1)  # [batch, seq_len]
#         weights = F.softmax(scores, dim=1)           # [batch, seq_len]
#         weighted_output = torch.bmm(weights.unsqueeze(1), lstm_output)  # [batch, 1, hidden_dim]
#         return weighted_output.squeeze(1)            # [batch, hidden_dim]
    
    
class TwoChannelLSTMClassifier(nn.Module):
    def __init__(self, input_channels=2, hidden_size=64, num_layers=2, num_classes=1, dropout_rate=0.3):
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

class EEGNet(nn.Module):
    """
    EEGNet implementation optimized for 2-channel motor imagery classification
    
    Based on: Lawhern et al. "EEGNet: a compact convolutional neural network 
    for EEG-based brain–computer interfaces" (2018)
    
    Optimized for:
    - 2 channels (TP9, TP10)
    - Motor imagery tasks (open/close fists, feet)
    - Input shape: (batch_size, 1, 2, time_samples)
    """
    
    def __init__(self, nb_classes=1, Chans=2, Samples=1024, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16,
                 norm_rate=0.25, dropoutType='Dropout'):
        super(EEGNet, self).__init__()
        
        if dropoutType == 'SpatialDropout2D':
            dropoutType = nn.Dropout2d
        elif dropoutType == 'Dropout':
            dropoutType = nn.Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout')
        
        # Store parameters
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        
        # Block 1: Temporal Convolution
        # Learn frequency-specific temporal patterns
        padding1 = (kernLength - 1) // 2  # Manual padding calculation
        self.firstconv = nn.Conv2d(1, F1, (1, kernLength), padding=(0, padding1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        
        # Block 2: Spatial Convolution (Depthwise)
        # Learn spatial patterns for each temporal feature
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.activation1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = dropoutType(dropoutRate)
        
        # Block 3: Separable Convolution
        # Efficiently combine features
        kernel_size = (1, 16)
        padding3 = (kernel_size[0] // 2, kernel_size[1] // 2)  # This gives us (0, 8)
        self.separableConv = nn.Conv2d(F1 * D, F2, kernel_size, padding=padding3, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.activation2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = dropoutType(dropoutRate)
        
        # Calculate the size after convolutions and pooling
        # For input (1, 2, 1025): after pooling1 (/4) and pooling2 (/8) = 1025/32 ≈ 32
        self.final_conv_length = Samples // 32
        
        # Classification layer
        self.flatten = nn.Flatten()
        self.classify = nn.Linear(F2 * self.final_conv_length, nb_classes)
        
        # Apply max norm constraint to classification layer
        self.max_norm_val = norm_rate
        
    def forward(self, x):
        # Input shape: (batch_size, 1, 2, time_samples)
        
        # Block 1: Temporal Convolution
        x = self.firstconv(x)
        x = self.batchnorm1(x)
        
        # Block 2: Spatial Convolution
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable Convolution
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Flatten and classify
        x = self.flatten(x)
        x = self.classify(x)
        
        return x
    
    def max_norm_constraint(self):
        """Apply max norm constraint to the final classification layer"""
        if hasattr(self.classify, 'weight'):
            with torch.no_grad():
                norm = self.classify.weight.norm(dim=1, keepdim=True)
                desired = torch.clamp(norm, 0, self.max_norm_val)
                self.classify.weight *= (desired / norm)


class EEGNet_MultiClass(nn.Module):
    """
    EEGNet variant for multi-class classification
    Use this if you want to classify: Rest, Open Fist, Close Fist, Feet (4 classes)
    """
    
    def __init__(self, nb_classes=4, Chans=2, Samples=1024, 
                 dropoutRate=0.25, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNet_MultiClass, self).__init__()
        
        # Same architecture as binary EEGNet
        self.eegnet_features = EEGNet(nb_classes=F2*Samples//32, Chans=Chans, 
                                     Samples=Samples, dropoutRate=dropoutRate,
                                     kernLength=kernLength, F1=F1, D=D, F2=F2)
        
        # Remove the classification layer from base EEGNet
        self.eegnet_features.classify = nn.Identity()
        
        # Custom multi-class classification head
        self.final_conv_length = Samples // 32
        self.classifier = nn.Sequential(
            nn.Dropout(dropoutRate),
            nn.Linear(F2 * self.final_conv_length, 64),
            nn.ELU(),
            nn.Dropout(dropoutRate * 0.5),
            nn.Linear(64, nb_classes)
        )
        
    def forward(self, x):
        # Extract features using EEGNet backbone
        x = self.eegnet_features(x)
        # Classify with multi-class head
        x = self.classifier(x)
        return x
    

# Training modifications for EEGNet
class EEGNetTrainer:
    """
    Training utilities specific to EEGNet
    """
    
    @staticmethod
    def apply_max_norm_constraint(model):
        """Apply max norm constraint during training"""
        if hasattr(model, 'max_norm_constraint'):
            model.max_norm_constraint()
    
    @staticmethod
    def get_eegnet_optimizer(model, lr=0.001, weight_decay=1e-4):
        """Recommended optimizer settings for EEGNet"""
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    @staticmethod
    def get_eegnet_scheduler(optimizer):
        """Recommended learning rate scheduler for EEGNet"""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    
def create_eegnet_model(task_type='binary', num_classes=1, samples=1024):
    """
    Factory function to create appropriate EEGNet model
    
    Args:
        task_type: 'binary' for your current setup, 'multiclass' for extended classification
        num_classes: 1 for binary classification, >1 for multiclass
        samples: number of time samples (your current: 1025)
    
    Returns:
        EEGNet model ready for training
    """
    
    if task_type == 'binary':
        model = EEGNet(
            nb_classes=num_classes,
            Chans=2,  # TP9, TP10
            Samples=samples,
            dropoutRate=0.25,  # Lower dropout for small dataset
            kernLength=64,     # Good for 256 Hz sampling rate
            F1=8,              # Number of temporal filters
            D=2,               # Depth multiplier
            F2=16              # Number of separable filters
        )
    elif task_type == 'multiclass':
        model = EEGNet_MultiClass(
            nb_classes=num_classes,
            Chans=2,
            Samples=samples,
            dropoutRate=0.25,
            kernLength=64,
            F1=8,
            D=2,
            F2=16
        )
    else:
        raise ValueError("task_type must be 'binary' or 'multiclass'")
    
    return model