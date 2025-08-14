import torch
import numpy as np
from model import create_eegnet_model

def decoder_predict(X_input: np.ndarray, trained_model_path: str):
    """
    Predict using real-time data with EEGNet model.
    X_input shape (2, 1024) where 2 is the num_channels and 1024 is the time.
    """
    # Check input shape
    if X_input.ndim != 2:
        raise ValueError("X_input must be 2D: (2, 1024).")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_eegnet_model(task_type='binary', num_classes=1, samples=X_input.shape[1])
    model.load_state_dict(torch.load(trained_model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Prepare input: (2, 1024) -> (1, 1, 2, 1024)
    X_input = X_input[np.newaxis, np.newaxis, ...]  # Add batch and EEGNet format dimensions
    X_tensor = torch.FloatTensor(X_input).to(device)
    print(X_tensor.shape)
    
    # Predict
    with torch.no_grad():
        output = model(X_tensor)
        prob = torch.sigmoid(output).cpu().numpy()[0, 0]  # Get probability for single sample
    
    prob_class1 = prob  # Probability for class 1
    pred_label = int(prob > 0.5) 
    
    return prob_class1, pred_label


if __name__ == "__main__":
    model_file = "best_fists_eegnet_model.pth"  # for open/close fists task
    
    # Get segment_4s from either .csv file or real-time
    # Simulating segment_4s data
    segment_4s = np.random.randn(1024, 4) * 1e-5  # Change this to real-time input
    
    X = segment_4s[:, [0, 3]].T  # TP9 & TP10
    y_prob, y_pred = decoder_predict(X_input=X, trained_model_path=model_file)
    print(f'prob for target class_1 is {y_prob}, predicted_label is {y_pred}')