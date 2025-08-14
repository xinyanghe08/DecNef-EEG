import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import json

# Import custom modules
from dataset import TwoChannelEEGDataset, EEGNetDataset
from model import TwoChannelLSTMClassifier, create_eegnet_model
from utils import set_all_seeds
from training import train_eegnet_model
from evaluation import evaluate_simple_model, plot_results, print_classification_results



def main(exp_number=1):
    # Set random seeds for reproducibility
    set_all_seeds(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if exp_number == 1:
        task = "openclosefists"
        experiment_name = "fists"
    else:
        task = "openclosefeet"
        experiment_name = "feet"

    dataset = TwoChannelEEGDataset(
        # Data directory - modify this path as needed
        data_dir = r"C:\Users\xhe\Documents\GitHub\DecNef-EEG\decoder\data\session1",
        exp_number=exp_number,
        run_number=[2],  # load runs
        task=task,
        window_size=1024,
        debug=False,
        normalize=True
        )
    
    norm_params = dataset.get_normalization_params()
    print(norm_params)

    with open("normalization.json", "w") as f:
        json.dump(norm_params, f, indent=2)

    print("Saved correct normalization.json.")

    test_dataset = TwoChannelEEGDataset(
        # Used for cross session
        data_dir=r"C:\Users\xhe\Documents\GitHub\DecNef-EEG\decoder\data\session2",
        #data_dir=data_dir,
        exp_number=exp_number,
        run_number=[1],  # load runs
        task=task,
        window_size=1024,
        debug=False,
        normalize=True
    )


    # Create proper train/test split
    train_indices, val_indices = dataset.create_proper_train_test_split(method='trial_based', test_size=0.1)

    print(f"Final train: {len(train_indices)} samples")
    print(f"Final validation: {len(val_indices)} samples")
    print(f"Final test: {len(test_dataset)} samples")

    # Create datasets using Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Wrap existing datasets
    eegnet_train_dataset = EEGNetDataset(train_dataset)
    eegnet_val_dataset = EEGNetDataset(val_dataset)
    eegnet_test_dataset = EEGNetDataset(test_dataset)

    # Create new data loaders
    batch_size=32
    train_loader = DataLoader(eegnet_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(eegnet_val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(eegnet_test_dataset, batch_size=batch_size, shuffle=False)

    model = create_eegnet_model(task_type='binary', num_classes=1, samples=1024).to(device)

    train_losses, val_losses, best_acc = train_eegnet_model(model, train_loader, val_loader, device, epochs=700, lr=0.001, experiment_name=experiment_name)

    # Evaluate model
    labels, preds, probs = evaluate_simple_model(model, test_loader, device)

    # Print results
    print_classification_results(labels, preds, probs, name=experiment_name)

    # Plot and save results
    auc_score = plot_results(train_losses, val_losses, labels, preds, probs, f'{experiment_name}_train_result.png')


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main(exp_number=1)
    # main(exp_number=2)