import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from datetime import datetime
from model import EEGNetTrainer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

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
    best_epoch = 0
    best_confusion_matrix = None
    best_predictions = None
    best_targets = None

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
        all_predictions = []
        all_targets = []

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

                # Store predictions and targets for confusion matrix
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target_reshaped.cpu().numpy().flatten())

        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            # Calculate and store confusion matrix for best performance
            best_predictions = np.array(all_predictions)
            best_targets = np.array(all_targets)
            best_confusion_matrix = confusion_matrix(best_targets, best_predictions)

            torch.save(model.state_dict(), 'best_fists_model.pth')

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Load best model
    model.load_state_dict(torch.load('best_fists_model.pth'))
    # Display results for best validation performance
    print(f'\n=== BEST VALIDATION PERFORMANCE ===')
    print(f'Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}')
    
    if best_confusion_matrix is not None:
        print(f'\nConfusion Matrix at Best Validation Accuracy:')
        print(best_confusion_matrix)
        
        # Calculate metrics
        tn, fp, fn, tp = best_confusion_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f'\nDetailed Metrics at Best Validation:')
        print(f'  Sensitivity (Recall): {sensitivity:.3f}')
        print(f'  Specificity: {specificity:.3f}')
        print(f'  Precision: {precision:.3f}')
        print(f'  F1-Score: {f1_score:.3f}')
        
        # Classification report
        class_names = ['Resting', 'Open/Close Fist']
        print(f'\nClassification Report at Best Validation:')
        print(classification_report(best_targets, best_predictions, 
                                  target_names=class_names, digits=3))

    return train_losses, val_losses, best_confusion_matrix, best_epoch


def train_simple_model_with_universal_best(model, train_loader, val_loader, device, epochs=100, lr=1e-3, 
                                          model_save_dir='models', experiment_name='eeg_classification'):
    """
    Training loop that keeps track of universal best model across multiple runs
    """
    
    # Create model directory if it doesn't exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Define file paths
    universal_best_path = os.path.join(model_save_dir, f'{experiment_name}_universal_best.pth')
    current_run_path = os.path.join(model_save_dir, f'{experiment_name}_current_run.pth')
    metadata_path = os.path.join(model_save_dir, f'{experiment_name}_best_metadata.json')
    
    # Load previous best performance if exists
    universal_best_acc = 0
    universal_best_metadata = {}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                universal_best_metadata = json.load(f)
                universal_best_acc = universal_best_metadata.get('best_accuracy', 0)
            print(f"📊 Previous universal best accuracy: {universal_best_acc:.4f}")
            print(f"   From run: {universal_best_metadata.get('run_date', 'Unknown')}")
        except:
            print(" Could not load previous best metadata")

    # Calculate class weights
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())

    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    pos_weight = torch.tensor(class_weights[1] / class_weights[0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Current run tracking
    best_val_acc_current_run = 0
    train_losses = []
    val_losses = []
    best_epoch = 0
    best_confusion_matrix = None
    best_predictions = None
    best_targets = None
    
    # Universal best tracking
    found_new_universal_best = False
    
    print(f"\n Starting training...")
    print(f" Target to beat: {universal_best_acc:.4f}")

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
        all_predictions = []
        all_targets = []

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

                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target_reshaped.cpu().numpy().flatten())

        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Check if this is the best for current run
        if val_acc > best_val_acc_current_run:
            best_val_acc_current_run = val_acc
            best_epoch = epoch

            best_predictions = np.array(all_predictions)
            best_targets = np.array(all_targets)
            best_confusion_matrix = confusion_matrix(best_targets, best_predictions)

            # Always save current run's best
            torch.save(model.state_dict(), current_run_path)
            
            # Check if this beats universal best
            if val_acc > universal_best_acc:
                universal_best_acc = val_acc
                found_new_universal_best = True
                
                # Save new universal best
                torch.save(model.state_dict(), universal_best_path)
                
                # Update metadata
                current_metadata = {
                    'best_accuracy': float(val_acc),
                    'epoch': epoch,
                    'run_date': datetime.now().isoformat(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'confusion_matrix': best_confusion_matrix.tolist(),
                    'hyperparameters': {
                        'epochs': epochs,
                        'lr': lr,
                        'model_type': type(model).__name__
                    }
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(current_metadata, f, indent=2)
                
                print(f" NEW UNIVERSAL BEST! Accuracy: {val_acc:.4f} (Previous: {universal_best_metadata.get('best_accuracy', 0):.4f})")

        if epoch % 10 == 0:
            status = "New Best" if val_acc > universal_best_acc else ""
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} {status}')

    # Load best model from current run
    model.load_state_dict(torch.load(current_run_path))
    
    # Print results
    print(f'\n=== CURRENT RUN RESULTS ===')
    print(f'Best validation accuracy this run: {best_val_acc_current_run:.4f} at epoch {best_epoch}')
    
    print(f'\n=== UNIVERSAL BEST COMPARISON ===')
    if found_new_universal_best:
        print(f'🎉 NEW UNIVERSAL BEST ACHIEVED!')
        print(f'   New best: {universal_best_acc:.4f}')
        print(f'   Saved to: {universal_best_path}')
    else:
        print(f'   Current run best: {best_val_acc_current_run:.4f}')
        print(f'   Universal best:   {universal_best_acc:.4f}')
        print(f'   Gap to beat:      {universal_best_acc - best_val_acc_current_run:.4f}')
    
    if best_confusion_matrix is not None:
        print(f'\nCurrent Run - Confusion Matrix at Best Validation:')
        print(best_confusion_matrix)
        
        # Calculate metrics
        tn, fp, fn, tp = best_confusion_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        print(f'\nCurrent Run - Detailed Metrics:')
        print(f'  Sensitivity (Recall): {sensitivity:.3f}')
        print(f'  Specificity: {specificity:.3f}')
        print(f'  Precision: {precision:.3f}')
        print(f'  F1-Score: {f1_score:.3f}')

    return train_losses, val_losses, best_confusion_matrix, best_epoch, found_new_universal_best

def load_universal_best_model(model, model_save_dir='models', experiment_name='eeg_classification'):
    """
    Load the universal best model across all runs
    """
    universal_best_path = os.path.join(model_save_dir, f'{experiment_name}_universal_best.pth')
    metadata_path = os.path.join(model_save_dir, f'{experiment_name}_best_metadata.json')
    
    if not os.path.exists(universal_best_path):
        print(f"No universal best model found at {universal_best_path}")
        return None, None
    
    # Load model
    model.load_state_dict(torch.load(universal_best_path))
    print(f"Loaded universal best model from {universal_best_path}")
    
    # Load metadata if available
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"  Universal best performance:")
        print(f"   Accuracy: {metadata['best_accuracy']:.4f}")
        print(f"   From run: {metadata['run_date']}")
        print(f"   Epoch: {metadata['epoch']}")
    
    return model, metadata

def compare_all_runs(model_save_dir='models', experiment_name='eeg_classification'):
    """
    Show performance history across all runs
    """
    metadata_path = os.path.join(model_save_dir, f'{experiment_name}_best_metadata.json')
    
    if not os.path.exists(metadata_path):
        print("No run history found.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n=== UNIVERSAL BEST MODEL HISTORY ===")
    print(f"Best Accuracy: {metadata['best_accuracy']:.4f}")
    print(f"Achieved on: {metadata['run_date']}")
    print(f"At epoch: {metadata['epoch']}")
    print(f"Model type: {metadata.get('hyperparameters', {}).get('model_type', 'Unknown')}")
    
    if 'confusion_matrix' in metadata:
        cm = np.array(metadata['confusion_matrix'])
        print(f"Confusion Matrix:")
        print(cm)

def clean_old_models(model_save_dir='models', experiment_name='eeg_classification', keep_universal_best=True):
    """
    Clean up old model files, optionally keeping the universal best
    """
    import glob
    
    pattern = os.path.join(model_save_dir, f'{experiment_name}_*.pth')
    model_files = glob.glob(pattern)
    
    universal_best_path = os.path.join(model_save_dir, f'{experiment_name}_universal_best.pth')
    
    removed_count = 0
    for file_path in model_files:
        if keep_universal_best and file_path == universal_best_path:
            continue
        
        os.remove(file_path)
        removed_count += 1
        print(f"Removed: {file_path}")
    
    print(f"Cleaned {removed_count} old model files")
    if keep_universal_best:
        print(f"Kept universal best: {universal_best_path}")

def train_eegnet_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, experiment_name='fists'):
    """
    Modified training loop for EEGNet with max norm constraint
    """
    
    # EEGNet-specific optimizer
    optimizer = EEGNetTrainer.get_eegnet_optimizer(model, lr=lr)
    scheduler = EEGNetTrainer.get_eegnet_scheduler(optimizer)
    
    # Your existing criterion (works fine with EEGNet)
    criterion = nn.BCEWithLogitsLoss()
    
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
            
            # Apply max norm constraint (EEGNet specific)
            EEGNetTrainer.apply_max_norm_constraint(model)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation (same as your current code)
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
        
        if val_acc > best_val_acc or val_acc == 1:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{experiment_name}_eegnet_model.pth')
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, best_val_acc



def train_multiclass_eegnet(model, train_loader, val_loader, device, class_weights=None, 
                           epochs=100, lr=0.001, experiment_name='multiclass'):
    """
    Training loop for multi-class EEGNet
    """
    
    # Setup optimizer and scheduler
    optimizer = EEGNetTrainer.get_eegnet_optimizer(model, lr=lr)
    scheduler = EEGNetTrainer.get_eegnet_scheduler(optimizer)
    
    # Loss function for multi-class
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_epoch_metrics = {}
    
    print(f"\nStarting multi-class training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Apply max norm constraint (EEGNet specific)
            EEGNetTrainer.apply_max_norm_constraint(model)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                # Store for metrics
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(target.cpu().numpy())
                all_val_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{experiment_name}_multiclass_model.pth')
            
            # Calculate additional metrics for best model
            all_val_labels = np.array(all_val_labels)
            all_val_preds = np.array(all_val_preds)
            all_val_probs = np.array(all_val_probs)
            
            # Confusion matrix
            cm = confusion_matrix(all_val_labels, all_val_preds)
            
            # Per-class metrics
            class_report = classification_report(
                all_val_labels, all_val_preds,
                target_names=['Resting', 'Fists', 'Feet'],
                output_dict=True
            )
            
            # Multi-class AUC (one-vs-rest)
            if len(np.unique(all_val_labels)) == 3:
                y_true_binary = label_binarize(all_val_labels, classes=[0, 1, 2])
                auc_scores = {}
                for i, class_name in enumerate(['Resting', 'Fists', 'Feet']):
                    auc_scores[class_name] = roc_auc_score(y_true_binary[:, i], all_val_probs[:, i])
                macro_auc = np.mean(list(auc_scores.values()))
            else:
                auc_scores = {}
                macro_auc = 0
            
            best_epoch_metrics = {
                'epoch': epoch,
                'val_accuracy': val_acc,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'auc_scores': auc_scores,
                'macro_auc': macro_auc
            }
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if epoch == best_epoch_metrics.get('epoch', -1):
                print(f'  *** New best model! ***')
    
    print(f'\n=== Training Complete ===')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}% at epoch {best_epoch_metrics["epoch"]}')
    
    # Print best model's confusion matrix
    if 'confusion_matrix' in best_epoch_metrics:
        print(f'\nBest Model Confusion Matrix:')
        cm = np.array(best_epoch_metrics['confusion_matrix'])
        print(f'        Predicted')
        print(f'        Rest  Fist  Feet')
        print(f'Rest    {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}')
        print(f'Fist    {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}')
        print(f'Feet    {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}')
        
        # Print per-class AUC
        if 'auc_scores' in best_epoch_metrics:
            print(f'\nPer-class AUC scores:')
            for class_name, auc in best_epoch_metrics['auc_scores'].items():
                print(f'  {class_name}: {auc:.3f}')
            print(f'  Macro-average AUC: {best_epoch_metrics["macro_auc"]:.3f}')
    
    return train_losses, val_losses, val_accuracies, best_epoch_metrics


def save_multiclass_config(experiment_name, config_dict):
    """Save configuration for multi-class model"""
    os.makedirs('models', exist_ok=True)
    
    config_path = f'models/{experiment_name}_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Saved configuration to {config_path}")


def evaluate_multiclass_model(model, test_loader, device, class_names=['Resting', 'Fists', 'Feet']):
    """
    Comprehensive evaluation for multi-class model
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels) * 100
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    
    # Multi-class AUC
    y_true_binary = label_binarize(all_labels, classes=[0, 1, 2])
    auc_scores = {}
    for i, class_name in enumerate(class_names):
        auc_scores[class_name] = roc_auc_score(y_true_binary[:, i], all_probs[:, i])
    macro_auc = np.mean(list(auc_scores.values()))
    
    # Print results
    print(f"\n=== TEST SET RESULTS ===")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"\nClassification Report:")
    print(report)
    
    print(f"\nConfusion Matrix:")
    print(f'        Predicted')
    print(f'        Rest  Fist  Feet')
    print(f'Rest    {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}')
    print(f'Fist    {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}')
    print(f'Feet    {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}')
    
    print(f"\nPer-class AUC scores:")
    for class_name, auc in auc_scores.items():
        print(f"  {class_name}: {auc:.3f}")
    print(f"  Macro-average AUC: {macro_auc:.3f}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs,
        'auc_scores': auc_scores,
        'macro_auc': macro_auc,
        'classification_report': report
    }

def train_stage_model(model, train_loader, val_loader, device, stage_name, 
                     class_weights=None, epochs=100, lr=0.001):
    """
    Training function for either stage of the two-stage classifier
    """
    # Setup optimizer and scheduler
    optimizer = EEGNetTrainer.get_eegnet_optimizer(model, lr=lr)
    scheduler = EEGNetTrainer.get_eegnet_scheduler(optimizer)
    
    # Loss function
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # For binary classification with 2 outputs
    if model.classify.out_features == 1:
        criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_metrics = {}
    
    print(f"\nTraining {stage_name} model for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            if model.classify.out_features == 1:
                target = target.float().view(-1, 1)
                loss = criterion(output, target)
                pred = (torch.sigmoid(output) > 0.5).float()
            else:
                loss = criterion(output, target)
                _, pred = torch.max(output, 1)
            
            loss.backward()
            
            # Apply max norm constraint
            EEGNetTrainer.apply_max_norm_constraint(model)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_total += target.size(0)
            
            if model.classify.out_features == 1:
                train_correct += (pred == target).sum().item()
            else:
                train_correct += (pred == target).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if model.classify.out_features == 1:
                    target = target.float().view(-1, 1)
                    loss = criterion(output, target)
                    probs = torch.sigmoid(output)
                    pred = (probs > 0.5).float()
                    
                    all_val_probs.extend(probs.cpu().numpy().flatten())
                else:
                    loss = criterion(output, target)
                    probs = torch.softmax(output, dim=1)
                    _, pred = torch.max(output, 1)
                    
                    all_val_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class
                
                val_loss += loss.item()
                val_total += target.size(0)
                
                if model.classify.out_features == 1:
                    val_correct += (pred == target).sum().item()
                    all_val_preds.extend(pred.cpu().numpy().flatten())
                    all_val_labels.extend(target.cpu().numpy().flatten())
                else:
                    val_correct += (pred == target).sum().item()
                    all_val_preds.extend(pred.cpu().numpy())
                    all_val_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{stage_name}_model.pth')
            
            # Calculate additional metrics
            all_val_labels = np.array(all_val_labels)
            all_val_preds = np.array(all_val_preds)
            all_val_probs = np.array(all_val_probs)
            
            cm = confusion_matrix(all_val_labels, all_val_preds)
            auc = roc_auc_score(all_val_labels, all_val_probs)
            
            best_metrics = {
                'epoch': epoch,
                'val_accuracy': val_acc,
                'confusion_matrix': cm.tolist(),
                'auc': auc
            }
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    print(f'\n{stage_name} Training Complete!')
    print(f'Best Validation Accuracy: {best_val_acc:.2f}% at epoch {best_metrics["epoch"]}')
    
    return train_losses, val_losses, val_accuracies, best_metrics


def evaluate_stage_model(model, test_loader, device, stage_name, class_names):
    """
    Evaluate a stage model
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            if model.classify.out_features == 1:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels) * 100
    cm = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)
    
    print(f"\n=== {stage_name} TEST RESULTS ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"AUC: {auc:.3f}")
    print(f"\nClassification Report:")
    print(report)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_probs': all_probs,
        'classification_report': report
    }


def evaluate_two_stage_system(stage1_model, stage2_model, test_loader, device, test_dataset):
    """
    Evaluate the complete two-stage system
    """
    stage1_model.eval()
    stage2_model.eval()
    
    all_true_labels = []
    all_pred_labels = []
    all_stage1_probs = []
    all_stage2_probs = []
    stage1_predictions = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            batch_size = data.size(0)
            
            # Get true 3-class labels for this batch
            start_idx = i * test_loader.batch_size
            end_idx = min(start_idx + batch_size, len(test_dataset))
            true_3class_labels = test_dataset.original_labels[start_idx:end_idx]
            all_true_labels.extend(true_3class_labels)
            
            # Stage 1: Rest vs Motor Imagery
            stage1_outputs = stage1_model(data)
            if stage1_model.classify.out_features == 1:
                stage1_probs = torch.sigmoid(stage1_outputs).squeeze()
                stage1_preds = (stage1_probs > 0.5).float()
            else:
                stage1_probs = torch.softmax(stage1_outputs, dim=1)[:, 1]
                stage1_preds = torch.argmax(stage1_outputs, dim=1).float()
            
            all_stage1_probs.extend(stage1_probs.cpu().numpy())
            stage1_predictions.extend(stage1_preds.cpu().numpy())
            
            # Initialize predictions
            batch_pred_labels = np.zeros(batch_size)
            
            # For samples predicted as motor imagery, apply stage 2
            motor_imagery_mask = stage1_preds == 1
            motor_imagery_indices = torch.where(motor_imagery_mask)[0]
            
            if len(motor_imagery_indices) > 0:
                motor_imagery_data = data[motor_imagery_indices]
                
                # Stage 2: Fists vs Feet
                stage2_outputs = stage2_model(motor_imagery_data)
                if stage2_model.classify.out_features == 1:
                    stage2_probs = torch.sigmoid(stage2_outputs).squeeze()
                    stage2_preds = (stage2_probs > 0.5).float()
                else:
                    stage2_probs = torch.softmax(stage2_outputs, dim=1)[:, 1]
                    stage2_preds = torch.argmax(stage2_outputs, dim=1).float()
                
                # Convert stage 2 predictions to 3-class labels
                # Stage 2: 0 = fists (class 1), 1 = feet (class 2)
                for idx, motor_idx in enumerate(motor_imagery_indices.cpu().numpy()):
                    if stage2_preds[idx] == 0:
                        batch_pred_labels[motor_idx] = 1  # Fists
                    else:
                        batch_pred_labels[motor_idx] = 2  # Feet
                
                # Store stage 2 probabilities for motor imagery samples
                stage2_prob_array = np.zeros(batch_size)
                stage2_prob_array[motor_imagery_indices.cpu().numpy()] = stage2_probs.cpu().numpy()
                all_stage2_probs.extend(stage2_prob_array)
            else:
                all_stage2_probs.extend(np.zeros(batch_size))
            
            all_pred_labels.extend(batch_pred_labels)
    
    all_true_labels = np.array(all_true_labels).astype(int)
    all_pred_labels = np.array(all_pred_labels).astype(int)
    
    # Calculate metrics
    accuracy = np.mean(all_true_labels == all_pred_labels) * 100
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    # Calculate per-class metrics
    class_names = ['Resting', 'Fists', 'Feet']
    report = classification_report(all_true_labels, all_pred_labels, 
                                 target_names=class_names, digits=3)
    
    print("\n=== TWO-STAGE SYSTEM RESULTS ===")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"\nClassification Report:")
    print(report)
    print(f"\nConfusion Matrix:")
    print(f'        Predicted')
    print(f'        Rest  Fist  Feet')
    for i, true_class in enumerate(['Rest', 'Fist', 'Feet']):
        print(f"{true_class:6s}  ", end="")
        for j in range(3):
            print(f"{cm[i,j]:4d}  ", end="")
        print()
    
    # Stage-wise analysis
    print(f"\n=== STAGE-WISE ANALYSIS ===")
    stage1_acc = np.mean(
        (np.array(stage1_predictions) == 0) == (all_true_labels == 0)
    ) * 100
    print(f"Stage 1 (Rest vs Motor) Accuracy: {stage1_acc:.2f}%")
    
    # Stage 2 accuracy (only for motor imagery samples)
    motor_mask = all_true_labels > 0
    if np.sum(motor_mask) > 0:
        stage2_acc = np.mean(
            all_true_labels[motor_mask] == all_pred_labels[motor_mask]
        ) * 100
        print(f"Stage 2 (Fists vs Feet) Accuracy: {stage2_acc:.2f}%")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'all_true_labels': all_true_labels,
        'all_pred_labels': all_pred_labels,
        'all_stage1_probs': np.array(all_stage1_probs),
        'all_stage2_probs': np.array(all_stage2_probs),
        'classification_report': report,
        'stage1_accuracy': stage1_acc,
        'stage2_accuracy': stage2_acc if np.sum(motor_mask) > 0 else None
    }


def save_two_stage_config(config_dict):
    """Save configuration for two-stage model"""
    os.makedirs('models', exist_ok=True)
    
    config_path = 'models/two_stage_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Saved configuration to {config_path}")