import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


def evaluate_simple_model(model, test_loader, device):
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
            preds = (probs > 0.5).astype(int)

            all_preds.extend(preds.flatten())
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_results(train_losses, val_losses, labels, preds, probs, save_path='fists_result.png'):
    """Plot training curves, confusion matrix, and ROC curve"""
    name = save_path.split('_')[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training curves
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title(f'Training Curves')

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Resting", "Open/Close Fist"],
                yticklabels=["Resting", "Open/Close Fist"], annot_kws={"size": 16})
    axes[1].tick_params(axis='x', labelsize=18)
    axes[1].tick_params(axis='y', labelsize=18)
    axes[1].set_xlabel("Predicted", fontsize=20)
    axes[1].set_ylabel("True", fontsize=20)
    axes[1].set_title(f"Confusion Matrix", fontsize=20)

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    axes[2].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[2].set_xlabel("False Positive Rate", fontsize=20)
    axes[2].set_ylabel("True Positive Rate", fontsize=20)
    axes[2].set_title(f"ROC Curve", fontsize=20)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return auc_score

def plot_results_short(labels, preds, probs, save_path='fists_result.png'):
    """Plot confusion matrix, and ROC curve"""
    name = save_path.split('_')[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Resting", f"Open/Close {name}"],
                yticklabels=["Resting", f"Open/Close {name}"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix {name}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"ROC Curve {name}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return auc_score

def print_classification_results(labels, preds, probs, name="fists"):
    """Print classification report and AUC score"""
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Resting", f"Open/Close {name}"]))
    
    auc_score = roc_auc_score(labels, probs)
    print(f"\nAUC Score: {auc_score:.3f}")
    
    return auc_score


def plot_multiclass_results(train_losses, val_losses, val_accuracies, test_results, save_path='multiclass_results.png'):
    """
    Plot comprehensive results for multi-class classification
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training curves
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(val_losses, label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation accuracy
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(val_accuracies, color='green', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    cm = test_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Rest', 'Fists', 'Feet'],
                yticklabels=['Rest', 'Fists', 'Feet'],
                cbar_kws={'label': 'Count'})
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title(f'Confusion Matrix (Acc: {test_results["accuracy"]:.1f}%)')
    
    # 4. Normalized Confusion Matrix
    ax4 = plt.subplot(2, 3, 4)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Rest', 'Fists', 'Feet'],
                yticklabels=['Rest', 'Fists', 'Feet'],
                cbar_kws={'label': 'Proportion'})
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    ax4.set_title('Normalized Confusion Matrix')
    
    # 5. Per-class metrics bar plot
    ax5 = plt.subplot(2, 3, 5)
    classes = ['Resting', 'Fists', 'Feet']
    auc_values = [test_results['auc_scores'][c] for c in classes]
    
    # Calculate per-class accuracy from confusion matrix
    class_accuracies = []
    for i in range(3):
        if cm[i].sum() > 0:
            class_accuracies.append(cm[i, i] / cm[i].sum() * 100)
        else:
            class_accuracies.append(0)
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, class_accuracies, width, label='Accuracy (%)', alpha=0.8)
    bars2 = ax5.bar(x + width/2, np.array(auc_values) * 100, width, label='AUC × 100', alpha=0.8)
    
    ax5.set_xlabel('Class')
    ax5.set_ylabel('Score')
    ax5.set_title('Per-Class Performance')
    ax5.set_xticks(x)
    ax5.set_xticklabels(classes)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
    
    # 6. Multi-class ROC curves
    ax6 = plt.subplot(2, 3, 6)
    
    # Compute ROC curve and ROC area for each class
    y_true = label_binarize(test_results['all_labels'], classes=[0, 1, 2])
    y_score = test_results['all_probs']
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, (color, class_name) in enumerate(zip(colors, classes)):
        ax6.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    ax6.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax6.set_xlim([0.0, 1.0])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel('False Positive Rate')
    ax6.set_ylabel('True Positive Rate')
    ax6.set_title('Multi-class ROC Curves')
    ax6.legend(loc="lower right")
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_results['macro_auc']


def plot_multiclass_predictions_distribution(test_results, save_path='multiclass_pred_dist.png'):
    """
    Plot the distribution of prediction probabilities for each class
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    class_names = ['Resting', 'Fists', 'Feet']
    
    all_probs = test_results['all_probs']
    all_labels = test_results['all_labels']
    
    for idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Get probabilities for this class
        class_probs = all_probs[:, idx]
        
        # Separate by true label
        for true_label in range(3):
            mask = all_labels == true_label
            probs = class_probs[mask]
            
            ax.hist(probs, bins=30, alpha=0.5, density=True,
                   label=f'True: {class_names[true_label]}')
        
        ax.set_xlabel(f'Probability of {class_name}')
        ax.set_ylabel('Density')
        ax.set_title(f'Prediction Distribution for {class_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_misclassifications(test_results, segment_metadata=None):
    """
    Analyze and report common misclassification patterns
    """
    all_labels = test_results['all_labels']
    all_preds = test_results['all_preds']
    all_probs = test_results['all_probs']
    
    misclassified_mask = all_labels != all_preds
    misclassified_indices = np.where(misclassified_mask)[0]
    
    print(f"\n=== MISCLASSIFICATION ANALYSIS ===")
    print(f"Total misclassifications: {len(misclassified_indices)} / {len(all_labels)} "
          f"({len(misclassified_indices)/len(all_labels)*100:.1f}%)")
    
    # Confusion patterns
    confusion_patterns = {}
    for idx in misclassified_indices:
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        pattern = f"{true_label}→{pred_label}"
        
        if pattern not in confusion_patterns:
            confusion_patterns[pattern] = {
                'count': 0,
                'avg_confidence': 0,
                'indices': []
            }
        
        confusion_patterns[pattern]['count'] += 1
        confusion_patterns[pattern]['avg_confidence'] += all_probs[idx, pred_label]
        confusion_patterns[pattern]['indices'].append(idx)
    
    # Calculate averages and sort by frequency
    class_names = ['Rest', 'Fists', 'Feet']
    print(f"\nMost common confusion patterns:")
    
    sorted_patterns = sorted(confusion_patterns.items(), 
                           key=lambda x: x[1]['count'], 
                           reverse=True)
    
    for pattern, data in sorted_patterns[:5]:
        true_idx, pred_idx = map(int, pattern.split('→'))
        avg_conf = data['avg_confidence'] / data['count']
        
        print(f"  {class_names[true_idx]} → {class_names[pred_idx]}: "
              f"{data['count']} times (avg confidence: {avg_conf:.3f})")
    
    # Low confidence predictions
    print(f"\nLow confidence predictions analysis:")
    max_probs = np.max(all_probs, axis=1)
    low_conf_threshold = 0.5
    low_conf_mask = max_probs < low_conf_threshold
    
    print(f"  Predictions with confidence < {low_conf_threshold}: "
          f"{np.sum(low_conf_mask)} ({np.mean(low_conf_mask)*100:.1f}%)")
    
    # Accuracy vs confidence
    confidence_bins = [(0.33, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    print(f"\nAccuracy by confidence level:")
    
    for conf_min, conf_max in confidence_bins:
        mask = (max_probs >= conf_min) & (max_probs < conf_max)
        if np.sum(mask) > 0:
            acc = np.mean(all_labels[mask] == all_preds[mask]) * 100
            count = np.sum(mask)
            print(f"  [{conf_min:.2f}, {conf_max:.2f}): {acc:.1f}% accuracy "
                  f"({count} samples, {count/len(all_labels)*100:.1f}%)")
    
    return confusion_patterns


def create_multiclass_summary_report(train_losses, val_losses, val_accuracies, 
                                   test_results, save_path='multiclass_report.txt'):
    """
    Create a comprehensive text report of the multi-class results
    """
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MULTI-CLASS EEG CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Model performance summary
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Test Accuracy: {test_results['accuracy']:.2f}%\n")
        f.write(f"Macro-average AUC: {test_results['macro_auc']:.3f}\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS\n")
        f.write("-"*40 + "\n")
        
        cm = test_results['confusion_matrix']
        class_names = ['Resting', 'Fists', 'Feet']
        
        for i, class_name in enumerate(class_names):
            if cm[i].sum() > 0:
                precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
                recall = cm[i, i] / cm[i].sum()
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {precision:.3f}\n")
                f.write(f"  Recall: {recall:.3f}\n")
                f.write(f"  F1-Score: {f1:.3f}\n")
                f.write(f"  AUC: {test_results['auc_scores'][class_name]:.3f}\n")
                f.write(f"  Support: {cm[i].sum()}\n")
        
        # Confusion matrix
        f.write("\nCONFUSION MATRIX\n")
        f.write("-"*40 + "\n")
        f.write("        Predicted\n")
        f.write("        Rest  Fist  Feet\n")
        for i, true_class in enumerate(['Rest', 'Fist', 'Feet']):
            f.write(f"{true_class:6s}  ")
            for j in range(3):
                f.write(f"{cm[i,j]:4d}  ")
            f.write("\n")
        
        # Training summary
        f.write("\nTRAINING SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Final training loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final validation loss: {val_losses[-1]:.4f}\n")
        f.write(f"Best validation accuracy: {max(val_accuracies):.2f}%\n")
        f.write(f"Best accuracy epoch: {np.argmax(val_accuracies)}\n")
        
        # Classification report
        f.write("\nDETAILED CLASSIFICATION REPORT\n")
        f.write("-"*40 + "\n")
        f.write(test_results['classification_report'])
        
    print(f"\nDetailed report saved to {save_path}")
    
    return save_path