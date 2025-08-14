import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import json

class TwoChannelEEGDataset(Dataset):
    def __init__(self, data_dir, run_number, exp_number=1, window_size=1024, overlap=0.5, 
                 task="openclosefists", debug=True, normalize=True, normalization_params=None):
        """
        EEG Dataset with consistent normalization
        
        Args:
            normalize: Whether to apply normalization
            normalization_params: Dict with 'mean' and 'std' for each channel. If None, will compute from data.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.normalize = normalize
        self.segments = []
        self.segment_labels = []
        self.segment_metadata = []
        
        # For storing normalization parameters
        self.normalization_params = normalization_params
        self.computed_norm_params = None

        if isinstance(run_number, int):
            run_number = [run_number]

        # First load all data
        all_tp9_data = []
        all_tp10_data = []
        all_labels = []
        
        for run in run_number:
            tp9_data, tp10_data, labels = self._load_run_data(data_dir, run, exp_number, task, debug)
            all_tp9_data.append(tp9_data)
            all_tp10_data.append(tp10_data)
            all_labels.append(labels)
        
        # Concatenate all data
        all_tp9_data = np.vstack(all_tp9_data)
        all_tp10_data = np.vstack(all_tp10_data)
        all_labels = np.hstack(all_labels)
        
        # Compute normalization parameters if needed
        if self.normalize and self.normalization_params is None:
            self.computed_norm_params = self._compute_normalization_params(all_tp9_data, all_tp10_data)
            if debug:
                print(f"\nComputed normalization parameters:")
                print(f"  TP9 - mean: {self.computed_norm_params['tp9_mean']:.6f}, std: {self.computed_norm_params['tp9_std']:.6f}")
                print(f"  TP10 - mean: {self.computed_norm_params['tp10_mean']:.6f}, std: {self.computed_norm_params['tp10_std']:.6f}")
        elif self.normalize:
            self.computed_norm_params = self.normalization_params
            if debug:
                print(f"\nUsing provided normalization parameters:")
                print(f"  TP9 - mean: {self.computed_norm_params['tp9_mean']:.6f}, std: {self.computed_norm_params['tp9_std']:.6f}")
                print(f"  TP10 - mean: {self.computed_norm_params['tp10_mean']:.6f}, std: {self.computed_norm_params['tp10_std']:.6f}")
        
        # Now process the data with normalization
        for i, run in enumerate(run_number):
            self._process_run(all_tp9_data[i*len(labels):i*len(labels)+len(labels)], 
                            all_tp10_data[i*len(labels):i*len(labels)+len(labels)], 
                            labels, run, debug)

        self.segments = np.array(self.segments)
        self.segment_labels = np.array(self.segment_labels)

        if debug:
            print(f"\n=== FINAL SEGMENTS ===")
            print(f"Created {len(self.segments)} segments from runs {run_number}")
            print(f"Segment shape: {self.segments.shape}")
            print(f"Label distribution: {np.bincount(self.segment_labels.astype(int))}")
            if self.normalize:
                print(f"Data range after normalization: [{np.min(self.segments):.3f}, {np.max(self.segments):.3f}]")

    def _load_run_data(self, data_dir, run_number, exp_number, task, debug):
        """Load raw data from files"""
        exp_dir = os.path.join(data_dir, f"exp_{exp_number}")
        tp9_file = f"{task}_run{run_number}_TP9.csv"
        tp10_file = f"{task}_run{run_number}_TP10.csv"
        label_file = f"{task}_run{run_number}_label.csv"

        tp9_data = pd.read_csv(os.path.join(exp_dir, tp9_file), header=None).values
        tp10_data = pd.read_csv(os.path.join(exp_dir, tp10_file), header=None).values
        labels = pd.read_csv(os.path.join(exp_dir, label_file), header=None).values.flatten()

        if debug:
            print(f"\n=== RUN {run_number} RAW DATA ===")
            print(f"TP9 shape: {tp9_data.shape}, range: [{np.min(tp9_data):.6f}, {np.max(tp9_data):.6f}]")
            print(f"TP10 shape: {tp10_data.shape}, range: [{np.min(tp10_data):.6f}, {np.max(tp10_data):.6f}]")

        return tp9_data, tp10_data, labels

    def _compute_normalization_params(self, tp9_data, tp10_data):
        """Compute mean and std for normalization"""
        return {
            'tp9_mean': np.mean(tp9_data),
            'tp9_std': np.std(tp9_data),
            'tp10_mean': np.mean(tp10_data),
            'tp10_std': np.std(tp10_data)
        }
    
    def _normalize_data(self, tp9_data, tp10_data):
        """Apply normalization using stored parameters"""
        if self.computed_norm_params is None:
            return tp9_data, tp10_data
        
        tp9_normalized = (tp9_data - self.computed_norm_params['tp9_mean']) / self.computed_norm_params['tp9_std']
        tp10_normalized = (tp10_data - self.computed_norm_params['tp10_mean']) / self.computed_norm_params['tp10_std']
        
        return tp9_normalized, tp10_normalized

    def _process_run(self, tp9_data, tp10_data, labels, run_number, debug):
        """Process data and create segments"""
        # Apply normalization if enabled
        if self.normalize:
            tp9_data, tp10_data = self._normalize_data(tp9_data, tp10_data)
            
        # Segment creation
        for i, (tp9_sample, tp10_sample, label) in enumerate(zip(tp9_data, tp10_data, labels)):
            data_length = len(tp9_sample)
            if self.window_size == data_length:
                segment = np.stack([tp9_sample, tp10_sample], axis=0)
                self.segments.append(segment)
                self.segment_labels.append(label)
                self.segment_metadata.append({
                    'original_trial': i,
                    'window_start': 0,
                    'window_end': data_length,
                    'run': run_number
                })
            else:
                step = int(self.window_size * (1 - self.overlap))
                for start in range(0, data_length - self.window_size + 1, step):
                    end = start + self.window_size
                    segment = np.stack([
                        tp9_sample[start:end],
                        tp10_sample[start:end]
                    ], axis=0)
                    self.segments.append(segment)
                    self.segment_labels.append(label)
                    self.segment_metadata.append({
                        'original_trial': i,
                        'window_start': start,
                        'window_end': end,
                        'run': run_number
                    })

    def get_normalization_params(self):
        """Return the normalization parameters used"""
        return self.computed_norm_params

    def save_normalization_params(self, filepath):
        """Save normalization parameters to file"""
        if self.computed_norm_params is not None:
            with open(filepath, 'w') as f:
                json.dump(self.computed_norm_params, f, indent=2)
            print(f"Saved normalization parameters to {filepath}")

    @staticmethod
    def load_normalization_params(filepath):
        """Load normalization parameters from file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    def create_proper_train_test_split(self, test_size=0.3, method='trial_based'):
        """Create proper train/test split to avoid data leakage"""
        if method == 'trial_based':
            unique_trials = np.unique([meta['original_trial'] for meta in self.segment_metadata])
            np.random.shuffle(unique_trials)
            
            n_test_trials = int(len(unique_trials) * test_size)
            test_trials = set(unique_trials[:n_test_trials])
            
            train_indices = []
            test_indices = []
            
            for i, meta in enumerate(self.segment_metadata):
                if meta['original_trial'] in test_trials:
                    test_indices.append(i)
                else:
                    train_indices.append(i)
            
            return train_indices, test_indices
        
        elif method == 'temporal':
            n_test = int(len(self.segments) * test_size)
            indices = np.arange(len(self.segments))
            
            train_indices = indices[:-n_test]
            test_indices = indices[-n_test:]
            
            return train_indices, test_indices
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        label = torch.tensor(self.segment_labels[idx], dtype=torch.float)
        return segment, label


class EEGNetDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that formats data for EEGNet
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        segment, label = self.base_dataset[idx]
        eegnet_segment = segment.unsqueeze(0)  # Shape: (1, 2, time_samples)
        label = torch.tensor(int(label), dtype=torch.long)
        return eegnet_segment, label
    

class MultiClassEEGDataset(Dataset):
    """
    Multi-class EEG Dataset that combines fists and feet experiments
    for 3-class classification: resting (0), fists (1), feet (2)
    """
    
    def __init__(self, data_dir, run_number, window_size=1025, overlap=0.5, 
                 debug=True, normalize=True, normalization_params=None):
        """
        Args:
            data_dir: Root directory containing exp_1 and exp_2
            run_number: Run number(s) to load (can be int or list)
            window_size: Size of each segment
            overlap: Overlap between segments
            debug: Whether to print debug information
            normalize: Whether to apply normalization
            normalization_params: Dict with normalization parameters for each experiment
        """
        self.window_size = window_size
        self.overlap = overlap
        self.normalize = normalize
        self.segments = []
        self.segment_labels = []
        self.segment_metadata = []
        
        if isinstance(run_number, int):
            run_number = [run_number]
        
        # Store normalization parameters
        self.normalization_params = normalization_params
        self.computed_norm_params = {}
        
        # Collect all data first for normalization computation
        all_data = {'fists': {'tp9': [], 'tp10': [], 'labels': []},
                   'feet': {'tp9': [], 'tp10': [], 'labels': []}}
        
        # Define experiments
        experiments = [
            {'exp_num': 1, 'task_short': 'fists', 'task_long': 'openclosefists'},
            {'exp_num': 2, 'task_short': 'feet', 'task_long': 'openclosefeet'}
        ]
        
        # Load data from both experiments
        for exp_info in experiments:
            exp_num = exp_info['exp_num']
            task_short = exp_info['task_short']
            task_long = exp_info['task_long']
            
            if debug:
                print(f"\n=== Loading {task_short.upper()} data (exp_{exp_num}) ===")
            
            for run in run_number:
                exp_dir = os.path.join(data_dir, f"exp_{exp_num}")
                tp9_file = f"{task_long}_run{run}_TP9.csv"
                tp10_file = f"{task_long}_run{run}_TP10.csv"
                label_file = f"{task_long}_run{run}_label.csv"
                
                tp9_data = pd.read_csv(os.path.join(exp_dir, tp9_file), header=None).values
                tp10_data = pd.read_csv(os.path.join(exp_dir, tp10_file), header=None).values
                labels = pd.read_csv(os.path.join(exp_dir, label_file), header=None).values.flatten()
                
                all_data[task_short]['tp9'].append(tp9_data)
                all_data[task_short]['tp10'].append(tp10_data)
                all_data[task_short]['labels'].append(labels)
                
                if debug:
                    print(f"  Run {run}: {len(labels)} trials loaded")
        
        # Concatenate all data
        for task in ['fists', 'feet']:
            if len(all_data[task]['tp9']) > 0:
                all_data[task]['tp9'] = np.vstack(all_data[task]['tp9'])
                all_data[task]['tp10'] = np.vstack(all_data[task]['tp10'])
                all_data[task]['labels'] = np.hstack(all_data[task]['labels'])
        
        # Compute normalization parameters if needed
        if self.normalize and self.normalization_params is None:
            self._compute_normalization_params(all_data, debug)
        elif self.normalize and self.normalization_params is not None:
            self.computed_norm_params = self.normalization_params
            if debug:
                print("\nUsing provided normalization parameters")
        
        # Process data with multi-class labels
        self._process_all_data(all_data, run_number, debug)
        
        self.segments = np.array(self.segments)
        self.segment_labels = np.array(self.segment_labels)
        
        if debug:
            self._print_dataset_stats()
    
    def _compute_normalization_params(self, all_data, debug):
        """Compute normalization parameters from all data"""
        if debug:
            print("\n=== Computing normalization parameters ===")
        
        # Combine all data for global normalization
        all_tp9 = np.vstack([all_data['fists']['tp9'], all_data['feet']['tp9']])
        all_tp10 = np.vstack([all_data['fists']['tp10'], all_data['feet']['tp10']])
        
        self.computed_norm_params = {
            'tp9_mean': np.mean(all_tp9),
            'tp9_std': np.std(all_tp9),
            'tp10_mean': np.mean(all_tp10),
            'tp10_std': np.std(all_tp10),
            'global_mean': np.mean(np.hstack([all_tp9.flatten(), all_tp10.flatten()])),
            'global_std': np.std(np.hstack([all_tp9.flatten(), all_tp10.flatten()]))
        }
        
        if debug:
            print(f"  TP9 - mean: {self.computed_norm_params['tp9_mean']:.6f}, "
                  f"std: {self.computed_norm_params['tp9_std']:.6f}")
            print(f"  TP10 - mean: {self.computed_norm_params['tp10_mean']:.6f}, "
                  f"std: {self.computed_norm_params['tp10_std']:.6f}")
            print(f"  Global - mean: {self.computed_norm_params['global_mean']:.6f}, "
                  f"std: {self.computed_norm_params['global_std']:.6f}")
    
    def _normalize_data(self, tp9_data, tp10_data):
        """Apply normalization to data"""
        if self.computed_norm_params is None:
            return tp9_data, tp10_data
        
        tp9_normalized = (tp9_data - self.computed_norm_params['tp9_mean']) / self.computed_norm_params['tp9_std']
        tp10_normalized = (tp10_data - self.computed_norm_params['tp10_mean']) / self.computed_norm_params['tp10_std']
        
        return tp9_normalized, tp10_normalized
    
    def _process_all_data(self, all_data, run_numbers, debug):
        """Process data and create segments with multi-class labels"""
        
        # Process fists data (label 1 for active, 0 for resting)
        for task_idx, task in enumerate(['fists', 'feet']):
            tp9_data = all_data[task]['tp9']
            tp10_data = all_data[task]['tp10']
            labels = all_data[task]['labels']
            
            # Apply normalization if enabled
            if self.normalize:
                tp9_data, tp10_data = self._normalize_data(tp9_data, tp10_data)
            
            # Create segments
            for i, (tp9_sample, tp10_sample, orig_label) in enumerate(zip(tp9_data, tp10_data, labels)):
                # Relabel: resting stays 0, active becomes 1 (fists) or 2 (feet)
                if orig_label == 0:
                    multi_label = 0  # Resting
                else:
                    multi_label = task_idx + 1  # 1 for fists, 2 for feet
                
                data_length = len(tp9_sample)
                
                if self.window_size == data_length:
                    segment = np.stack([tp9_sample, tp10_sample], axis=0)
                    self.segments.append(segment)
                    self.segment_labels.append(multi_label)
                    self.segment_metadata.append({
                        'original_trial': i,
                        'window_start': 0,
                        'window_end': data_length,
                        'task': task,
                        'original_label': orig_label
                    })
                else:
                    step = int(self.window_size * (1 - self.overlap))
                    for start in range(0, data_length - self.window_size + 1, step):
                        end = start + self.window_size
                        segment = np.stack([
                            tp9_sample[start:end],
                            tp10_sample[start:end]
                        ], axis=0)
                        self.segments.append(segment)
                        self.segment_labels.append(multi_label)
                        self.segment_metadata.append({
                            'original_trial': i,
                            'window_start': start,
                            'window_end': end,
                            'task': task,
                            'original_label': orig_label
                        })
    
    def _print_dataset_stats(self):
        """Print detailed dataset statistics"""
        print(f"\n=== MULTI-CLASS DATASET STATISTICS ===")
        print(f"Total segments: {len(self.segments)}")
        print(f"Segment shape: {self.segments.shape}")
        
        # Class distribution
        print(f"\nClass distribution:")
        unique_labels, counts = np.unique(self.segment_labels, return_counts=True)
        class_names = ['Resting', 'Fists', 'Feet']
        
        for label, count in zip(unique_labels, counts):
            percentage = count / len(self.segment_labels) * 100
            print(f"  Class {label} ({class_names[int(label)]}): {count} samples ({percentage:.1f}%)")
        
        # Task distribution
        print(f"\nTask distribution:")
        task_counts = {'fists': 0, 'feet': 0}
        for meta in self.segment_metadata:
            task_counts[meta['task']] += 1
        
        for task, count in task_counts.items():
            percentage = count / len(self.segment_metadata) * 100
            print(f"  {task.capitalize()}: {count} samples ({percentage:.1f}%)")
        
        if self.normalize:
            print(f"\nNormalized data range: [{np.min(self.segments):.3f}, {np.max(self.segments):.3f}]")
    
    def get_normalization_params(self):
        """Return the normalization parameters used"""
        return self.computed_norm_params
    
    def save_normalization_params(self, filepath):
        """Save normalization parameters to file"""
        if self.computed_norm_params is not None:
            with open(filepath, 'w') as f:
                json.dump(self.computed_norm_params, f, indent=2)
            print(f"Saved normalization parameters to {filepath}")
    
    @staticmethod
    def load_normalization_params(filepath):
        """Load normalization parameters from file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def create_stratified_split(self, test_size=0.2, random_state=42):
        """Create stratified train/test split maintaining class balance"""
        from sklearn.model_selection import train_test_split
        
        # Get unique trials per task
        fists_trials = set()
        feet_trials = set()
        
        for i, meta in enumerate(self.segment_metadata):
            if meta['task'] == 'fists':
                fists_trials.add(meta['original_trial'])
            else:
                feet_trials.add(meta['original_trial'])
        
        # Split trials for each task
        fists_trials = list(fists_trials)
        feet_trials = list(feet_trials)
        
        fists_train, fists_test = train_test_split(
            fists_trials, test_size=test_size, random_state=random_state
        )
        feet_train, feet_test = train_test_split(
            feet_trials, test_size=test_size, random_state=random_state
        )
        
        # Create index lists
        train_indices = []
        test_indices = []
        
        for i, meta in enumerate(self.segment_metadata):
            if meta['task'] == 'fists':
                if meta['original_trial'] in fists_train:
                    train_indices.append(i)
                else:
                    test_indices.append(i)
            else:
                if meta['original_trial'] in feet_train:
                    train_indices.append(i)
                else:
                    test_indices.append(i)
        
        return train_indices, test_indices
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(self.segment_labels)
        weights = compute_class_weight('balanced', classes=classes, y=self.segment_labels)
        
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        label = torch.tensor(self.segment_labels[idx], dtype=torch.long)  # Long for multi-class
        return segment, label
        

class TwoStageEEGDataset(Dataset):
    """
    Dataset for two-stage classification:
    Stage 1: Rest (0) vs Motor Imagery (1)
    Stage 2: Fists (0) vs Feet (1) - only for motor imagery samples
    """
    
    def __init__(self, data_dir, run_number, stage='stage1', window_size=1025, 
                 overlap=0.5, debug=True, normalize=True, normalization_params=None):
        """
        Args:
            stage: 'stage1' for rest vs motor imagery, 'stage2' for fists vs feet
        """
        self.stage = stage
        self.window_size = window_size
        self.overlap = overlap
        self.normalize = normalize
        self.segments = []
        self.segment_labels = []
        self.segment_metadata = []
        self.original_labels = []  # Store original 3-class labels
        
        if isinstance(run_number, int):
            run_number = [run_number]
        
        # Store normalization parameters
        self.normalization_params = normalization_params
        self.computed_norm_params = {}
        
        # Collect all data
        all_data = {'fists': {'tp9': [], 'tp10': [], 'labels': []},
                   'feet': {'tp9': [], 'tp10': [], 'labels': []}}
        
        # Define experiments
        experiments = [
            {'exp_num': 1, 'task_short': 'fists', 'task_long': 'openclosefists'},
            {'exp_num': 2, 'task_short': 'feet', 'task_long': 'openclosefeet'}
        ]
        
        # Load data from both experiments
        for exp_info in experiments:
            exp_num = exp_info['exp_num']
            task_short = exp_info['task_short']
            task_long = exp_info['task_long']
            
            if debug:
                print(f"\n=== Loading {task_short.upper()} data (exp_{exp_num}) ===")
            
            for run in run_number:
                exp_dir = os.path.join(data_dir, f"exp_{exp_num}")
                tp9_file = f"{task_long}_run{run}_TP9.csv"
                tp10_file = f"{task_long}_run{run}_TP10.csv"
                label_file = f"{task_long}_run{run}_label.csv"
                
                tp9_data = pd.read_csv(os.path.join(exp_dir, tp9_file), header=None).values
                tp10_data = pd.read_csv(os.path.join(exp_dir, tp10_file), header=None).values
                labels = pd.read_csv(os.path.join(exp_dir, label_file), header=None).values.flatten()
                
                all_data[task_short]['tp9'].append(tp9_data)
                all_data[task_short]['tp10'].append(tp10_data)
                all_data[task_short]['labels'].append(labels)
                
                if debug:
                    print(f"  Run {run}: {len(labels)} trials loaded")
        
        # Concatenate all data
        for task in ['fists', 'feet']:
            if len(all_data[task]['tp9']) > 0:
                all_data[task]['tp9'] = np.vstack(all_data[task]['tp9'])
                all_data[task]['tp10'] = np.vstack(all_data[task]['tp10'])
                all_data[task]['labels'] = np.hstack(all_data[task]['labels'])
        
        # Compute normalization parameters if needed
        if self.normalize and self.normalization_params is None:
            self._compute_normalization_params(all_data, debug)
        elif self.normalize and self.normalization_params is not None:
            self.computed_norm_params = self.normalization_params
            if debug:
                print("\nUsing provided normalization parameters")
        
        # Process data based on stage
        if self.stage == 'stage1':
            self._process_stage1_data(all_data, debug)
        else:  # stage2
            self._process_stage2_data(all_data, debug)
        
        self.segments = np.array(self.segments)
        self.segment_labels = np.array(self.segment_labels)
        self.original_labels = np.array(self.original_labels)
        
        if debug:
            self._print_dataset_stats()
    
    def _compute_normalization_params(self, all_data, debug):
        """Compute normalization parameters from all data"""
        if debug:
            print("\n=== Computing normalization parameters ===")
        
        # Combine all data for global normalization
        all_tp9 = np.vstack([all_data['fists']['tp9'], all_data['feet']['tp9']])
        all_tp10 = np.vstack([all_data['fists']['tp10'], all_data['feet']['tp10']])
        
        self.computed_norm_params = {
            'tp9_mean': np.mean(all_tp9),
            'tp9_std': np.std(all_tp9),
            'tp10_mean': np.mean(all_tp10),
            'tp10_std': np.std(all_tp10)
        }
        
        if debug:
            print(f"  TP9 - mean: {self.computed_norm_params['tp9_mean']:.6f}, "
                  f"std: {self.computed_norm_params['tp9_std']:.6f}")
            print(f"  TP10 - mean: {self.computed_norm_params['tp10_mean']:.6f}, "
                  f"std: {self.computed_norm_params['tp10_std']:.6f}")
    
    def _normalize_data(self, tp9_data, tp10_data):
        """Apply normalization to data"""
        if self.computed_norm_params is None:
            return tp9_data, tp10_data
        
        tp9_normalized = (tp9_data - self.computed_norm_params['tp9_mean']) / self.computed_norm_params['tp9_std']
        tp10_normalized = (tp10_data - self.computed_norm_params['tp10_mean']) / self.computed_norm_params['tp10_std']
        
        return tp9_normalized, tp10_normalized
    
    def _process_stage1_data(self, all_data, debug):
        """Process data for Stage 1: Rest (0) vs Motor Imagery (1)"""
        if debug:
            print(f"\n=== Processing Stage 1 Data (Rest vs Motor Imagery) ===")
        
        for task_idx, task in enumerate(['fists', 'feet']):
            tp9_data = all_data[task]['tp9']
            tp10_data = all_data[task]['tp10']
            labels = all_data[task]['labels']
            
            # Apply normalization if enabled
            if self.normalize:
                tp9_data, tp10_data = self._normalize_data(tp9_data, tp10_data)
            
            # Create segments
            for i, (tp9_sample, tp10_sample, orig_label) in enumerate(zip(tp9_data, tp10_data, labels)):
                # For stage 1: 0 = rest, 1 = motor imagery (both fists and feet)
                stage1_label = 0 if orig_label == 0 else 1
                
                # Store original 3-class label for reference
                if orig_label == 0:
                    original_3class = 0  # Rest
                else:
                    original_3class = task_idx + 1  # 1 for fists, 2 for feet
                
                data_length = len(tp9_sample)
                
                if self.window_size == data_length:
                    segment = np.stack([tp9_sample, tp10_sample], axis=0)
                    self.segments.append(segment)
                    self.segment_labels.append(stage1_label)
                    self.original_labels.append(original_3class)
                    self.segment_metadata.append({
                        'original_trial': i,
                        'window_start': 0,
                        'window_end': data_length,
                        'task': task,
                        'original_binary_label': orig_label
                    })
                else:
                    step = int(self.window_size * (1 - self.overlap))
                    for start in range(0, data_length - self.window_size + 1, step):
                        end = start + self.window_size
                        segment = np.stack([
                            tp9_sample[start:end],
                            tp10_sample[start:end]
                        ], axis=0)
                        self.segments.append(segment)
                        self.segment_labels.append(stage1_label)
                        self.original_labels.append(original_3class)
                        self.segment_metadata.append({
                            'original_trial': i,
                            'window_start': start,
                            'window_end': end,
                            'task': task,
                            'original_binary_label': orig_label
                        })
    
    def _process_stage2_data(self, all_data, debug):
        """Process data for Stage 2: Fists (0) vs Feet (1) - only motor imagery samples"""
        if debug:
            print(f"\n=== Processing Stage 2 Data (Fists vs Feet) ===")
        
        for task_idx, task in enumerate(['fists', 'feet']):
            tp9_data = all_data[task]['tp9']
            tp10_data = all_data[task]['tp10']
            labels = all_data[task]['labels']
            
            # Apply normalization if enabled
            if self.normalize:
                tp9_data, tp10_data = self._normalize_data(tp9_data, tp10_data)
            
            # Create segments - ONLY for motor imagery samples
            for i, (tp9_sample, tp10_sample, orig_label) in enumerate(zip(tp9_data, tp10_data, labels)):
                # Skip resting samples for stage 2
                if orig_label == 0:
                    continue
                
                # For stage 2: 0 = fists, 1 = feet
                stage2_label = task_idx  # 0 for fists, 1 for feet
                
                # Original 3-class label
                original_3class = task_idx + 1  # 1 for fists, 2 for feet
                
                data_length = len(tp9_sample)
                
                if self.window_size == data_length:
                    segment = np.stack([tp9_sample, tp10_sample], axis=0)
                    self.segments.append(segment)
                    self.segment_labels.append(stage2_label)
                    self.original_labels.append(original_3class)
                    self.segment_metadata.append({
                        'original_trial': i,
                        'window_start': 0,
                        'window_end': data_length,
                        'task': task,
                        'original_binary_label': orig_label
                    })
                else:
                    step = int(self.window_size * (1 - self.overlap))
                    for start in range(0, data_length - self.window_size + 1, step):
                        end = start + self.window_size
                        segment = np.stack([
                            tp9_sample[start:end],
                            tp10_sample[start:end]
                        ], axis=0)
                        self.segments.append(segment)
                        self.segment_labels.append(stage2_label)
                        self.original_labels.append(original_3class)
                        self.segment_metadata.append({
                            'original_trial': i,
                            'window_start': start,
                            'window_end': end,
                            'task': task,
                            'original_binary_label': orig_label
                        })
    
    def _print_dataset_stats(self):
        """Print dataset statistics"""
        print(f"\n=== {self.stage.upper()} DATASET STATISTICS ===")
        print(f"Total segments: {len(self.segments)}")
        print(f"Segment shape: {self.segments.shape}")
        
        # Class distribution
        print(f"\nClass distribution:")
        unique_labels, counts = np.unique(self.segment_labels, return_counts=True)
        
        if self.stage == 'stage1':
            class_names = ['Rest', 'Motor Imagery']
        else:
            class_names = ['Fists', 'Feet']
        
        for label, count in zip(unique_labels, counts):
            percentage = count / len(self.segment_labels) * 100
            print(f"  Class {label} ({class_names[int(label)]}): {count} samples ({percentage:.1f}%)")
        
        if self.normalize:
            print(f"\nNormalized data range: [{np.min(self.segments):.3f}, {np.max(self.segments):.3f}]")
    
    def create_stratified_split(self, test_size=0.2, random_state=42):
        """Create stratified train/test split"""
        from sklearn.model_selection import train_test_split
        
        # Get unique trials per class
        class_trials = {}
        
        for i, meta in enumerate(self.segment_metadata):
            label = self.segment_labels[i]
            if label not in class_trials:
                class_trials[label] = set()
            class_trials[label].add((meta['task'], meta['original_trial']))
        
        # Split trials for each class
        train_indices = []
        test_indices = []
        
        for label, trials in class_trials.items():
            trials_list = list(trials)
            if len(trials_list) > 1:
                train_trials, test_trials = train_test_split(
                    trials_list, test_size=test_size, random_state=random_state
                )
                test_trials_set = set(test_trials)
                
                for i, meta in enumerate(self.segment_metadata):
                    if self.segment_labels[i] == label:
                        trial_key = (meta['task'], meta['original_trial'])
                        if trial_key in test_trials_set:
                            test_indices.append(i)
                        else:
                            train_indices.append(i)
        
        return train_indices, test_indices
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(self.segment_labels)
        weights = compute_class_weight('balanced', classes=classes, y=self.segment_labels)
        
        return torch.FloatTensor(weights)
    
    def get_normalization_params(self):
        """Return the normalization parameters used"""
        return self.computed_norm_params
    
    def save_normalization_params(self, filepath):
        """Save normalization parameters to file"""
        if self.computed_norm_params is not None:
            with open(filepath, 'w') as f:
                json.dump(self.computed_norm_params, f, indent=2)
            print(f"Saved normalization parameters to {filepath}")
    
    @staticmethod
    def load_normalization_params(filepath):
        """Load normalization parameters from file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        label = torch.tensor(self.segment_labels[idx], dtype=torch.long)
        return segment, label
