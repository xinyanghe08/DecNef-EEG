import torch
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from dataset import TwoChannelEEGDataset, EEGNetDataset

class CSVWindowProcessor:
    """
    Processor that saves windows as CSV files and uses TwoChannelEEGDataset
    """
    
    def __init__(self, normalization_params=None, temp_dir=None):
        """
        Initialize processor
        
        Args:
            normalization_params (dict): Normalization parameters from training
            temp_dir (str): Directory for temporary CSV files (if None, uses system temp)
        """
        self.normalization_params = normalization_params
        
        # Create temporary directory for CSV files
        if temp_dir:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="eeg_windows_")
        
        print(f"Using temporary directory: {self.temp_dir}")
    
    def save_window_as_csv(self, window_data, window_id=0):
        """
        Save a single window as CSV files in the format expected by TwoChannelEEGDataset
        
        Args:
            window_data (np.ndarray): Window data shape (n_channels, window_size)
            window_id (int): Unique identifier for this window
            
        Returns:
            str: Path to the temporary experiment directory
        """
        # Extract TP9 and TP10 if 4 channels provided
        if window_data.shape[0] == 4:
            # Assume standard Muse order: [TP9, AF7, AF8, TP10]
            tp9_data = window_data[0, :]
            tp10_data = window_data[3, :]
        elif window_data.shape[0] == 2:
            # Already TP9, TP10
            tp9_data = window_data[0, :]
            tp10_data = window_data[1, :]
        else:
            raise ValueError(f"Expected 2 or 4 channels, got {window_data.shape[0]}")
        
        # Create experiment directory structure (like your training data)
        exp_dir = os.path.join(self.temp_dir, f"window_{window_id}", "exp_1")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Save TP9 and TP10 data as CSV files (single trial each)
        tp9_path = os.path.join(exp_dir, "openclosefists_run1_TP9.csv")
        tp10_path = os.path.join(exp_dir, "openclosefists_run1_TP10.csv")
        label_path = os.path.join(exp_dir, "openclosefists_run1_label.csv")
        
        # Save as single-row DataFrames (1 trial)
        pd.DataFrame([tp9_data]).to_csv(tp9_path, index=False, header=False)
        pd.DataFrame([tp10_data]).to_csv(tp10_path, index=False, header=False)
        
        # Create dummy label (doesn't matter for prediction, just needs to exist)
        pd.DataFrame([0]).to_csv(label_path, index=False, header=False)
        
        return os.path.join(self.temp_dir, f"window_{window_id}")
    
    def process_window_with_dataset(self, window_data, window_id=0):
        """
        Process window using TwoChannelEEGDataset pipeline
        
        Args:
            window_data (np.ndarray): Window data shape (n_channels, window_size)
            window_id (int): Unique identifier for this window
            
        Returns:
            torch.Tensor: Processed tensor ready for EEGNet, shape (1, 2, window_size)
        """
        # Save window as CSV files
        data_dir = self.save_window_as_csv(window_data, window_id)
        
        # Create TwoChannelEEGDataset
        dataset = TwoChannelEEGDataset(
            data_dir=data_dir,
            run_number=1,
            exp_number=1,
            window_size=window_data.shape[1],
            overlap=0.0,
            task="openclosefists",
            debug=False,
            normalize=False,
            normalization_params=self.normalization_params
        )
        
        # Wrap with EEGNetDataset
        eegnet_dataset = EEGNetDataset(dataset)
        
        # Get the processed data (should be only 1 sample)
        if len(eegnet_dataset) != 1:
            raise ValueError(f"Expected 1 sample, got {len(eegnet_dataset)}")
        
        processed_tensor, _ = eegnet_dataset[0]  # Get (tensor, label)
        
        return processed_tensor
    
    def cleanup(self):
        """Remove temporary directory and files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()


class RealTimeEEGAnalyzerCSV:
    """
    Real-time analyzer that uses CSV files and TwoChannelEEGDataset
    """
    
    def __init__(self, model_path, experiment_type='fists', device=None, 
                 window_size=1024, normalization_params=None, temp_dir=None):
        """
        Initialize analyzer
        
        Args:
            model_path (str): Path to trained model
            experiment_type (str): 'fists' or 'feet'
            device: torch device
            window_size (int): Window size in samples
            normalization_params (dict): Normalization parameters from training
            temp_dir (str): Directory for temporary files
        """
        self.experiment_type = experiment_type
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.sfreq = 256
        self.n_channels = 2
        self.segment_duration = 4.0
        
        # Initialize CSV processor
        self.csv_processor = CSVWindowProcessor(
            normalization_params=normalization_params,
            temp_dir=temp_dir
        )
        
        # Load model
        try:
            from model import create_eegnet_model
            self.model = create_eegnet_model(task_type='binary', num_classes=1, samples=self.window_size)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully for {experiment_type} classification")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        print(f"Using device: {self.device}")
        print(f"Window size: {self.window_size} samples ({self.segment_duration} seconds)")
        print(f"Using TwoChannelEEGDataset preprocessing pipeline")
    
    def extract_4sec_windows(self, signal, overlap=0.0):
        """Extract 4-second windows from signal"""
        n_channels, n_samples = signal.shape
        
        if n_samples < self.window_size:
            raise ValueError(f"Signal too short: {n_samples} samples, need at least {self.window_size}")
        
        windows = []
        step_size = int(self.window_size * (1 - overlap))
        start_idx = 0
        
        while start_idx + self.window_size <= n_samples:
            end_idx = start_idx + self.window_size
            window = signal[:, start_idx:end_idx]
            windows.append(window)
            start_idx += step_size
        
        return windows
    
    def predict_single_window(self, window_data, window_id=None):
        """
        Predict on a single window using TwoChannelEEGDataset pipeline
        
        Args:
            window_data (np.ndarray): Window data shape (n_channels, window_size)
            window_id (int): Unique ID for this window (for temp file naming)
        
        Returns:
            tuple: (prediction, probability, label)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if window_data.shape[1] != self.window_size:
            raise ValueError(f"Window size must be {self.window_size}")
        
        # Use unique window ID if not provided
        if window_id is None:
            window_id = np.random.randint(0, 1000000)
        
        print(f"Processing window {window_id}")
        print(f"Input window shape: {window_data.shape}")
        print(f"Input range: [{window_data.min():.6f}, {window_data.max():.6f}]")
        
        # Process window using TwoChannelEEGDataset
        try:
            processed_tensor = self.csv_processor.process_window_with_dataset(window_data, window_id)
            
            print(f"Processed tensor shape: {processed_tensor.shape}")
            print(f"Processed range: [{processed_tensor.min():.6f}, {processed_tensor.max():.6f}]")
            
            # Add batch dimension and move to device
            input_tensor = processed_tensor.unsqueeze(0).to(self.device)  # Shape: (1, 1, 2, window_size)
            
            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)
                print(f"Raw model output: {output.cpu().numpy()}")
                probability = torch.sigmoid(output).cpu().numpy()[0, 0]
                prediction = (probability > 0.0001).astype(int)
                print(f"Probability after sigmoid: {probability}")
            
            # Determine label
            if prediction == 1:
                label = f"Open/Close {self.experiment_type}"
            else:
                label = "Resting"
            
            return prediction, probability, label
            
        except Exception as e:
            print(f"Error processing window {window_id}: {e}")
            raise
    
    def analyze_eeg_data(self, signal, overlap=0.0):
        """
        Analyze EEG data using TwoChannelEEGDataset preprocessing
        
        Args:
            signal (np.ndarray): EEG signal shape (n_channels, n_samples)
            overlap (float): Overlap between windows
        
        Returns:
            dict: Analysis results
        """
        # Extract windows
        windows = self.extract_4sec_windows(signal, overlap)
        
        if len(windows) == 0:
            raise ValueError("No complete windows could be extracted")
        
        results = {
            'total_windows': len(windows),
            'window_results': [],
            'summary': {},
            'signal_info': {
                'total_samples': signal.shape[1],
                'duration_seconds': signal.shape[1] / self.sfreq,
                'channels': signal.shape[0],
                'overlap_used': overlap
            }
        }
        
        predictions = []
        probabilities = []
        labels = []
        
        # Analyze each window
        for i, window in enumerate(windows):
            pred, prob, label = self.predict_single_window(window, window_id=i)
            
            start_time = i * self.window_size * (1 - overlap) / self.sfreq
            end_time = start_time + self.segment_duration
            
            window_result = {
                'window_number': i + 1,
                'time_range': f"{start_time:.1f}-{end_time:.1f}s",
                'prediction': pred,
                'probability': prob,
                'label': label
            }
            
            results['window_results'].append(window_result)
            predictions.append(pred)
            probabilities.append(prob)
            labels.append(label)
            
            print(f"Window {i+1} ({start_time:.1f}-{end_time:.1f}s): {label} (prob: {prob:.3f})")

        # Calculate summary
        positive_predictions = sum(predictions)
        avg_probability = np.mean(probabilities)
        consensus_prediction = 1 if positive_predictions > len(predictions) / 2 else 0
        consensus_label = f"Open/Close {self.experiment_type}" if consensus_prediction == 1 else "Resting"
        
        results['summary'] = {
            'positive_windows': positive_predictions,
            'total_windows': len(predictions),
            'positive_rate': positive_predictions / len(predictions),
            'average_probability': avg_probability,
            'consensus_prediction': consensus_prediction,
            'consensus_label': consensus_label
        }
        
        return results
    
    def analyze_csv_file(self, csv_path, channels=['TP9', 'AF7', 'AF8', 'TP10'], overlap=0.0):
        """
        Analyze CSV file using TwoChannelEEGDataset preprocessing
        
        Args:
            csv_path (str): Path to CSV file
            channels (list): Channel names
            overlap (float): Window overlap
        
        Returns:
            dict: Analysis results
        """
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Extract channels
        if 'TP9' in df.columns and 'TP10' in df.columns:
            eeg_data = df[['TP9', 'TP10']].T.values
            channels_used = ['TP9', 'TP10']
        elif len(channels) >= 2:
            available_channels = [ch for ch in channels if ch in df.columns]
            if len(available_channels) >= 2:
                eeg_data = df[available_channels[:2]].T.values
                channels_used = available_channels[:2]
            else:
                raise ValueError(f"Not enough channels available. Found: {available_channels}")
        else:
            raise ValueError("Cannot find required channels in CSV")
        
        total_duration = eeg_data.shape[1] / self.sfreq
        print(f"Loaded CSV: {eeg_data.shape[1]} samples ({total_duration:.1f} seconds)")
        
        # Analyze using TwoChannelEEGDataset preprocessing
        results = self.analyze_eeg_data(eeg_data, overlap=overlap)
        
        # Add file info
        results['file_info'] = {
            'csv_path': csv_path,
            'channels_used': channels_used,
            'duration_seconds': total_duration,
            'sampling_rate': self.sfreq
        }
        
        return results
    
    def cleanup(self):
        """Clean up temporary files"""
        self.csv_processor.cleanup()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()


def create_csv_analyzer(experiment_type='fists', model_base_path=r"C:\Users\xhe\Documents\GitHub\DecNef-EEG\decoder",
                       normalization_params=None, window_size=1024, temp_dir=None):
    """
    Create analyzer that uses CSV files and TwoChannelEEGDataset
    
    Args:
        experiment_type (str): 'fists' or 'feet'
        model_base_path (str): Path to model files
        normalization_params (dict): Normalization parameters from training
        window_size (int): Window size in samples
        temp_dir (str): Directory for temporary files
    
    Returns:
        RealTimeEEGAnalyzerCSV: Configured analyzer
    """
    #model_path = f"{model_base_path}best_{experiment_type}_eegnet_model.pth"
    model_path = r"C:\Users\xhe\Documents\GitHub\DecNef-EEG\decoder\best_fists_eegnet_model.pth"
    return RealTimeEEGAnalyzerCSV(
        model_path=model_path,
        experiment_type=experiment_type,
        window_size=window_size,
        normalization_params=normalization_params,
        temp_dir=temp_dir
    )


# Example usage
def example_csv_analysis():
    """Example using CSV-based analyzer"""
    normalization_params = TwoChannelEEGDataset.load_normalization_params(r'C:\Users\xhe\Documents\GitHub\DecNef-EEG\decoder\normalization.json')
    # Create CSV-based analyzer
    analyzer = create_csv_analyzer(
        experiment_type='fists',
        normalization_params=normalization_params,
        window_size=1024,  
        temp_dir="./temp_eeg_windows"  # Specify temp directory
    )
    
    # Test CSV analysis
    csv_path = r"C:\Users\xhe\Documents\GitHub\DecNef-EEG\decoder\data\sub-EB-43_EEG_recording_2025-06-13-00.22.42.csv"
    
    try:
        results = analyzer.analyze_csv_file(csv_path, overlap=0.0)
        
        print("\n=== CSV-Based Analysis Results ===")
        print(f"Duration: {results['file_info']['duration_seconds']:.1f} seconds")
        print(f"Windows: {results['total_windows']}")
        print(f"Open/Close Fists: {results['summary']['positive_windows']}")
        print(f"Rest: {np.subtract(results['summary']['total_windows'], results['summary']['positive_windows'])}")
        print(f"Positive rate: {results['summary']['positive_rate']:.3f}")
        
        # Show first few windows
        print("\nFirst 5 windows:")
        for window_result in results['window_results'][:5]:
            print(f"  {window_result['time_range']}: {window_result['label']} (prob: {window_result['probability']:.3f})")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        # Clean up temporary files
        analyzer.cleanup()


if __name__ == "__main__":
    print("Testing CSV-based real-time analyzer...")
    example_csv_analysis()
