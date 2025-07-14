import os 
# import mne #only for filtering
import numpy as np
from mne import Epochs, find_events
from matplotlib import pyplot as plt
from model_utils import model
import utils
import re
from sklearn.model_selection import train_test_split

#based on main.py in the decoder
def remove_outliers_with_mean(data, threshold):
    """
    Quality control of outliers, remove abnormally big data
    """
    data_clean = data.copy()
    replaced_count = 0

    n_epochs, n_channels, n_times = data_clean.shape
    for i in range(n_epochs):
        for c in range(n_channels):
            channel_data = data_clean[i, c, :]
            outliers = np.abs(channel_data) > threshold
            inliers = ~outliers
            if np.any(inliers):
                mean_inliers = channel_data[inliers].mean()
            replaced_count += np.sum(outliers)

            channel_data[outliers] = mean_inliers

    print(f"Total replaced points: {replaced_count}")
    return data_clean

def bandpass_filter_ndarray(data: np.ndarray, sfreq:float=256, l_freq:float=1., h_freq:float=30.,fir_design:str= 'firwin'):
    """
    Convert a np ndarray first to raw then call the bandpassfilter
    """
    n_channels, n_times = data.shape
    ch_names = [f'ch{i}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design=fir_design)
    data_filt = raw.get_data()
    return data_filt

def decoder_predict(X_input:np.ndarray, trained_model: model, qc:bool=False, threshold:float=100*1e-6, filter:bool=False):
    """
    Predict using real-time data.
    X_input shape (2, 512) where 2 is the num_channels and 512 is the time.
    """
    # perform qc and filter if requested
    if X_input.ndim !=2:
        raise ValueError("X_input must be 2D: (2, 512).")

    # if qc:
    #     X_input = remove_outliers_with_mean(X_input, threshold=threshold)
    # if filter:
    #     X_input = bandpass_filter_ndarray(X_input)

    # expand 1 dimension to use sklearn structure
    X_input = X_input[np.newaxis, ...]
    print(X_input.shape)
    prob = trained_model.best_model_.predict_proba(X_input)[0] # （sampleSize, n_classes), get the first sample, i.e., [0]
    prob_class1 = prob[1] # prob[0]: class0, prob[0]: class1, select the probability for class 1
    pred_label = trained_model.best_model_.predict(X_input)[0]
    return prob_class1, pred_label


if __name__ == "__main__":
    import pickle
    # sample usage: decoder_predict
    #model_file = "my_model_exp_2_sub_1.pkl" #for eye blink model
    model_file = "openCloseFistsFeet_model_exp_1_sub_1.pkl" #open/close fists task
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # example_input = all_X[0][0][3]
    # y_prob, y_pred = decoder_predict(X_input=example_input, trained_model=loaded_model)
    # print(f'prob for target class_1 is {y_prob}, predicted_label is {y_pred}')

    #get segment_4s from either .csv file or real-time
    """Run one 4 s segment (1024×4) through decoder_predict."""
    X = segment_4s[:, [0, 3]].T  # TP9 & TP10
    y_prob, y_pred =decoder_predict(X_input=X, trained_model=loaded_model)
    print(f'prob for target class_1 is {y_prob}, predicted_label is {y_pred}')


    
   



    



 



    
