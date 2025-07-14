#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Neurofeedback PsychoPy Experiment

This experiment displays:
  - A welcome page (press Space to continue, or Esc to quit)
  - Then N trials, each with:
      1. Task Page (6 s): display task instructions; send marker 101 at start, 102 at end.
      2. Black Fixation Cross (6 s).
      3. Compute avg prediction over last 6 s of EEG (three 2 s segments).
      4. Result Page (2 s): show central **green** feedback circle with outer max‐size ring; send marker 101 if label>0 else 102.
      5. Black Fixation Cross (6 s).
  - An end page (press Space to exit, or Esc to quit)

Pressing Esc at any point sends a 999 marker and exits immediately.
"""

from psychopy import visual, core, event, gui
import pandas as pd
import numpy as np
import time
import random
from pylsl import StreamInfo, StreamOutlet
from multiprocessing import shared_memory
import pickle
import os

# add these two for the tkinter popup
import tkinter as tk
from tkinter import messagebox

# ----- LSL Marker Setup -----
info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')
outlet = StreamOutlet(info)

# choose model variant
ifUseTemplateBasedModel = 0  # regular
#ifUseTemplateBasedModel = 1  # template matching
if ifUseTemplateBasedModel > 0:
    from DecNef_NF_predict_template import decoder_predict
else:
    from DecNef_NF_predict import decoder_predict

# ----- Configuration Variables -----
eeg_csv_path = r'D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\decoder\EB-Data\sub-EB-34\Muse_data\sub-EB-34_EEG_recording.csv'
n_rows_for_prediction     = 256 * 2
model_path_template       = r"D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\decoder\dtw_model_exp_2_sub_1.pkl"
model_path_plain          = r"D:\Faculty\ColumbiaUniversity\dataprocess\EEG\dataProcessing\Interaxon\museS\LSL\Python\muse-lsl-python\decoder\my_model_exp_2_sub_1.pkl"
model_path                = model_path_template if ifUseTemplateBasedModel > 0 else model_path_plain

eye_blinking_count = 0

# Trial timings
n_trials                 = 2  # testing
beforeTrial_cross_duration = 6
task_duration            = 6.0
cross_duration           = 6.0
result_duration          = 2.0
intertrial_interval      = 6.0

# Circle display parameters (fractions of screen height)
min_radius = 0.05
max_radius = 0.3  # maximum green circle

# ----- Shared Memory Setup -----
try:
    shm_eeg    = shared_memory.SharedMemory(name='eeg_data')
    eeg_shared = np.ndarray((12 * 256, 5), dtype=np.float32, buffer=shm_eeg.buf)
    shm_lock   = shared_memory.SharedMemory(name='eeg_lock')
    lock_flag  = np.ndarray((1,), dtype=np.int8, buffer=shm_lock.buf)
except Exception as e:
    print("Shared memory unavailable, falling back to CSV:", e)
    shm_eeg = shm_lock = None

# ----- Helper Functions -----
def safe_quit(win):
    """Send marker 999, then close window and quit."""
    try:
        outlet.push_sample([999], time.time())
    except:
        pass
    win.close()
    core.quit()

def load_model():
    """Load and return the pickled model (or None on fail)."""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print("Model load error:", e)
        return None

def predict_segment(segment_2s, model):
    """Run one 2 s segment (512×4) through decoder_predict."""
    X = segment_2s[:, [0, 3]].T  # TP9 & TP10
    return decoder_predict(X_input=X, trained_model=model)

def show_black_cross(win, duration):
    """Draw a large black '+' for `duration` seconds, allow ESC to quit."""
    cross = visual.TextStim(win, text='+', color='black', height=0.1, bold=True)
    t0 = core.getTime()
    while core.getTime() - t0 < duration:
        if 'escape' in event.getKeys():
            safe_quit(win)
        cross.draw()
        win.flip()
        core.wait(0.01)

def compute_avg_prediction():
    """
    Take the last 6 s from shared memory, split into 3×2 s segments,
    predict on each, and return (prob_avg, label_avg).
    """
    if shm_eeg is None or shm_lock is None:
        # fallback to single-shot CSV prediction
        if not os.path.exists(eeg_csv_path):
            return 0.0, 0
        df = pd.read_csv(eeg_csv_path)
        if len(df) < n_rows_for_prediction:
            return 0.0, 0
        seg   = df.tail(n_rows_for_prediction)[['TP9','TP10']].to_numpy().T
        model = load_model()
        return decoder_predict(X_input=seg, trained_model=model)

    # wait if updating
    while lock_flag[0] == 1:
        core.wait(0.001)
    data  = eeg_shared.copy()            # shape (3072,5)
    eeg6  = data[-int(6*256):, :4]       # last 1536×4
    model = load_model()
    preds = []
    for i in range(3):
        segment = eeg6[i*512:(i+1)*512, :]
        p, _ = predict_segment(segment, model)
        preds.append(p)
    prob_avg  = float(np.mean(preds))
    label_avg = int(prob_avg > 0.5)
    return prob_avg, label_avg

def show_result(win, prob, label, duration):
    """Display **green** feedback circle with outer ring; send appropriate marker."""
    # outer ring
    outer = visual.Circle(
        win,
        units='height',
        radius=max_radius,
        edges=128,
        lineColor='green',
        lineWidth=2,
        fillColor=None
    )
    # feedback circle always green
    radius = min_radius + (max_radius - min_radius) * prob
    circle = visual.Circle(
        win,
        units='height',
        radius=radius,
        fillColor='green',
        lineColor='green',
        edges=128,
        pos=(0, 0)
    )

    t0 = core.getTime()
    while core.getTime() - t0 < duration:
        if 'escape' in event.getKeys():
            safe_quit(win)
        outer.draw()
        circle.draw()
        win.flip()
        core.wait(0.01)

# ------ Main Experiment Function ------
def trial_experiment():
    global eye_blinking_count

    # ---- replace PsychoPy Dlg with tkinter popup ----
    root = tk.Tk()
    root.withdraw()
    ok = messagebox.askokcancel(
        "Neurofeedback Experiment",
        "Click OK to proceed to the experiment."
    )
    root.destroy()
    if not ok:
        core.quit()

    win = visual.Window(fullscr=True, color='grey', units='norm')

    # --- Welcome Page ---
    welcome = visual.TextStim(
        win,
        text="Welcome to the experiment.\n\nPress Space to continue (Esc to quit).",
        color='black',
        height=0.08
    )
    welcome.draw()
    win.flip()
    while True:
        k = event.waitKeys(keyList=['space', 'escape'])
        if 'escape' in k:
            safe_quit(win)
        if 'space' in k:
            break

    # show cross before trial 
    show_black_cross(win, beforeTrial_cross_duration)

    # --- N trials with 5 Stages ---
    for trial in range(n_trials):
        # 1) Task Page
        outlet.push_sample([101], time.time())
        task = visual.TextStim(win, text="Task: please do the task.", color='black', height=0.08)
        task.draw()
        win.flip()
        core.wait(task_duration)
        outlet.push_sample([102], time.time())
        if 'escape' in event.getKeys():
            safe_quit(win)

        # 2) Black Fixation Cross
        show_black_cross(win, cross_duration)

        # 3) Compute avg prediction
        prob, label = compute_avg_prediction()

        # 4) Result Page
        show_result(win, prob, label, result_duration)
        if label > 0:
            eye_blinking_count += 1

        # 5) Intertrial Black Fixation Cross
        show_black_cross(win, intertrial_interval)

    # --- End Page ---
    end = visual.TextStim(
        win,
        text=f"Total eye blinking count: {eye_blinking_count}\n\n"
             "Press Space to exit (Esc to quit).",
        color='black',
        height=0.08
    )
    end.draw()
    win.flip()
    while True:
        k = event.waitKeys(keyList=['space', 'escape'])
        if 'escape' in k:
            safe_quit(win)
        if 'space' in k:
            break

    outlet.push_sample([999], time.time())
    win.close()
    core.quit()

# ------ trial with Error Screening ------
if __name__ == '__main__':
    try:
        trial_experiment()
    except Exception as e:
        # ensure we send the final marker if possible
        try:
            outlet.push_sample([999], time.time())
        except:
            pass
        import traceback
        traceback.print_exc()
        gui.Dlg(title="Experiment Error", labelButtonOK='OK') \
           .addText(f"{e}") \
           .show()
        core.quit()
