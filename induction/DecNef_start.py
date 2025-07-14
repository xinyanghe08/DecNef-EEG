import muselsl
import time
import datetime
import numpy as np
import subprocess
import threading
import os
import psutil
import muselsl.winctrlc
import pandas as pd
from pathlib import Path
from time import strftime, gmtime
from multiprocessing import shared_memory

# %% Shared Memory Setup for Latest 12 Seconds of EEG Data (4 Channels + Timestamp)
eeg_rows = int(12 * 256)   # 3072 rows
eeg_cols = 5               # 4 EEG channels + 1 timestamp column
dtype = np.float32

# Create shared memory for EEG (only EEG channels + timestamp)
shm_eeg = shared_memory.SharedMemory(
    create=True,
    size=np.zeros((eeg_rows, eeg_cols), dtype=dtype).nbytes,
    name='eeg_data'
)
eeg_shared = np.ndarray((eeg_rows, eeg_cols), dtype=dtype, buffer=shm_eeg.buf)
eeg_shared[:] = 0  # init to zeros

# Simple lock flag (0=not updating, 1=updating)
shm_lock = shared_memory.SharedMemory(
    create=True,
    size=np.zeros((1,), dtype=np.int8).nbytes,
    name='eeg_lock'
)
lock_flag = np.ndarray((1,), dtype=np.int8, buffer=shm_lock.buf)
lock_flag[0] = 0

# %% Existing Setup Code
t_init = int(datetime.datetime.now().timestamp())
participant = 'EB-41'
name = 'MuseS-6B97'
cwd = os.getcwd()

# Path to your venv activate script
venv_activate = os.path.join(cwd, 'venv', 'Scripts', 'activate.bat')
# We'll invoke like: cmd /k "<venv_activate> & python DecNef_museS.py -p ..."

data_folder = os.path.join(cwd, '..', 'Data')
os.makedirs(data_folder, exist_ok=True)
data_path = os.path.join(data_folder, f"sub-{participant}", "Muse_data_DecNef")
os.makedirs(data_path, exist_ok=True)

# LSL stream/record commands
stream_cmd = f"muselsl stream -n {name} -p -c -g"
record_base = f'muselsl record -s {t_init} -p {participant} -d "{data_path}" -t'

filename = os.path.join(data_path, f"sub-{participant}_EEG_recording.csv")

# -------------------------------------------------------------------------
# 1) Launch the PsychoPy experiment and capture its stdout/stderr
# -------------------------------------------------------------------------

# Build our command: activate venv, then run experiment script
exp_cmd = (
    f'cmd /k "{venv_activate} & '
    f'python DecNef_museS.py -p {participant} -s {t_init}"'
)

# Start the experiment process with pipes
exp_proc = subprocess.Popen(
    exp_cmd,
    cwd=cwd,
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
)

def _stream_reader(pipe, tag):
    """Continuously read lines from pipe and print them prefixed with tag."""
    for line in iter(pipe.readline, ''):
        print(f"[{tag}] {line}", end='')
    pipe.close()

# Spin up threads to forward stdout/stderr
threading.Thread(target=_stream_reader, args=(exp_proc.stdout, "EXP-OUT"), daemon=True).start()
threading.Thread(target=_stream_reader, args=(exp_proc.stderr, "EXP-ERR"), daemon=True).start()

# Give the experiment a bit to start
time.sleep(5)

# %% 2) Start your MuseSL streaming + recording pipelines
stream_p = muselsl.winctrlc.Popen(stream_cmd, cwd)
time.sleep(15)

ppg_p  = muselsl.winctrlc.Popen(record_base + ' PPG', cwd)
acc_p  = muselsl.winctrlc.Popen(record_base + ' ACC', cwd)
gyro_p = muselsl.winctrlc.Popen(record_base + ' GYRO', cwd)

[inlet, inlet_marker, marker_time_correction,
 chunk_length, ch, ch_names] = muselsl.start_record(t_init=t_init)

# %% 3) Main Data Acquisition Loop
res = []
timestamps = []
markers = []
flag = 0
data_flag = 1
time_correction = inlet.time_correction()
last_written_timestamp = None

print(f'Start recording at time t={t_init:.3f}')
print('Time correction:', time_correction)

while True:
    old_len = len(timestamps)
    flag += 1
    try:
        data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=chunk_length)
        if timestamp:
            res.append(data)
            timestamps.extend(timestamp)
        if inlet_marker:
            m, mt = inlet_marker.pull_sample(timeout=0.0)
            if mt:
                markers.append([m, mt + marker_time_correction])
        # if experiment sent final marker, break
        if markers and markers[-1][0] == [999]:
            break

        # periodic save every 5s
        if last_written_timestamp is None or last_written_timestamp + 5 < timestamps[-1]:
            muselsl.save_ongoing(
                res, timestamps, time_correction, False,
                inlet_marker, markers, ch_names,
                last_written_timestamp=last_written_timestamp,
                participant=participant, filename=filename
            )
            last_written_timestamp = timestamps[-1]

        # update shared memory
        if res:
            all_data = np.concatenate(res, axis=0)  # (total_samples,5)
            eeg_only = all_data[:, :4]
            ts_arr = (np.array(timestamps) + time_correction).reshape(-1,1)
            num_rows = int(12*256)
            if eeg_only.shape[0] >= num_rows:
                latest_eeg = eeg_only[-num_rows:]
                latest_ts  = ts_arr[-num_rows:]
            else:
                pad = num_rows - eeg_only.shape[0]
                latest_eeg = np.vstack((np.zeros((pad,4)), eeg_only))
                latest_ts  = np.vstack((np.zeros((pad,1)), ts_arr))
            latest_chunk = np.hstack((latest_eeg, latest_ts))
            lock_flag[0] = 1
            eeg_shared[:] = latest_chunk
            lock_flag[0] = 0

    except KeyboardInterrupt:
        break

# Final save
muselsl.save_ongoing(
    res, timestamps, time_correction, False,
    inlet_marker, markers, ch_names,
    last_written_timestamp=last_written_timestamp,
    participant=participant, filename=filename
)

# shut down all subprocesses
ppg_p.send_ctrl_c()
acc_p.send_ctrl_c()
gyro_p.send_ctrl_c()
stream_p.send_ctrl_c()
exp_proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)  # politely ask experiment to quit
exp_proc.wait()

# Cleanup shared memory
shm_eeg.close(); shm_eeg.unlink()
shm_lock.close(); shm_lock.unlink()
