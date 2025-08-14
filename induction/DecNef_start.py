import muselsl
import time
import datetime
import numpy as np
import subprocess
import os
import psutil
import muselsl.winctrlc
import pandas as pd
from pathlib import Path
from time import strftime, gmtime
from multiprocessing import shared_memory

# %% Shared Memory Setup for Latest 12 Seconds of EEG Data (4 Channels + Timestamp)
# At 256 Hz, 12 seconds correspond to 3072 rows.
eeg_rows = int(12 * 256)  # 3072 rows
eeg_cols = 5              # 4 EEG channels + 1 timestamp column
dtype = np.float32

# Create shared memory for the EEG data (only EEG channels, no timestamp)
shm_eeg = shared_memory.SharedMemory(create=True, size=np.zeros((eeg_rows, eeg_cols), dtype=dtype).nbytes,
                                     name='eeg_data')
eeg_shared = np.ndarray((eeg_rows, eeg_cols), dtype=dtype, buffer=shm_eeg.buf)
# Initialize to zeros
eeg_shared[:] = 0

# %% Shared Memory Setup for a Simple Lock Flag
# A one-element array; 0 means "not updating", 1 means "updating"
shm_lock = shared_memory.SharedMemory(create=True, size=np.zeros((1,), dtype=np.int8).nbytes, name='eeg_lock')
lock_flag = np.ndarray((1,), dtype=np.int8, buffer=shm_lock.buf)
lock_flag[0] = 0  # initial state: not updating

# %% Existing Setup Code
# t_init = int(datetime.datetime(2024, 3, 21, 15, 0, 0).timestamp())  # Year, Month, Day, Hour, Minutes
t_init = int(datetime.datetime.now().timestamp())
# participant = '200'
# name = 'MuseS-6408'
windowLength_saving2csv=5 #5s
#participant = 'EB-41'
participant = 'EB-51' #XF
#name = 'MuseS-6B97' # get it through  https://eegedu.com/
name = 'MuseS-62E9' # get it through  https://eegedu.com/
cwd = os.getcwd()
venv_path = ''.join(['\"', os.path.join(cwd, 'venv', 'Scripts', 'activate.bat'), '\"'])
data_folder = os.path.join(cwd, '..', 'Data')
try:
    os.makedirs(data_folder)
except OSError as error:
    print(error)
#data_path_relative = os.path.join('..', 'Data', 'sub-' + participant, 'Muse_data')
data_path_relative = os.path.join('..', 'Data', 'sub-' + participant, 'Muse_data_DecNef')
data_path = os.path.join(cwd, data_path_relative)
try:
    os.makedirs(data_path)
except OSError as error:
    print(error)

data_path_quotes = '\"' + data_path + '\"'
stream_command = ''.join(['muselsl stream -n ', name, ' -p -c -g'])
record_command = (
    f'muselsl record -s {t_init} '
    f'-p {participant} '
    f'-d "{data_path}" '
    '-t'
)
# filename = os.path.join(data_path, "%s_%s_recording.csv" %
#                         (''.join(['sub-', participant]),
#                          'EEG'))
#adding a time stamp to the filename
filename = os.path.join(data_path, "%s_%s_recording_%s.csv" %
                            (''.join(['sub-', participant]),
                             'EEG',
                             time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime())))

# for windows power shell/cmd
# command = ''.join(
#     ['start ', venv_path, ' ; .\\venv\\Scripts\\python.exe DecNef_museS.py -p ',
#       participant, ' -s ', str(t_init)])

# #4/25/2025, didn't work!
# command = (
#     f'start "" {venv_path} & '
#     r'.\venv\Scripts\python.exe DecNef_museS.py '
#     f'-p {participant} -s {t_init}'
# )
# p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
# time.sleep(5)
# stream_p = muselsl.winctrlc.Popen(stream_command, cwd=cwd, shell=True)
# time.sleep(15)

#6/11/2025
command = ''.join(
    ['start ', venv_path, ' ; .\\venv\\Scripts\\python.exe DecNef_museS.py -p ', participant, ' -s ', str(t_init)])
print(command)
p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
#p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE) #for debugging
time.sleep(5)
stream_p = muselsl.winctrlc.Popen(stream_command, cwd)
time.sleep(15)

ppg_command = ''.join([record_command, ' PPG'])
ppg_p = muselsl.winctrlc.Popen(ppg_command, cwd)

acc_command = ''.join([record_command, ' ACC'])
acc_p = muselsl.winctrlc.Popen(acc_command, cwd)

gyro_command = ''.join([record_command, ' GYRO'])
gyro_p = muselsl.winctrlc.Popen(gyro_command, cwd)

[inlet, inlet_marker, marker_time_correction, chunk_length, ch, ch_names] = muselsl.start_record(t_init=t_init)

res = []
timestamps = []
markers = []
flag = 0
data_flag = 1
time_correction = inlet.time_correction()
last_written_timestamp = None
print('Start recording at time t=%.3f' % t_init)
print('Time correction: ', time_correction)

# %% Main Data Acquisition Loop with Shared Memory Update
while 1:
    old_len = len(timestamps)
    flag += 1
    try:
        data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=chunk_length)
        if timestamp:
            res.append(data)
            timestamps.extend(timestamp)
            tr = time.time()
        if inlet_marker:
            marker, marker_timestamp = inlet_marker.pull_sample(timeout=0.0)
            if marker_timestamp:
                markers.append([marker, marker_timestamp])
                marker_timestamp = marker_timestamp + marker_time_correction
        if marker == [999]:
            break

        # Save every 5s
        if data_flag == 1:
            pass
        elif last_written_timestamp is None or last_written_timestamp + windowLength_saving2csv < timestamps[-1]:
            muselsl.save_ongoing(
                res,
                timestamps,
                time_correction,
                False,
                inlet_marker,
                markers,
                ch_names,
                last_written_timestamp=last_written_timestamp,
                participant=participant,
                filename=filename
            )
            last_written_timestamp = timestamps[-1]

        if len(timestamps) > old_len:
            flag = 0
            if len(timestamps) > 256:
                data_flag = np.sum(np.asarray(res[-1]))

        # --- Update Shared Memory with Latest 12 Seconds of EEG Data (EEG channels + Timestamp) ---
        if res:
            # Concatenate all EEG data pulled from the stream.
            all_data = np.concatenate(res, axis=0)  # shape: (total_samples, 5) [Note: the 5th column may be dummy]

            # Extract EEG channels (first 4 columns).
            eeg_only = all_data[:, 0:4]  # shape: (total_samples, 4)

            # Convert the timestamps list to a column vector.
            # Apply LSL time_correction so shared‐memory stamps match time.time()
            timestamps_arr = (np.array(timestamps) + time_correction).reshape(-1, 1)
            # shape: (total_samples, 1)

            # Define the number of rows corresponding to 12 seconds at 256 Hz.
            num_rows = int(12 * 256)  # 3072 rows

            # Get the latest num_rows rows from the EEG data and corresponding timestamps.
            if eeg_only.shape[0] >= num_rows:
                latest_eeg = eeg_only[-num_rows:, :]
                latest_timestamps = timestamps_arr[-num_rows:, :]
            else:
                pad_rows = num_rows - eeg_only.shape[0]
                latest_eeg = np.vstack((np.zeros((pad_rows, 4), dtype=dtype), eeg_only))
                latest_timestamps = np.vstack((np.zeros((pad_rows, 1), dtype=dtype), timestamps_arr))

            # Combine the EEG data with the timestamps as the 5th column.
            latest_chunk = np.hstack((latest_eeg, latest_timestamps))  # shape: (num_rows, 5)

            # Update the shared memory with the new combined data.
            lock_flag[0] = 1  # set lock: update in progress
            eeg_shared[:] = latest_chunk
            lock_flag[0] = 0  # clear lock

        if data_flag != 0 and flag <= 5:
            pass
        else:
            print(data_flag, flag, len(timestamps))
            flag = 0
            data_flag = 1
            ppg_p.send_ctrl_c()
            acc_p.send_ctrl_c()
            gyro_p.send_ctrl_c()
            while 1:
                try:
                    stream_p.send_ctrl_c()
                    time.sleep(2)
                except OSError as error:
                    print(error)
                    break
            stream_p = muselsl.winctrlc.Popen(stream_command, cwd)
            time.sleep(15)
            ppg_p = muselsl.winctrlc.Popen(ppg_command, cwd)
            acc_p = muselsl.winctrlc.Popen(acc_command, cwd)
            gyro_p = muselsl.winctrlc.Popen(gyro_command, cwd)
    except KeyboardInterrupt:
        break

# Final save of the remaining data
muselsl.save_ongoing(
    res,
    timestamps,
    time_correction,
    False,
    inlet_marker,
    markers,
    ch_names,
    last_written_timestamp=last_written_timestamp,
    participant=participant,
    filename=filename
)

ppg_p.send_ctrl_c()
acc_p.send_ctrl_c()
gyro_p.send_ctrl_c()
while 1:
    try:
        stream_p.send_ctrl_c()
        time.sleep(2)
    except OSError as error:
        print(error)
        break

# Cleanup shared memory objects
shm_eeg.close()
shm_eeg.unlink()
shm_lock.close()
shm_lock.unlink()
