import numpy as np
import pandas as pd
# ONLINE DATA
# 12-Feb 

import os
import time
from pynput import keyboard
from scipy.signal import butter, filtfilt, find_peaks
from scipy.integrate import simpson
from sklearn.svm import SVC
import joblib
import spidev
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
from threading import Thread
from matplotlib.animation import FuncAnimation
from collections import Counter
from sklearn.preprocessing import StandardScaler


# SUMMMARY
# Step 1
# Signal is captured (read_adc) and saved to a dataframe (data_acquisition_thread)

# SPI Configuration for ADC
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

# Function to read data from ADC
def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

# Data acquisition thread to read sensors 
# Change the channels if needed read_adc(0) etc
def data_acquisition_thread():
    start_time = time.time()
    while acquisition_running:
        current_time = time.time()
        elapsed_time = current_time - start_time
        tribo_dataRA = read_adc(0)  # Sensor RA
        tribo_dataLA = read_adc(2)  # Sensor LA
        tribo_dataBA = read_adc(4)  # Sensor Back
        live_buffer.append({
            'Temps': elapsed_time,
            'Capteur Triboélectrique RA': tribo_dataRA,
            'Capteur Triboélectrique LA': tribo_dataLA,
            'Capteur Triboélectrique BA': tribo_dataBA
        })
        time.sleep(1 / 20)                   # sampling rate = 20


# Sampling settings
sampling_rate = 20
fs = 20
cutoff_freq = 8          # Cutoff frequency here
window_duration = 2
delay_duration = 4       # so that the signal can properly stabilize when started, can be changed if needed
window_size = window_duration * sampling_rate
buffer_size = 200
live_buffer = deque(maxlen=buffer_size)
acquisition_running = True
prev_tribo_filtered = None
processing_started = False



# Step 2
# The signal is filtered by (butter_lowpass_filter)

# Butterworth filter function
def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

# Step 2B
# In the Main(), differential of the signal is used as well for features calculation

# Step 3
# FFT is calculated for filtered signals (extract_fft_features) fs = sampling rate
# Then FFT is divided into 10 sections and the mean of the 10 secions is calculated and used as features for training

# Function to extract FFT features
def extract_fft_features(signal, fs):
    n = len(signal)
    fft_signal = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, d=1/fs)
    positive_freq_idx = fft_freq > 0
    fft_magnitude = np.abs(fft_signal[positive_freq_idx])
    section_length = len(fft_magnitude) // 10
    fft_features = [np.mean(fft_magnitude[i * section_length:(i + 1) * section_length]) for i in range(10)]
    return fft_features


# Step 4
# Other features are calculated (2 for tribo and 2 for differential signal/ same features)
# Pplus 1 more features which is the 5th features, only for diffrential signal


def compute_five_features(segment, signal_column, differential_column, time_column="Temps"):

    # Extract signal, differential, and time from the DataFrame
    signal = segment[signal_column].values
    diff_signal = segment[differential_column].values  # Access the differential from the DataFrame
    time = segment[time_column].values  # Optional, included for possible time-dependent processing

    # Find top and bottom peaks for the original (tribo) signal
    top_peaks, _ = find_peaks(signal)
    bottom_peaks, _ = find_peaks(-signal)  # Inverted signal for bottom peaks

    # Find top and bottom peaks for the differential signal
    top_peaks_diff, _ = find_peaks(diff_signal)
    bottom_peaks_diff, _ = find_peaks(-diff_signal)

    # Tribo features
    # Feature 1: Mean of top peaks - mean of bottom peaks for original signal
    top_mean = np.mean(signal[top_peaks]) if len(top_peaks) > 0 else 0
    bottom_mean = np.mean(signal[bottom_peaks]) if len(bottom_peaks) > 0 else 0
    feature1 = top_mean - bottom_mean

    # Feature 2: Sum of deviations from the minimum of bottom peaks for original signal
    bottom_min = np.min(signal[bottom_peaks]) if len(bottom_peaks) > 0 else 0
    deviations = signal - bottom_min
    feature2 = np.sum(deviations)

    # Differential features
    # Feature 3: Mean of top peaks - mean of bottom peaks for differential signal
    top_mean_diff = np.mean(diff_signal[top_peaks_diff]) if len(top_peaks_diff) > 0 else 0
    bottom_mean_diff = np.mean(diff_signal[bottom_peaks_diff]) if len(bottom_peaks_diff) > 0 else 0
    feature3 = top_mean_diff - bottom_mean_diff

    # Feature 4: Sum of deviations from the minimum of bottom peaks for differential signal
    bottom_min_diff = np.min(diff_signal[bottom_peaks_diff]) if len(bottom_peaks_diff) > 0 else 0
    deviations_diff = diff_signal - bottom_min_diff
    feature4 = np.sum(deviations_diff)

    # Feature 5: Sum of the absolute values of the differential signal (absolute changes)
    feature5 = np.sum(np.abs(diff_signal))

    return feature1, feature2, feature3, feature4, feature5




# Step 5
# Load the models for each sensor and features names 

# Load the pre-trained model
model_RA = joblib.load("tribo_differential_18Feb_RA_gradient_boosting_model.pkl")
model_LA = joblib.load("tribo_differential_18Feb_LA_gradient_boosting_model.pkl")
model_BA = joblib.load("tribo_differential_18Feb_BACK_random_forest_model.pkl")

# Feature column names
feature_names = [
    "tribo_f1", "tribo_f2", "tribo_f3", "tribo_f4", "tribo_f5",
    "tribo_f6", "tribo_f7", "tribo_f8", "tribo_f9", "tribo_f10",
    "diff_tribo_f1", "diff_tribo_f2", "diff_tribo_f3", "diff_tribo_f4", "diff_tribo_f5",
    "diff_tribo_f6", "diff_tribo_f7", "diff_tribo_f8", "diff_tribo_f9", "diff_tribo_f10",
    "tribo_feature1", "tribo_feature2", "tribo_feature3", "tribo_feature4", "tribo_feature5"
]


# Step 6 - if needed uncomment the section in the Main()
# Define the path where you want to save the CSV file
output_csv_path = 'Data_online_12_FEB_Test.csv'


# To change mean threshold if needed
touch_threshold = 450



def on_press(key):
    global acquisition_running
    if hasattr(key, 'char') and key.char == 'q':
        acquisition_running = False
        return False


# List to hold the filtered triboelectric signal
filtered_tribo_data = []


listener = keyboard.Listener(on_press=on_press)
listener.start()
acquisition_thread = Thread(target=data_acquisition_thread)

acquisition_thread.start()

is_processing_LA = False
is_processing_RA = False
prediction_bufferLA = []
prediction_bufferRA = []
prediction_bufferBA = []



# MAIN()
try:
    start_time = time.time()
    last_prediction_time = time.time()
    last_summary_time = time.time()

    while acquisition_running:
        current_time = time.time()
        elapsed_time = current_time - start_time


        # Delay before starting processing (To allow signal to stabilize)
        if not processing_started and elapsed_time > delay_duration:
            processing_started = True
            print("Encoder starting")

        # Process live data when buffer is full
        if processing_started and len(live_buffer) >= window_size:
            # Convert the live buffer to a DataFrame
            df_live = pd.DataFrame(list(live_buffer))
            recent_data = df_live.iloc[-window_size:]

            # Predict every 0.05 seconds  (Every 0.05 second for window of 2 seconds) That's why we take last 40 sampples
            # So every 0.05 seconds, we take the last 40 values of the signal, filter it, calculate features
            # and get a prediction every 0.05 seconds. All predictions are stored in a list
            # Also when doing data collection, the features are saved in a dataframe.

            if current_time - last_prediction_time >= 0.05:
                last_prediction_time = current_time

                if len(live_buffer) >= 40:                                  
                    # Get the last 40 samples
                    last_samples = df_live.iloc[-window_size:]
                
                    # Apply Butterworth filter to the triboelectric signal LA
                    filtered_LA = butter_lowpass_filter(
                        last_samples['Capteur Triboélectrique LA'], cutoff_freq, sampling_rate
                    )
                    # Apply Butterworth filter to the triboelectric signal RA
                    filtered_RA = butter_lowpass_filter(
                        last_samples['Capteur Triboélectrique RA'], cutoff_freq, sampling_rate
                    )
                    # Apply Butterworth filter to the triboelectric signal BA
                    filtered_BA = butter_lowpass_filter(
                        last_samples['Capteur Triboélectrique BA'], cutoff_freq, sampling_rate
                    )


                    # Compute the differential of the triboelectric signal LA
                    differential_LA = np.diff(filtered_LA, prepend=filtered_LA[0])
                    # Compute the differential of the triboelectric signal RA
                    differential_RA = np.diff(filtered_RA, prepend=filtered_RA[0])
                     # Compute the differential of the triboelectric signal bA
                    differential_BA = np.diff(filtered_BA, prepend=filtered_BA[0])
            

                    # Add differential_LA as a new column in the DataFrame
                    last_samples = last_samples.copy()  # Ensure it's a copy, not a view
                    last_samples.loc[:, 'Tribo Differential LA'] = differential_LA[-40:]
                    last_samples.loc[:, 'Tribo Differential RA'] = differential_RA[-40:]
                    last_samples.loc[:, 'Tribo Differential BA'] = differential_BA[-40:]



                    # Extract FFT features for last_tribo_filtered and tribo_differential LA
                    fft_features_last_tribo_filtered_LA = extract_fft_features(filtered_LA[-40:], fs)
                    fft_features_tribo_differential_LA = extract_fft_features(differential_LA[-40:], fs)
                    # Combine them into a single feature vector (20 features) LA
                    total_features_LA = fft_features_last_tribo_filtered_LA + fft_features_tribo_differential_LA

                    # Extract FFT features for last_tribo_filtered and tribo_differential RA
                    fft_features_last_tribo_filtered_RA = extract_fft_features(filtered_RA[-40:], fs)
                    fft_features_tribo_differential_RA = extract_fft_features(differential_RA[-40:], fs)
                    # Combine them into a single feature vector (20 features) RA
                    total_features_RA = fft_features_last_tribo_filtered_RA + fft_features_tribo_differential_RA

                    # Extract FFT features for last_tribo_filtered and tribo_differential BA
                    fft_features_last_tribo_filtered_BA = extract_fft_features(filtered_BA[-40:], fs)
                    fft_features_tribo_differential_BA = extract_fft_features(differential_BA[-40:], fs)
                    # Combine them into a single feature vector (20 features) BA
                    total_features_BA = fft_features_last_tribo_filtered_BA + fft_features_tribo_differential_BA



                    # Compute five additional features LA
                    feature1LA, feature2LA, feature3LA, feature4LA, feature5LA = compute_five_features(
                        last_samples, 'Capteur Triboélectrique LA', 'Tribo Differential LA')
                    # Combine the 20 FFT features with the 5 additional features to form the 25 total features LA
                    final_feature_vector_LA = total_features_LA + [feature1LA, feature2LA, feature3LA, feature4LA, feature5LA]

                    # Compute five additional features RA
                    feature1RA, feature2RA, feature3RA, feature4RA, feature5RA = compute_five_features(
                        last_samples, 'Capteur Triboélectrique RA', 'Tribo Differential RA')
                    # Combine the 20 FFT features with the 5 additional features to form the 25 total features RA
                    final_feature_vector_RA = total_features_RA + [feature1RA, feature2RA, feature3RA, feature4RA, feature5RA]

                    # Compute five additional features BA
                    feature1BA, feature2BA, feature3BA, feature4BA, feature5BA = compute_five_features(
                        last_samples, 'Capteur Triboélectrique BA', 'Tribo Differential BA')
                    # Combine the 20 FFT features with the 5 additional features to form the 25 total features BA
                    final_feature_vector_BA = total_features_BA + [feature1BA, feature2BA, feature3BA, feature4BA, feature5BA]



                    # Create the DataFrame for data collection
                    last_feature_df_LA = pd.DataFrame([final_feature_vector_LA], columns=feature_names)
                    last_feature_df_RA = pd.DataFrame([final_feature_vector_RA], columns=feature_names)
                    last_feature_df_BA = pd.DataFrame([final_feature_vector_BA], columns=feature_names)

                    # Create a separate DataFrame for saving (with elapsed time)
                    save_feature_df_LA = last_feature_df_LA.copy()
                    save_feature_df_RA = last_feature_df_RA.copy()
                    save_feature_df_BA = last_feature_df_BA.copy()
                    save_feature_df_LA["Temps"] = elapsed_time
                    save_feature_df_RA["Temps"] = elapsed_time
                    save_feature_df_BA["Temps"] = elapsed_time


                    # # Save to CSV (UNCOMMENT BELOW IF USED FOR DATA COLLECTION/ DONT FORGET TO CHANGE PATH NAME)
                    # save_feature_df_LA.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)
                    #save_feature_df_RA.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)
                    # save_feature_df_BA.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)


                    # Make predictions (without elapsed time column)
                    predictionLA = model_LA.predict(last_feature_df_LA)[0]
                    predictionRA = model_RA.predict(last_feature_df_RA)[0]
                    predictionBA = model_BA.predict(last_feature_df_BA)[0]

                    # Track predictions over time
                    prediction_bufferLA.append(predictionLA)
                    prediction_bufferRA.append(predictionRA)
                    prediction_bufferBA.append(predictionBA)


            # Every 2 seconds, count the most frequent prediction and print the results. 
            # The mean of all the sensors is calculated (40samples).
            # The sensor being touched will have the highest mean (this part works correctly)
            # A result will be printed if the mean of the sensor is higher than a certain threshold. (to avoid getting unecessary wrong results)

            if current_time - last_summary_time >= 2.0:
            
                last_summary_time = current_time

               # Compute mean values for filtered signals
                mean_RA = np.mean(filtered_RA) if len(filtered_RA) > 0 else 0
                mean_LA = np.mean(filtered_LA) if len(filtered_LA) > 0 else 0
                mean_BA = np.mean(filtered_BA) if len(filtered_BA) > 0 else 0

                # Find the sensor with the highest mean above the threshold  (TO CHANGE THRESHOLD IF NEEDED)
                sensor_means = {'LA': mean_LA, 'RA': mean_RA, 'BA': mean_BA}
                max_sensor = max(sensor_means, key=sensor_means.get)


                # Threshold logic here + results is printed and sent to QT

                if sensor_means[max_sensor] > touch_threshold:
                    # print(f"Most affected sensor: {max_sensor} ({sensor_means[max_sensor]:.2f})")

                    # Get most common prediction from the buffer of the most affected sensor
                    if max_sensor == "LA":
                        most_common = Counter(prediction_bufferLA[-36:]).most_common(1)
                
                        prediction_bufferLA.clear()
                    elif max_sensor == "RA":
                        most_common = Counter(prediction_bufferRA[-36:]).most_common(1)
                        prediction_bufferRA.clear()
                    else:
                        most_common = Counter(prediction_bufferBA[-36:]).most_common(1)
                        prediction_bufferBA.clear()

                    if most_common:
                        # print(f"Most frequent prediction for {max_sensor}: {most_common[0][0]}")
                        result = f"{max_sensor}:{most_common[0][0]}"
                        print(result)



except KeyboardInterrupt:
    acquisition_running = False
    print("The end - Thank You.")

finally:
    acquisition_thread.join()
    listener.stop()
