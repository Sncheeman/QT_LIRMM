import numpy as np
import pandas as pd
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




# TO change the channel in data_acquisition_thread()




# Butterworth filter function
def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

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



# SPI Configuration for ADC
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

# Function to read data from ADC
def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data



# Feature column names
feature_names = [
    "tribo_f1", "tribo_f2", "tribo_f3", "tribo_f4", "tribo_f5",
    "tribo_f6", "tribo_f7", "tribo_f8", "tribo_f9", "tribo_f10",
    "diff_tribo_f1", "diff_tribo_f2", "diff_tribo_f3", "diff_tribo_f4", "diff_tribo_f5",
    "diff_tribo_f6", "diff_tribo_f7", "diff_tribo_f8", "diff_tribo_f9", "diff_tribo_f10",
    "tribo_feature1", "tribo_feature2", "tribo_feature3", "tribo_feature4", "tribo_feature5"
]


# Sampling settings
sampling_rate = 20
fs = 20
cutoff_freq = 8
window_duration = 2
delay_duration = 4
window_size = window_duration * sampling_rate
buffer_size = 200
live_buffer = deque(maxlen=buffer_size)
acquisition_running = True
prev_tribo_filtered = None
processing_started = False

def data_acquisition_thread():
    start_time = time.time()
    while acquisition_running:
        current_time = time.time()
        elapsed_time = current_time - start_time
        tribo_dataLF = read_adc(0)   # CHANGE THE CHANNEL NUMBER, TO VISUALISE THE FILTERED SIGNAL IN REAL TIME
        pressure_dataLF = read_adc(1)
        live_buffer.append({
            'Temps': elapsed_time,
            'Capteur Triboélectrique RF': tribo_dataLF,
            'Capteur de Pression RF': pressure_dataLF
        })
        time.sleep(1 / 20)

def on_press(key):
    global acquisition_running
    if hasattr(key, 'char') and key.char == 'q':
        acquisition_running = False
        return False

listener = keyboard.Listener(on_press=on_press)
listener.start()
acquisition_thread = Thread(target=data_acquisition_thread)
acquisition_thread.start()

# Initialize the real-time plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
x_data, y_data = [], []  # Lists to store x and y data points
line, = ax.plot([], [], 'bo-', lw=1, label="Filtered Triboelectric Signal")  # Use points with 'bo-' style
ax.set_title("Real-Time Filtered Triboelectric Signal")
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Amplitude")
ax.set_xlim(0, 40)  # Initial x-axis range
ax.set_ylim(0, 1100)  # Adjust y-axis range based on expected signal amplitude
ax.legend(loc="upper right")

# List to hold the filtered triboelectric signal
filtered_tribo_data = []

columns_with_labels = feature_names + ["Predicted_Label"]
stored_features_df = pd.DataFrame(columns=columns_with_labels)

# Path to save the CSV file
output_file = "Data_online_LOW_TEST_all_HZ_18_FEB.csv"           # WARNING, Do not forget to rename the output

# Load the pre-trained model
# model = joblib.load("tribo_differential_dec16_svm_(rbf_kernel)_model.pkl")
model = joblib.load("tribo_differential_18Feb_LA_random_forest_model.pkl")

# Initialize the CSV file with headers if it doesn't exist
if not os.path.exists(output_file):
    with open(output_file, "w") as f:
        header = ",".join(["Temps"] + feature_names) + "\n"
        f.write(header)

try:
    start_time = time.time()
    last_prediction_time = time.time()
    last_frame_time = time.time()

    # Buffer for storing predictions
    prediction_buffer = []

    while acquisition_running:
        current_time = time.time()
        elapsed_time = current_time - start_time

        if not processing_started and elapsed_time > delay_duration:
            processing_started = True
            print("Feature extraction and classification started.")

        # Ensure sufficient samples are available for processing
        if processing_started and len(live_buffer) >= window_size:
            # Convert the live buffer to a DataFrame
            df_live = pd.DataFrame(list(live_buffer))
            recent_data = df_live.iloc[-window_size:]

            # Perform prediction every 0.1 seconds
            if current_time - last_prediction_time >= 0.1:
                last_prediction_time = current_time

                if len(live_buffer) >= 40:
                    # Get the last 40 samples
                    last_samples = df_live.iloc[-window_size:].copy()

                    # Remove -200 offset from raw signals
                    last_samples['Capteur Triboélectrique RF'] = last_samples['Capteur Triboélectrique RF']
             


                    # Apply Butterworth filter to the triboelectric signal
                    filtered_tribo = butter_lowpass_filter(
                        last_samples['Capteur Triboélectrique RF'], cutoff_freq, sampling_rate
                    )

                    # Compute the differential of the triboelectric signal
                    tribo_differential = np.diff(filtered_tribo, prepend=filtered_tribo[0])

                    # Add tribo_differential as a new column in the DataFrame
                    last_samples = last_samples.copy()  # Ensure it's a copy, not a view
                    last_samples.loc[:, 'Tribo Differential'] = tribo_differential[-40:]

                    # Extract FFT features for last_tribo_filtered and tribo_differential
                    fft_features_last_tribo_filtered = extract_fft_features(filtered_tribo[-40:], fs)
                    fft_features_tribo_differential = extract_fft_features(tribo_differential[-40:], fs)

                    # Combine them into a single feature vector (20 features)
                    total_features = fft_features_last_tribo_filtered + fft_features_tribo_differential

                    # Compute five additional features
                    feature1, feature2, feature3, feature4, feature5 = compute_five_features(
                        last_samples, 'Capteur Triboélectrique RF', 'Tribo Differential'
                    )

                    # Combine the 20 FFT features with the 5 additional features to form the 25 total features
                    final_feature_vector = total_features + [feature1, feature2, feature3, feature4, feature5]

                    # Create the DataFrame with the correct shape (1, 25)
                    last_feature_df = pd.DataFrame([final_feature_vector], columns=feature_names)

                    # Make prediction using the 4-class model
                    prediction = model.predict(last_feature_df)[0]

                    # Track predictions over time
                    prediction_buffer.append(prediction)

                    prediction_proba = model.predict_proba(last_feature_df)[0]

                    # Append features and label to storage DataFrame
                    features_with_label = list(last_feature_df.iloc[0]) + [prediction]
                    stored_features_df.loc[len(stored_features_df)] = features_with_label

                    # Add elapsed time and save to CSV
                    with open(output_file, "a") as f:
                        row = [elapsed_time] + final_feature_vector
                        f.write(",".join(map(str, row)) + "\n")

            # Update filtered_tribo_data with the latest filtered signal
            filtered_tribo_data = filtered_tribo[-40:]

            # Update the plot
            line.set_ydata(filtered_tribo_data)
            line.set_xdata(range(len(filtered_tribo_data)))
            ax.relim()
            ax.autoscale_view()  # Auto-scale the view for changes in amplitude
            plt.draw()
            plt.pause(0.1)  # Pause for a short duration to allow the plot to update

            # Every 2 seconds, calculate the most frequent prediction
            if current_time - last_frame_time >= 2.0:
                most_common = Counter(prediction_buffer[-18:]).most_common(1)
                if most_common:
                    print(f"Most frequent prediction: {most_common[0][0]}")
                prediction_buffer.clear()
                last_frame_time = current_time

except KeyboardInterrupt:
    acquisition_running = False
    print("Data acquisition stopped.")

finally:
    acquisition_thread.join()
    listener.stop()
    plt.ioff()  # Turn off interactive mode
    plt.show()
