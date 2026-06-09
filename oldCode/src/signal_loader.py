import numpy as np
import csv
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os

class PatientSignals:
    def __init__(self, folder):
        self.folder = folder
        self.signal_file = folder + '/Signals/New_Data.csv'
        self.gt_file = folder + '/gt/NOM_ECG_ELEC_POTL_IIWaveExport.csv'
        self.data = self._read_signal_data()
        self.gt_ecg = self._read_gt_data()

        # Extract signals
        if self.data is not None:
            self.cecg = self.data[:, 1]
            self.ppg1 = self.data[:, 12]
            self.ppg2 = self.data[:, 24]
            self.bcg1 = self.data[:, 2]
            self.bcg2 = self.data[:, 3]
        else:
            self.cecg = None
            self.ppg1 = None
            self.ppg2 = None
            self.bcg1 = None
            self.bcg2 = None

        # Filtered signals (initialized as None)
        self.cecg_filt = None
        self.ppg1_filt = None
        self.ppg2_filt = None
        self.bcg1_filt = None
        self.bcg2_filt = None
        self.gt_ecg_filt = None

    def _read_signal_data(self):
        data = []
        try:
            with open(self.signal_file, mode='r') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader)
                for row in csv_reader:
                    data.append(row)
            return np.array(data, dtype=np.float64)
        except Exception as e:
            print(f"Error reading signal data: {e}")
            return None

    def _read_gt_data(self):
        gt_data = []
        try:
            with open(self.gt_file, mode='r') as gt_file:
                csv_reader = csv.reader(gt_file)
                for row in csv_reader:
                    gt_data.append(row)
            gt_data = np.array(gt_data)
            gt_ecg = gt_data[:, 3].astype(np.uint16)
            return gt_ecg
        except Exception as e:
            print(f"Error reading ground truth data: {e}")
            return None

    def bp_filter(self, signal, lowcut, highcut, fs, order=4):
        b, a = butter(order, [lowcut, highcut], fs=fs, btype='bandpass')
        return filtfilt(b, a, signal)

    def filter_all(self, fs=128):
        if self.cecg is not None:
            self.cecg_filt = self.bp_filter(self.cecg, 0.5, 35, fs, order=4)
            # b,a = butter(4, [16, 18], fs=fs, btype='stop')
            # self.cecg_filt = filtfilt(b, a, self.cecg_filt)
            # b,a = butter(4, [26, 28], fs=fs, btype='stop')
            # self.cecg_filt = filtfilt(b, a, self.cecg_filt)
            if np.median(self.cecg_filt) > np.mean(self.cecg_filt):
                self.cecg_filt = -self.cecg_filt
        if self.ppg1 is not None:
            self.ppg1_filt = self.bp_filter(self.ppg1, 0.6, 3.6, fs, order=4)
        if self.ppg2 is not None:
            self.ppg2_filt = self.bp_filter(self.ppg2, 0.6, 3.6, fs, order=4)
        if self.bcg1 is not None:
            self.bcg1_filt = self.bp_filter(self.bcg1, 0.6, 10, fs, order=4)
        if self.bcg2 is not None:
            self.bcg2_filt = self.bp_filter(self.bcg2, 0.6, 10, fs, order=4)
        if self.gt_ecg is not None:
            self.gt_ecg_filt = self.bp_filter(self.gt_ecg, 0.5, 35, 500, order=4)


    def offset_correction(self):
        offset_file = os.path.join(self.folder, "Signals", "offsets.txt")
        if not os.path.exists(offset_file):
            print(f"Offset file not found: {offset_file}")
            return

        with open(offset_file, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                print("Offset file must have two lines (initial and final offset).")
                return
            initial_offset = int(float(lines[0].strip()))
            final_offset = int(float(lines[1].strip()))

        def correct_signal(signal, initial_offset, final_offset):
            shifted_signal = np.pad(signal.copy(), (initial_offset, 0), mode='constant')[:len(signal)]
            total_samples = len(shifted_signal)
            offset_change = final_offset - initial_offset
            orig_indices = np.arange(total_samples)
            corrected_indices = orig_indices + np.linspace(0, offset_change, total_samples)
            from scipy.interpolate import interp1d
            interp_func = interp1d(corrected_indices, shifted_signal, bounds_error=False, fill_value=0)
            return interp_func(orig_indices)

        # Apply to all filtered signals except gt_ecg_filt
        if self.cecg_filt is not None:
            self.cecg_filt = correct_signal(self.cecg_filt, initial_offset, final_offset)
        if self.ppg1_filt is not None:
            self.ppg1_filt = correct_signal(self.ppg1_filt, initial_offset, final_offset)
        if self.ppg2_filt is not None:
            self.ppg2_filt = correct_signal(self.ppg2_filt, initial_offset, final_offset)
        if self.bcg1_filt is not None:
            self.bcg1_filt = correct_signal(self.bcg1_filt, initial_offset, final_offset)
        if self.bcg2_filt is not None:
            self.bcg2_filt = correct_signal(self.bcg2_filt, initial_offset, final_offset)
  
    
    def plot_all_filtered(self):
        if self.data is None or self.gt_ecg_filt is None:
            print("No filtered data to plot.")
            return

        t_128 = np.arange(self.data.shape[0]) / 128.0
        t_500 = np.arange(len(self.gt_ecg_filt)) / 500.0

        fig, axs = plt.subplots(6, 1, figsize=(15, 12), sharex=True)
        axs[0].plot(t_500, self.gt_ecg_filt, label='Filtered GT ECG', color='r')
        axs[1].plot(t_128, self.cecg_filt, label='Filtered CECG')
        axs[2].plot(t_128, self.ppg1_filt, label='Filtered PPG1')
        axs[3].plot(t_128, self.ppg2_filt, label='Filtered PPG2')
        axs[4].plot(t_128, self.bcg1_filt, label='Filtered BCG1')
        axs[5].plot(t_128, self.bcg2_filt, label='Filtered BCG2')
    
        for ax in axs:
            ax.legend()
            ax.grid(True)
            ax.set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()
