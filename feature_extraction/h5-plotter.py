import h5py
import numpy as np
import matplotlib.pyplot as plt


def load_fbe_data(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        fbe1 = h5_file['FBE1'][:]
        fbe2 = h5_file['FBE2'][:]
        fbe3 = h5_file['FBE3'][:]
        fbe4 = h5_file['FBE4'][:]

        # Try to get the pulse rate
        pulse_rate = None
        possible_paths = ['DAQ/RepetitionFrequency', 'DAQ/PulseRate', 'ProcessingServer/DataRate']
        for path in possible_paths:
            if path in h5_file:
                pulse_rate = h5_file[path][()]
                break

        if pulse_rate is None:
            print("Warning: Pulse rate not found. Using default value of 1000 Hz.")
            pulse_rate = 1000

    return fbe1, fbe2, fbe3, fbe4, pulse_rate


def plot_fbe_data(fbe1, fbe2, fbe3, fbe4, pulse_rate, channel_start=0, channel_end=None, time_start=0, time_end=None,
                  figsize=(16, 4)):
    fbe_data = [fbe1, fbe2, fbe3, fbe4]
    fbe_names = ['FBE1 (4-8 Hz)', 'FBE2 (8-20 Hz)', 'FBE3 (20-48 Hz)', 'FBE4 (48-100 Hz)']

    max_time, max_channels = fbe1.shape
    window_size = 0.25  # Assuming 0.25s windows as in your original code

    time_end_idx = int(time_end / window_size) if time_end is not None else max_time
    time_end_idx = min(time_end_idx, max_time)

    if channel_end is None:
        channel_end = max_channels
    else:
        channel_end = min(channel_end, max_channels)

    channel_start = max(0, min(channel_start, channel_end - 1))

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)

    for ax, fbe, name in zip(axes, fbe_data, fbe_names):
        plot_data = fbe[int(time_start / window_size):time_end_idx, channel_start:channel_end]

        im = ax.imshow(plot_data.T, aspect='auto', origin='lower', cmap='viridis',
                       extent=[time_start, time_end_idx * window_size, channel_start, channel_end])

        ax.set_title(name)
        ax.set_xlabel('Time (s)')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('FBE (dB)')

    fig.text(0.04, 0.5, 'Channel', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = '/Users/jamesramsay/Downloads/0000028223_2023-09-15_07.43.24.54298.hdf5'

    fbe1, fbe2, fbe3, fbe4, pulse_rate = load_fbe_data(file_path)

    plot_fbe_data(fbe1, fbe2, fbe3, fbe4, pulse_rate,
                  channel_start=0,
                  channel_end=2000,
                  time_start=0,
                  time_end=300,
                  figsize=(16, 4))