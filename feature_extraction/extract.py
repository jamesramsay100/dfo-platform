import h5py
import numpy as np
from scipy.fft import fft
from typing import Tuple, Optional


def extract_features(
        h5_file_path: str,
        window_size: float = 1.0,
        normalize: bool = False,
        freq_range: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract centroid and spread features from acoustic data in an H5 file.

    Args:
    h5_file_path (str): Path to the H5 file containing acoustic data.
    window_size (float): Size of the time window in seconds (default: 1.0).
    normalize (bool): Whether to normalize the data before FFT (default: False).
    freq_range (tuple): Optional frequency range for feature calculation (min_freq, max_freq).

    Returns:
    Tuple[np.ndarray, np.ndarray]: Centroid and spread features.
    """
    with h5py.File(h5_file_path, 'r') as h5_file:
        das_data = h5_file.get('DAS')[:]
        sample_rate = h5_file.get('DAS').attrs['Rate'][0]

    # Calculate window parameters
    samples_per_window = int(window_size * sample_rate)
    num_windows = das_data.shape[0] // samples_per_window
    num_channels = das_data.shape[1]

    # Initialize output arrays
    centroids = np.zeros((num_windows, num_channels))
    spreads = np.zeros((num_windows, num_channels))

    # Calculate frequency array for FFT (positive frequencies only)
    freqs = np.fft.rfftfreq(samples_per_window, 1 / sample_rate)

    # Apply frequency range if specified
    if freq_range:
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    else:
        freq_mask = np.ones_like(freqs, dtype=bool)

    masked_freqs = freqs[freq_mask]

    for window in range(num_windows):
        start = window * samples_per_window
        end = start + samples_per_window
        window_data = das_data[start:end, :]

        if normalize:
            window_data = (window_data - np.mean(window_data, axis=0)) / np.std(window_data, axis=0)

        # Compute FFT (real FFT)
        fft_data = np.abs(np.fft.rfft(window_data, axis=0))

        # Apply frequency mask
        masked_fft = fft_data[freq_mask, :]

        # Calculate centroid and spread
        total_power = np.sum(masked_fft, axis=0)
        centroids[window] = np.sum(masked_freqs[:, np.newaxis] * masked_fft, axis=0) / total_power
        spreads[window] = np.sqrt(
            np.sum(((masked_freqs[:, np.newaxis] - centroids[window]) ** 2) * masked_fft, axis=0) / total_power)

    return centroids, spreads


# Example usage
if __name__ == "__main__":
    h5_file_path = '/Users/jamesramsay/Downloads/OneDrive_1_09-07-2024/0000000005_2024-07-03_09.30.30.84400.hdf5'
    centroids, spreads = extract_features(h5_file_path, window_size=2.0, normalize=True)
    print("Centroids shape:", centroids.shape)
    print("Spreads shape:", spreads.shape)