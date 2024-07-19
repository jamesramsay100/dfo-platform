import h5py
import numpy as np
from scipy.signal.windows import blackmanharris
from scipy.fft import rfft
from typing import Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class DASFileLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> Tuple[np.ndarray, float]:
        pass


class StandardH5Loader(DASFileLoader):
    def load(self, file_path: str) -> Tuple[np.ndarray, float]:
        with h5py.File(file_path, 'r') as h5_file:
            if 'DAS' in h5_file:
                dphase_s16 = h5_file['DAS'][:]
            else:
                raise KeyError("Unable to find 'DAS' dataset in the HDF5 file.")

                # Load pulse rate (try different possible paths)
            pulse_rate = None
            possible_paths = ['DAQ/RepetitionFrequency', 'DAQ/PulseRate', 'ProcessingServer/DataRate']
            for path in possible_paths:
                if path in h5_file:
                    pulse_rate = h5_file[path][()]
                    break

            if pulse_rate is None:
                raise KeyError(f"Unable to find pulse rate in the HDF5 file. Tried paths: {possible_paths}")

        # Convert differential phase to phase
        phase_s64 = np.cumsum(dphase_s16.astype(np.int64), axis=0)
        phase_f64 = phase_s64 * np.pi / 2 ** 15

        return phase_f64, pulse_rate


class Preprocessor:
    def __init__(self, window_size: float = 0.25):
        self.window_size = window_size

    def preprocess(self, data: np.ndarray, rate: float) -> np.ndarray:
        # Ensure rate is a scalar
        rate = float(np.asarray(rate).item())
        samples_per_window = int(self.window_size * rate)
        num_windows = data.shape[0] // samples_per_window

        # Reshape data into windows
        windowed_data = data[:num_windows * samples_per_window].reshape(num_windows, samples_per_window, -1)

        # Apply Blackman-Harris window
        window = blackmanharris(samples_per_window)[:, np.newaxis]
        windowed_data = windowed_data * window

        # Remove DC (mean) from each window
        windowed_data = windowed_data - np.mean(windowed_data, axis=1, keepdims=True)

        return windowed_data


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, data: np.ndarray, rate: float, **kwargs) -> Dict[str, np.ndarray]:
        pass


class FBEExtractor(FeatureExtractor):
    def __init__(self, freq_bands: List[Tuple[float, float]] = None):
        self.freq_bands = freq_bands or [(4, 8), (8, 20), (20, 48), (48, 100), (100, None)]

    def extract(self, data: np.ndarray, rate: float, **kwargs) -> Dict[str, np.ndarray]:
        # Ensure rate and window_size are scalars
        rate = float(np.asarray(rate).item())
        window_size = float(np.asarray(kwargs.get('window_size', 0.25)).item())
        samples_per_window = int(window_size * rate)
        num_windows, _, num_channels = data.shape

        freqs = np.fft.rfftfreq(samples_per_window, 1 / rate)
        fft_data = np.fft.rfft(data, axis=1)
        psd = np.abs(fft_data)**2 / (rate * samples_per_window)

        fbe_results = {}
        for i, (low, high) in enumerate(self.freq_bands):
            if high is None:
                high = rate / 2
            mask = (freqs >= low) & (freqs < high)
            fbe = np.mean(psd[:, mask, :], axis=1)
            fbe_results[f'fbe_{low}_{high}'] = fbe

        return fbe_results


class FeatureSaver(ABC):
    @abstractmethod
    def save(self, features: Dict[str, np.ndarray], file_path: str):
        pass


class NpzSaver(FeatureSaver):
    def save(self, features: Dict[str, np.ndarray], file_path: str):
        np.savez(file_path, **features)


class DAS:
    def __init__(self, file_loader: DASFileLoader = StandardH5Loader()):
        self.das_data: Optional[np.ndarray] = None
        self.rate: Optional[float] = None
        self.preprocessed_data: Optional[np.ndarray] = None
        self.extracted_features: Dict[str, np.ndarray] = {}
        self.file_loader = file_loader
        self.preprocessor = Preprocessor()

    def load_raw_data(self, file_path: str):
        self.das_data, self.rate = self.file_loader.load(file_path)
        # Ensure rate is a scalar
        self.rate = float(np.asarray(self.rate).item())

    def preprocess(self, window_size: float = 0.25):
        if self.das_data is None or self.rate is None:
            raise ValueError("Raw data not loaded. Call load_raw_data() first.")
        self.preprocessor.window_size = window_size
        self.preprocessed_data = self.preprocessor.preprocess(self.das_data, self.rate)

    def extract_features(self, extractor: FeatureExtractor, **kwargs):
        if self.preprocessed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")

        features = extractor.extract(self.preprocessed_data, self.rate, **kwargs)
        self.extracted_features.update(features)

    def save_features(self, file_path: str, saver: FeatureSaver = NpzSaver()):
        if not self.extracted_features:
            raise ValueError("No features extracted. Call extract_features() first.")

        saver.save(self.extracted_features, file_path)

    def plot_features(self, feature_names: List[str], channel_start: int = 0, channel_end: Optional[int] = None,
                      time_start: float = 0, time_end: Optional[float] = None, figsize: Tuple[int, int] = (16, 6)):
        if not feature_names:
            raise ValueError("No features specified for plotting.")

        for feature_name in feature_names:
            if feature_name not in self.extracted_features:
                raise ValueError(f"Feature '{feature_name}' not found. Extract it first.")

        if channel_end is None:
            channel_end = next(iter(self.extracted_features.values())).shape[1]
        if time_end is None:
            time_end = self.preprocessed_data.shape[0] * self.preprocessor.window_size

        # Convert time indices to window indices
        time_start_idx = int(time_start / self.preprocessor.window_size)
        time_end_idx = int(time_end / self.preprocessor.window_size)

        # Create subplots
        n_features = len(feature_names)
        fig, axes = plt.subplots(1, n_features, figsize=figsize, sharey=True)
        if n_features == 1:
            axes = [axes]  # Make axes iterable for single subplot

        for ax, feature_name in zip(axes, feature_names):
            feature_data = self.extracted_features[feature_name]
            plot_data = feature_data[time_start_idx:time_end_idx, channel_start:channel_end]

            im = ax.imshow(plot_data.T, aspect='auto', origin='lower', cmap='viridis',
                           extent=[time_start, time_end, channel_start, channel_end])

            ax.set_title(f'{feature_name.capitalize()}')
            ax.set_xlabel('Time (s)')

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(feature_name.capitalize())

        # Set common y-label
        fig.text(0.04, 0.5, 'Channel', va='center', rotation='vertical')

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    path = '/Users/jamesramsay/Downloads/OneDrive_1_09-07-2024/0000000005_2024-07-03_09.30.30.84400.hdf5'
    # path = '/Users/jamesramsay/Downloads/ap-sensing-synthetic-2-mins-1000m.hdf5'
    window_size = 0.25  # 250 ms window

    das = DAS()
    das.load_raw_data(path)
    das.preprocess(window_size=window_size)

    fbe_extractor = FBEExtractor()
    das.extract_features(fbe_extractor, window_size=window_size)

    print("Features extracted successfully.")
    print("Available features:", list(das.extracted_features.keys()))
    for feature, data in das.extracted_features.items():
        print(f"{feature} shape:", data.shape)

    # Plot the FBE features
    das.plot_features(
        ['fbe_4_8', 'fbe_8_20', 'fbe_20_48', 'fbe_48_100'],
        channel_start=0,
        channel_end=4020,
        time_start=0,
        time_end=30,
        figsize=(16, 4)
    )
