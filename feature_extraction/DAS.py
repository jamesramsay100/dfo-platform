import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import dask
from dask.distributed import Client, progress
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

            pulse_rate = None
            possible_paths = ['DAQ/RepetitionFrequency', 'DAQ/PulseRate', 'ProcessingServer/DataRate']
            for path in possible_paths:
                if path in h5_file:
                    pulse_rate = h5_file[path][()]
                    break

            if pulse_rate is None:
                raise KeyError(f"Unable to find pulse rate in the HDF5 file. Tried paths: {possible_paths}")

        phase_s64 = np.cumsum(dphase_s16.astype(np.int64), axis=0)
        phase_f64 = phase_s64 * np.pi / 2 ** 15

        return phase_f64, pulse_rate


class Preprocessor:
    def __init__(self, window_size: float = 0.25):
        self.window_size = window_size

    def preprocess(self, data: np.ndarray, rate: float) -> np.ndarray:
        rate = float(np.asarray(rate).item())
        samples_per_window = int(self.window_size * rate)
        num_windows = data.shape[0] // samples_per_window

        windowed_data = data[:num_windows * samples_per_window].reshape(num_windows, samples_per_window, -1)

        window = np.blackman(samples_per_window)[:, np.newaxis]
        windowed_data = windowed_data * window

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
        rate = float(np.asarray(rate).item())
        window_size = float(np.asarray(kwargs.get('window_size', 0.25)).item())
        samples_per_window = int(window_size * rate)

        freqs = np.fft.rfftfreq(samples_per_window, 1 / rate)
        fft_data = np.fft.rfft(data, axis=1)
        psd = np.abs(fft_data) ** 2 / (rate * samples_per_window)

        fbe_results = {}
        for i, (low, high) in enumerate(self.freq_bands):
            if high is None:
                high = rate / 2
            mask = (freqs >= low) & (freqs < high)
            fbe_linear = np.mean(psd[:, mask, :], axis=1)
            fbe_db = 10 * np.log10(fbe_linear)  # Convert to dB
            fbe_results[f'fbe_{low}_{high}'] = fbe_db

        return fbe_results


class DAS:
    def __init__(self, file_loader: DASFileLoader = StandardH5Loader()):
        self.das_data: Optional[np.ndarray] = None
        self.rate: Optional[float] = None
        self.preprocessed_data: Optional[np.ndarray] = None
        self.extracted_features: Dict[str, np.ndarray] = {}
        self.file_loader = file_loader
        self.preprocessor = Preprocessor()
        self.extractor = FBEExtractor()

    def load_and_process_chunk(self, file_path: str, start_index: int, chunk_size: int, window_size: float):
        with h5py.File(file_path, 'r') as h5_file:
            chunk_data = h5_file['DAS'][start_index:start_index + chunk_size, :]
            pulse_rate = h5_file['DAQ/PulseRate'][()]

        # phase_s64 = np.cumsum(chunk_data.astype(np.int64), axis=0)
        phase_s64 = chunk_data.astype(np.int64)
        phase_f64 = phase_s64 * np.pi / 2 ** 15

        preprocessed = self.preprocessor.preprocess(phase_f64, pulse_rate)
        features = self.extractor.extract(preprocessed, pulse_rate, window_size=window_size)

        return features

    def process_file(self, file_path: str, chunk_size: int = 100000, window_size: float = 0.25):
        with h5py.File(file_path, 'r') as h5_file:
            total_traces = h5_file['DAS'].shape[0]

        lazy_results = []
        for start_index in range(0, total_traces, chunk_size):
            lazy_result = dask.delayed(self.load_and_process_chunk)(file_path, start_index, chunk_size, window_size)
            lazy_results.append(lazy_result)

        return lazy_results

    def plot_features(self, feature_names: List[str], results: List[Dict[str, np.ndarray]],
                      channel_start: int = 0, channel_end: Optional[int] = None,
                      time_start: float = 0, time_end: Optional[float] = None,
                      figsize: Tuple[int, int] = (16, 6)):
        # Combine results from all chunks
        combined_features = {name: np.concatenate([r[name] for r in results], axis=0) for name in feature_names}

        max_time, max_channels = next(iter(combined_features.values())).shape

        time_end_idx = int(time_end * 4) if time_end is not None else max_time  # Assuming 0.25s windows
        time_end_idx = min(time_end_idx, max_time)

        if channel_end is None:
            channel_end = max_channels
        else:
            channel_end = min(channel_end, max_channels)

        channel_start = max(0, min(channel_start, channel_end - 1))

        n_features = len(feature_names)
        fig, axes = plt.subplots(1, n_features, figsize=figsize, sharey=True)
        if n_features == 1:
            axes = [axes]

        for ax, feature_name in zip(axes, feature_names):
            feature_data = combined_features[feature_name]
            plot_data = feature_data[int(time_start * 4):time_end_idx, channel_start:channel_end]

            im = ax.imshow(plot_data.T, aspect='auto', origin='lower', cmap='viridis',
                           extent=[time_start, time_end_idx / 4, channel_start, channel_end])

            ax.set_title(f'{feature_name.capitalize()}')
            ax.set_xlabel('Time (s)')

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(feature_name.capitalize())

        fig.text(0.04, 0.5, 'Channel', va='center', rotation='vertical')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # file_path = '/Users/jamesramsay/Downloads/OneDrive_1_09-07-2024/0000000005_2024-07-03_09.30.30.84400.hdf5'
    file_path = '/Users/jamesramsay/Downloads/0000028222_2023-09-15_07.43.24.54298.hdf5'
    chunk_size = 2500  # Adjust based on your memory constraints
    window_size = 0.25  # 250 ms window

    das = DAS()

    # Set up the Dask client
    client = Client(threads_per_worker=2, n_workers=8)
    print(client)
    print(client.dashboard_link)

    # Process the file in parallel
    lazy_results = das.process_file(file_path, chunk_size, window_size)
    futures = client.persist(lazy_results)
    progress(futures)  # This will display a progress bar
    results = client.compute(futures, sync=True)

    print("Features extracted successfully.")
    print("Number of chunks processed:", len(results))

    # Plot the FBE features
    das.plot_features(
        ['fbe_4_8', 'fbe_8_20', 'fbe_20_48', 'fbe_48_100'],
        results,
        channel_start=0,
        channel_end=2000,
        time_start=0,
        time_end=300,
        figsize=(16, 4)
    )

    client.close()
