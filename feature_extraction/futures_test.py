import h5py
import numpy as np
import dask
from dask.distributed import Client, progress
import time
from scipy.fftpack import fft


def calculate_block_fft_average(file_path, start_index, block_size):
    with h5py.File(file_path, 'r') as f:
        das_data = f['DAS'][start_index:start_index + block_size, :]

    # Perform FFT on each channel (column)
    fft_data = np.abs(fft(das_data, axis=0))

    # Calculate the average of the FFT magnitudes
    return np.mean(fft_data)


def process_h5_file(file_path, block_size=1000):
    with h5py.File(file_path, 'r') as f:
        total_traces = f['DAS'].shape[0]

    lazy_results = []
    for start_index in range(0, total_traces, block_size):
        lazy_result = dask.delayed(calculate_block_fft_average)(file_path, start_index, block_size)
        lazy_results.append(lazy_result)

    futures = client.persist(lazy_results)
    progress(futures)  # This will display a progress bar
    results = client.compute(futures, sync=True)

    return results


# Example usage
if __name__ == "__main__":
    # file_path = '/Users/jamesramsay/Downloads/OneDrive_1_09-07-2024/0000000005_2024-07-03_09.30.30.84400.hdf5'
    file_path = '/Users/jamesramsay/Downloads/0000028222_2023-09-15_07.43.24.54298.hdf5'

    start_time = time.perf_counter()
    # Set up the Dask client
    client = Client(threads_per_worker=1, n_workers=8)
    print(client)  # This will display information about the client
    print(client.dashboard_link)

    block_averages = process_h5_file(file_path)
    print(f"Number of blocks processed: {len(block_averages)}")
    print(f"Average values for each block: {block_averages}")

    client.close()  # Don't forget to close the client when you're done

    end_time = time.perf_counter()
    print(f"Total time : {end_time - start_time}")