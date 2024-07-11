import h5py

# Path to your H5 file
file_path = '/Users/jamesramsay/Downloads/OneDrive_1_09-07-2024/0000000005_2024-07-03_09.30.30.84400.hdf5'

# Open the H5 file
with h5py.File(file_path, 'r') as h5_file:
    # List all groups
    print("Keys: %s" % h5_file.keys())

    dataset_name = 'Metadata'
    if dataset_name in h5_file:
        # Get the data
        data = h5_file[dataset_name][:]
        print(f"Data in '{dataset_name}':")
        print(data)
    else:
        print(f"Dataset '{dataset_name}' not found in the H5 file.")
